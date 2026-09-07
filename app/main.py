from __future__ import annotations

import asyncio
import csv
import hmac
import io
import logging
import os
import shutil
import sqlite3
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from .paths import PROJECT_ROOT, collection_pending_dir, resolve_project_path

load_dotenv(PROJECT_ROOT / ".env", override=False)

from .classifier import ModelUnavailableError, WasteClassifier
from .detector import DetectorUnavailableError, WasteDetector
from .database import (
    add_scans_if_history_generation,
    clear_scans,
    delete_scan,
    get_history_generation,
    get_history_page,
    get_learning_examples,
    get_learning_stats,
    get_scan_thumbnail_state,
    initialize_database,
    record_feedback,
    set_scan_thumbnail_name,
    store_scan_embedding,
)
from .learning import (
    LEARNING_ENABLED,
    LEARNING_MAX_EXAMPLES,
    apply_feedback_memory,
)
from .waste_rules import LEARNABLE_RULE_KEYS, RULE_BY_KEY, WASTE_RULES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_IMAGE_BYTES = 8 * 1024 * 1024
STATIC_DIR = Path(__file__).resolve().parent / "static"
RESERVED_CLIENT_IDS = {"anonymous"}


class FeedbackPayload(BaseModel):
    scan_id: int = Field(ge=1)
    correct_key: str = Field(min_length=1, max_length=32)


def _bool_env(name: str, default: bool) -> bool:
    raw_value = os.getenv(name, "1" if default else "0").strip().lower()
    if raw_value in {"1", "true", "yes", "on"}:
        return True
    if raw_value in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be true/false; received: {raw_value!r}")


def _positive_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a positive integer; received: {raw_value!r}") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be greater than 0; received: {value}")
    return value


def _bounded_int_env(name: str, default: int, minimum: int, maximum: int) -> int:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer; received: {raw_value!r}") from exc
    if value < minimum or value > maximum:
        raise RuntimeError(
            f"{name} must be in the range {minimum}..{maximum}; received: {value}"
        )
    return value


MAX_IMAGE_PIXELS = _positive_int_env("MAX_IMAGE_PIXELS", 20_000_000)
PRELOAD_MODEL = _bool_env("PRELOAD_MODEL", True)
CLASSIFIER_IMAGE_MAX_DIMENSION = _positive_int_env("CLASSIFIER_IMAGE_MAX_DIMENSION", 1024)
CLASSIFIER_CROP_SOURCE_MAX_DIMENSION = _positive_int_env(
    "CLASSIFIER_CROP_SOURCE_MAX_DIMENSION", 1600
)
CLASSIFIER_MIN_OBJECT_SHORT_SIDE = _positive_int_env(
    "CLASSIFIER_MIN_OBJECT_SHORT_SIDE", 160
)
THUMBNAIL_MAX_DIMENSION = _positive_int_env("THUMBNAIL_MAX_DIMENSION", 480)
THUMBNAIL_JPEG_QUALITY = _bounded_int_env("THUMBNAIL_JPEG_QUALITY", 80, 40, 95)
HISTORY_DELETE_PASSWORD = os.getenv("HISTORY_DELETE_PASSWORD", "").strip()
def _project_managed_path(raw_value: str, default: Path) -> Path:
    return resolve_project_path(raw_value or default)


_SCAN_THUMBNAIL_DIR_ENV = os.getenv("SCAN_THUMBNAIL_DIR", "").strip()
SCAN_THUMBNAIL_DIR = _project_managed_path(
    _SCAN_THUMBNAIL_DIR_ENV, PROJECT_ROOT / "data" / "scans"
)

# Real-world dataset collection. A high-quality, EXIF-corrected JPEG is staged
# for each saved scan. It becomes a durable labeled training sample only after
# the user confirms or corrects the label through /api/feedback.
DATASET_COLLECTION_ENABLED = _bool_env("DATASET_COLLECTION_ENABLED", True)
COLLECTED_IMAGE_MAX_DIMENSION = _positive_int_env("COLLECTED_IMAGE_MAX_DIMENSION", 1600)
COLLECTED_JPEG_QUALITY = _bounded_int_env("COLLECTED_JPEG_QUALITY", 92, 70, 98)
MAX_COLLECTION_IMAGE_BYTES = _positive_int_env(
    "MAX_COLLECTION_IMAGE_BYTES", 16 * 1024 * 1024
)
_COLLECTED_DATA_DIR_ENV = os.getenv("COLLECTED_DATA_DIR", "").strip()
COLLECTED_DATA_DIR = _project_managed_path(
    _COLLECTED_DATA_DIR_ENV, PROJECT_ROOT / "data" / "collected"
)
COLLECTED_PENDING_DIR = collection_pending_dir(COLLECTED_DATA_DIR)
COLLECTED_METADATA_PATH = COLLECTED_DATA_DIR / "metadata.csv"


def _require_history_delete_password(candidate: str | None) -> None:
    # Do not expose whether the delete password is configured. Missing, blank,
    # and incorrect credentials intentionally share the same public response.
    supplied = (candidate or "").strip()
    if (
        not HISTORY_DELETE_PASSWORD
        or not supplied
        or not hmac.compare_digest(supplied, HISTORY_DELETE_PASSWORD)
    ):
        raise HTTPException(status_code=401, detail="Incorrect password.")
classifier = WasteClassifier()
detector = WasteDetector()
# Gate requests before they enter Starlette's shared threadpool. The classifier
# still keeps its internal threading lock as a second line of protection.
classification_gate = asyncio.Semaphore(1)
# Serialize scan persistence, feedback mutations and history deletion so a shared
# multi-device deployment cannot clear/delete a row halfway through saving it.
history_mutation_gate = asyncio.Lock()


def _normalize_client_id(client_id: str | None) -> str:
    """Validate and return the required browser/device history scope."""
    if client_id is None:
        raise HTTPException(status_code=400, detail="Missing X-Client-ID header.")

    value = client_id.strip()
    if not value:
        raise HTTPException(status_code=400, detail="X-Client-ID cannot be empty.")
    if value.lower() in RESERVED_CLIENT_IDS:
        raise HTTPException(status_code=400, detail="X-Client-ID uses a reserved value.")
    return value


def _save_scan_thumbnail(scan_id: int, client_id: str, image: Image.Image) -> bool:
    """Persist a small history preview without exposing it as a public static file."""
    filename = f"scan_{scan_id}.jpg"
    target = SCAN_THUMBNAIL_DIR / filename
    temporary = SCAN_THUMBNAIL_DIR / f".scan_{scan_id}_{os.getpid()}.tmp"
    try:
        SCAN_THUMBNAIL_DIR.mkdir(parents=True, exist_ok=True)
        preview_image = image.copy()
        preview_image.thumbnail(
            (THUMBNAIL_MAX_DIMENSION, THUMBNAIL_MAX_DIMENSION),
            Image.Resampling.LANCZOS,
        )
        preview_image.save(
            temporary,
            format="JPEG",
            quality=THUMBNAIL_JPEG_QUALITY,
            optimize=True,
        )
        temporary.replace(target)
        if not set_scan_thumbnail_name(scan_id, client_id, filename):
            target.unlink(missing_ok=True)
            return False
        return True
    except (OSError, sqlite3.Error):
        logger.exception("Could not save scan thumbnail for scan_id=%s", scan_id)
        temporary.unlink(missing_ok=True)
        target.unlink(missing_ok=True)
        return False


def _decode_supported_image(content: bytes, *, label: str) -> Image.Image:
    """Validate an uploaded image and return an EXIF-corrected RGB copy."""
    if not content:
        raise HTTPException(status_code=400, detail=f"{label} is empty.")
    try:
        with Image.open(io.BytesIO(content)) as source:
            image_format = (source.format or "").upper()
            if image_format not in {"JPEG", "PNG", "WEBP"}:
                raise HTTPException(
                    status_code=415,
                    detail=f"{label} only supports JPEG, PNG, or WebP.",
                )
            width, height = source.size
            if width <= 0 or height <= 0:
                raise HTTPException(status_code=400, detail=f"{label} has invalid dimensions.")
            if width * height > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"{label} has an excessively large resolution. "
                        f"The limit is {MAX_IMAGE_PIXELS:,} pixel."
                    ),
                )
            source.verify()

        with Image.open(io.BytesIO(content)) as source:
            return ImageOps.exif_transpose(source).convert("RGB")
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise HTTPException(status_code=400, detail=f"{label} is not a valid image.") from exc


def _save_pending_collection_image(scan_id: int, image: Image.Image) -> bool:
    """Stage a training-quality image until its label is confirmed by feedback."""
    if not DATASET_COLLECTION_ENABLED:
        return False
    target = COLLECTED_PENDING_DIR / f"scan_{scan_id}.jpg"
    temporary = COLLECTED_PENDING_DIR / f".scan_{scan_id}_{os.getpid()}.tmp"
    try:
        COLLECTED_PENDING_DIR.mkdir(parents=True, exist_ok=True)
        sample = image.copy()
        sample.thumbnail(
            (COLLECTED_IMAGE_MAX_DIMENSION, COLLECTED_IMAGE_MAX_DIMENSION),
            Image.Resampling.LANCZOS,
        )
        sample.save(
            temporary,
            format="JPEG",
            quality=COLLECTED_JPEG_QUALITY,
            optimize=True,
        )
        temporary.replace(target)
        return True
    except OSError:
        logger.exception("Could not stage dataset image for scan_id=%s", scan_id)
        temporary.unlink(missing_ok=True)
        return False


def _collection_sample_candidates(scan_id: int) -> list[Path]:
    filename = f"scan_{scan_id}.jpg"
    candidates: list[Path] = []
    for key in RULE_BY_KEY:
        path = COLLECTED_DATA_DIR / key / filename
        try:
            if path.is_file():
                candidates.append(path)
        except OSError:
            continue
    return candidates


def _update_collection_metadata(
    scan_id: int,
    *,
    label: str,
    predicted_key: str,
    is_correct: bool,
    image_path: Path,
) -> None:
    """Upsert one row in metadata.csv using an atomic replace."""
    fields = ("scan_id", "image_path", "label", "predicted_key", "is_correct", "updated_at")
    rows: dict[int, dict[str, str]] = {}
    if COLLECTED_METADATA_PATH.is_file():
        try:
            with COLLECTED_METADATA_PATH.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    try:
                        row_id = int(row.get("scan_id", ""))
                    except (TypeError, ValueError):
                        continue
                    rows[row_id] = {field: str(row.get(field, "")) for field in fields}
        except OSError:
            logger.warning("Could not read collection metadata; rebuilding it", exc_info=True)

    rows[scan_id] = {
        "scan_id": str(scan_id),
        "image_path": image_path.relative_to(COLLECTED_DATA_DIR).as_posix(),
        "label": label,
        "predicted_key": predicted_key,
        "is_correct": "1" if is_correct else "0",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    COLLECTED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    temporary = COLLECTED_DATA_DIR / f".metadata_{os.getpid()}.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row_id in sorted(rows):
            writer.writerow(rows[row_id])
    temporary.replace(COLLECTED_METADATA_PATH)


def _commit_collected_sample(
    scan_id: int,
    corrected_key: str,
    predicted_key: str,
    is_correct: bool,
) -> dict[str, Any]:
    """Promote a staged image into the confirmed real-world dataset.

    Colab commonly stages pending images under /content while the durable
    collection lives on the Google Drive mount under /content/drive. A direct
    Path.replace()/os.replace() cannot move a file across those filesystems and
    raises EXDEV ("Invalid cross-device link"). Copy into a temporary file in
    the destination directory first, atomically replace the final file there,
    then remove the source only after the destination has been committed.
    """
    if not DATASET_COLLECTION_ENABLED:
        return {
            "enabled": False,
            "saved": False,
            "image_path": None,
            "reason": "disabled",
        }
    if corrected_key not in RULE_BY_KEY:
        return {
            "enabled": True,
            "saved": False,
            "image_path": None,
            "reason": "invalid_label",
        }

    filename = f"scan_{scan_id}.jpg"
    pending = COLLECTED_PENDING_DIR / filename
    target_dir = COLLECTED_DATA_DIR / corrected_key
    target = target_dir / filename
    temporary = target_dir / f".scan_{scan_id}_{os.getpid()}.tmp"
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        existing = _collection_sample_candidates(scan_id)
        source: Path | None = pending if pending.is_file() else (existing[0] if existing else None)
        if source is None:
            return {
                "enabled": True,
                "saved": False,
                "image_path": None,
                "reason": "source_missing",
            }

        if source != target:
            temporary.unlink(missing_ok=True)
            shutil.copy2(source, temporary)
            temporary.replace(target)
            source.unlink(missing_ok=True)

        # Remove any stale copy left in a class directory after a label edit.
        for stale in _collection_sample_candidates(scan_id):
            if stale != target:
                stale.unlink(missing_ok=True)

        _update_collection_metadata(
            scan_id,
            label=corrected_key,
            predicted_key=predicted_key,
            is_correct=is_correct,
            image_path=target,
        )
        return {
            "enabled": True,
            "saved": True,
            "image_path": target.relative_to(COLLECTED_DATA_DIR).as_posix(),
            "reason": None,
        }
    except OSError:
        logger.exception("Could not promote collected dataset image for scan_id=%s", scan_id)
        temporary.unlink(missing_ok=True)
        return {
            "enabled": True,
            "saved": False,
            "image_path": None,
            "reason": "storage_error",
        }


def _delete_pending_collection_image(scan_id: int) -> int:
    path = COLLECTED_PENDING_DIR / f"scan_{scan_id}.jpg"
    try:
        if path.is_file() or path.is_symlink():
            path.unlink()
            return 1
    except OSError:
        logger.warning("Could not delete pending dataset image: %s", path, exc_info=True)
    return 0


def _delete_all_pending_collection_images() -> int:
    deleted = 0
    try:
        candidates = list(COLLECTED_PENDING_DIR.glob("scan_*.jpg"))
    except OSError:
        return 0
    for path in candidates:
        try:
            if path.is_file() or path.is_symlink():
                path.unlink()
                deleted += 1
        except OSError:
            logger.warning("Could not delete pending dataset image: %s", path, exc_info=True)
    return deleted


def _remove_collection_metadata(scan_id: int | None = None) -> int:
    """Remove one scan row, or all rows, from collection metadata."""
    if not COLLECTED_METADATA_PATH.is_file():
        return 0
    if scan_id is None:
        try:
            with COLLECTED_METADATA_PATH.open("r", encoding="utf-8", newline="") as handle:
                removed = sum(1 for _ in csv.DictReader(handle))
            COLLECTED_METADATA_PATH.unlink(missing_ok=True)
            return removed
        except OSError:
            logger.warning("Could not clear collection metadata", exc_info=True)
            return 0

    fields = ("scan_id", "image_path", "label", "predicted_key", "is_correct", "updated_at")
    kept: list[dict[str, str]] = []
    removed = 0
    try:
        with COLLECTED_METADATA_PATH.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    row_id = int(row.get("scan_id", ""))
                except (TypeError, ValueError):
                    kept.append({field: str(row.get(field, "")) for field in fields})
                    continue
                if row_id == scan_id:
                    removed += 1
                else:
                    kept.append({field: str(row.get(field, "")) for field in fields})

        temporary = COLLECTED_DATA_DIR / f".metadata_delete_{os.getpid()}.tmp"
        if kept:
            COLLECTED_DATA_DIR.mkdir(parents=True, exist_ok=True)
            with temporary.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(kept)
            temporary.replace(COLLECTED_METADATA_PATH)
        else:
            temporary.unlink(missing_ok=True)
            COLLECTED_METADATA_PATH.unlink(missing_ok=True)
        return removed
    except OSError:
        logger.warning("Could not update collection metadata during delete", exc_info=True)
        return 0


def _delete_collection_sample(scan_id: int) -> dict[str, int]:
    deleted_images = _delete_pending_collection_image(scan_id)
    for path in _collection_sample_candidates(scan_id):
        try:
            if path.is_file() or path.is_symlink():
                path.unlink()
                deleted_images += 1
        except OSError:
            logger.warning("Could not delete collected dataset image: %s", path, exc_info=True)
    return {
        "images_deleted": deleted_images,
        "metadata_rows_deleted": _remove_collection_metadata(scan_id),
    }


def _delete_all_collection_samples() -> dict[str, int]:
    deleted_images = _delete_all_pending_collection_images()
    for key in RULE_BY_KEY:
        directory = COLLECTED_DATA_DIR / key
        try:
            candidates = list(directory.glob("scan_*.jpg"))
        except OSError:
            continue
        for path in candidates:
            try:
                if path.is_file() or path.is_symlink():
                    path.unlink()
                    deleted_images += 1
            except OSError:
                logger.warning("Could not delete collected dataset image: %s", path, exc_info=True)
    return {
        "images_deleted": deleted_images,
        "metadata_rows_deleted": _remove_collection_metadata(None),
    }


def _thumbnail_path(filename: str) -> Path | None:
    if not filename or Path(filename).name != filename:
        return None
    path = SCAN_THUMBNAIL_DIR / filename
    try:
        return path if path.is_file() else None
    except OSError:
        return None


def _delete_thumbnail_files(filenames: list[str]) -> int:
    deleted = 0
    for filename in set(filenames):
        if not filename or Path(filename).name != filename:
            continue
        path = SCAN_THUMBNAIL_DIR / filename
        try:
            if path.is_file() or path.is_symlink():
                path.unlink()
                deleted += 1
        except OSError:
            logger.warning("Could not delete thumbnail: %s", path, exc_info=True)
    return deleted


def _delete_all_managed_thumbnail_files() -> int:
    deleted = 0
    try:
        candidates = list(SCAN_THUMBNAIL_DIR.iterdir())
    except FileNotFoundError:
        return 0
    except OSError:
        logger.warning("Could not inspect thumbnail directory: %s", SCAN_THUMBNAIL_DIR, exc_info=True)
        return 0

    for path in candidates:
        name = path.name
        managed = (
            name.startswith("scan_")
            and name.endswith(".jpg")
            and name[len("scan_") : -len(".jpg")].isdigit()
        )
        if not managed:
            continue
        try:
            if path.is_file() or path.is_symlink():
                path.unlink()
                deleted += 1
        except OSError:
            logger.warning("Could not delete thumbnail: %s", path, exc_info=True)
    return deleted


async def _preload_models() -> None:
    # Let the web UI become responsive first, then warm both AI stages in the
    # background so the first multi-object scan avoids model-load latency.
    await asyncio.sleep(0.35)
    try:
        await run_in_threadpool(detector.warmup)
    except DetectorUnavailableError as exc:
        logger.warning("Background detector preload failed: %s", exc)
    try:
        await run_in_threadpool(classifier.warmup)
    except ModelUnavailableError as exc:
        logger.warning("Background classifier preload failed: %s", exc)


@asynccontextmanager
async def lifespan(app_: FastAPI):
    initialize_database()
    preload_task = asyncio.create_task(_preload_models()) if PRELOAD_MODEL else None
    app_.state.model_preload_task = preload_task
    yield
    if preload_task and not preload_task.done():
        preload_task.cancel()


app = FastAPI(
    title="Waste Scanner AI",
    description="Camera-based waste scanning and classification application.",
    version="3.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def disable_ui_cache(request, call_next):
    response = await call_next(request)

    # The scanner UI is edited frequently while running locally. Force the
    # browser to fetch HTML/CSS/JS again on every navigation/reload so stale
    # assets cannot survive between server restarts.
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


def _service_status_payload() -> dict[str, Any]:
    classifier_status = classifier.status
    detector_status = detector.status
    states = (detector_status["state"], classifier_status["state"])
    ready = all(state == "ready" for state in states)
    retry_available = any(state == "retry_available" for state in states)
    if ready:
        status = "ok"
    elif any(state in {"error", "retry_available"} for state in states):
        status = "degraded"
    else:
        status = "starting"
    return {
        "status": status,
        "app": "waste-scanner-ai",
        "version": "3.0.0",
        "launch_token": os.getenv("WASTE_SCANNER_LAUNCH_TOKEN", ""),
        "pid": os.getpid(),
        "ready": ready,
        "retry_available": retry_available,
        "detector": detector_status,
        "classifier": classifier_status,
        "learning": {
            "enabled": LEARNING_ENABLED,
            "mode": "shared-feedback-knn-11class",
            "max_examples": LEARNING_MAX_EXAMPLES,
        },
        "dataset_collection": {
            "enabled": DATASET_COLLECTION_ENABLED,
            "directory": str(COLLECTED_DATA_DIR),
            "pending_directory": str(COLLECTED_PENDING_DIR),
        },
    }


@app.get("/api/health")
def health() -> JSONResponse:
    """Liveness endpoint: HTTP 200 means the FastAPI process is alive."""
    return JSONResponse(status_code=200, content=_service_status_payload())


@app.get("/api/ready")
def readiness() -> JSONResponse:
    """Readiness endpoint: HTTP 200 only after the AI model is usable."""
    payload = _service_status_payload()
    return JSONResponse(status_code=200 if payload["ready"] else 503, content=payload)


@app.get("/api/categories")
def categories() -> list[dict[str, str]]:
    return [rule.public_dict() for rule in WASTE_RULES]


def _expanded_bbox_normalized(
    bbox: tuple[float, float, float, float],
    padding: float,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    pad_x = width * max(0.0, padding)
    pad_y = height * max(0.0, padding)
    return (
        max(0.0, x1 - pad_x),
        max(0.0, y1 - pad_y),
        min(1.0, x2 + pad_x),
        min(1.0, y2 + pad_y),
    )


def _crop_normalized(
    image: Image.Image,
    bbox: tuple[float, float, float, float],
) -> Image.Image:
    width, height = image.size
    x1, y1, x2, y2 = bbox
    left = max(0, min(width - 1, int(round(x1 * width))))
    top = max(0, min(height - 1, int(round(y1 * height))))
    right = max(left + 1, min(width, int(round(x2 * width))))
    bottom = max(top + 1, min(height, int(round(y2 * height))))
    return image.crop((left, top, right, bottom))


def _bbox_pixel_size(
    image: Image.Image,
    bbox: tuple[float, float, float, float],
) -> tuple[int, int]:
    """Approximate detector-box pixel size on the classifier source image."""
    width, height = image.size
    x1, y1, x2, y2 = bbox
    box_width = max(0, int(round(max(0.0, x2 - x1) * width)))
    box_height = max(0, int(round(max(0.0, y2 - y1) * height)))
    return box_width, box_height


def _object_has_enough_pixels(
    image: Image.Image,
    bbox: tuple[float, float, float, float],
) -> bool:
    box_width, box_height = _bbox_pixel_size(image, bbox)
    return min(box_width, box_height) >= CLASSIFIER_MIN_OBJECT_SHORT_SIDE


def _same_aspect_ratio(
    first: Image.Image,
    second: Image.Image,
    tolerance: float = 0.005,
) -> bool:
    """Return True when normalized detector boxes safely map between two images."""
    first_width, first_height = first.size
    second_width, second_height = second.size
    if min(first_width, first_height, second_width, second_height) <= 0:
        return False
    first_ratio = first_width / first_height
    second_ratio = second_width / second_height
    return abs(first_ratio - second_ratio) / max(first_ratio, second_ratio) <= tolerance


def _public_bbox(bbox: tuple[float, float, float, float]) -> dict[str, float]:
    x1, y1, x2, y2 = bbox
    return {
        "x1": round(x1, 6),
        "y1": round(y1, 6),
        "x2": round(x2, 6),
        "y2": round(y2, 6),
        "width": round(max(0.0, x2 - x1), 6),
        "height": round(max(0.0, y2 - y1), 6),
    }


def _needs_paper_consistency_check(result: Any) -> bool:
    """Paper/cardboard are common background/material confusions; always verify them."""
    return getattr(result, "key", None) in {"paper", "cardboard"}


def _mark_result_uncertain(result: Any, reason: str) -> Any:
    """Preserve model scores while safely abstaining after conflicting crop views."""
    analysis = dict(getattr(result, "analysis", {}) or {})
    reasons = list(analysis.get("uncertainty_reasons", []))
    if reason not in reasons:
        reasons.append(reason)
    analysis["uncertainty_reasons"] = reasons
    return replace(result, uncertain=True, analysis=analysis)


def _object_notice(
    *,
    result: Any,
    memory_info: dict[str, Any],
    memory_applied: bool,
    is_demo: bool,
    persistence_enabled: bool,
    history_saved: bool,
    fallback_full_frame: bool = False,
) -> str:
    if result.uncertain:
        notice = (
            "Low-confidence result — the AI's best guess is shown. "
            "Capture a clearer, well-lit image for higher accuracy."
        )
    else:
        notice = "AI results are guidance only; always prioritize local waste-sorting regulations."

    if fallback_full_frame:
        notice += (
            " No object was clearly detected, so the entire image was classified."
            " For better accuracy, move the object closer to the camera."
        )
    if memory_applied:
        matched = int(memory_info.get("matched_examples", 0))
        notice += f" This result referenced {matched} confirmed samples in shared learning memory."
    if is_demo:
        notice += (
            " This is a demo image: the result is not saved to history, "
            "does not use feedback memory, and is not added to the dataset."
        )
    elif not persistence_enabled:
        notice += (
            " This result was requested without persistence, so it is not included in history, "
            "feedback memory, or the dataset."
        )
    elif not history_saved:
        notice += (
            " History was cleared while the AI was processing the image, so this result was not "
            "saved and cannot receive feedback; scan again if you want to keep the result."
        )
    return notice


@app.post("/api/classify")
async def classify_image(
    image: Annotated[UploadFile, File(description="Image captured from the camera")],
    classifier_image: Annotated[
        UploadFile | None,
        File(description="Optional high-quality image dedicated to the classifier"),
    ] = None,
    collection_image: Annotated[
        UploadFile | None,
        File(description="Optional high-quality image used only for dataset storage"),
    ] = None,
    persist: Annotated[bool, Form(description="Whether to persist history/feedback for this scan")] = True,
    source: Annotated[str, Form(description="Image source: user or demo")] = "user",
    client_id: Annotated[str | None, Header(alias="X-Client-ID", max_length=128)] = None,
) -> dict[str, Any]:
    history_scope = _normalize_client_id(client_id)
    request_source = source.strip().lower()[:32] or "user"
    is_demo = request_source == "demo"
    persistence_enabled = bool(persist) and not is_demo

    history_generation: int | None = None
    if persistence_enabled:
        async with history_mutation_gate:
            history_generation = await run_in_threadpool(get_history_generation)

    content = await image.read(MAX_IMAGE_BYTES + 1)
    if not content:
        raise HTTPException(status_code=400, detail="Image is empty.")
    if len(content) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 8 MB limit.")

    primary_full_image = _decode_supported_image(content, label="Classification image")

    # Keep detector inference lightweight, but classify each normalized detector
    # box from a higher-resolution source so material texture/reflections are not
    # destroyed before the 224x224 classifier preprocessing.
    detector_image = primary_full_image.copy()
    detector_image.thumbnail(
        (CLASSIFIER_IMAGE_MAX_DIMENSION, CLASSIFIER_IMAGE_MAX_DIMENSION),
        Image.Resampling.LANCZOS,
    )
    classifier_source_image = primary_full_image.copy()
    classifier_source_image.thumbnail(
        (CLASSIFIER_CROP_SOURCE_MAX_DIMENSION, CLASSIFIER_CROP_SOURCE_MAX_DIMENSION),
        Image.Resampling.LANCZOS,
    )

    if classifier_image is not None:
        try:
            classifier_content = await classifier_image.read(MAX_COLLECTION_IMAGE_BYTES + 1)
            if len(classifier_content) > MAX_COLLECTION_IMAGE_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        "High-quality classifier image exceeds the limit "
                        f"{MAX_COLLECTION_IMAGE_BYTES // (1024 * 1024)} MB."
                    ),
                )
            high_quality_classifier_image = _decode_supported_image(
                classifier_content,
                label="High-quality classifier image",
            )
            if _same_aspect_ratio(primary_full_image, high_quality_classifier_image):
                classifier_source_image = high_quality_classifier_image.copy()
                classifier_source_image.thumbnail(
                    (
                        CLASSIFIER_CROP_SOURCE_MAX_DIMENSION,
                        CLASSIFIER_CROP_SOURCE_MAX_DIMENSION,
                    ),
                    Image.Resampling.LANCZOS,
                )
            else:
                logger.warning(
                    "Ignoring optional classifier image because its aspect ratio differs "
                    "from the detector image: primary=%s classifier=%s",
                    primary_full_image.size,
                    high_quality_classifier_image.size,
                )
        except HTTPException as exc:
            logger.warning("Ignoring optional classifier image: %s", exc.detail)

    collection_source_image = primary_full_image
    collection_source = "inference_upload"
    if persistence_enabled and collection_image is not None:
        try:
            collection_content = await collection_image.read(MAX_COLLECTION_IMAGE_BYTES + 1)
            if len(collection_content) > MAX_COLLECTION_IMAGE_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        "High-quality image exceeds the limit "
                        f"{MAX_COLLECTION_IMAGE_BYTES // (1024 * 1024)} MB."
                    ),
                )
            high_quality_collection_image = _decode_supported_image(
                collection_content,
                label="High-quality image for dataset storage",
            )
            if _same_aspect_ratio(primary_full_image, high_quality_collection_image):
                collection_source_image = high_quality_collection_image
                collection_source = "high_quality_upload"
            else:
                logger.warning(
                    "Ignoring optional collection image because its aspect ratio differs "
                    "from the detector image: primary=%s collection=%s",
                    primary_full_image.size,
                    high_quality_collection_image.size,
                )
        except HTTPException as exc:
            logger.warning("Ignoring optional collection image: %s", exc.detail)

    records: list[dict[str, Any]] = []
    detections: list[Any] = []
    fallback_full_frame = False
    skipped_small_detections = 0

    try:
        async with classification_gate:
            detections = await run_in_threadpool(detector.detect, detector_image)
            if not detections:
                # No YOLO detections — fall back to classifying the entire frame so
                # the user always gets a result instead of a dead-end message.
                fallback_full_frame = True
                detections = []

            eligible_detections = [
                detection
                for detection in detections
                if _object_has_enough_pixels(
                    classifier_source_image,
                    detection.bbox_normalized,
                )
            ]
            eligible_detection_count = len(eligible_detections)
            skipped_small_detections = len(detections) - eligible_detection_count
            if not eligible_detections:
                # Either no detections at all, or every box was too small.
                # Fall back to the full frame so the user always receives a result.
                fallback_full_frame = True
                # Synthesise a single full-frame "detection" with a sentinel bbox.
                from dataclasses import dataclass as _dc

                @_dc(frozen=True)
                class _FullFrameDetection:
                    confidence: float = 1.0
                    bbox_xyxy: tuple = (0.0, 0.0, 1.0, 1.0)
                    bbox_normalized: tuple = (0.0, 0.0, 1.0, 1.0)

                eligible_detections = [_FullFrameDetection()]

            for object_index, detection in enumerate(eligible_detections, start=1):
                bbox = detection.bbox_normalized
                crop_bbox = _expanded_bbox_normalized(bbox, detector.crop_padding)
                object_image = _crop_normalized(classifier_source_image, crop_bbox)
                detector_confidence = detection.confidence

                base_result = await run_in_threadpool(
                    classifier.classify,
                    object_image,
                    allow_framing_rescue=False,
                )

                crop_retry = {
                    "attempted": False,
                    "applied": False,
                    "reason": "not_needed",
                    "views": [],
                }
                if _needs_paper_consistency_check(base_result):
                    crop_retry["attempted"] = True
                    crop_retry["reason"] = "no_strong_consensus"
                    retry_views = [
                        (detector.crop_padding, crop_bbox, object_image, base_result),
                    ]
                    for retry_padding in (0.02, 0.16):
                        if abs(retry_padding - detector.crop_padding) < 1e-6:
                            continue
                        retry_bbox = _expanded_bbox_normalized(bbox, retry_padding)
                        retry_image = _crop_normalized(classifier_source_image, retry_bbox)
                        retry_result = await run_in_threadpool(
                            classifier.classify,
                            retry_image,
                            allow_framing_rescue=False,
                        )
                        retry_views.append(
                            (retry_padding, retry_bbox, retry_image, retry_result)
                        )

                    for padding_value, _, _, retry_result in retry_views:
                        crop_retry["views"].append(
                            {
                                "padding": round(float(padding_value), 4),
                                "key": retry_result.key,
                                "confidence": retry_result.confidence,
                                "uncertain": retry_result.uncertain,
                            }
                        )

                    # Paper/cardboard are common background confusions. Never turn an
                    # uncertain paper-like prediction into a confident one from a
                    # single alternate crop. Only accept a different class when two
                    # crop views agree and at least one of them passes all classifier
                    # uncertainty/OOD checks.
                    candidate_keys = {
                        retry_result.key
                        for _, _, _, retry_result in retry_views
                        if retry_result.key not in {"paper", "cardboard"}
                    }
                    selected_view = None
                    for candidate_key in candidate_keys:
                        agreeing = [
                            view for view in retry_views if view[3].key == candidate_key
                        ]
                        certain = [view for view in agreeing if not view[3].uncertain]
                        if len(agreeing) < 2 or not certain:
                            continue
                        best_view = max(certain, key=lambda view: view[3].confidence)
                        if selected_view is None or best_view[3].confidence > selected_view[3].confidence:
                            selected_view = best_view

                    if selected_view is not None:
                        selected_padding, crop_bbox, object_image, base_result = selected_view
                        crop_retry["applied"] = True
                        crop_retry["reason"] = "two_crop_views_agree_on_non_paper_class"
                        crop_retry["selected_padding"] = round(float(selected_padding), 4)
                        crop_retry["selected_key"] = base_result.key
                        crop_retry["selected_confidence"] = base_result.confidence
                    elif not base_result.uncertain:
                        # A confident paper/cardboard result is not trustworthy when
                        # both alternate framings independently classify it as
                        # something else. If those alternates disagree with each
                        # other, abstain instead of inventing an override class.
                        alternate_non_paper = [
                            view
                            for view in retry_views[1:]
                            if view[3].key not in {"paper", "cardboard"}
                        ]
                        if len(alternate_non_paper) == len(retry_views) - 1:
                            base_result = _mark_result_uncertain(
                                base_result,
                                "paper_crop_inconsistency",
                            )
                            crop_retry["reason"] = "alternate_crops_reject_paper_without_consensus"

                records.append(
                    {
                        "object_index": object_index,
                        "detected": not fallback_full_frame,
                        "detector_confidence": detector_confidence,
                        "bbox": bbox,
                        "crop_bbox": crop_bbox,
                        "object_image": object_image,
                        "base_result": base_result,
                        "result": base_result,
                        "crop_retry": crop_retry,
                        "scan_id": None,
                        "thumbnail_stored": False,
                        "embedding_stored": False,
                        "collection_pending_stored": False,
                    }
                )
    except DetectorUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    generation_still_current = False
    if persistence_enabled and history_generation is not None:
        generation_still_current = (
            await run_in_threadpool(get_history_generation) == history_generation
        )

    learning_examples_cache: dict[str, list[dict[str, Any]]] = {}
    if persistence_enabled and generation_still_current and LEARNING_ENABLED:
        for record in records:
            result = record["result"]
            if not result.embedding or not result.embedding_kind:
                continue
            examples = learning_examples_cache.get(result.embedding_kind)
            if examples is None:
                examples = await run_in_threadpool(
                    get_learning_examples,
                    result.embedding_kind,
                    LEARNING_MAX_EXAMPLES,
                )
                learning_examples_cache[result.embedding_kind] = examples
            record["result"] = await run_in_threadpool(
                apply_feedback_memory,
                result,
                embedding=result.embedding,
                embedding_kind=result.embedding_kind,
                examples=examples,
                unknown_threshold=classifier.unknown_threshold,
                uncertainty_margin=classifier.uncertainty_margin,
            )

    history_saved = False
    if persistence_enabled and history_generation is not None:
        async with history_mutation_gate:
            scan_rows: list[dict[str, Any]] = []
            for record in records:
                base_result = record["base_result"]
                result = record["result"]
                rule = RULE_BY_KEY[result.key]
                memory_info = result.analysis.get("learning_memory", {})
                scan_rows.append(
                    {
                        "waste_key": rule.key,
                        "display_name": rule.display_name,
                        "category": rule.category,
                        "confidence": result.confidence,
                        "uncertain": result.uncertain,
                        "model_waste_key": base_result.key,
                        "model_confidence": base_result.confidence,
                        "model_uncertain": base_result.uncertain,
                        "effective_score": result.confidence,
                        "memory_applied": bool(memory_info.get("applied")),
                    }
                )

            scan_ids = await run_in_threadpool(
                add_scans_if_history_generation,
                scan_rows,
                history_scope,
                history_generation,
            )
            if scan_ids is not None and len(scan_ids) == len(records):
                history_saved = True
                for record, scan_id in zip(records, scan_ids):
                    record["scan_id"] = scan_id
                    record["thumbnail_stored"] = await run_in_threadpool(
                        _save_scan_thumbnail,
                        scan_id,
                        history_scope,
                        record["object_image"],
                    )
                    if DATASET_COLLECTION_ENABLED:
                        collection_crop = _crop_normalized(
                            collection_source_image,
                            record["crop_bbox"],
                        )
                        record["collection_pending_stored"] = await run_in_threadpool(
                            _save_pending_collection_image,
                            scan_id,
                            collection_crop,
                        )
                    result = record["result"]
                    if LEARNING_ENABLED and result.embedding and result.embedding_kind:
                        try:
                            record["embedding_stored"] = await run_in_threadpool(
                                store_scan_embedding,
                                scan_id,
                                history_scope,
                                result.embedding_kind,
                                result.embedding,
                            )
                        except sqlite3.Error:
                            logger.exception("Could not store embedding for scan_id=%s", scan_id)
                            record["embedding_stored"] = False

    if persistence_enabled and not history_saved:
        # A full-history clear after inference invalidates feedback-memory fusion
        # for every object from this camera frame. Keep the visible frame internally
        # consistent by reverting all objects to their calibrated model prediction.
        for record in records:
            record["result"] = record["base_result"]
            record["scan_id"] = None
            record["thumbnail_stored"] = False
            record["embedding_stored"] = False
            record["collection_pending_stored"] = False

    objects: list[dict[str, Any]] = []
    for record in records:
        base_result = record["base_result"]
        result = record["result"]
        rule = RULE_BY_KEY[result.key]
        base_rule = RULE_BY_KEY[base_result.key]
        memory_info = result.analysis.get("learning_memory", {})
        memory_applied = bool(memory_info.get("applied"))
        public_rule = rule.public_dict()
        if result.uncertain:
            # Still show the best prediction — just flag it as low-confidence so
            # the user can make an informed decision rather than seeing nothing.
            uncertainty_reasons = result.analysis.get("uncertainty_reasons", [])
            public_rule = {
                **public_rule,
                "low_confidence": True,
                "uncertainty_reasons": uncertainty_reasons,
                "instruction": (
                    f"Low-confidence result ({rule.display_name}). "
                    "Verify with local waste-sorting regulations or try again with a clearer image."
                ),
            }

        object_payload = {
            "object_index": int(record["object_index"]),
            "object_id": f"object-{int(record['object_index'])}",
            "scan_id": record["scan_id"],
            "history_saved": history_saved,
            "source": request_source,
            "persistence_enabled": persistence_enabled,
            "detected": bool(record["detected"]),
            "fallback_full_frame": fallback_full_frame,
            "detector_confidence": record["detector_confidence"],
            "bbox": _public_bbox(record["bbox"]),
            "crop_bbox": _public_bbox(record["crop_bbox"]),
            **public_rule,
            "predicted_category": rule.category,
            "confidence": result.confidence,
            "uncertain": result.uncertain,
            "model_uncertain": base_result.uncertain,
            "alternatives": result.alternatives,
            "model_prediction": {
                "key": base_result.key,
                "display_name": base_rule.display_name,
                "category": base_rule.category,
                "confidence": base_result.confidence,
                "uncertain": base_result.uncertain,
                "alternatives": base_result.alternatives,
            },
            "effective_prediction": {
                "key": result.key,
                "display_name": rule.display_name,
                "category": rule.category,
                "score": result.confidence,
                "uncertain": result.uncertain,
                "alternatives": result.alternatives,
                "memory_applied": memory_applied,
            },
            "effective_score": result.confidence,
            "analysis": result.analysis,
            "crop_retry": record.get("crop_retry", {}),
            "learning": {
                "enabled": LEARNING_ENABLED,
                "feedback_available": bool(record["embedding_stored"]),
                "memory_applied": memory_applied,
                "matched_examples": int(memory_info.get("matched_examples", 0)),
            },
            "history_thumbnail_available": bool(record["thumbnail_stored"]),
            "dataset_collection": {
                "enabled": DATASET_COLLECTION_ENABLED,
                "pending": bool(record["collection_pending_stored"]),
                "saved": False,
                "source": collection_source if record["collection_pending_stored"] else None,
            },
            "notice": _object_notice(
                result=result,
                memory_info=memory_info,
                memory_applied=memory_applied,
                is_demo=is_demo,
                persistence_enabled=persistence_enabled,
                history_saved=history_saved,
                fallback_full_frame=fallback_full_frame,
            ),
        }
        objects.append(object_payload)

    primary = objects[0]
    return {
        **primary,
        "object_count": len(objects),
        "objects": objects,
        "scan_ids": [item["scan_id"] for item in objects if item.get("scan_id")],
        "history_saved": history_saved,
        "detector": {
            "detected_count": len(detections),
            "eligible_count": eligible_detection_count,
            "skipped_small_detections": skipped_small_detections,
            "fallback_full_frame": fallback_full_frame,
            "confidence_threshold": detector.confidence_threshold,
            "iou_threshold": detector.iou_threshold,
            "image_size": detector.image_size,
            "max_detections": detector.max_detections,
            "min_box_area_ratio": detector.min_box_area_ratio,
            "multi_pass_enabled": detector.multi_pass_enabled,
            "multi_pass_trigger_count": detector.multi_pass_trigger_count,
            "multi_pass_confidence": detector.multi_pass_confidence,
            "multi_pass_splits": detector.multi_pass_splits,
            "multi_pass_overlap": detector.multi_pass_overlap,
            "merge_iou_threshold": detector.merge_iou_threshold,
            "classifier_min_object_short_side": CLASSIFIER_MIN_OBJECT_SHORT_SIDE,
        },
    }

@app.post("/api/feedback")
async def submit_feedback(
    payload: FeedbackPayload,
) -> dict[str, Any]:
    correct_key = payload.correct_key.strip()
    if correct_key not in RULE_BY_KEY:
        raise HTTPException(status_code=400, detail="The feedback waste type is invalid.")

    collection_result: dict[str, Any] = {
        "enabled": DATASET_COLLECTION_ENABLED,
        "saved": False,
        "image_path": None,
        "reason": "not_attempted",
    }
    async with history_mutation_gate:
        saved = await run_in_threadpool(
            record_feedback,
            payload.scan_id,
            correct_key,
        )
        if saved is not None and DATASET_COLLECTION_ENABLED:
            collection_result = await run_in_threadpool(
                _commit_collected_sample,
                payload.scan_id,
                correct_key,
                str(saved["predicted_key"]),
                bool(saved["is_correct"]),
            )
    if saved is None:
        raise HTTPException(status_code=404, detail="This scan was not found in shared history.")

    compatible_kinds = classifier.learning_embedding_kinds()
    embedding_kind = saved.get("embedding_kind")
    has_embedding = bool(embedding_kind)
    learnable = bool(
        correct_key in LEARNABLE_RULE_KEYS
        and has_embedding
        and embedding_kind in compatible_kinds
    )
    try:
        statistics = await run_in_threadpool(
            get_learning_stats,
            compatible_kinds,
        )
    except sqlite3.Error:
        # record_feedback() commits before this best-effort statistics lookup.
        # Preserve the successful feedback response if the secondary read fails.
        logger.exception("Could not load learning statistics after feedback")
        statistics = {}
    target_rule = RULE_BY_KEY[correct_key]
    target_name = target_rule.display_name
    if correct_key not in LEARNABLE_RULE_KEYS:
        message = f"Feedback saved: {target_name}, but this label is not part of the current model schema."
    elif learnable and LEARNING_ENABLED:
        message = f"Feedback learned: {target_name}. Similar images can use this sample to adjust future results."
    elif not has_embedding:
        message = f"Feedback saved: {target_name}, but this scan has no embedding to use as a learning sample."
    elif not learnable:
        message = f"Feedback saved: {target_name}, but the old embedding is incompatible with the current model and will not be used to adjust future scans."
    else:
        message = f"Feedback saved: {target_name}. Feedback learning is currently disabled, so this sample is not being used to adjust future scans."

    if collection_result.get("saved"):
        message += " The image was saved to the real-world dataset for the next fine-tuning run."
    elif DATASET_COLLECTION_ENABLED:
        collection_reason = collection_result.get("reason")
        if collection_reason == "storage_error":
            message += (
                " Feedback was saved, but the image could not be written to the real-world dataset "
                "because of a storage error. Please try again later."
            )
        elif collection_reason == "source_missing":
            message += (
                " The source image could not be found for addition to the real-world dataset "
                "(this may be a history entry created before this update)."
            )

    return {
        **saved,
        **target_rule.public_dict(),
        "learnable": learnable,
        "learning_enabled": LEARNING_ENABLED,
        "statistics": statistics,
        "dataset_collection": collection_result,
        "message": message,
    }


@app.get("/api/learning/stats")
def learning_stats() -> dict[str, Any]:
    return {
        "enabled": LEARNING_ENABLED,
        "scope": "shared",
        **get_learning_stats(classifier.learning_embedding_kinds()),
    }


@app.get("/api/history")
def history(
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    before_id: Annotated[int | None, Query(ge=1)] = None,
    q: Annotated[str, Query(max_length=100)] = "",
) -> dict[str, Any]:
    page = get_history_page(limit, before_id, q)
    items = page["items"]
    for item in items:
        thumbnail_name = item.pop("_thumbnail_name", None)
        item["thumbnail_available"] = bool(
            thumbnail_name and _thumbnail_path(str(thumbnail_name)) is not None
        )
    next_cursor = items[-1]["id"] if items and page["has_more"] else None
    return {
        "items": items,
        "limit": limit,
        "before_id": before_id,
        "matched_total": page["matched_total"],
        "history_total": page["history_total"],
        "next_cursor": next_cursor,
        "has_more": page["has_more"],
        "statistics": page["statistics"],
        "scope": "shared",
    }


@app.get("/api/history/{scan_id}/thumbnail")
def history_thumbnail(scan_id: int) -> FileResponse:
    exists, filename = get_scan_thumbnail_state(scan_id)
    if not exists:
        raise HTTPException(status_code=404, detail="This scan was not found in history.")
    if not filename:
        raise HTTPException(status_code=404, detail="This scan has no preview image.")
    path = _thumbnail_path(filename)
    if path is None:
        raise HTTPException(status_code=404, detail="The preview image no longer exists.")
    return FileResponse(
        path,
        media_type="image/jpeg",
        headers={"Cache-Control": "private, no-store, max-age=0"},
    )


@app.delete("/api/history/{scan_id}")
async def delete_history_item(
    scan_id: int,
    delete_password: Annotated[str | None, Header(alias="X-Delete-Password")] = None,
) -> dict[str, Any]:
    if scan_id <= 0:
        raise HTTPException(status_code=400, detail="Invalid scan ID.")
    _require_history_delete_password(delete_password)

    async with history_mutation_gate:
        deleted = await run_in_threadpool(delete_scan, scan_id)
        if deleted is None:
            raise HTTPException(status_code=404, detail="This scan was not found in shared history.")

        filenames = [f"scan_{scan_id}.jpg"]
        linked_thumbnail = deleted.get("thumbnail_name")
        if linked_thumbnail:
            filenames.append(str(linked_thumbnail))
        thumbnails_deleted = await run_in_threadpool(_delete_thumbnail_files, filenames)
        collection_deleted = await run_in_threadpool(_delete_collection_sample, scan_id)

    return {
        "deleted": 1,
        "scan_id": scan_id,
        "thumbnails_deleted": thumbnails_deleted,
        "dataset_images_deleted": int(collection_deleted["images_deleted"]),
        "collection_metadata_rows_deleted": int(collection_deleted["metadata_rows_deleted"]),
        "remaining": int(deleted["remaining"]),
        "sequence_reset": False,
        "next_scan_id": None,
    }


@app.delete("/api/history")
async def delete_history(
    delete_password: Annotated[str | None, Header(alias="X-Delete-Password")] = None,
) -> dict[str, Any]:
    _require_history_delete_password(delete_password)
    async with history_mutation_gate:
        deleted = await run_in_threadpool(clear_scans)
        thumbnails_deleted = await run_in_threadpool(_delete_all_managed_thumbnail_files)
        collection_deleted = await run_in_threadpool(_delete_all_collection_samples)
    return {
        "deleted": deleted,
        "thumbnails_deleted": thumbnails_deleted,
        "dataset_images_deleted": int(collection_deleted["images_deleted"]),
        "collection_metadata_rows_deleted": int(collection_deleted["metadata_rows_deleted"]),
        "sequence_reset": False,
        "next_scan_id": None,
    }


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
