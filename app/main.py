from __future__ import annotations

import asyncio
import csv
import hmac
import io
import logging
import os
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=False)

from .classifier import ModelUnavailableError, WasteClassifier
from .database import (
    DB_PATH,
    add_scan_if_history_generation,
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
    raise RuntimeError(f"{name} phải là true/false, nhận được: {raw_value!r}")


def _positive_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise RuntimeError(f"{name} phải là số nguyên dương, nhận được: {raw_value!r}") from exc
    if value <= 0:
        raise RuntimeError(f"{name} phải lớn hơn 0, nhận được: {value}")
    return value


def _bounded_int_env(name: str, default: int, minimum: int, maximum: int) -> int:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise RuntimeError(f"{name} phải là số nguyên, nhận được: {raw_value!r}") from exc
    if value < minimum or value > maximum:
        raise RuntimeError(
            f"{name} phải nằm trong khoảng {minimum}..{maximum}, nhận được: {value}"
        )
    return value


MAX_IMAGE_PIXELS = _positive_int_env("MAX_IMAGE_PIXELS", 20_000_000)
PRELOAD_MODEL = _bool_env("PRELOAD_MODEL", True)
CLASSIFIER_IMAGE_MAX_DIMENSION = _positive_int_env("CLASSIFIER_IMAGE_MAX_DIMENSION", 1024)
THUMBNAIL_MAX_DIMENSION = _positive_int_env("THUMBNAIL_MAX_DIMENSION", 480)
THUMBNAIL_JPEG_QUALITY = _bounded_int_env("THUMBNAIL_JPEG_QUALITY", 80, 40, 95)
HISTORY_DELETE_PASSWORD = os.getenv("HISTORY_DELETE_PASSWORD", "").strip()
_SCAN_THUMBNAIL_DIR_ENV = os.getenv("SCAN_THUMBNAIL_DIR", "").strip()
SCAN_THUMBNAIL_DIR = (
    Path(_SCAN_THUMBNAIL_DIR_ENV).expanduser()
    if _SCAN_THUMBNAIL_DIR_ENV
    else DB_PATH.parent / "scans"
)

# Real-world dataset collection. A high-quality, EXIF-corrected JPEG is staged
# for each saved scan. It becomes a durable labeled training sample only after
# the user confirms or corrects the label through /api/feedback.
DATASET_COLLECTION_ENABLED = _bool_env("DATASET_COLLECTION_ENABLED", True)
COLLECTED_IMAGE_MAX_DIMENSION = _positive_int_env("COLLECTED_IMAGE_MAX_DIMENSION", 1600)
COLLECTED_JPEG_QUALITY = _bounded_int_env("COLLECTED_JPEG_QUALITY", 92, 70, 98)
_COLLECTED_DATA_DIR_ENV = os.getenv("COLLECTED_DATA_DIR", "").strip()
_default_collected_dir = DB_PATH.parent / "collected"
if not _default_collected_dir.is_absolute():
    _default_collected_dir = PROJECT_ROOT / _default_collected_dir
COLLECTED_DATA_DIR = (
    Path(_COLLECTED_DATA_DIR_ENV).expanduser()
    if _COLLECTED_DATA_DIR_ENV
    else _default_collected_dir
)
if not COLLECTED_DATA_DIR.is_absolute():
    COLLECTED_DATA_DIR = PROJECT_ROOT / COLLECTED_DATA_DIR
COLLECTED_PENDING_DIR = COLLECTED_DATA_DIR / "_pending"
COLLECTED_METADATA_PATH = COLLECTED_DATA_DIR / "metadata.csv"


def _require_history_delete_password(candidate: str | None) -> None:
    if not HISTORY_DELETE_PASSWORD:
        raise HTTPException(
            status_code=503,
            detail="Chức năng xóa đang bị khóa vì HISTORY_DELETE_PASSWORD chưa được cấu hình trong .env.",
        )
    supplied = (candidate or "").strip()
    if not supplied or not hmac.compare_digest(supplied, HISTORY_DELETE_PASSWORD):
        raise HTTPException(status_code=401, detail="Mật khẩu xóa không đúng.")
classifier = WasteClassifier()
# Gate requests before they enter Starlette's shared threadpool. The classifier
# still keeps its internal threading lock as a second line of protection.
classification_gate = asyncio.Semaphore(1)
# Serialize scan persistence, feedback mutations and history deletion so a shared
# multi-device deployment cannot clear/delete a row halfway through saving it.
history_mutation_gate = asyncio.Lock()


def _normalize_client_id(client_id: str | None) -> str:
    """Validate and return the required browser/device history scope."""
    if client_id is None:
        raise HTTPException(status_code=400, detail="Thiếu header X-Client-ID.")

    value = client_id.strip()
    if not value:
        raise HTTPException(status_code=400, detail="X-Client-ID không được để trống.")
    if value.lower() in RESERVED_CLIENT_IDS:
        raise HTTPException(status_code=400, detail="X-Client-ID sử dụng giá trị dành riêng.")
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
    """Promote a staged image into the confirmed real-world dataset."""
    if not DATASET_COLLECTION_ENABLED:
        return {"enabled": False, "saved": False, "image_path": None}
    if corrected_key not in RULE_BY_KEY:
        return {"enabled": True, "saved": False, "image_path": None}

    filename = f"scan_{scan_id}.jpg"
    pending = COLLECTED_PENDING_DIR / filename
    target_dir = COLLECTED_DATA_DIR / corrected_key
    target = target_dir / filename
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        existing = _collection_sample_candidates(scan_id)
        source: Path | None = pending if pending.is_file() else (existing[0] if existing else None)
        if source is None:
            return {"enabled": True, "saved": False, "image_path": None}

        if source != target:
            if target.exists():
                target.unlink()
            source.replace(target)

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
        }
    except OSError:
        logger.exception("Could not promote collected dataset image for scan_id=%s", scan_id)
        return {"enabled": True, "saved": False, "image_path": None}


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


async def _preload_classifier() -> None:
    # Let the web UI become responsive first, then warm the heavy model in the
    # background so the first scan usually avoids model-load latency.
    await asyncio.sleep(0.35)
    try:
        await run_in_threadpool(classifier.warmup)
    except ModelUnavailableError as exc:
        logger.warning("Background model preload failed: %s", exc)


@asynccontextmanager
async def lifespan(app_: FastAPI):
    initialize_database()
    preload_task = asyncio.create_task(_preload_classifier()) if PRELOAD_MODEL else None
    app_.state.model_preload_task = preload_task
    yield
    if preload_task and not preload_task.done():
        preload_task.cancel()


app = FastAPI(
    title="Waste Scanner AI",
    description="Ứng dụng quét và phân loại rác bằng camera.",
    version="2.0.0",
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
    state = classifier_status["state"]
    if state == "ready":
        status = "ok"
    elif state in {"error", "retry_available"}:
        status = "degraded"
    else:
        status = "starting"
    return {
        "status": status,
        "app": "waste-scanner-ai",
        "version": "2.0.0",
        "launch_token": os.getenv("WASTE_SCANNER_LAUNCH_TOKEN", ""),
        "pid": os.getpid(),
        "ready": state == "ready",
        "retry_available": state == "retry_available",
        "classifier": classifier_status,
        "learning": {
            "enabled": LEARNING_ENABLED,
            "mode": "shared-feedback-knn-11class",
            "max_examples": LEARNING_MAX_EXAMPLES,
        },
        "dataset_collection": {
            "enabled": DATASET_COLLECTION_ENABLED,
            "directory": str(COLLECTED_DATA_DIR),
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


@app.post("/api/classify")
async def classify_image(
    image: Annotated[UploadFile, File(description="Ảnh chụp từ camera")],
    client_id: Annotated[str | None, Header(alias="X-Client-ID", max_length=128)] = None,
) -> dict[str, Any]:
    history_scope = _normalize_client_id(client_id)
    # Register this request before reading/decoding the image. Once registered, a
    # later full-history clear invalidates this request's persistence even if the
    # clear happens before model inference actually begins.
    async with history_mutation_gate:
        history_generation = await run_in_threadpool(get_history_generation)

    content = await image.read(MAX_IMAGE_BYTES + 1)
    if not content:
        raise HTTPException(status_code=400, detail="Ảnh rỗng.")
    if len(content) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Ảnh vượt quá giới hạn 8 MB.")

    try:
        with Image.open(io.BytesIO(content)) as source:
            image_format = (source.format or "").upper()
            if image_format not in {"JPEG", "PNG", "WEBP"}:
                raise HTTPException(
                    status_code=415,
                    detail="Chỉ hỗ trợ ảnh JPEG, PNG hoặc WebP.",
                )

            width, height = source.size
            if width <= 0 or height <= 0:
                raise HTTPException(status_code=400, detail="Ảnh có kích thước không hợp lệ.")
            if width * height > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        "Ảnh có độ phân giải quá lớn. "
                        f"Giới hạn là {MAX_IMAGE_PIXELS:,} pixel."
                    ),
                )
            source.verify()

        with Image.open(io.BytesIO(content)) as source:
            collection_image = ImageOps.exif_transpose(source).convert("RGB")
            pil_image = collection_image.copy()
            pil_image.thumbnail(
                (CLASSIFIER_IMAGE_MAX_DIMENSION, CLASSIFIER_IMAGE_MAX_DIMENSION),
                Image.Resampling.LANCZOS,
            )
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise HTTPException(status_code=400, detail="Tệp tải lên không phải ảnh hợp lệ.") from exc

    try:
        async with classification_gate:
            base_result = await run_in_threadpool(classifier.classify, pil_image)
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    result = base_result
    generation_still_current = (
        await run_in_threadpool(get_history_generation) == history_generation
    )
    if (
        generation_still_current
        and LEARNING_ENABLED
        and result.embedding
        and result.embedding_kind
    ):
        examples = await run_in_threadpool(
            get_learning_examples,
            result.embedding_kind,
            LEARNING_MAX_EXAMPLES,
        )
        result = await run_in_threadpool(
            apply_feedback_memory,
            result,
            embedding=result.embedding,
            embedding_kind=result.embedding_kind,
            examples=examples,
            unknown_threshold=classifier.unknown_threshold,
            uncertainty_margin=classifier.uncertainty_margin,
        )

    rule = RULE_BY_KEY[result.key]
    base_rule = RULE_BY_KEY[base_result.key]
    memory_info = result.analysis.get("learning_memory", {})
    memory_applied = bool(memory_info.get("applied"))
    stored_category = "Chưa xác định" if result.uncertain else rule.category
    thumbnail_stored = False
    embedding_stored = False
    collection_pending_stored = False
    history_saved = False
    scan_id: int | None = None
    async with history_mutation_gate:
        scan_id = await run_in_threadpool(
            add_scan_if_history_generation,
            rule.key,
            rule.display_name,
            stored_category,
            base_result.confidence,
            result.uncertain,
            base_result.key,
            base_result.confidence,
            base_result.uncertain,
            result.confidence,
            memory_applied,
            history_scope,
            history_generation,
        )
        if scan_id is not None:
            history_saved = True
            thumbnail_stored = await run_in_threadpool(
                _save_scan_thumbnail,
                scan_id,
                history_scope,
                pil_image,
            )
            if DATASET_COLLECTION_ENABLED:
                collection_pending_stored = await run_in_threadpool(
                    _save_pending_collection_image,
                    scan_id,
                    collection_image,
                )
            if LEARNING_ENABLED and result.embedding and result.embedding_kind:
                try:
                    embedding_stored = await run_in_threadpool(
                        store_scan_embedding,
                        scan_id,
                        history_scope,
                        result.embedding_kind,
                        result.embedding,
                    )
                except sqlite3.Error:
                    # The scan itself has already been committed. Embedding persistence is
                    # optional metadata, so a transient DB failure must not make the client
                    # believe that the whole classification failed and retry the scan.
                    logger.exception("Could not store embedding for scan_id=%s", scan_id)
                    embedding_stored = False

    if not history_saved:
        # A clear occurred after this request started. Do not let feedback memory
        # from the pre-clear generation influence the visible post-clear result.
        result = base_result
        rule = RULE_BY_KEY[result.key]
        base_rule = rule
        memory_info = result.analysis.get("learning_memory", {})
        memory_applied = False
        scan_id = None
        thumbnail_stored = False
        embedding_stored = False
        collection_pending_stored = False

    if result.uncertain:
        notice = (
            "AI chưa đủ chắc chắn về kết quả này. Hãy chụp gần hơn, đủ sáng và chỉ để "
            "một vật thể trong khung trước khi quyết định cách phân loại."
        )
    else:
        notice = "Kết quả AI chỉ mang tính gợi ý; hãy ưu tiên quy định phân loại tại địa phương."

    if memory_applied:
        matched = int(memory_info.get("matched_examples", 0))
        notice += f" Kết quả này đã tham khảo {matched} mẫu bạn từng xác nhận/sửa trước đó."

    if not history_saved:
        notice += (
            " Lịch sử đã được xóa trong lúc AI xử lý ảnh nên kết quả này không được "
            "lưu và không thể phản hồi; hãy quét lại nếu bạn muốn lưu kết quả."
        )

    public_rule = rule.public_dict()
    if result.uncertain:
        public_rule = {
            **public_rule,
            "category": "Chưa xác định",
            "bin_name": "Chưa xác định",
            "instruction": "Chưa đưa ra hướng dẫn xử lý cho đến khi vật thể được nhận dạng chắc chắn hơn.",
        }

    return {
        "scan_id": scan_id,
        "history_saved": history_saved,
        **public_rule,
        # Keep the top-level prediction internally consistent: identity, score,
        # alternatives and uncertainty all describe the effective result after
        # feedback-memory fusion. The original calibrated model prediction is
        # preserved separately under ``model_prediction``.
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
        "learning": {
            "enabled": LEARNING_ENABLED,
            "feedback_available": embedding_stored,
            "memory_applied": memory_applied,
            "matched_examples": int(memory_info.get("matched_examples", 0)),
        },
        "history_thumbnail_available": thumbnail_stored,
        "dataset_collection": {
            "enabled": DATASET_COLLECTION_ENABLED,
            "pending": collection_pending_stored,
            "saved": False,
        },
        "notice": notice,
    }


@app.post("/api/feedback")
async def submit_feedback(
    payload: FeedbackPayload,
) -> dict[str, Any]:
    correct_key = payload.correct_key.strip()
    if correct_key not in RULE_BY_KEY:
        raise HTTPException(status_code=400, detail="Loại rác phản hồi không hợp lệ.")

    collection_result: dict[str, Any] = {
        "enabled": DATASET_COLLECTION_ENABLED,
        "saved": False,
        "image_path": None,
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
        raise HTTPException(status_code=404, detail="Không tìm thấy lần quét này trong lịch sử dùng chung.")

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
        message = f"Đã lưu phản hồi: {target_name}, nhưng nhãn này không thuộc schema model hiện tại."
    elif learnable and LEARNING_ENABLED:
        message = f"Đã ghi nhớ phản hồi: {target_name}. Các ảnh tương tự sau này có thể dùng mẫu này để điều chỉnh kết quả."
    elif not has_embedding:
        message = f"Đã lưu phản hồi: {target_name}, nhưng lần quét này không có embedding để dùng làm mẫu học."
    elif not learnable:
        message = f"Đã lưu phản hồi: {target_name}, nhưng embedding cũ không tương thích với model hiện tại nên không được dùng để điều chỉnh lần quét sau."
    else:
        message = f"Đã lưu phản hồi: {target_name}. Chức năng học từ phản hồi hiện đang tắt nên mẫu này chưa được dùng để điều chỉnh các lần quét sau."

    if collection_result.get("saved"):
        message += " Ảnh đã được lưu vào bộ dữ liệu thực tế để dùng cho lần fine-tune sau."
    elif DATASET_COLLECTION_ENABLED:
        message += " Không tìm thấy ảnh nguồn để thêm vào bộ dữ liệu thực tế (có thể đây là lịch sử tạo trước bản cập nhật này)."

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
        raise HTTPException(status_code=404, detail="Không tìm thấy lần quét này trong lịch sử.")
    if not filename:
        raise HTTPException(status_code=404, detail="Lần quét này không có ảnh xem trước.")
    path = _thumbnail_path(filename)
    if path is None:
        raise HTTPException(status_code=404, detail="Ảnh xem trước không còn tồn tại.")
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
        raise HTTPException(status_code=400, detail="ID lần quét không hợp lệ.")
    _require_history_delete_password(delete_password)

    async with history_mutation_gate:
        deleted = await run_in_threadpool(delete_scan, scan_id)
        if deleted is None:
            raise HTTPException(status_code=404, detail="Không tìm thấy lần quét này trong lịch sử dùng chung.")

        filenames = [f"scan_{scan_id}.jpg"]
        linked_thumbnail = deleted.get("thumbnail_name")
        if linked_thumbnail:
            filenames.append(str(linked_thumbnail))
        thumbnails_deleted = await run_in_threadpool(_delete_thumbnail_files, filenames)
        pending_dataset_images_deleted = await run_in_threadpool(
            _delete_pending_collection_image, scan_id
        )

    return {
        "deleted": 1,
        "scan_id": scan_id,
        "thumbnails_deleted": thumbnails_deleted,
        "pending_dataset_images_deleted": pending_dataset_images_deleted,
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
        pending_dataset_images_deleted = await run_in_threadpool(
            _delete_all_pending_collection_images
        )
    return {
        "deleted": deleted,
        "thumbnails_deleted": thumbnails_deleted,
        "pending_dataset_images_deleted": pending_dataset_images_deleted,
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
