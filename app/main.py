from __future__ import annotations

import asyncio
import hashlib
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

from .classifier import ModelUnavailableError, WasteClassifier
from .database import (
    DB_PATH,
    add_scan_if_history_generation,
    clear_scans,
    delete_scan,
    get_learning_examples,
    get_learning_stats,
    get_history_clear_timestamp,
    get_history_generation,
    get_history_page,
    get_max_scan_id,
    get_scan_thumbnail_name,
    get_scan_thumbnail_state,
    initialize_database,
    list_all_scan_thumbnail_names,
    list_scan_ids_without_thumbnail,
    list_scan_recovery_tombstones,
    migrate_legacy_embedding_kind,
    record_feedback,
    recover_missing_scan_from_thumbnail,
    recover_scan_thumbnail_name,
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
RESERVED_CLIENT_IDS = {"anonymous", "legacy"}


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
MIGRATE_LEGACY_THUMBNAILS = _bool_env("MIGRATE_LEGACY_THUMBNAILS", False)
CLASSIFIER_IMAGE_MAX_DIMENSION = _positive_int_env("CLASSIFIER_IMAGE_MAX_DIMENSION", 1024)
THUMBNAIL_MAX_DIMENSION = _positive_int_env("THUMBNAIL_MAX_DIMENSION", 480)
THUMBNAIL_JPEG_QUALITY = _bounded_int_env("THUMBNAIL_JPEG_QUALITY", 80, 40, 95)
LEGACY_SCAN_THUMBNAIL_DIR = DB_PATH.parent / "scans"
_SCAN_THUMBNAIL_DIR_ENV = os.getenv("SCAN_THUMBNAIL_DIR", "").strip()
if _SCAN_THUMBNAIL_DIR_ENV:
    SCAN_THUMBNAIL_DIR = Path(_SCAN_THUMBNAIL_DIR_ENV).expanduser()
else:
    db_namespace_hash = hashlib.sha256(DB_PATH.name.encode("utf-8")).hexdigest()[:8]
    db_namespace = f"{DB_PATH.stem}-{db_namespace_hash}"
    SCAN_THUMBNAIL_DIR = LEGACY_SCAN_THUMBNAIL_DIR / db_namespace
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


def _thumbnail_directories() -> tuple[Path, ...]:
    """Runtime thumbnail access is isolated to this database namespace only."""
    return (SCAN_THUMBNAIL_DIR,)


def _thumbnail_path(filename: str) -> Path | None:
    if not filename or Path(filename).name != filename:
        return None
    path = SCAN_THUMBNAIL_DIR / filename
    try:
        return path if path.is_file() else None
    except OSError:
        return None


def _exact_scan_thumbnail_path(scan_id: int) -> Path | None:
    if scan_id <= 0:
        return None
    return _thumbnail_path(f"scan_{scan_id}.jpg")


def _recoverable_exact_scan_thumbnail_path(
    scan_id: int,
    tombstones: dict[int, str] | None = None,
    last_clear_at: str | None = None,
) -> Path | None:
    """Return scan_<id>.jpg only when delete/clear recovery barriers allow it."""
    path = _exact_scan_thumbnail_path(scan_id)
    if path is None:
        return None
    try:
        active_tombstones = (
            tombstones if tombstones is not None else list_scan_recovery_tombstones()
        )
        active_last_clear = (
            last_clear_at if tombstones is not None else get_history_clear_timestamp()
        )
    except sqlite3.Error:
        logger.warning(
            "Could not validate thumbnail recovery barrier for scan_id=%s",
            scan_id,
            exc_info=True,
        )
        return None
    return (
        path
        if _thumbnail_is_recoverable(
            scan_id, path, active_tombstones, active_last_clear
        )
        else None
    )


def _delete_thumbnail_files(filenames: list[str]) -> int:
    """Delete specific thumbnails only inside this database namespace.

    Legacy scan_<id>.jpg files are intentionally never deleted here because a
    legacy filename does not prove which database originally owned the image.
    Legacy cleanup must therefore be an explicit migration/maintenance action.
    """
    deleted = 0
    for filename in set(filenames):
        if not filename or Path(filename).name != filename:
            continue
        path = SCAN_THUMBNAIL_DIR / filename
        try:
            if not path.is_file() and not path.is_symlink():
                continue
            path.unlink()
            deleted += 1
        except OSError:
            logger.warning("Could not delete thumbnail: %s", path, exc_info=True)
    return deleted


def _delete_all_managed_thumbnail_files() -> int:
    """Delete managed thumbnails only inside this database namespace."""
    deleted = 0
    try:
        candidates = list(SCAN_THUMBNAIL_DIR.iterdir())
    except FileNotFoundError:
        return 0
    except OSError:
        logger.warning(
            "Could not inspect thumbnail directory: %s",
            SCAN_THUMBNAIL_DIR,
            exc_info=True,
        )
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


def _parse_deleted_at(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _thumbnail_is_recoverable(
    scan_id: int,
    path: Path,
    tombstones: dict[int, str],
    last_clear_at: str | None,
) -> bool:
    """Reject stale JPEGs left behind by explicit delete/clear operations.

    A tombstone is permanent for that scan id. Filesystem mtimes are not trusted
    to resurrect an explicitly deleted id because copied/restored files can carry
    timestamps newer than the delete itself.
    """
    if scan_id in tombstones:
        return False

    clear_barrier = _parse_deleted_at(last_clear_at or "")
    if clear_barrier is None:
        return True
    try:
        modified_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return False
    return modified_at > clear_barrier

def _recover_unlinked_thumbnail_rows() -> int:
    """Reconnect exact namespaced scan_<id>.jpg files to existing rows."""
    try:
        missing_ids = list_scan_ids_without_thumbnail()
        tombstones = list_scan_recovery_tombstones()
        last_clear_at = get_history_clear_timestamp()
    except sqlite3.Error:
        logger.exception("Could not inspect unlinked thumbnail rows")
        return 0

    recovered = 0
    for scan_id in missing_ids:
        filename = f"scan_{scan_id}.jpg"
        candidate = SCAN_THUMBNAIL_DIR / filename
        try:
            if not candidate.is_file():
                continue
        except OSError:
            logger.warning(
                "Could not inspect thumbnail candidate for scan_id=%s",
                scan_id,
                exc_info=True,
            )
            continue
        if not _thumbnail_is_recoverable(scan_id, candidate, tombstones, last_clear_at):
            continue
        try:
            if recover_scan_thumbnail_name(scan_id, filename):
                recovered += 1
        except sqlite3.Error:
            logger.warning(
                "Could not reconnect thumbnail for scan_id=%s",
                scan_id,
                exc_info=True,
            )

    if recovered:
        logger.info("Reconnected %s existing scan thumbnail(s) to history rows", recovered)
    return recovered

def _recover_missing_tail_scan_rows() -> int:
    """Restore a contiguous missing history tail from namespaced thumbnails.

    Explicitly deleted ids are permanently protected by DB tombstones.
    """
    try:
        max_scan_id = get_max_scan_id()
        tombstones = list_scan_recovery_tombstones()
        last_clear_at = get_history_clear_timestamp()
    except sqlite3.Error:
        logger.exception("Could not inspect history tail for thumbnail recovery")
        return 0

    files_by_id: dict[int, Path] = {}
    try:
        candidates = list(SCAN_THUMBNAIL_DIR.iterdir())
    except FileNotFoundError:
        candidates = []
    except OSError:
        logger.warning(
            "Could not inspect recovery directory: %s",
            SCAN_THUMBNAIL_DIR,
            exc_info=True,
        )
        return 0

    for path in candidates:
        name = path.name
        if not (name.startswith("scan_") and name.endswith(".jpg")):
            continue
        raw_id = name[len("scan_") : -len(".jpg")]
        if raw_id.isdigit():
            files_by_id[int(raw_id)] = path

    recovered = 0
    next_id = max_scan_id + 1
    max_file_id = max(files_by_id, default=0)
    while next_id <= max_file_id:
        path = files_by_id.get(next_id)
        if path is None:
            # A missing id is safe to cross only when the DB remembers that the user
            # explicitly deleted it. Any unexplained gap still ends tail recovery.
            if next_id in tombstones:
                next_id += 1
                continue
            break
        if not _thumbnail_is_recoverable(
            next_id, path, tombstones, last_clear_at
        ):
            # This id is blocked by an explicit delete or clear recovery barrier;
            # skip the stale JPEG and keep scanning the known contiguous tail.
            next_id += 1
            continue
        try:
            modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
            if recover_missing_scan_from_thumbnail(next_id, path.name, modified):
                recovered += 1
        except (OSError, sqlite3.Error):
            logger.warning(
                "Could not restore missing history row for scan_id=%s",
                next_id,
                exc_info=True,
            )
            break
        next_id += 1

    if recovered:
        logger.warning(
            "Recovered %s missing history row(s) from contiguous namespaced thumbnails; "
            "labels/confidence are intentionally marked unknown",
            recovered,
        )
    return recovered

def _migrate_legacy_thumbnails() -> int:
    """Optionally copy legacy shared thumbnails into this DB namespace.

    Disabled by default because a shared legacy scan_<id>.jpg filename cannot prove
    which database originally owned the image. Enable only for a known single-DB
    upgrade where the legacy directory is trusted.
    """
    if (
        not MIGRATE_LEGACY_THUMBNAILS
        or _SCAN_THUMBNAIL_DIR_ENV
        or SCAN_THUMBNAIL_DIR == LEGACY_SCAN_THUMBNAIL_DIR
    ):
        return 0

    try:
        referenced = list_all_scan_thumbnail_names()
        if not referenced or not LEGACY_SCAN_THUMBNAIL_DIR.exists():
            return 0
        SCAN_THUMBNAIL_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.exception("Could not prepare legacy thumbnail migration")
        return 0

    migrated = 0
    for filename in referenced:
        source = LEGACY_SCAN_THUMBNAIL_DIR / filename
        target = SCAN_THUMBNAIL_DIR / filename
        try:
            if target.is_file():
                continue
            if not source.is_file():
                continue
            # Copy rather than move: another legacy database may still refer to
            # the same shared path. New writes are isolated in the namespaced
            # folder, so leaving the old source is the safest upgrade path.
            target.write_bytes(source.read_bytes())
            migrated += 1
        except OSError:
            logger.warning("Could not migrate legacy thumbnail: %s", source, exc_info=True)
    if migrated:
        logger.info("Migrated %s legacy thumbnail(s) into %s", migrated, SCAN_THUMBNAIL_DIR)
    return migrated


def _cleanup_orphan_thumbnails() -> int:
    """Delete scanner thumbnails that are no longer referenced by this database."""
    try:
        referenced = set(list_all_scan_thumbnail_names())
        candidates = list(SCAN_THUMBNAIL_DIR.iterdir())
    except FileNotFoundError:
        return 0
    except (OSError, sqlite3.Error):
        logger.exception("Could not inspect scan thumbnails for orphan cleanup")
        return 0

    deleted = 0
    for path in candidates:
        name = path.name
        is_managed_thumbnail = (
            name.startswith("scan_")
            and name.endswith(".jpg")
            and name[len("scan_") : -len(".jpg")].isdigit()
        )
        if not is_managed_thumbnail or name in referenced:
            continue
        try:
            if path.is_file() or path.is_symlink():
                path.unlink()
                deleted += 1
        except OSError:
            logger.warning("Could not delete orphan thumbnail: %s", path, exc_info=True)

    if deleted:
        logger.info("Deleted %s orphan scan thumbnail(s)", deleted)
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
    migrated_embeddings = 0
    for source_kind, target_kind, dimension in classifier.legacy_embedding_migrations():
        migrated_embeddings += await run_in_threadpool(
            migrate_legacy_embedding_kind,
            source_kind,
            target_kind,
            dimension,
        )
    app_.state.legacy_embeddings_migrated = migrated_embeddings
    if migrated_embeddings:
        logger.info("Migrated %s compatible legacy feedback embedding(s)", migrated_embeddings)
    # Legacy files are consulted only during this explicit-reference migration.
    # All normal reads/recovery below are namespaced to the current database.
    app_.state.legacy_thumbnails_migrated = await run_in_threadpool(
        _migrate_legacy_thumbnails
    )
    app_.state.thumbnail_links_recovered = await run_in_threadpool(
        _recover_unlinked_thumbnail_rows
    )
    app_.state.missing_history_rows_recovered = await run_in_threadpool(
        _recover_missing_tail_scan_rows
    )
    # Do not delete unreferenced JPEGs automatically at startup. An orphan can be
    # the only surviving evidence of a row lost when a WAL file was not copied.
    # Managed thumbnails are deleted only by the explicit DELETE /api/history flow.
    app_.state.orphan_thumbnails_deleted = 0
    preload_task = asyncio.create_task(_preload_classifier()) if PRELOAD_MODEL else None
    app_.state.model_preload_task = preload_task
    yield
    if preload_task and not preload_task.done():
        preload_task.cancel()


app = FastAPI(
    title="Waste Scanner AI",
    description="Ứng dụng quét và phân loại rác bằng camera.",
    version="1.10.6",
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


@app.get("/api/health")
def health() -> JSONResponse:
    classifier_status = classifier.status
    state = classifier_status["state"]
    # A retry window expiring does not mean the classifier recovered; only a
    # successful model load can move it back to a healthy state. Keep reporting
    # degraded until that actually happens so external health checks do not flip
    # to green merely because another retry is allowed.
    degraded = state in {"error", "retry_available"}
    payload = {
        "status": "degraded" if degraded else "ok",
        "app": "waste-scanner-ai",
        "version": "1.10.6",
        "launch_token": os.getenv("WASTE_SCANNER_LAUNCH_TOKEN", ""),
        "pid": os.getpid(),
        "ready": state == "ready",
        "retry_available": state == "retry_available",
        "classifier": classifier_status,
        "learning": {
            "enabled": LEARNING_ENABLED,
            "mode": "shared-feedback-knn",
            "max_examples": LEARNING_MAX_EXAMPLES,
        },
    }
    return JSONResponse(status_code=503 if degraded else 200, content=payload)


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
            pil_image = ImageOps.exif_transpose(source).convert("RGB")
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
    stored_category = "Chưa xác định" if result.uncertain else rule.category
    thumbnail_stored = False
    embedding_stored = False
    history_saved = False
    scan_id: int | None = None
    async with history_mutation_gate:
        scan_id = await run_in_threadpool(
            add_scan_if_history_generation,
            rule.key,
            rule.display_name,
            stored_category,
            result.confidence,
            result.uncertain,
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
        scan_id = None
        thumbnail_stored = False
        embedding_stored = False

    memory_info = result.analysis.get("learning_memory", {})
    memory_applied = bool(memory_info.get("applied"))

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
        "predicted_category": rule.category,
        "confidence": result.confidence,
        "uncertain": result.uncertain,
        "alternatives": result.alternatives,
        "analysis": result.analysis,
        "learning": {
            "enabled": LEARNING_ENABLED,
            "feedback_available": embedding_stored,
            "memory_applied": memory_applied,
            "matched_examples": int(memory_info.get("matched_examples", 0)),
        },
        "history_thumbnail_available": thumbnail_stored,
        "notice": notice,
    }


@app.post("/api/feedback")
async def submit_feedback(
    payload: FeedbackPayload,
) -> dict[str, Any]:
    correct_key = payload.correct_key.strip()
    if correct_key not in RULE_BY_KEY:
        raise HTTPException(status_code=400, detail="Loại rác phản hồi không hợp lệ.")

    async with history_mutation_gate:
        saved = await run_in_threadpool(
            record_feedback,
            payload.scan_id,
            correct_key,
        )
    if saved is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy lần quét này trong lịch sử dùng chung.")

    compatible_kinds = classifier.learning_embedding_kinds()
    embedding_kind = saved.get("embedding_kind")
    feedback_kind = str(saved.get("feedback_kind") or "evaluation")
    has_embedding = bool(embedding_kind)
    learnable = bool(
        feedback_kind == "evaluation"
        and correct_key in LEARNABLE_RULE_KEYS
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
    if feedback_kind == "assignment":
        message = (
            f"Đã gán nhãn cho ảnh khôi phục: {target_name}. "
            "Bản ghi này không được tính là một lần AI đoán đúng/sai."
        )
    elif correct_key not in LEARNABLE_RULE_KEYS:
        message = (
            f"Đã lưu phản hồi: {target_name}. Nhóm này là nhãn dự phòng nên không được "
            "dùng làm mẫu học để tránh AI biến 'Rác còn lại' thành một lớp nhận dạng trực tiếp."
        )
    elif learnable and LEARNING_ENABLED:
        message = f"Đã ghi nhớ phản hồi: {target_name}. Các ảnh tương tự sau này có thể dùng mẫu này để điều chỉnh kết quả."
    elif not has_embedding:
        message = f"Đã lưu phản hồi: {target_name}, nhưng lần quét này không có embedding để dùng làm mẫu học."
    elif not learnable:
        message = f"Đã lưu phản hồi: {target_name}, nhưng embedding cũ không tương thích với model hiện tại nên không được dùng để điều chỉnh lần quét sau."
    else:
        message = f"Đã lưu phản hồi: {target_name}. Chức năng học từ phản hồi hiện đang tắt nên mẫu này chưa được dùng để điều chỉnh các lần quét sau."

    return {
        **saved,
        **target_rule.public_dict(),
        "learnable": learnable,
        "learning_enabled": LEARNING_ENABLED,
        "statistics": statistics,
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

    # A filename stored in SQLite is only a reference, not proof that the JPEG
    # still exists. Validate every linked thumbnail against the filesystem before
    # advertising it to the frontend. Unlinked rows may still use the guarded
    # exact-name recovery path.
    needs_recovery_check = any(not item.get("_thumbnail_name") for item in items)
    tombstones: dict[int, str] | None = None
    last_clear_at: str | None = None
    recovery_barriers_loaded = True
    if needs_recovery_check:
        try:
            tombstones = list_scan_recovery_tombstones()
            last_clear_at = get_history_clear_timestamp()
        except sqlite3.Error:
            logger.warning("Could not load thumbnail recovery barriers for history", exc_info=True)
            recovery_barriers_loaded = False

    for item in items:
        linked_thumbnail = item.pop("_thumbnail_name", None)
        if linked_thumbnail:
            item["thumbnail_available"] = _thumbnail_path(str(linked_thumbnail)) is not None
        else:
            item["thumbnail_available"] = bool(
                recovery_barriers_loaded
                and _recoverable_exact_scan_thumbnail_path(
                    int(item["id"]), tombstones, last_clear_at
                )
                is not None
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
def history_thumbnail(
    scan_id: int,
) -> FileResponse:
    exists, filename = get_scan_thumbnail_state(scan_id)
    if not exists:
        raise HTTPException(status_code=404, detail="Không tìm thấy lần quét này trong lịch sử.")
    fallback_thumbnail = filename is None
    if fallback_thumbnail:
        filename = f"scan_{scan_id}.jpg"
        path = _recoverable_exact_scan_thumbnail_path(scan_id)
    else:
        path = _thumbnail_path(filename)
    if path is None:
        raise HTTPException(status_code=404, detail="Ảnh xem trước không còn tồn tại.")
    if fallback_thumbnail:
        try:
            recover_scan_thumbnail_name(scan_id, filename)
        except sqlite3.Error:
            logger.warning("Could not relink thumbnail for scan_id=%s", scan_id, exc_info=True)
    return FileResponse(
        path,
        media_type="image/jpeg",
        headers={"Cache-Control": "private, no-store, max-age=0"},
    )


@app.delete("/api/history/{scan_id}")
async def delete_history_item(scan_id: int) -> dict[str, Any]:
    if scan_id <= 0:
        raise HTTPException(status_code=400, detail="ID lần quét không hợp lệ.")

    async with history_mutation_gate:
        deleted = await run_in_threadpool(delete_scan, scan_id)
        if deleted is None:
            raise HTTPException(status_code=404, detail="Không tìm thấy lần quét này trong lịch sử dùng chung.")

        filenames = [f"scan_{scan_id}.jpg"]
        linked_thumbnail = deleted.get("thumbnail_name")
        if linked_thumbnail:
            filenames.append(str(linked_thumbnail))
        thumbnails_deleted = await run_in_threadpool(_delete_thumbnail_files, filenames)

    return {
        "deleted": 1,
        "scan_id": scan_id,
        "thumbnails_deleted": thumbnails_deleted,
        "remaining": int(deleted["remaining"]),
        "sequence_reset": False,
        "next_scan_id": None,
    }


@app.delete("/api/history")
async def delete_history() -> dict[str, Any]:
    async with history_mutation_gate:
        deleted = await run_in_threadpool(clear_scans)
        # Runtime deletion is restricted to this database's thumbnail namespace.
        # The shared legacy root may contain scan_<id>.jpg files owned by another
        # database, so it must never be cleaned implicitly from this endpoint.
        thumbnails_deleted = await run_in_threadpool(_delete_all_managed_thumbnail_files)
    return {
        "deleted": deleted,
        "thumbnails_deleted": thumbnails_deleted,
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
