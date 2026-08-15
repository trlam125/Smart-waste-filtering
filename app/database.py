from __future__ import annotations

import os
import sqlite3
import unicodedata
from array import array
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .waste_rules import LEARNABLE_RULE_KEYS, RULE_BY_KEY, WASTE_RULES

DB_PATH = Path(os.getenv("DATABASE_PATH", "data/waste_scanner.db"))
MAX_HISTORY_PAGE_SIZE = 100
LEGACY_CLIENT_ID = "legacy"
RECYCLABLE_KEYS = (
    "plastic_rigid",
    "plastic_film",
    "paper",
    "cardboard",
    "metal",
    "glass",
)


def _nonnegative_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise RuntimeError(f"{name} phải là số nguyên >= 0, nhận được: {raw_value!r}") from exc
    if value < 0:
        raise RuntimeError(f"{name} phải >= 0, nhận được: {value}")
    return value


SQLITE_BUSY_TIMEOUT_MS = _nonnegative_int_env("SQLITE_BUSY_TIMEOUT_MS", 5000)
SQLITE_JOURNAL_MODE = os.getenv("SQLITE_JOURNAL_MODE", "DELETE").strip().upper()
if SQLITE_JOURNAL_MODE not in {"DELETE", "WAL"}:
    raise RuntimeError(
        "SQLITE_JOURNAL_MODE chỉ hỗ trợ DELETE hoặc WAL, nhận được: "
        f"{SQLITE_JOURNAL_MODE!r}"
    )


def _normalize_search_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    without_marks = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(without_marks.replace("đ", "d").split())


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=SQLITE_BUSY_TIMEOUT_MS / 1000)
    connection.row_factory = sqlite3.Row
    connection.execute(f"PRAGMA busy_timeout = {SQLITE_BUSY_TIMEOUT_MS}")
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _table_columns(connection: sqlite3.Connection, table_name: str) -> set[str]:
    return {str(row["name"]) for row in connection.execute(f"PRAGMA table_info({table_name})")}


def _migrate_scans_schema(connection: sqlite3.Connection) -> None:
    """Add v2 scan columns to older databases without discarding scan history."""
    columns = _table_columns(connection, "scans")
    additions = (
        ("display_name_search", "TEXT NOT NULL DEFAULT ''"),
        ("category_search", "TEXT NOT NULL DEFAULT ''"),
        ("thumbnail_name", "TEXT"),
        ("model_waste_key", "TEXT NOT NULL DEFAULT ''"),
        ("model_confidence", "REAL"),
        ("model_uncertain", "INTEGER"),
        ("effective_score", "REAL"),
        ("memory_applied", "INTEGER NOT NULL DEFAULT 0"),
        ("client_id", f"TEXT NOT NULL DEFAULT '{LEGACY_CLIENT_ID}'"),
    )
    for column_name, definition in additions:
        if column_name not in columns:
            connection.execute(
                f"ALTER TABLE scans ADD COLUMN {column_name} {definition}"
            )
            columns.add(column_name)

    # Older rows need normalized search values after the columns are introduced.
    # Also repair blank values left by an interrupted/partial migration.
    rows = connection.execute(
        """
        SELECT id, display_name, category
        FROM scans
        WHERE display_name_search = '' OR category_search = ''
        """
    ).fetchall()
    if rows:
        connection.executemany(
            """
            UPDATE scans
            SET display_name_search = ?, category_search = ?
            WHERE id = ?
            """,
            [
                (
                    _normalize_search_text(str(row["display_name"])),
                    _normalize_search_text(str(row["category"])),
                    int(row["id"]),
                )
                for row in rows
            ],
        )

    # v3 separates the original calibrated model prediction from the effective
    # result after feedback-memory reranking. Older rows cannot reconstruct that
    # lineage, so backfill them as model == effective without losing history.
    connection.execute(
        """
        UPDATE scans
        SET model_waste_key = CASE
                WHEN model_waste_key = '' THEN waste_key ELSE model_waste_key END,
            model_confidence = COALESCE(model_confidence, confidence),
            model_uncertain = COALESCE(model_uncertain, uncertain),
            effective_score = COALESCE(effective_score, confidence)
        WHERE model_waste_key = ''
           OR model_confidence IS NULL
           OR model_uncertain IS NULL
           OR effective_score IS NULL
        """
    )
    connection.execute(
        "UPDATE scans SET client_id = ? WHERE TRIM(client_id) = ''",
        (LEGACY_CLIENT_ID,),
    )


def _migrate_child_client_id(connection: sqlite3.Connection, table_name: str) -> None:
    """Add/backfill client_id on v2 child tables created by older app builds."""
    columns = _table_columns(connection, table_name)
    if "client_id" not in columns:
        connection.execute(
            f"ALTER TABLE {table_name} ADD COLUMN client_id "
            f"TEXT NOT NULL DEFAULT '{LEGACY_CLIENT_ID}'"
        )

    # Keep child ownership aligned with the parent scan. This also repairs blank
    # or legacy values left by partial migrations without changing scan history.
    connection.execute(
        f"""
        UPDATE {table_name}
        SET client_id = COALESCE(
            (SELECT scans.client_id FROM scans WHERE scans.id = {table_name}.scan_id),
            client_id
        )
        WHERE client_id = ? OR TRIM(client_id) = ''
        """,
        (LEGACY_CLIENT_ID,),
    )


def initialize_database() -> None:
    """Create the current schema and migrate compatible older databases in place."""
    with closing(_connect()) as connection:
        connection.execute(f"PRAGMA journal_mode = {SQLITE_JOURNAL_MODE}")
        connection.execute("PRAGMA synchronous = NORMAL")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                waste_key TEXT NOT NULL,
                display_name TEXT NOT NULL,
                display_name_search TEXT NOT NULL,
                category TEXT NOT NULL,
                category_search TEXT NOT NULL,
                confidence REAL NOT NULL,
                uncertain INTEGER NOT NULL DEFAULT 0,
                model_waste_key TEXT NOT NULL,
                model_confidence REAL NOT NULL,
                model_uncertain INTEGER NOT NULL DEFAULT 0,
                effective_score REAL NOT NULL,
                memory_applied INTEGER NOT NULL DEFAULT 0,
                client_id TEXT NOT NULL,
                thumbnail_name TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        _migrate_scans_schema(connection)
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS scan_embeddings (
                scan_id INTEGER PRIMARY KEY,
                client_id TEXT NOT NULL,
                embedding_kind TEXT NOT NULL,
                dimension INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(scan_id) REFERENCES scans(id) ON DELETE CASCADE
            )
            """
        )
        _migrate_child_client_id(connection, "scan_embeddings")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                scan_id INTEGER PRIMARY KEY,
                client_id TEXT NOT NULL,
                predicted_key TEXT NOT NULL,
                corrected_key TEXT NOT NULL,
                is_correct INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(scan_id) REFERENCES scans(id) ON DELETE CASCADE
            )
            """
        )
        _migrate_child_client_id(connection, "feedback")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS history_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT OR IGNORE INTO history_state (key, value) VALUES ('generation', '0')"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_scans_id ON scans(id DESC)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_embeddings_kind_scan "
            "ON scan_embeddings(embedding_kind, scan_id DESC)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_feedback_corrected_updated "
            "ON feedback(corrected_key, updated_at DESC, scan_id DESC)"
        )
        connection.commit()


def get_history_generation() -> int:
    with closing(_connect()) as connection:
        row = connection.execute(
            "SELECT value FROM history_state WHERE key = 'generation'"
        ).fetchone()
    try:
        return max(0, int(str(row["value"]))) if row is not None else 0
    except (TypeError, ValueError):
        return 0


def add_scan_if_history_generation(
    waste_key: str,
    display_name: str,
    category: str,
    confidence: float,
    uncertain: bool,
    model_waste_key: str,
    model_confidence: float,
    model_uncertain: bool,
    effective_score: float,
    memory_applied: bool,
    client_id: str,
    expected_generation: int,
) -> int | None:
    """Store a scan only if history was not cleared while inference was running."""
    created_at = datetime.now(timezone.utc).isoformat()
    with closing(_connect()) as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT value FROM history_state WHERE key = 'generation'"
        ).fetchone()
        try:
            current_generation = max(0, int(str(row["value"]))) if row is not None else 0
        except (TypeError, ValueError):
            current_generation = 0
        if current_generation != max(0, int(expected_generation)):
            connection.rollback()
            return None

        cursor = connection.execute(
            """
            INSERT INTO scans (
                waste_key, display_name, display_name_search, category, category_search,
                confidence, uncertain, model_waste_key, model_confidence, model_uncertain,
                effective_score, memory_applied, client_id, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                waste_key,
                display_name,
                _normalize_search_text(display_name),
                category,
                _normalize_search_text(category),
                float(confidence),
                int(uncertain),
                model_waste_key,
                float(model_confidence),
                int(model_uncertain),
                float(effective_score),
                int(memory_applied),
                client_id,
                created_at,
            ),
        )
        connection.commit()
        return int(cursor.lastrowid)


def set_scan_thumbnail_name(scan_id: int, client_id: str, thumbnail_name: str) -> bool:
    name = Path(thumbnail_name).name
    if not name or name != thumbnail_name or len(name) > 128:
        return False
    with closing(_connect()) as connection:
        cursor = connection.execute(
            "UPDATE scans SET thumbnail_name = ? WHERE id = ? AND client_id = ?",
            (name, scan_id, client_id),
        )
        connection.commit()
        return cursor.rowcount == 1


def get_scan_thumbnail_state(scan_id: int) -> tuple[bool, str | None]:
    with closing(_connect()) as connection:
        row = connection.execute(
            "SELECT thumbnail_name FROM scans WHERE id = ?", (scan_id,)
        ).fetchone()
    if row is None:
        return False, None
    name = str(row["thumbnail_name"] or "")
    if not name or Path(name).name != name:
        return True, None
    return True, name


def _serialize_embedding(embedding: tuple[float, ...] | list[float]) -> tuple[bytes, int]:
    values = array("f")
    for item in embedding:
        value = float(item)
        if not (-3.4028235e38 <= value <= 3.4028235e38):
            raise ValueError("Embedding contains a non-finite/out-of-range value.")
        values.append(value)
    if not values or len(values) > 4096:
        raise ValueError("Embedding dimension is invalid.")
    return values.tobytes(), len(values)


def _deserialize_embedding(blob: bytes, dimension: int) -> tuple[float, ...] | None:
    if dimension <= 0 or dimension > 4096 or len(blob) != dimension * 4:
        return None
    values = array("f")
    values.frombytes(blob)
    if len(values) != dimension:
        return None
    return tuple(float(value) for value in values)


def store_scan_embedding(
    scan_id: int,
    client_id: str,
    embedding_kind: str,
    embedding: tuple[float, ...] | list[float],
) -> bool:
    kind = embedding_kind.strip()
    if not kind or len(kind) > 32:
        return False
    try:
        blob, dimension = _serialize_embedding(embedding)
    except (TypeError, ValueError, OverflowError):
        return False

    created_at = datetime.now(timezone.utc).isoformat()
    with closing(_connect()) as connection:
        scan = connection.execute(
            "SELECT 1 FROM scans WHERE id = ? AND client_id = ?", (scan_id, client_id)
        ).fetchone()
        if scan is None:
            return False
        connection.execute(
            """
            INSERT INTO scan_embeddings (
                scan_id, client_id, embedding_kind, dimension, embedding, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(scan_id) DO UPDATE SET
                client_id = excluded.client_id,
                embedding_kind = excluded.embedding_kind,
                dimension = excluded.dimension,
                embedding = excluded.embedding,
                created_at = excluded.created_at
            """,
            (scan_id, client_id, kind, dimension, sqlite3.Binary(blob), created_at),
        )
        connection.commit()
    return True


def record_feedback(scan_id: int, corrected_key: str) -> dict[str, Any] | None:
    now = datetime.now(timezone.utc).isoformat()
    with closing(_connect()) as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT waste_key, client_id FROM scans WHERE id = ?", (scan_id,)
        ).fetchone()
        if row is None:
            connection.rollback()
            return None

        predicted_key = str(row["waste_key"])
        source_client_id = str(row["client_id"])
        is_correct = predicted_key == corrected_key
        connection.execute(
            """
            INSERT INTO feedback (
                scan_id, client_id, predicted_key, corrected_key, is_correct,
                created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(scan_id) DO UPDATE SET
                client_id = excluded.client_id,
                predicted_key = excluded.predicted_key,
                corrected_key = excluded.corrected_key,
                is_correct = excluded.is_correct,
                updated_at = excluded.updated_at
            """,
            (
                scan_id,
                source_client_id,
                predicted_key,
                corrected_key,
                int(is_correct),
                now,
                now,
            ),
        )
        embedding_row = connection.execute(
            "SELECT embedding_kind FROM scan_embeddings WHERE scan_id = ?", (scan_id,)
        ).fetchone()
        connection.commit()

    return {
        "scan_id": scan_id,
        "predicted_key": predicted_key,
        "corrected_key": corrected_key,
        "is_correct": is_correct,
        "embedding_kind": (
            str(embedding_row["embedding_kind"]) if embedding_row is not None else None
        ),
    }


def get_learning_examples(
    embedding_kind: str,
    limit: int = 500,
) -> list[dict[str, Any]]:
    """Return recent feedback examples with a per-class cap to protect rare classes."""
    learnable_keys = tuple(sorted(LEARNABLE_RULE_KEYS))
    if not learnable_keys:
        return []
    # The memory budget must be large enough to reserve at least one slot per class.
    # Clamp defensively here even though app.learning validates the environment value.
    safe_limit = max(len(learnable_keys), min(int(limit), 5000))

    # A global LIMIT lets frequent classes evict rare classes from memory. Reserve an
    # approximately equal quota for every class so every learnable class can retain
    # recent examples even if common classes dominate.
    base_quota, remainder = divmod(safe_limit, len(learnable_keys))
    rows: list[sqlite3.Row] = []
    with closing(_connect()) as connection:
        for index, corrected_key in enumerate(learnable_keys):
            class_quota = base_quota + (1 if index < remainder else 0)
            if class_quota <= 0:
                continue
            rows.extend(
                connection.execute(
                    """
                    SELECT feedback.scan_id, feedback.corrected_key, feedback.is_correct,
                           scan_embeddings.embedding_kind, scan_embeddings.dimension,
                           scan_embeddings.embedding, feedback.updated_at
                    FROM feedback
                    JOIN scan_embeddings ON scan_embeddings.scan_id = feedback.scan_id
                    WHERE scan_embeddings.embedding_kind = ?
                      AND feedback.corrected_key = ?
                    ORDER BY feedback.updated_at DESC, feedback.scan_id DESC
                    LIMIT ?
                    """,
                    (embedding_kind, corrected_key, class_quota),
                ).fetchall()
            )

    rows.sort(key=lambda row: (str(row["updated_at"]), int(row["scan_id"])), reverse=True)

    examples: list[dict[str, Any]] = []
    for row in rows:
        vector = _deserialize_embedding(bytes(row["embedding"]), int(row["dimension"]))
        if vector is None:
            continue
        examples.append(
            {
                "scan_id": int(row["scan_id"]),
                "corrected_key": str(row["corrected_key"]),
                "is_correct": bool(row["is_correct"]),
                "embedding_kind": str(row["embedding_kind"]),
                "embedding": vector,
                "updated_at": str(row["updated_at"]),
            }
        )
    return examples


def get_learning_stats(
    compatible_embedding_kinds: tuple[str, ...] | list[str] | None = None,
) -> dict[str, int]:
    compatible = tuple(
        kind.strip() for kind in (compatible_embedding_kinds or ()) if kind.strip()
    )
    learnable_keys = tuple(sorted(LEARNABLE_RULE_KEYS))
    key_placeholders = ",".join("?" for _ in learnable_keys)

    with closing(_connect()) as connection:
        summary = connection.execute(
            """
            SELECT COUNT(*) AS feedback_total,
                   COALESCE(SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END), 0) AS confirmed,
                   COALESCE(SUM(CASE WHEN is_correct = 0 THEN 1 ELSE 0 END), 0) AS corrected
            FROM feedback
            """
        ).fetchone()
        stored = int(
            connection.execute(
                "SELECT COUNT(*) FROM feedback "
                "JOIN scan_embeddings ON scan_embeddings.scan_id = feedback.scan_id"
            ).fetchone()[0]
        )

        if compatible and learnable_keys:
            kind_placeholders = ",".join("?" for _ in compatible)
            learnable = int(
                connection.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM feedback
                    JOIN scan_embeddings ON scan_embeddings.scan_id = feedback.scan_id
                    WHERE scan_embeddings.embedding_kind IN ({kind_placeholders})
                      AND feedback.corrected_key IN ({key_placeholders})
                    """,
                    (*compatible, *learnable_keys),
                ).fetchone()[0]
            )
        elif learnable_keys:
            learnable = int(
                connection.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM feedback
                    JOIN scan_embeddings ON scan_embeddings.scan_id = feedback.scan_id
                    WHERE feedback.corrected_key IN ({key_placeholders})
                    """,
                    learnable_keys,
                ).fetchone()[0]
            )
        else:
            learnable = 0

    return {
        "feedback_total": int(summary["feedback_total"]),
        "confirmed": int(summary["confirmed"]),
        "corrected": int(summary["corrected"]),
        "stored_embedding_examples": stored,
        "learnable_examples": learnable,
        "incompatible_examples": max(0, stored - learnable),
    }


def _matching_rule_keys_for_search(clean_query: str) -> tuple[str, ...]:
    if not clean_query:
        return ()
    matched: list[str] = []
    for rule in WASTE_RULES:
        searchable = " ".join(
            (
                _normalize_search_text(rule.display_name),
                _normalize_search_text(rule.category),
                _normalize_search_text(rule.bin_name),
            )
        )
        if clean_query in searchable:
            matched.append(rule.key)
    return tuple(matched)


def _history_filter(query: str) -> tuple[str, tuple[str, ...]]:
    clean_query = _normalize_search_text(query.strip())
    if not clean_query:
        return "", ()
    escaped = clean_query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    pattern = f"%{escaped}%"
    matched_keys = _matching_rule_keys_for_search(clean_query)
    rule_sql = ""
    rule_params: tuple[str, ...] = ()
    if matched_keys:
        placeholders = ", ".join("?" for _ in matched_keys)
        rule_sql = (
            f" OR scans.waste_key IN ({placeholders})"
            f" OR feedback.corrected_key IN ({placeholders})"
        )
        rule_params = (*matched_keys, *matched_keys)
    return (
        " AND (scans.display_name_search LIKE ? ESCAPE '\\' "
        "OR scans.category_search LIKE ? ESCAPE '\\'"
        f"{rule_sql})",
        (pattern, pattern, *rule_params),
    )


def _list_scans_from_connection(
    connection: sqlite3.Connection,
    limit: int = 20,
    before_id: int | None = None,
    query: str = "",
) -> tuple[list[dict[str, Any]], bool]:
    safe_limit = max(1, min(limit, MAX_HISTORY_PAGE_SIZE))
    filter_sql, filter_params = _history_filter(query)
    cursor_sql = ""
    cursor_params: tuple[int, ...] = ()
    if before_id is not None:
        cursor_sql = " AND scans.id < ?"
        cursor_params = (before_id,)

    rows = connection.execute(
        f"""
        SELECT scans.id, scans.waste_key, scans.display_name, scans.category,
               scans.confidence, scans.uncertain, scans.model_waste_key,
               scans.model_confidence, scans.model_uncertain, scans.effective_score,
               scans.memory_applied, scans.created_at, scans.thumbnail_name,
               feedback.corrected_key, feedback.is_correct
        FROM scans
        LEFT JOIN feedback ON feedback.scan_id = scans.id
        WHERE 1 = 1{filter_sql}{cursor_sql}
        ORDER BY scans.id DESC
        LIMIT ?
        """,
        (*filter_params, *cursor_params, safe_limit + 1),
    ).fetchall()

    has_more = len(rows) > safe_limit
    items = [dict(row) for row in rows[:safe_limit]]
    for item in items:
        # Keep the scan-time effective prediction intact for audit/debugging.
        # The public history identity below may be replaced by a later user
        # correction, but these fields always describe what the app actually
        # showed immediately after inference (including feedback-memory rerank).
        effective_key = str(item["waste_key"])
        effective_display_name = str(item["display_name"])
        effective_category = str(item["category"])
        effective_uncertain = bool(item["uncertain"])

        item["effective_waste_key"] = effective_key
        item["effective_display_name"] = effective_display_name
        item["effective_category"] = effective_category
        item["effective_uncertain"] = effective_uncertain

        item["uncertain"] = effective_uncertain
        item["model_uncertain"] = bool(item.get("model_uncertain"))
        item["memory_applied"] = bool(item.get("memory_applied"))
        item["model_confidence"] = float(
            item["model_confidence"]
            if item.get("model_confidence") is not None
            else item["confidence"]
        )
        item["effective_score"] = float(
            item["effective_score"]
            if item.get("effective_score") is not None
            else item["confidence"]
        )
        model_key = str(item.get("model_waste_key") or item["waste_key"])
        item["model_waste_key"] = model_key
        model_rule = RULE_BY_KEY.get(model_key)
        item["model_display_name"] = (
            model_rule.display_name if model_rule is not None else item["display_name"]
        )
        item["model_category"] = (
            model_rule.category if model_rule is not None else item["category"]
        )

        # Feedback is the user's canonical label for this historical item. Do
        # not mutate the stored scan prediction: resolve the visible history
        # identity at read time so the original AI/effective lineage remains
        # available in the fields above.
        corrected_key = str(item.get("corrected_key") or "")
        corrected_rule = RULE_BY_KEY.get(corrected_key)
        if corrected_rule is not None:
            item["waste_key"] = corrected_rule.key
            item["display_name"] = corrected_rule.display_name
            item["category"] = corrected_rule.category
            # A user-confirmed/corrected label is no longer an unresolved AI
            # result, even if the original effective prediction was uncertain.
            item["uncertain"] = False

        item["_thumbnail_name"] = item.pop("thumbnail_name", None)
        if item.get("is_correct") is not None:
            item["is_correct"] = bool(item["is_correct"])
    return items, has_more


def _count_scans_from_connection(connection: sqlite3.Connection, query: str = "") -> int:
    filter_sql, filter_params = _history_filter(query)
    row = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM scans
        LEFT JOIN feedback ON feedback.scan_id = scans.id
        WHERE 1 = 1{filter_sql}
        """,
        filter_params,
    ).fetchone()
    return int(row[0])


def _scan_statistics_from_connection(
    connection: sqlite3.Connection,
    query: str = "",
) -> dict[str, int | float]:
    filter_sql, filter_params = _history_filter(query)
    placeholders = ",".join("?" for _ in RECYCLABLE_KEYS)
    row = connection.execute(
        f"""
        SELECT COUNT(*) AS total,
               COALESCE(SUM(
                   CASE
                       WHEN COALESCE(feedback.corrected_key, scans.waste_key) IN ({placeholders})
                            AND (feedback.scan_id IS NOT NULL OR scans.uncertain = 0)
                       THEN 1 ELSE 0
                   END
               ), 0) AS recycled,
               COALESCE(AVG(COALESCE(scans.model_confidence, scans.confidence)), 0.0) AS average_confidence
        FROM scans
        LEFT JOIN feedback ON feedback.scan_id = scans.id
        WHERE 1 = 1{filter_sql}
        """,
        (*RECYCLABLE_KEYS, *filter_params),
    ).fetchone()
    return {
        "total": int(row["total"]),
        "recycled": int(row["recycled"]),
        "average_confidence": round(float(row["average_confidence"]), 4),
    }


def get_history_page(
    limit: int = 20,
    before_id: int | None = None,
    query: str = "",
) -> dict[str, Any]:
    with closing(_connect()) as connection:
        connection.execute("BEGIN")
        try:
            matched_total = _count_scans_from_connection(connection, query)
            history_total = matched_total if not query.strip() else _count_scans_from_connection(connection)
            items, has_more = _list_scans_from_connection(connection, limit, before_id, query)
            statistics = _scan_statistics_from_connection(connection, query)
            connection.commit()
        except Exception:
            connection.rollback()
            raise
    return {
        "items": items,
        "matched_total": matched_total,
        "history_total": history_total,
        "has_more": has_more,
        "statistics": statistics,
    }


def delete_scan(scan_id: int) -> dict[str, Any] | None:
    if scan_id <= 0:
        return None
    with closing(_connect()) as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT id, thumbnail_name FROM scans WHERE id = ?", (scan_id,)
        ).fetchone()
        if row is None:
            connection.rollback()
            return None
        connection.execute("DELETE FROM scans WHERE id = ?", (scan_id,))
        remaining = int(connection.execute("SELECT COUNT(*) FROM scans").fetchone()[0])
        connection.commit()

    thumbnail_name = str(row["thumbnail_name"] or "")
    if thumbnail_name and Path(thumbnail_name).name != thumbnail_name:
        thumbnail_name = ""
    return {
        "id": int(row["id"]),
        "thumbnail_name": thumbnail_name or None,
        "remaining": remaining,
    }


def clear_scans() -> int:
    """Clear history and invalidate in-flight classifications without reusing IDs."""
    with closing(_connect()) as connection:
        connection.execute("BEGIN IMMEDIATE")
        count = int(connection.execute("SELECT COUNT(*) FROM scans").fetchone()[0])
        connection.execute(
            """
            INSERT INTO history_state (key, value)
            VALUES ('generation', '1')
            ON CONFLICT(key) DO UPDATE SET
                value = CAST(COALESCE(NULLIF(value, ''), '0') AS INTEGER) + 1
            """
        )
        connection.execute("DELETE FROM scans")
        connection.commit()
        return count
