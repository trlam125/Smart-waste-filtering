from __future__ import annotations

import os
import sqlite3
import unicodedata
from array import array
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .waste_rules import LEARNABLE_RULE_KEYS, WASTE_RULES

DB_PATH = Path(os.getenv("DATABASE_PATH", "data/waste_scanner.db"))
LEGACY_CLIENT_ID = "legacy"
MAX_HISTORY_PAGE_SIZE = 100


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
    """Normalize Vietnamese text for accent-insensitive, case-insensitive search."""
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    without_marks = "".join(char for char in decomposed if not unicodedata.combining(char))
    # Vietnamese đ/Đ does not decompose into an ASCII base character.
    return " ".join(without_marks.replace("đ", "d").split())


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(
        DB_PATH,
        timeout=SQLITE_BUSY_TIMEOUT_MS / 1000,
    )
    connection.row_factory = sqlite3.Row
    connection.execute(f"PRAGMA busy_timeout = {SQLITE_BUSY_TIMEOUT_MS}")
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database() -> None:
    with closing(_connect()) as connection:
        # DELETE is the default because this project is commonly copied/zipped as a
        # single folder. With WAL, recently committed rows can still live only in
        # waste_scanner.db-wal; copying just waste_scanner.db then silently loses
        # those rows while JPEG thumbnails remain on disk. WAL remains opt-in via
        # SQLITE_JOURNAL_MODE for deployments that need more read/write concurrency.
        connection.execute(f"PRAGMA journal_mode = {SQLITE_JOURNAL_MODE}")
        connection.execute("PRAGMA synchronous = NORMAL")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                waste_key TEXT NOT NULL,
                display_name TEXT NOT NULL,
                display_name_search TEXT NOT NULL DEFAULT '',
                category TEXT NOT NULL,
                category_search TEXT NOT NULL DEFAULT '',
                confidence REAL NOT NULL,
                uncertain INTEGER NOT NULL DEFAULT 0,
                recovered INTEGER NOT NULL DEFAULT 0,
                client_id TEXT NOT NULL DEFAULT 'legacy',
                thumbnail_name TEXT,
                created_at TEXT NOT NULL
            )
            """
        )

        # Migrate older databases without deleting existing history.
        columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(scans)").fetchall()
        }
        if "uncertain" not in columns:
            connection.execute(
                "ALTER TABLE scans ADD COLUMN uncertain INTEGER NOT NULL DEFAULT 0"
            )
        if "client_id" not in columns:
            connection.execute(
                "ALTER TABLE scans ADD COLUMN client_id TEXT NOT NULL DEFAULT 'legacy'"
            )
        if "recovered" not in columns:
            connection.execute(
                "ALTER TABLE scans ADD COLUMN recovered INTEGER NOT NULL DEFAULT 0"
            )
        if "display_name_search" not in columns:
            connection.execute(
                "ALTER TABLE scans ADD COLUMN display_name_search TEXT NOT NULL DEFAULT ''"
            )
        if "category_search" not in columns:
            connection.execute(
                "ALTER TABLE scans ADD COLUMN category_search TEXT NOT NULL DEFAULT ''"
            )
        if "thumbnail_name" not in columns:
            connection.execute("ALTER TABLE scans ADD COLUMN thumbnail_name TEXT")

        # Backfill normalized search fields for rows created by older versions.
        stale_rows = connection.execute(
            """
            SELECT id, display_name, category
            FROM scans
            WHERE display_name_search = '' OR category_search = ''
            """
        ).fetchall()
        if stale_rows:
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
                    for row in stale_rows
                ],
            )

        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_scans_client_id_id "
            "ON scans(client_id, id DESC)"
        )
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
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                scan_id INTEGER PRIMARY KEY,
                client_id TEXT NOT NULL,
                predicted_key TEXT NOT NULL,
                corrected_key TEXT NOT NULL,
                is_correct INTEGER NOT NULL,
                feedback_kind TEXT NOT NULL DEFAULT 'evaluation',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(scan_id) REFERENCES scans(id) ON DELETE CASCADE
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_embeddings_client_kind "
            "ON scan_embeddings(client_id, embedding_kind, scan_id DESC)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_embeddings_kind_scan "
            "ON scan_embeddings(embedding_kind, scan_id DESC)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_feedback_client_scan "
            "ON feedback(client_id, scan_id DESC)"
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS scan_recovery_tombstones (
                scan_id INTEGER PRIMARY KEY,
                deleted_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS history_recovery_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )

        connection.execute(
            """
            INSERT OR IGNORE INTO history_recovery_state (key, value)
            VALUES ('history_generation', '0')
            """
        )

        # Older builds reset sqlite_sequence after clearing history. If such a DB
        # is upgraded while empty, preserve every id remembered by tombstones so
        # a stale browser tab can never target a newly-created scan with that id.
        max_known_scan_id = int(
            connection.execute(
                """
                SELECT MAX(value)
                FROM (
                    SELECT COALESCE(MAX(id), 0) AS value FROM scans
                    UNION ALL
                    SELECT COALESCE(MAX(scan_id), 0) AS value FROM scan_recovery_tombstones
                )
                """
            ).fetchone()[0]
            or 0
        )
        sequence_row = connection.execute(
            "SELECT seq FROM sqlite_sequence WHERE name = 'scans'"
        ).fetchone()
        current_sequence = int(sequence_row["seq"]) if sequence_row is not None else 0
        if max_known_scan_id > current_sequence:
            if sequence_row is None:
                connection.execute(
                    "INSERT INTO sqlite_sequence(name, seq) VALUES ('scans', ?)",
                    (max_known_scan_id,),
                )
            else:
                connection.execute(
                    "UPDATE sqlite_sequence SET seq = ? WHERE name = 'scans'",
                    (max_known_scan_id,),
                )

        feedback_columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(feedback)").fetchall()
        }
        if "feedback_kind" not in feedback_columns:
            connection.execute(
                "ALTER TABLE feedback ADD COLUMN feedback_kind TEXT NOT NULL DEFAULT 'evaluation'"
            )
        # Older releases could record feedback on recovered JPEG-only rows as if it
        # were an AI correction. Mark those rows as assignments during migration.
        connection.execute(
            """
            UPDATE feedback
            SET feedback_kind = 'assignment'
            WHERE scan_id IN (SELECT id FROM scans WHERE recovered = 1)
            """
        )

        connection.commit()


def migrate_legacy_embedding_kind(
    source_kind: str,
    target_kind: str,
    expected_dimension: int,
) -> int:
    """Rename a legacy embedding schema only when its shape is known compatible."""
    source = source_kind.strip()
    target = target_kind.strip()
    if not source or not target or source == target or expected_dimension <= 0:
        return 0
    if len(target) > 32:
        raise ValueError("embedding_kind không được dài quá 32 ký tự")

    with closing(_connect()) as connection:
        cursor = connection.execute(
            """
            UPDATE scan_embeddings
            SET embedding_kind = ?
            WHERE embedding_kind = ? AND dimension = ?
            """,
            (target, source, int(expected_dimension)),
        )
        connection.commit()
        return int(cursor.rowcount)


def add_scan(
    waste_key: str,
    display_name: str,
    category: str,
    confidence: float,
    uncertain: bool = False,
    client_id: str = LEGACY_CLIENT_ID,
) -> int:
    created_at = datetime.now(timezone.utc).isoformat()
    with closing(_connect()) as connection:
        cursor = connection.execute(
            """
            INSERT INTO scans (
                waste_key,
                display_name,
                display_name_search,
                category,
                category_search,
                confidence,
                uncertain,
                client_id,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                waste_key,
                display_name,
                _normalize_search_text(display_name),
                category,
                _normalize_search_text(category),
                confidence,
                int(uncertain),
                client_id,
                created_at,
            ),
        )
        connection.commit()
        return int(cursor.lastrowid)



def get_history_generation() -> int:
    """Return the monotonically increasing full-history clear generation."""
    with closing(_connect()) as connection:
        row = connection.execute(
            "SELECT value FROM history_recovery_state WHERE key = 'history_generation'"
        ).fetchone()
    if row is None:
        return 0
    try:
        return max(0, int(str(row["value"])))
    except (TypeError, ValueError):
        # Treat a malformed legacy value as generation zero. The next clear repairs it.
        return 0


def add_scan_if_history_generation(
    waste_key: str,
    display_name: str,
    category: str,
    confidence: float,
    uncertain: bool,
    client_id: str,
    expected_generation: int,
) -> int | None:
    """Insert a scan only if no full-history clear happened since request start."""
    created_at = datetime.now(timezone.utc).isoformat()
    with closing(_connect()) as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT value FROM history_recovery_state WHERE key = 'history_generation'"
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
                confidence, uncertain, client_id, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                waste_key,
                display_name,
                _normalize_search_text(display_name),
                category,
                _normalize_search_text(category),
                confidence,
                int(uncertain),
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


def get_max_scan_id() -> int:
    """Return the largest persisted scan id, or 0 when history is empty."""
    with closing(_connect()) as connection:
        row = connection.execute("SELECT COALESCE(MAX(id), 0) FROM scans").fetchone()
    return int(row[0])


def list_scan_recovery_tombstones() -> dict[int, str]:
    """Return explicit-delete timestamps used to reject stale JPEG recovery."""
    with closing(_connect()) as connection:
        rows = connection.execute(
            "SELECT scan_id, deleted_at FROM scan_recovery_tombstones"
        ).fetchall()
    return {int(row["scan_id"]): str(row["deleted_at"]) for row in rows}


def get_history_clear_timestamp() -> str | None:
    """Return the most recent explicit full-history clear timestamp, if any."""
    with closing(_connect()) as connection:
        row = connection.execute(
            "SELECT value FROM history_recovery_state WHERE key = 'last_clear_at'"
        ).fetchone()
    return str(row["value"]) if row is not None else None


def recover_missing_scan_from_thumbnail(
    scan_id: int,
    thumbnail_name: str,
    created_at: str,
) -> bool:
    """Restore a missing tail history row from an exact scan_<id>.jpg file.

    The original AI label/confidence cannot be reconstructed from a JPEG alone, so
    restored rows are intentionally marked uncertain and shown as "Ảnh khôi phục".
    The user can then inspect the image and assign the correct label in the UI.
    """
    if scan_id <= 0:
        return False
    name = Path(thumbnail_name).name
    expected = f"scan_{scan_id}.jpg"
    if name != thumbnail_name or name != expected:
        return False

    timestamp = created_at.strip()
    if not timestamp:
        return False

    with closing(_connect()) as connection:
        cursor = connection.execute(
            """
            INSERT OR IGNORE INTO scans (
                id, waste_key, display_name, display_name_search, category,
                category_search, confidence, uncertain, recovered, client_id,
                thumbnail_name, created_at
            )
            VALUES (?, 'other', 'Ảnh khôi phục', ?, 'Chưa xác định', ?, 0.0, 1, 1, ?, ?, ?)
            """,
            (
                scan_id,
                _normalize_search_text("Ảnh khôi phục"),
                _normalize_search_text("Chưa xác định"),
                LEGACY_CLIENT_ID,
                name,
                timestamp,
            ),
        )
        connection.commit()
        return cursor.rowcount == 1


def list_scan_ids_without_thumbnail() -> list[int]:
    """Return scan ids whose DB row is not linked to a thumbnail yet."""
    with closing(_connect()) as connection:
        rows = connection.execute(
            """
            SELECT id
            FROM scans
            WHERE thumbnail_name IS NULL OR TRIM(thumbnail_name) = ''
            ORDER BY id
            """
        ).fetchall()
    return [int(row["id"]) for row in rows]


def recover_scan_thumbnail_name(scan_id: int, thumbnail_name: str) -> bool:
    """Link an exact legacy scan_<id>.jpg file to an existing unlinked row."""
    if scan_id <= 0:
        return False
    name = Path(thumbnail_name).name
    expected = f"scan_{scan_id}.jpg"
    if name != thumbnail_name or name != expected:
        return False
    with closing(_connect()) as connection:
        cursor = connection.execute(
            """
            UPDATE scans
            SET thumbnail_name = ?
            WHERE id = ? AND (thumbnail_name IS NULL OR TRIM(thumbnail_name) = '')
            """,
            (name, scan_id),
        )
        connection.commit()
        return cursor.rowcount == 1


def get_scan_thumbnail_state(scan_id: int) -> tuple[bool, str | None]:
    """Return whether a scan exists and its validated thumbnail filename."""
    with closing(_connect()) as connection:
        row = connection.execute(
            "SELECT thumbnail_name FROM scans WHERE id = ?",
            (scan_id,),
        ).fetchone()
    if row is None:
        return False, None
    if not row["thumbnail_name"]:
        return True, None
    name = str(row["thumbnail_name"])
    return True, name if Path(name).name == name else None


def get_scan_thumbnail_name(scan_id: int) -> str | None:
    """Return a validated thumbnail name for an existing shared-history scan."""
    _exists, thumbnail_name = get_scan_thumbnail_state(scan_id)
    return thumbnail_name


def list_scan_thumbnail_names() -> list[str]:
    """Return thumbnail names for the shared history."""
    with closing(_connect()) as connection:
        rows = connection.execute(
            "SELECT thumbnail_name FROM scans WHERE thumbnail_name IS NOT NULL"
        ).fetchall()
    names: list[str] = []
    for row in rows:
        name = str(row["thumbnail_name"] or "")
        if name and Path(name).name == name:
            names.append(name)
    return names


def list_all_scan_thumbnail_names() -> list[str]:
    """Return every valid thumbnail filename currently referenced by the database."""
    with closing(_connect()) as connection:
        rows = connection.execute(
            "SELECT thumbnail_name FROM scans WHERE thumbnail_name IS NOT NULL"
        ).fetchall()
    names: list[str] = []
    for row in rows:
        name = str(row["thumbnail_name"] or "")
        if name and Path(name).name == name:
            names.append(name)
    return names



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
            "SELECT 1 FROM scans WHERE id = ? AND client_id = ?",
            (scan_id, client_id),
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
    """Store feedback or a label assignment for a recovered history row."""
    now = datetime.now(timezone.utc).isoformat()
    with closing(_connect()) as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT waste_key, client_id, recovered FROM scans WHERE id = ?",
            (scan_id,),
        ).fetchone()
        if row is None:
            connection.rollback()
            return None
        predicted_key = str(row["waste_key"])
        source_client_id = str(row["client_id"])
        feedback_kind = "assignment" if bool(row["recovered"]) else "evaluation"
        # is_correct remains populated for backwards-compatible storage, but it is
        # semantically ignored for recovered JPEG-only assignments.
        stored_is_correct = predicted_key == corrected_key
        public_is_correct: bool | None = (
            None if feedback_kind == "assignment" else stored_is_correct
        )
        connection.execute(
            """
            INSERT INTO feedback (
                scan_id, client_id, predicted_key, corrected_key, is_correct,
                created_at, updated_at, feedback_kind
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(scan_id) DO UPDATE SET
                client_id = excluded.client_id,
                predicted_key = excluded.predicted_key,
                corrected_key = excluded.corrected_key,
                is_correct = excluded.is_correct,
                updated_at = excluded.updated_at,
                feedback_kind = excluded.feedback_kind
            """,
            (
                scan_id,
                source_client_id,
                predicted_key,
                corrected_key,
                int(stored_is_correct),
                now,
                now,
                feedback_kind,
            ),
        )
        embedding_row = connection.execute(
            "SELECT embedding_kind FROM scan_embeddings WHERE scan_id = ?",
            (scan_id,),
        ).fetchone()
        connection.commit()
    return {
        "scan_id": scan_id,
        "predicted_key": predicted_key,
        "corrected_key": corrected_key,
        "is_correct": public_is_correct,
        "feedback_kind": feedback_kind,
        "embedding_kind": (
            str(embedding_row["embedding_kind"]) if embedding_row is not None else None
        ),
    }

def get_learning_examples(
    embedding_kind: str,
    limit: int = 500,
) -> list[dict[str, Any]]:
    """Return compatible direct-class feedback examples from every device."""
    safe_limit = max(1, min(int(limit), 5000))
    learnable_keys = tuple(sorted(LEARNABLE_RULE_KEYS))
    if not learnable_keys:
        return []
    key_placeholders = ",".join("?" for _ in learnable_keys)
    with closing(_connect()) as connection:
        rows = connection.execute(
            f"""
            SELECT
                feedback.scan_id,
                feedback.corrected_key,
                feedback.is_correct,
                scan_embeddings.embedding_kind,
                scan_embeddings.dimension,
                scan_embeddings.embedding,
                feedback.updated_at
            FROM feedback
            JOIN scan_embeddings ON scan_embeddings.scan_id = feedback.scan_id
            WHERE scan_embeddings.embedding_kind = ?
              AND feedback.feedback_kind = 'evaluation'
              AND feedback.corrected_key IN ({key_placeholders})
            ORDER BY feedback.updated_at DESC, feedback.scan_id DESC
            LIMIT ?
            """,
            (embedding_kind, *learnable_keys, safe_limit),
        ).fetchall()

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
    """Return feedback-memory statistics across the shared scanner database."""
    compatible = tuple(
        kind.strip() for kind in (compatible_embedding_kinds or ()) if kind.strip()
    )
    with closing(_connect()) as connection:
        row = connection.execute(
            """
            SELECT
                COUNT(*) AS feedback_total,
                COALESCE(SUM(CASE WHEN feedback_kind = 'evaluation' AND is_correct = 1 THEN 1 ELSE 0 END), 0) AS confirmed,
                COALESCE(SUM(CASE WHEN feedback_kind = 'evaluation' AND is_correct = 0 THEN 1 ELSE 0 END), 0) AS corrected,
                COALESCE(SUM(CASE WHEN feedback_kind = 'assignment' THEN 1 ELSE 0 END), 0) AS assigned
            FROM feedback
            """
        ).fetchone()
        stored = int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM feedback
                JOIN scan_embeddings ON scan_embeddings.scan_id = feedback.scan_id
                WHERE feedback.feedback_kind = 'evaluation'
                """
            ).fetchone()[0]
        )
        learnable_keys = tuple(sorted(LEARNABLE_RULE_KEYS))
        key_placeholders = ",".join("?" for _ in learnable_keys)
        direct_stored = int(
            connection.execute(
                f"""
                SELECT COUNT(*)
                FROM feedback
                JOIN scan_embeddings ON scan_embeddings.scan_id = feedback.scan_id
                WHERE feedback.feedback_kind = 'evaluation'
                  AND feedback.corrected_key IN ({key_placeholders})
                """,
                learnable_keys,
            ).fetchone()[0]
        ) if learnable_keys else 0
        if compatible:
            placeholders = ",".join("?" for _ in compatible)
            learnable = int(
                connection.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM feedback
                    JOIN scan_embeddings ON scan_embeddings.scan_id = feedback.scan_id
                    WHERE feedback.feedback_kind = 'evaluation'
                      AND scan_embeddings.embedding_kind IN ({placeholders})
                      AND feedback.corrected_key IN ({key_placeholders})
                    """,
                    (*compatible, *learnable_keys),
                ).fetchone()[0]
            ) if learnable_keys else 0
        else:
            learnable = direct_stored
    return {
        "feedback_total": int(row["feedback_total"]),
        "confirmed": int(row["confirmed"]),
        "corrected": int(row["corrected"]),
        "assigned": int(row["assigned"]),
        "stored_embedding_examples": stored,
        "learnable_examples": learnable,
        "fallback_examples": max(0, stored - direct_stored),
        "incompatible_examples": max(0, direct_stored - learnable),
    }

def _matching_rule_keys_for_search(clean_query: str) -> tuple[str, ...]:
    """Return rule keys whose user-facing label/category matches the query."""
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
    escaped = (
        clean_query.replace("\\", "\\\\")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )
    pattern = f"%{escaped}%"
    matched_keys = _matching_rule_keys_for_search(clean_query)
    rule_sql = ""
    rule_params: tuple[str, ...] = ()
    if matched_keys:
        placeholders = ", ".join("?" for _ in matched_keys)
        # bin_name is not stored on each scan. Translate a bin/category label
        # search back to rule keys and match both the original AI prediction and
        # any user-corrected label.
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
        SELECT
            scans.id, scans.waste_key, scans.display_name, scans.category,
            scans.confidence, scans.uncertain, scans.recovered, scans.created_at,
            scans.thumbnail_name,
            feedback.corrected_key, feedback.is_correct, feedback.feedback_kind
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
        item["uncertain"] = bool(item["uncertain"])
        item["recovered"] = bool(item.get("recovered", False))
        # Keep the DB reference internal so the API layer can verify that the
        # referenced JPEG really exists before exposing thumbnail_available.
        item["_thumbnail_name"] = item.pop("thumbnail_name", None)
        if item.get("feedback_kind") == "assignment":
            item["is_correct"] = None
        elif item.get("is_correct") is not None:
            item["is_correct"] = bool(item["is_correct"])
    return items, has_more


def list_scans(
    limit: int = 20,
    before_id: int | None = None,
    query: str = "",
) -> tuple[list[dict[str, Any]], bool]:
    """Return one stable page from the shared history."""
    with closing(_connect()) as connection:
        return _list_scans_from_connection(connection, limit, before_id, query)


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


def count_scans(query: str = "") -> int:
    with closing(_connect()) as connection:
        return _count_scans_from_connection(connection, query)


def _scan_statistics_from_connection(
    connection: sqlite3.Connection,
    query: str = "",
) -> dict[str, int | float]:
    filter_sql, filter_params = _history_filter(query)
    row = connection.execute(
        f"""
        SELECT
            COUNT(*) AS total,
            COALESCE(
                SUM(
                    CASE
                        WHEN COALESCE(feedback.corrected_key, scans.waste_key)
                             IN ('plastic', 'nylon', 'paper', 'metal', 'glass')
                             AND (feedback.scan_id IS NOT NULL OR scans.uncertain = 0)
                        THEN 1 ELSE 0
                    END
                ),
                0
            ) AS recycled,
            COALESCE(
                AVG(CASE WHEN scans.recovered = 0 THEN scans.confidence END),
                0.0
            ) AS average_confidence
        FROM scans
        LEFT JOIN feedback ON feedback.scan_id = scans.id
        WHERE 1 = 1{filter_sql}
        """,
        filter_params,
    ).fetchone()
    return {
        "total": int(row["total"]),
        "recycled": int(row["recycled"]),
        "average_confidence": round(float(row["average_confidence"]), 4),
    }


def get_scan_statistics(query: str = "") -> dict[str, int | float]:
    with closing(_connect()) as connection:
        return _scan_statistics_from_connection(connection, query)


def get_history_page(
    limit: int = 20,
    before_id: int | None = None,
    query: str = "",
) -> dict[str, Any]:
    """Read counts, page rows and statistics from one SQLite snapshot."""
    with closing(_connect()) as connection:
        connection.execute("BEGIN")
        try:
            matched_total = _count_scans_from_connection(connection, query)
            history_total = (
                matched_total if not query.strip() else _count_scans_from_connection(connection)
            )
            items, has_more = _list_scans_from_connection(
                connection, limit, before_id, query
            )
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
    """Delete one scan while atomically blocking stale-thumbnail recovery."""
    if scan_id <= 0:
        return None

    deleted_at = datetime.now(timezone.utc).isoformat()
    with closing(_connect()) as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT id, thumbnail_name FROM scans WHERE id = ?",
            (scan_id,),
        ).fetchone()
        if row is None:
            connection.rollback()
            return None

        connection.execute(
            """
            INSERT INTO scan_recovery_tombstones (scan_id, deleted_at)
            VALUES (?, ?)
            ON CONFLICT(scan_id) DO UPDATE SET deleted_at = excluded.deleted_at
            """,
            (scan_id, deleted_at),
        )
        connection.execute("DELETE FROM scans WHERE id = ?", (scan_id,))
        remaining = int(connection.execute("SELECT COUNT(*) FROM scans").fetchone()[0])
        # Never reset AUTOINCREMENT. Reusing a scan id can make a stale browser tab
        # submit feedback/delete requests against a different, newly-created scan.
        sequence_reset = False
        connection.commit()

    thumbnail_name = str(row["thumbnail_name"] or "")
    if thumbnail_name and Path(thumbnail_name).name != thumbnail_name:
        thumbnail_name = ""
    return {
        "id": int(row["id"]),
        "thumbnail_name": thumbnail_name or None,
        "remaining": remaining,
        "sequence_reset": sequence_reset,
    }


def clear_scans() -> int:
    """Delete all history and tombstone each id without ever reusing scan ids."""
    deleted_at = datetime.now(timezone.utc).isoformat()
    with closing(_connect()) as connection:
        connection.execute("BEGIN IMMEDIATE")
        count = int(connection.execute("SELECT COUNT(*) FROM scans").fetchone()[0])
        connection.execute(
            """
            INSERT INTO scan_recovery_tombstones (scan_id, deleted_at)
            SELECT id, ? FROM scans
            WHERE 1 = 1
            ON CONFLICT(scan_id) DO UPDATE SET deleted_at = excluded.deleted_at
            """,
            (deleted_at,),
        )
        connection.execute(
            """
            INSERT INTO history_recovery_state (key, value)
            VALUES ('last_clear_at', ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (deleted_at,),
        )
        connection.execute(
            """
            INSERT INTO history_recovery_state (key, value)
            VALUES ('history_generation', '1')
            ON CONFLICT(key) DO UPDATE SET
                value = CAST(COALESCE(NULLIF(value, ''), '0') AS INTEGER) + 1
            """
        )
        connection.execute("DELETE FROM scans")
        # Keep sqlite_sequence intact so ids are never reused after a clear.
        connection.commit()
        return count

