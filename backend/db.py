"""SQLite-backed chat log for the /ask endpoint.

Single table; one INSERT per /ask call. We log the question text + reply
+ session_id + latency_ms. We deliberately do NOT log IP / User-Agent /
cookies — the API is unauthenticated in v1 and we don't want accidental
PII.

The DB lives at data/external/api/chat_log.sqlite, kept separate from
the Phase 9 news files so vacuum/wipe operations on one don't affect
the other.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from src.utils.config import EXTERNAL_DATA_DIR

DB_PATH = EXTERNAL_DATA_DIR / "api" / "chat_log.sqlite"


SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,           -- ISO 8601 UTC, e.g. 2026-08-17T10:30:00+00:00
    session_id  TEXT,                    -- nullable; client-supplied
    question    TEXT NOT NULL,
    reply       TEXT NOT NULL,
    latency_ms  INTEGER
);
CREATE INDEX IF NOT EXISTS chat_log_ts_idx      ON chat_log(ts);
CREATE INDEX IF NOT EXISTS chat_log_session_idx ON chat_log(session_id);
"""


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db() -> None:
    """Create the table on startup. Idempotent."""
    with _connect() as c:
        c.executescript(SCHEMA)


def log_chat(*, question: str, reply: str,
             session_id: str | None, latency_ms: int) -> None:
    """Insert one row. Never raises — logs the error and returns."""
    try:
        with _connect() as c:
            c.execute(
                "INSERT INTO chat_log "
                "(ts, session_id, question, reply, latency_ms) "
                "VALUES (?,?,?,?,?)",
                (datetime.now(timezone.utc).isoformat(),
                 session_id, question, reply, int(latency_ms)),
            )
    except sqlite3.Error as exc:
        # A logging failure must not break the API response.
        from src.utils.logger import get_logger
        get_logger("backend.db").error(f"log_chat failed: {exc}")


def count_chats() -> int:
    """Test helper — total rows."""
    with _connect() as c:
        return c.execute("SELECT COUNT(*) FROM chat_log").fetchone()[0]
