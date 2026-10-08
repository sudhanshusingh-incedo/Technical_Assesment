"""Server-side conversation history, keyed by an opaque conversation id.

Why server-side: history can't be forged by the client, survives page reloads and works for any
API client. Only what follow-up resolution needs is stored (question, answer, status), never the
retrieved passages, with a TTL and a per-conversation turn cap to bound retention.

SQLite (stdlib) is the default: zero extra infrastructure, persistent across restarts, fine for a
single API instance. Several API replicas would need shared storage: implement the same
`ConversationStore` protocol on Redis or Postgres.
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from contextlib import closing
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel


class ConversationTurn(BaseModel):
    question: str
    answer: str
    status: str = "answered"
    rewritten_question: str | None = None
    created_at: float = 0.0


class ConversationNotFoundError(KeyError):
    """Unknown or expired conversation id."""


class ConversationStore(Protocol):
    def create(self) -> str: ...

    def exists(self, conversation_id: str) -> bool: ...

    def turns(self, conversation_id: str, limit: int | None = None) -> list[ConversationTurn]: ...

    def append(self, conversation_id: str, turn: ConversationTurn) -> None: ...

    def delete(self, conversation_id: str) -> bool: ...


_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    status TEXT NOT NULL,
    rewritten_question TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_turns_conversation ON turns(conversation_id, id);
"""


class SqliteConversationStore:
    def __init__(self, path: str | Path, ttl_hours: float = 24, max_turns: int = 20):
        self.path = str(path)
        self.ttl_s = ttl_hours * 3600
        self.max_turns = max_turns
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn, conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        # A short-lived connection per operation: safe across FastAPI's worker threads.
        conn = sqlite3.connect(self.path, timeout=10)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    def _purge_expired(self, conn: sqlite3.Connection) -> None:
        conn.execute("DELETE FROM conversations WHERE updated_at < ?", (time.time() - self.ttl_s,))

    def create(self) -> str:
        conversation_id = str(uuid.uuid4())
        now = time.time()
        with closing(self._connect()) as conn, conn:
            self._purge_expired(conn)
            conn.execute("INSERT INTO conversations (id, created_at, updated_at) VALUES (?, ?, ?)",
                         (conversation_id, now, now))
        return conversation_id

    def exists(self, conversation_id: str) -> bool:
        with closing(self._connect()) as conn, conn:
            self._purge_expired(conn)
            row = conn.execute("SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
        return row is not None

    def turns(self, conversation_id: str, limit: int | None = None) -> list[ConversationTurn]:
        """Oldest first; with `limit`, only the most recent `limit` turns."""
        if not self.exists(conversation_id):
            raise ConversationNotFoundError(conversation_id)
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT question, answer, status, rewritten_question, created_at FROM turns "
                "WHERE conversation_id = ? ORDER BY id DESC LIMIT ?",
                (conversation_id, limit if limit is not None else -1),
            ).fetchall()
        return [ConversationTurn(question=q, answer=a, status=s, rewritten_question=r, created_at=t)
                for q, a, s, r, t in reversed(rows)]

    def append(self, conversation_id: str, turn: ConversationTurn) -> None:
        now = time.time()
        with closing(self._connect()) as conn, conn:
            updated = conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?",
                                   (now, conversation_id)).rowcount
            if not updated:
                raise ConversationNotFoundError(conversation_id)
            conn.execute(
                "INSERT INTO turns (conversation_id, question, answer, status, rewritten_question, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (conversation_id, turn.question, turn.answer, turn.status, turn.rewritten_question, now),
            )
            # Keep only the newest `max_turns` turns of this conversation.
            conn.execute(
                "DELETE FROM turns WHERE conversation_id = ? AND id NOT IN "
                "(SELECT id FROM turns WHERE conversation_id = ? ORDER BY id DESC LIMIT ?)",
                (conversation_id, conversation_id, self.max_turns),
            )

    def delete(self, conversation_id: str) -> bool:
        with closing(self._connect()) as conn, conn:
            return conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,)).rowcount > 0
