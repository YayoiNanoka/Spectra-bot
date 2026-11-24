"""Session state persistence and prefix-stable OpenAI context projection.

Owns the ``sessions`` SQLite table migration, block-rollover cursor advancement
(``window_max_messages`` -> ``window_keep_messages``), retention protection
watermark, and strict alternating ``user`` / ``assistant`` context formatting.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import aiosqlite

from spectra.config import SessionConfig
from spectra.db import Database
from spectra.messages import MessageRecord, MessageStore

_MIGRATIONS: tuple[str, ...] = (
    """
    CREATE TABLE sessions (
        session_key TEXT PRIMARY KEY,
        window_start_id INTEGER NOT NULL DEFAULT 0,
        summary TEXT NOT NULL DEFAULT '',
        last_extracted_id INTEGER NOT NULL DEFAULT 0,
        group_impression TEXT NOT NULL DEFAULT '',
        updated_at INTEGER NOT NULL
    );
    """,
)

_SESSION_COLUMNS = (
    "session_key, window_start_id, summary, last_extracted_id, "
    "group_impression, updated_at"
)
_MESSAGE_COLUMNS = "id, group_id, user_id, sender_name, role, content, created_at"


@dataclass(slots=True, frozen=True)
class SessionRecord:
    """Immutable row from the ``sessions`` table."""

    session_key: str
    window_start_id: int
    summary: str
    last_extracted_id: int
    group_impression: str
    updated_at: int


def _row_to_session(row: aiosqlite.Row) -> SessionRecord:
    return SessionRecord(
        session_key=str(row["session_key"]),
        window_start_id=int(row["window_start_id"]),
        summary=str(row["summary"]),
        last_extracted_id=int(row["last_extracted_id"]),
        group_impression=str(row["group_impression"]),
        updated_at=int(row["updated_at"]),
    )


def _row_to_message(row: aiosqlite.Row) -> MessageRecord:
    return MessageRecord(
        id=int(row["id"]),
        group_id=row["group_id"],
        user_id=str(row["user_id"]),
        sender_name=str(row["sender_name"]),
        role=row["role"],
        content=str(row["content"]),
        created_at=int(row["created_at"]),
    )


class SessionStore:
    """Async store for per-session cursors and OpenAI chat context building.

    Args:
        db: Connected :class:`spectra.db.Database` instance.
        message_store: Initialized :class:`spectra.messages.MessageStore`.
        config: :class:`spectra.config.SessionConfig` controlling window bounds
            and display timezone.
    """

    def __init__(
        self,
        db: Database,
        message_store: MessageStore,
        config: SessionConfig,
    ) -> None:
        self._db = db
        self._message_store = message_store
        self._config = config
        self._tz = ZoneInfo(config.timezone)

    async def init(self) -> None:
        """Apply schema migrations for the ``sessions`` module."""
        await self._db.migrate("sessions", _MIGRATIONS)

    async def get_or_create(
        self,
        session_key: str,
        *,
        now: int | None = None,
    ) -> SessionRecord:
        """Return an existing session row or atomically create one with defaults.

        Args:
            session_key: Canonical session identifier (``group:{id}`` or
                ``private:{id}``).
            now: Optional Unix timestamp in seconds for initial ``updated_at``.

        Returns:
            The persisted :class:`SessionRecord`.
        """
        row = await self._db.fetchone(
            f"SELECT {_SESSION_COLUMNS} FROM sessions WHERE session_key = ?",
            (session_key,),
        )
        if row is not None:
            return _row_to_session(row)

        ts = int(time.time()) if now is None else int(now)
        await self._db.execute(
            """
            INSERT OR IGNORE INTO sessions (
                session_key, window_start_id, summary, last_extracted_id,
                group_impression, updated_at
            )
            VALUES (?, 0, '', 0, '', ?)
            """,
            (session_key, ts),
        )
        row = await self._db.fetchone(
            f"SELECT {_SESSION_COLUMNS} FROM sessions WHERE session_key = ?",
            (session_key,),
        )
        assert row is not None
        return _row_to_session(row)

    async def touch(
        self,
        session_key: str,
        *,
        now: int | None = None,
    ) -> None:
        """Update ``updated_at`` for a session, creating it if absent."""
        ts = int(time.time()) if now is None else int(now)
        await self._db.execute(
            """
            INSERT INTO sessions (
                session_key, window_start_id, summary, last_extracted_id,
                group_impression, updated_at
            )
            VALUES (?, 0, '', 0, '', ?)
            ON CONFLICT(session_key) DO UPDATE SET
                updated_at = excluded.updated_at
            """,
            (session_key, ts),
        )

    async def update_window(
        self,
        session_key: str,
        *,
        window_start_id: int,
        summary: str | None = None,
        now: int | None = None,
    ) -> None:
        """Advance ``window_start_id`` (and optionally update ``summary``).

        Args:
            session_key: Canonical session identifier.
            window_start_id: New non-negative active window start cursor.
            summary: Optional replacement rolling summary text. When ``None``,
                any existing summary is preserved.
            now: Optional Unix timestamp in seconds for ``updated_at``.

        Raises:
            ValueError: If ``window_start_id`` is negative.
        """
        if window_start_id < 0:
            raise ValueError(
                f"window_start_id must be non-negative, got {window_start_id}."
            )
        ts = int(time.time()) if now is None else int(now)
        if summary is None:
            await self._db.execute(
                """
                INSERT INTO sessions (
                    session_key, window_start_id, summary, last_extracted_id,
                    group_impression, updated_at
                )
                VALUES (?, ?, '', 0, '', ?)
                ON CONFLICT(session_key) DO UPDATE SET
                    window_start_id = excluded.window_start_id,
                    updated_at = excluded.updated_at
                """,
                (session_key, int(window_start_id), ts),
            )
        else:
            await self._db.execute(
                """
                INSERT INTO sessions (
                    session_key, window_start_id, summary, last_extracted_id,
                    group_impression, updated_at
                )
                VALUES (?, ?, ?, 0, '', ?)
                ON CONFLICT(session_key) DO UPDATE SET
                    window_start_id = excluded.window_start_id,
                    summary = excluded.summary,
                    updated_at = excluded.updated_at
                """,
                (session_key, int(window_start_id), summary, ts),
            )

    async def min_protected_message_id(self) -> int | None:
        """Return the lowest message ID still protected by any session cursor.

        Computes ``MIN(MIN(window_start_id, last_extracted_id))`` across all
        rows in ``sessions`` so expired message cleanup never deletes messages
        still needed by an active context window or pending extraction.

        Returns:
            Minimum protected message ID, or ``None`` if no sessions exist.
        """
        row = await self._db.fetchone(
            """
            SELECT MIN(
                CASE
                    WHEN window_start_id < last_extracted_id THEN window_start_id
                    ELSE last_extracted_id
                END
            ) AS min_id
            FROM sessions
            """
        )
        if row is None or row["min_id"] is None:
            return None
        return int(row["min_id"])

    async def _fetch_active_messages(
        self,
        *,
        group_id: str | None,
        user_id: str,
        window_start_id: int,
        up_to_id: int | None,
        limit: int,
    ) -> list[MessageRecord]:
        """Fetch bounded active window messages in ascending ``id`` order."""
        if group_id is not None:
            if up_to_id is None:
                sql = (
                    f"SELECT {_MESSAGE_COLUMNS} FROM messages "
                    "WHERE group_id = ? AND id >= ? "
                    "ORDER BY id DESC LIMIT ?"
                )
                params: tuple[Any, ...] = (str(group_id), window_start_id, limit)
            else:
                sql = (
                    f"SELECT {_MESSAGE_COLUMNS} FROM messages "
                    "WHERE group_id = ? AND id >= ? AND id <= ? "
                    "ORDER BY id DESC LIMIT ?"
                )
                params = (str(group_id), window_start_id, int(up_to_id), limit)
        else:
            if up_to_id is None:
                sql = (
                    f"SELECT {_MESSAGE_COLUMNS} FROM messages "
                    "WHERE group_id IS NULL AND user_id = ? AND id >= ? "
                    "ORDER BY id DESC LIMIT ?"
                )
                params = (str(user_id), window_start_id, limit)
            else:
                sql = (
                    f"SELECT {_MESSAGE_COLUMNS} FROM messages "
                    "WHERE group_id IS NULL AND user_id = ? AND id >= ? AND id <= ? "
                    "ORDER BY id DESC LIMIT ?"
                )
                params = (str(user_id), window_start_id, int(up_to_id), limit)

        rows = await self._db.fetchall(sql, params)
        rows.reverse()
        return [_row_to_message(row) for row in rows]

    def _format_message(self, rec: MessageRecord, *, is_group: bool) -> str:
        if is_group and rec.role == "user":
            hh_mm = datetime.fromtimestamp(rec.created_at, tz=self._tz).strftime(
                "%H:%M"
            )
            return f"[{rec.sender_name}({rec.user_id}) {hh_mm}]: {rec.content}"
        return rec.content

    async def build_context(
        self,
        *,
        session_key: str,
        group_id: str | None,
        user_id: str,
        system_prompt: str,
        up_to_id: int | None = None,
        now: int | None = None,
    ) -> list[dict[str, Any]]:
        """Project a session's active message window into OpenAI chat messages.

        Performs prefix-stable block rollover when the active window exceeds
        ``config.window_max_messages``, formats group user messages with
        speaker/timestamp metadata, strips leading assistant turns, and
        coalesces adjacent same-role turns into strict alternating
        ``user`` / ``assistant`` messages after the ``system`` message.

        Args:
            session_key: Canonical session identifier.
            group_id: Group QQ ID for group chat, or ``None`` for private chat.
            user_id: Peer QQ ID (used for private chat message lookup).
            system_prompt: Base persona system prompt text.
            up_to_id: Optional inclusive upper bound on ``messages.id``.
            now: Optional Unix timestamp in seconds for session creation or
                rollover updates.

        Returns:
            OpenAI-formatted ``messages`` list starting with ``role="system"``.
        """
        session = await self.get_or_create(session_key, now=now)
        active = await self._fetch_active_messages(
            group_id=group_id,
            user_id=user_id,
            window_start_id=session.window_start_id,
            up_to_id=up_to_id,
            limit=self._config.window_max_messages + 1,
        )

        if len(active) > self._config.window_max_messages:
            kept = active[-self._config.window_keep_messages :]
            while len(kept) > 1 and kept[0].role == "assistant":
                kept = kept[1:]
            await self.update_window(
                session_key,
                window_start_id=kept[0].id,
                now=now,
            )
            active = kept

        sys_parts: list[str] = []
        base_prompt = system_prompt.strip()
        if base_prompt:
            sys_parts.append(base_prompt)
        if group_id is not None and session.group_impression.strip():
            sys_parts.append(
                f"## 当前群聊氛围与印象\n{session.group_impression.strip()}"
            )
        if session.summary.strip():
            sys_parts.append(f"## 近期对话前情摘要\n{session.summary.strip()}")
        sys_text = "\n\n".join(sys_parts)

        messages: list[dict[str, Any]] = [{"role": "system", "content": sys_text}]

        start_idx = 0
        while start_idx < len(active) and active[start_idx].role == "assistant":
            start_idx += 1

        is_group = group_id is not None
        for rec in active[start_idx:]:
            formatted = self._format_message(rec, is_group=is_group)
            if messages[-1]["role"] == rec.role:
                messages[-1]["content"] = f"{messages[-1]['content']}\n{formatted}"
            else:
                messages.append({"role": rec.role, "content": formatted})

        return messages
