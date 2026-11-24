"""Append-only chat message store backed by :class:`spectra.db.Database`.

Owns the ``messages`` SQLite table and its incremental schema migrations.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import aiosqlite

from spectra.db import Database

MessageRole = Literal["user", "assistant"]

_MIGRATIONS: tuple[str, ...] = (
    """
    CREATE TABLE messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        group_id TEXT,
        user_id TEXT NOT NULL,
        sender_name TEXT NOT NULL,
        role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
        content TEXT NOT NULL,
        created_at INTEGER NOT NULL
    );
    CREATE INDEX idx_messages_group_id_id ON messages (group_id, id);
    CREATE INDEX idx_messages_private_user_id ON messages (user_id, id)
        WHERE group_id IS NULL;
    CREATE INDEX idx_messages_created_at ON messages (created_at);
    """,
)

_SELECT_COLUMNS = "id, group_id, user_id, sender_name, role, content, created_at"


@dataclass(slots=True, frozen=True)
class MessageRecord:
    """Immutable chat message row from the ``messages`` table.

    Attributes:
        id: Monotonically increasing primary key and cursor.
        group_id: Group QQ ID for group messages, or ``None`` for private chat.
        user_id: Speaker QQ ID in group chat; peer user QQ ID in private chat
            (including when ``role == 'assistant'``).
        sender_name: Snapshot of sender display name at message time.
        role: Message role (``'user'`` or ``'assistant'``).
        content: Flattened text content of the message.
        created_at: Unix timestamp in seconds.
    """

    id: int
    group_id: str | None
    user_id: str
    sender_name: str
    role: MessageRole
    content: str
    created_at: int


def _row_to_record(row: aiosqlite.Row) -> MessageRecord:
    return MessageRecord(
        id=int(row["id"]),
        group_id=row["group_id"],
        user_id=str(row["user_id"]),
        sender_name=str(row["sender_name"]),
        role=row["role"],
        content=str(row["content"]),
        created_at=int(row["created_at"]),
    )


class MessageStore:
    """Async store for appending, querying, and pruning chat messages.

    Args:
        db: Connected :class:`spectra.db.Database` instance.
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    async def init(self) -> None:
        """Apply schema migrations for the ``messages`` module."""
        await self._db.migrate("messages", _MIGRATIONS)

    async def append_message(
        self,
        *,
        group_id: str | None,
        user_id: str,
        sender_name: str,
        role: MessageRole,
        content: str,
        created_at: int | None = None,
    ) -> int:
        """Insert a single chat message row and return its generated ID.

        Args:
            group_id: Group QQ ID for group chat, or ``None`` for private chat.
            user_id: Speaker QQ ID in group chat, or the peer user's QQ ID in
                private chat (even for ``role='assistant'``).
            sender_name: Snapshot of the sender's display name.
            role: Either ``'user'`` or ``'assistant'``.
            content: Flattened message text.
            created_at: Optional Unix timestamp in seconds; defaults to now.

        Returns:
            Auto-incremented ``messages.id`` for the inserted row.

        Raises:
            ValueError: If ``role`` is not ``'user'`` or ``'assistant'``.
        """
        if role not in ("user", "assistant"):
            raise ValueError(f"Invalid message role: {role!r}")
        ts = int(time.time()) if created_at is None else int(created_at)
        result = await self._db.execute(
            """
            INSERT INTO messages (
                group_id, user_id, sender_name, role, content, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (group_id, str(user_id), sender_name, role, content, ts),
        )
        return result.lastrowid

    async def get_group_messages(
        self,
        group_id: str,
        *,
        limit: int = 50,
        after_id: int | None = None,
    ) -> list[MessageRecord]:
        """Fetch messages for a group in ascending ``id`` order.

        Args:
            group_id: Group QQ ID.
            limit: Maximum number of rows to return.
            after_id: When ``None``, returns the latest ``limit`` rows in
                chronological order. When provided, returns the first ``limit``
                rows with ``id > after_id`` in chronological order.

        Returns:
            List of :class:`MessageRecord` objects ordered by ``id ASC``.

        Raises:
            ValueError: If ``limit`` is negative.
        """
        if limit < 0:
            raise ValueError("limit must be non-negative.")
        if limit == 0:
            return []

        if after_id is None:
            rows = await self._db.fetchall(
                f"""
                SELECT {_SELECT_COLUMNS}
                FROM messages
                WHERE group_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (str(group_id), limit),
            )
            rows.reverse()
        else:
            rows = await self._db.fetchall(
                f"""
                SELECT {_SELECT_COLUMNS}
                FROM messages
                WHERE group_id = ? AND id > ?
                ORDER BY id ASC
                LIMIT ?
                """,
                (str(group_id), after_id, limit),
            )
        return [_row_to_record(row) for row in rows]

    async def get_private_messages(
        self,
        user_id: str,
        *,
        limit: int = 50,
        after_id: int | None = None,
    ) -> list[MessageRecord]:
        """Fetch private chat messages for a peer user in ascending ``id`` order.

        Args:
            user_id: Peer user's QQ ID.
            limit: Maximum number of rows to return.
            after_id: When ``None``, returns the latest ``limit`` rows in
                chronological order. When provided, returns the first ``limit``
                rows with ``id > after_id`` in chronological order.

        Returns:
            List of :class:`MessageRecord` objects ordered by ``id ASC``.

        Raises:
            ValueError: If ``limit`` is negative.
        """
        if limit < 0:
            raise ValueError("limit must be non-negative.")
        if limit == 0:
            return []

        if after_id is None:
            rows = await self._db.fetchall(
                f"""
                SELECT {_SELECT_COLUMNS}
                FROM messages
                WHERE group_id IS NULL AND user_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (str(user_id), limit),
            )
            rows.reverse()
        else:
            rows = await self._db.fetchall(
                f"""
                SELECT {_SELECT_COLUMNS}
                FROM messages
                WHERE group_id IS NULL AND user_id = ? AND id > ?
                ORDER BY id ASC
                LIMIT ?
                """,
                (str(user_id), after_id, limit),
            )
        return [_row_to_record(row) for row in rows]

    async def cleanup_expired_messages(
        self,
        before_created_at: int,
        *,
        max_deletable_id: int | None = None,
    ) -> int:
        """Delete expired messages older than ``before_created_at``.

        Args:
            before_created_at: Exclusive Unix timestamp cutoff in seconds.
            max_deletable_id: Optional exclusive upper bound on ``id`` to
                protect messages not yet compacted or extracted.

        Returns:
            Number of deleted rows.
        """
        if max_deletable_id is None:
            result = await self._db.execute(
                "DELETE FROM messages WHERE created_at < ?",
                (int(before_created_at),),
            )
        else:
            result = await self._db.execute(
                "DELETE FROM messages WHERE created_at < ? AND id < ?",
                (int(before_created_at), int(max_deletable_id)),
            )
        return result.rowcount
