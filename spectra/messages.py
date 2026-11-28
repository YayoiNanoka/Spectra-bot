"""Append-only chat message store backed by :class:`spectra.db.Database`.

Owns the ``messages`` and ``message_platform_ids`` SQLite tables and their
incremental schema migrations.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
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
        created_at INTEGER NOT NULL,
        platform_message_id TEXT
            CHECK (
                platform_message_id IS NULL
                OR length(trim(platform_message_id)) > 0
            ),
        reply_to_message_id TEXT
            CHECK (
                reply_to_message_id IS NULL
                OR length(trim(reply_to_message_id)) > 0
            )
    );
    CREATE INDEX idx_messages_group_id_id ON messages (group_id, id);
    CREATE INDEX idx_messages_private_user_id ON messages (user_id, id)
        WHERE group_id IS NULL;
    CREATE INDEX idx_messages_created_at ON messages (created_at);
    CREATE INDEX idx_messages_platform_id ON messages (platform_message_id)
        WHERE platform_message_id IS NOT NULL;

    CREATE TABLE message_platform_ids (
        message_id INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
        platform_message_id TEXT NOT NULL
            CHECK (length(trim(platform_message_id)) > 0),
        PRIMARY KEY (message_id, platform_message_id)
    );
    CREATE INDEX idx_message_platform_ids_lookup
        ON message_platform_ids (platform_message_id, message_id);
    """,
)

_SELECT_COLUMNS = (
    "id, group_id, user_id, sender_name, role, content, created_at, "
    "platform_message_id, reply_to_message_id"
)


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
        platform_message_id: Primary OneBot platform message ID, if known.
        reply_to_message_id: Quoted OneBot platform message ID, if any.
    """

    id: int
    group_id: str | None
    user_id: str
    sender_name: str
    role: MessageRole
    content: str
    created_at: int
    platform_message_id: str | None = None
    reply_to_message_id: str | None = None


def _normalize_optional_id(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _row_to_record(row: aiosqlite.Row) -> MessageRecord:
    return MessageRecord(
        id=int(row["id"]),
        group_id=row["group_id"],
        user_id=str(row["user_id"]),
        sender_name=str(row["sender_name"]),
        role=row["role"],
        content=str(row["content"]),
        created_at=int(row["created_at"]),
        platform_message_id=row["platform_message_id"],
        reply_to_message_id=row["reply_to_message_id"],
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
        platform_message_id: str | None = None,
        reply_to_message_id: str | None = None,
        extra_platform_ids: Sequence[str] = (),
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
            platform_message_id: Optional primary platform message ID.
            reply_to_message_id: Optional quoted platform message ID.
            extra_platform_ids: Optional additional platform message IDs (e.g.
                all bubble IDs from a segmented outbound reply).

        Returns:
            Auto-incremented ``messages.id`` for the inserted row.

        Raises:
            ValueError: If ``role`` is not ``'user'`` or ``'assistant'``.
            TypeError: If ``extra_platform_ids`` is a raw string/bytes value.
        """
        if role not in ("user", "assistant"):
            raise ValueError(f"Invalid message role: {role!r}")
        if isinstance(extra_platform_ids, (str, bytes, bytearray)):
            raise TypeError(
                "extra_platform_ids must be a sequence of IDs, not a single str."
            )

        ts = int(time.time()) if created_at is None else int(created_at)
        norm_platform_id = _normalize_optional_id(platform_message_id)
        norm_reply_to = _normalize_optional_id(reply_to_message_id)

        platform_ids: list[str] = []
        seen_ids: set[str] = set()
        if norm_platform_id is not None:
            platform_ids.append(norm_platform_id)
            seen_ids.add(norm_platform_id)
        for raw_pid in extra_platform_ids:
            pid = _normalize_optional_id(raw_pid)
            if pid is not None and pid not in seen_ids:
                platform_ids.append(pid)
                seen_ids.add(pid)

        primary_platform_id = platform_ids[0] if platform_ids else None

        if platform_ids:
            async with self._db.transaction() as tx:
                result = await tx.execute(
                    """
                    INSERT INTO messages (
                        group_id, user_id, sender_name, role, content,
                        created_at, platform_message_id, reply_to_message_id
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        group_id,
                        str(user_id),
                        sender_name,
                        role,
                        content,
                        ts,
                        primary_platform_id,
                        norm_reply_to,
                    ),
                )
                msg_id = result.lastrowid
                for pid in platform_ids:
                    await tx.execute(
                        """
                        INSERT OR IGNORE INTO message_platform_ids (
                            message_id, platform_message_id
                        )
                        VALUES (?, ?)
                        """,
                        (msg_id, pid),
                    )
            return msg_id

        result = await self._db.execute(
            """
            INSERT INTO messages (
                group_id, user_id, sender_name, role, content,
                created_at, platform_message_id, reply_to_message_id
            )
            VALUES (?, ?, ?, ?, ?, ?, NULL, ?)
            """,
            (group_id, str(user_id), sender_name, role, content, ts, norm_reply_to),
        )
        return result.lastrowid

    async def resolve_platform_message_id(
        self,
        *,
        platform_message_id: str,
        group_id: str | None,
        user_id: str,
    ) -> int | None:
        """Resolve a platform message ID to the latest ``messages.id`` in the same chat.

        Checks both ``messages.platform_message_id`` and
        ``message_platform_ids.platform_message_id`` strictly scoped to the same
        group or private peer chat.

        Args:
            platform_message_id: Platform message ID to look up.
            group_id: Group QQ ID for group chat, or ``None`` for private chat.
            user_id: Peer user QQ ID (used when ``group_id is None``).

        Returns:
            Latest matching ``messages.id``, or ``None`` if blank or not found.
        """
        norm_pid = _normalize_optional_id(platform_message_id)
        if norm_pid is None:
            return None

        if group_id is not None:
            gid = str(group_id)
            row = await self._db.fetchone(
                """
                SELECT MAX(id) AS msg_id
                FROM (
                    SELECT id
                    FROM messages
                    WHERE group_id = ? AND platform_message_id = ?
                    UNION ALL
                    SELECT m.id
                    FROM message_platform_ids mpi
                    JOIN messages m ON m.id = mpi.message_id
                    WHERE m.group_id = ? AND mpi.platform_message_id = ?
                )
                """,
                (gid, norm_pid, gid, norm_pid),
            )
        else:
            uid = str(user_id)
            row = await self._db.fetchone(
                """
                SELECT MAX(id) AS msg_id
                FROM (
                    SELECT id
                    FROM messages
                    WHERE group_id IS NULL AND user_id = ? AND platform_message_id = ?
                    UNION ALL
                    SELECT m.id
                    FROM message_platform_ids mpi
                    JOIN messages m ON m.id = mpi.message_id
                    WHERE m.group_id IS NULL
                      AND m.user_id = ?
                      AND mpi.platform_message_id = ?
                )
                """,
                (uid, norm_pid, uid, norm_pid),
            )

        if row is None or row["msg_id"] is None:
            return None
        return int(row["msg_id"])

    async def get_messages_in_range(
        self,
        *,
        group_id: str | None,
        user_id: str,
        start_id: int,
        end_id: int,
    ) -> list[MessageRecord]:
        """Fetch messages within inclusive ``[start_id, end_id]`` ordered by ``id ASC``.

        Args:
            group_id: Group QQ ID for group chat, or ``None`` for private chat.
            user_id: Peer user QQ ID (used when ``group_id is None``).
            start_id: Inclusive lower bound on ``messages.id``.
            end_id: Inclusive upper bound on ``messages.id``.

        Returns:
            List of :class:`MessageRecord` objects ordered by ``id ASC``
            (empty when ``end_id < start_id``).
        """
        start = int(start_id)
        end = int(end_id)
        if end < start:
            return []

        if group_id is not None:
            rows = await self._db.fetchall(
                f"""
                SELECT {_SELECT_COLUMNS}
                FROM messages
                WHERE group_id = ? AND id >= ? AND id <= ?
                ORDER BY id ASC
                """,
                (str(group_id), start, end),
            )
        else:
            rows = await self._db.fetchall(
                f"""
                SELECT {_SELECT_COLUMNS}
                FROM messages
                WHERE group_id IS NULL AND user_id = ? AND id >= ? AND id <= ?
                ORDER BY id ASC
                """,
                (str(user_id), start, end),
            )
        return [_row_to_record(row) for row in rows]

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
