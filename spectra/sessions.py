"""Session state, short-episode segmentation, and prefix-stable context projection.

Owns the ``sessions`` SQLite table migration, short-episode boundary detection
(idle/hard timeouts, in-window quote shortcuts, Jev continuity checks, and
lead-in cluster lookback), frozen profile snapshot injection, intra-episode
block-rollover cursor advancement, and strict alternating ``user`` /
``assistant`` OpenAI context formatting.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import aiosqlite

from spectra.config import SessionConfig
from spectra.db import Database
from spectra.jev import JevClient
from spectra.messages import MessageRecord, MessageStore

_MIGRATIONS: tuple[str, ...] = (
    """
    CREATE TABLE sessions (
        session_key TEXT PRIMARY KEY,
        episode_id INTEGER NOT NULL DEFAULT 1,
        episode_start_id INTEGER NOT NULL DEFAULT 0,
        window_start_id INTEGER NOT NULL DEFAULT 0,
        last_message_at INTEGER NOT NULL DEFAULT 0,
        last_extracted_id INTEGER NOT NULL DEFAULT 0,
        profile_snapshot TEXT NOT NULL DEFAULT '',
        updated_at INTEGER NOT NULL
    );
    """,
)

_SESSION_COLUMNS = (
    "session_key, episode_id, episode_start_id, window_start_id, "
    "last_message_at, last_extracted_id, profile_snapshot, updated_at"
)
_MESSAGE_COLUMNS = (
    "id, group_id, user_id, sender_name, role, content, created_at, "
    "platform_message_id, reply_to_message_id"
)

WEEKDAY_NAMES: tuple[str, ...] = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)


def format_datetime_reminder(timestamp: int, tz: ZoneInfo, tz_name: str) -> str:
    """Format an AstrBot-aligned ``<system_reminder>`` timestamp block.

    Args:
        timestamp: Unix epoch timestamp in seconds.
        tz: Resolved :class:`zoneinfo.ZoneInfo` timezone.
        tz_name: Canonical IANA timezone name fallback if ``%Z`` is empty.

    Returns:
        Multi-line ``<system_reminder>`` block with date, time, timezone,
        and English weekday name.
    """
    dt = datetime.fromtimestamp(int(timestamp), tz=tz)
    tz_abbr_or_name = dt.strftime("%Z") or tz_name
    current_time = f"{dt.strftime('%Y-%m-%d %H:%M')} ({tz_abbr_or_name})"
    weekday = WEEKDAY_NAMES[dt.weekday()]
    return (
        "<system_reminder>\n"
        f"Current datetime: {current_time}, Weekday: {weekday}\n"
        "</system_reminder>"
    )


@dataclass(slots=True, frozen=True)
class SessionRecord:
    """Immutable row from the ``sessions`` table."""

    session_key: str
    episode_id: int
    episode_start_id: int
    window_start_id: int
    last_message_at: int
    last_extracted_id: int
    profile_snapshot: str
    updated_at: int


@dataclass(slots=True, frozen=True)
class ClosedEpisode:
    """Immutable message span of an episode closed by segmentation."""

    session_key: str
    group_id: str | None
    user_id: str
    episode_id: int
    start_msg_id: int
    end_msg_id: int
    closed_at: int


@dataclass(slots=True, frozen=True)
class EpisodeResolution:
    """Outcome of :meth:`SessionStore.resolve_episode` for an inbound batch."""

    session: SessionRecord
    new_episode: bool
    reason: str
    continuity_probability: float | None = None
    closed_episode: ClosedEpisode | None = None


def _row_to_session(row: aiosqlite.Row) -> SessionRecord:
    return SessionRecord(
        session_key=str(row["session_key"]),
        episode_id=int(row["episode_id"]),
        episode_start_id=int(row["episode_start_id"]),
        window_start_id=int(row["window_start_id"]),
        last_message_at=int(row["last_message_at"]),
        last_extracted_id=int(row["last_extracted_id"]),
        profile_snapshot=str(row["profile_snapshot"]),
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
        platform_message_id=row["platform_message_id"],
        reply_to_message_id=row["reply_to_message_id"],
    )


class SessionStore:
    """Async store for per-session episode cursors and OpenAI context building.

    Args:
        db: Connected :class:`spectra.db.Database` instance.
        message_store: Initialized :class:`spectra.messages.MessageStore`.
        config: :class:`spectra.config.SessionConfig` controlling timeouts,
            window bounds, lead-in lookback, and display timezone.
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
                session_key, episode_id, episode_start_id, window_start_id,
                last_message_at, last_extracted_id, profile_snapshot, updated_at
            )
            VALUES (?, 1, 0, 0, 0, 0, '', ?)
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
                session_key, episode_id, episode_start_id, window_start_id,
                last_message_at, last_extracted_id, profile_snapshot, updated_at
            )
            VALUES (?, 1, 0, 0, 0, 0, '', ?)
            ON CONFLICT(session_key) DO UPDATE SET
                updated_at = excluded.updated_at
            """,
            (session_key, ts),
        )

    async def record_turn(
        self,
        session_key: str,
        *,
        message_at: int,
        now: int | None = None,
    ) -> None:
        """Monotonically advance ``last_message_at`` and refresh ``updated_at``.

        Args:
            session_key: Canonical session identifier.
            message_at: Message timestamp in seconds.
            now: Optional Unix timestamp in seconds for ``updated_at``. When
                omitted, stays in the message timestamp domain using
                ``MAX(updated_at, message_at)``.
        """
        msg_ts = int(message_at)
        if now is None:
            await self._db.execute(
                """
                INSERT INTO sessions (
                    session_key, episode_id, episode_start_id, window_start_id,
                    last_message_at, last_extracted_id, profile_snapshot, updated_at
                )
                VALUES (?, 1, 0, 0, ?, 0, '', ?)
                ON CONFLICT(session_key) DO UPDATE SET
                    last_message_at = MAX(last_message_at, excluded.last_message_at),
                    updated_at = MAX(updated_at, excluded.updated_at)
                """,
                (session_key, msg_ts, msg_ts),
            )
        else:
            ts = int(now)
            await self._db.execute(
                """
                INSERT INTO sessions (
                    session_key, episode_id, episode_start_id, window_start_id,
                    last_message_at, last_extracted_id, profile_snapshot, updated_at
                )
                VALUES (?, 1, 0, 0, ?, 0, '', ?)
                ON CONFLICT(session_key) DO UPDATE SET
                    last_message_at = MAX(last_message_at, excluded.last_message_at),
                    updated_at = excluded.updated_at
                """,
                (session_key, msg_ts, ts),
            )

    async def update_window(
        self,
        session_key: str,
        *,
        window_start_id: int,
        now: int | None = None,
    ) -> None:
        """Advance ``window_start_id`` while preserving episode metadata.

        Args:
            session_key: Canonical session identifier.
            window_start_id: New non-negative active window start cursor.
            now: Optional Unix timestamp in seconds for ``updated_at``.

        Raises:
            ValueError: If ``window_start_id`` is negative.
        """
        if window_start_id < 0:
            raise ValueError(
                f"window_start_id must be non-negative, got {window_start_id}."
            )
        ts = int(time.time()) if now is None else int(now)
        await self._db.execute(
            """
            INSERT INTO sessions (
                session_key, episode_id, episode_start_id, window_start_id,
                last_message_at, last_extracted_id, profile_snapshot, updated_at
            )
            VALUES (?, 1, 0, ?, 0, 0, '', ?)
            ON CONFLICT(session_key) DO UPDATE SET
                window_start_id = excluded.window_start_id,
                updated_at = excluded.updated_at
            """,
            (session_key, int(window_start_id), ts),
        )

    async def advance_extracted_cursor(
        self,
        session_key: str,
        last_extracted_id: int,
        *,
        now: int | None = None,
    ) -> None:
        """Monotonically advance ``last_extracted_id`` for a session.

        Args:
            session_key: Canonical session identifier.
            last_extracted_id: Non-negative extracted message ID watermark.
            now: Optional Unix timestamp in seconds for ``updated_at``.

        Raises:
            ValueError: If ``last_extracted_id`` is negative.
        """
        if last_extracted_id < 0:
            raise ValueError(
                f"last_extracted_id must be non-negative, got {last_extracted_id}."
            )
        ts = int(time.time()) if now is None else int(now)
        await self._db.execute(
            """
            INSERT INTO sessions (
                session_key, episode_id, episode_start_id, window_start_id,
                last_message_at, last_extracted_id, profile_snapshot, updated_at
            )
            VALUES (?, 1, 0, 0, 0, ?, '', ?)
            ON CONFLICT(session_key) DO UPDATE SET
                last_extracted_id = MAX(last_extracted_id, excluded.last_extracted_id),
                updated_at = excluded.updated_at
            """,
            (session_key, int(last_extracted_id), ts),
        )

    async def min_protected_message_id(
        self,
        *,
        active_since: int | None = None,
    ) -> int | None:
        """Return the lowest message ID still protected by active session cursors.

        Computes
        ``MIN(MIN(window_start_id, last_extracted_id, CASE WHEN episode_start_id > 0
        THEN episode_start_id ELSE window_start_id END))`` across ``sessions``
        so expired message cleanup never deletes messages still needed by an
        active context window, open episode, or pending extraction.

        Args:
            active_since: Optional inclusive timestamp lower bound applied to
                ``MAX(updated_at, last_message_at)`` so dormant sessions do not
                block retention cleanup indefinitely.

        Returns:
            Minimum protected message ID, or ``None`` if no matching sessions exist.
        """
        sql = """
            SELECT MIN(
                MIN(
                    window_start_id,
                    last_extracted_id,
                    CASE
                        WHEN episode_start_id > 0 THEN episode_start_id
                        ELSE window_start_id
                    END
                )
            ) AS min_id
            FROM sessions
        """
        params: tuple[Any, ...] = ()
        if active_since is not None:
            sql += " WHERE MAX(updated_at, last_message_at) >= ?"
            params = (int(active_since),)
        row = await self._db.fetchone(sql, params)
        if row is None or row["min_id"] is None:
            return None
        return int(row["min_id"])

    async def _find_cluster_start_id(
        self,
        *,
        group_id: str | None,
        user_id: str,
        floor_id: int,
        batch_min_id: int,
        batch_first_ts: int,
    ) -> int:
        """Walk backwards from ``batch_min_id`` to absorb user lead-in rows."""
        if batch_min_id <= floor_id:
            return batch_min_id

        query_limit = self._config.lead_in_max_messages + 1
        if group_id is not None:
            rows = await self._db.fetchall(
                f"""
                SELECT {_MESSAGE_COLUMNS}
                FROM messages
                WHERE group_id = ? AND id >= ? AND id < ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (str(group_id), int(floor_id), int(batch_min_id), query_limit),
            )
        else:
            rows = await self._db.fetchall(
                f"""
                SELECT {_MESSAGE_COLUMNS}
                FROM messages
                WHERE group_id IS NULL AND user_id = ? AND id >= ? AND id < ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (str(user_id), int(floor_id), int(batch_min_id), query_limit),
            )

        cluster_start_id = batch_min_id
        next_ts = batch_first_ts
        taken = 0
        for row in rows:
            if taken >= self._config.lead_in_max_messages:
                break
            rec = _row_to_message(row)
            if rec.role != "user" or not rec.content.strip():
                break
            if (next_ts - rec.created_at) > self._config.lead_in_gap_sec:
                break
            cluster_start_id = rec.id
            next_ts = rec.created_at
            taken += 1
        return cluster_start_id

    async def _fetch_tail_messages(
        self,
        *,
        group_id: str | None,
        user_id: str,
        start_id: int,
        end_id: int,
        limit: int,
    ) -> list[MessageRecord]:
        """Fetch up to ``limit`` non-empty tail rows in ``[start_id, end_id]``."""
        if end_id < start_id or limit <= 0:
            return []
        if group_id is not None:
            rows = await self._db.fetchall(
                f"""
                SELECT {_MESSAGE_COLUMNS}
                FROM messages
                WHERE group_id = ?
                  AND id >= ?
                  AND id <= ?
                  AND length(trim(content)) > 0
                ORDER BY id DESC
                LIMIT ?
                """,
                (str(group_id), int(start_id), int(end_id), int(limit)),
            )
        else:
            rows = await self._db.fetchall(
                f"""
                SELECT {_MESSAGE_COLUMNS}
                FROM messages
                WHERE group_id IS NULL
                  AND user_id = ?
                  AND id >= ?
                  AND id <= ?
                  AND length(trim(content)) > 0
                ORDER BY id DESC
                LIMIT ?
                """,
                (str(user_id), int(start_id), int(end_id), int(limit)),
            )
        rows.reverse()
        return [_row_to_message(row) for row in rows]

    async def _fetch_episode_bounds(
        self,
        *,
        group_id: str | None,
        user_id: str,
        start_id: int,
        end_id: int,
    ) -> tuple[int, int] | None:
        """Return ``(min_id, max_id)`` of chat rows within ``[start_id, end_id]``."""
        if end_id < start_id:
            return None
        if group_id is not None:
            row = await self._db.fetchone(
                """
                SELECT MIN(id) AS min_id, MAX(id) AS max_id
                FROM messages
                WHERE group_id = ? AND id >= ? AND id <= ?
                """,
                (str(group_id), int(start_id), int(end_id)),
            )
        else:
            row = await self._db.fetchone(
                """
                SELECT MIN(id) AS min_id, MAX(id) AS max_id
                FROM messages
                WHERE group_id IS NULL AND user_id = ? AND id >= ? AND id <= ?
                """,
                (str(user_id), int(start_id), int(end_id)),
            )
        if row is None or row["min_id"] is None or row["max_id"] is None:
            return None
        return int(row["min_id"]), int(row["max_id"])

    async def resolve_episode(
        self,
        *,
        session_key: str,
        group_id: str | None,
        user_id: str,
        batch_messages: Sequence[Any],
        jev_client: JevClient | None = None,
        profile_loader: Callable[[], Awaitable[str]] | None = None,
        advance_extracted_on_close: bool = True,
    ) -> EpisodeResolution:
        """Decide whether an inbound batch continues or starts a new episode.

        All timeout comparisons stay strictly in the message timestamp domain.
        External network or profile-loading I/O is always performed outside
        SQLite transactions.

        Args:
            session_key: Canonical session identifier.
            group_id: Group QQ ID for group chat, or ``None`` for private chat.
            user_id: Peer/sender QQ ID.
            batch_messages: Non-empty sequence of inbound messages with
                ``db_id``, ``timestamp``, ``content``, ``sender_name``,
                ``user_id``, and optional ``reply_to_message_id``.
            jev_client: Optional :class:`JevClient` used for continuity checks
                when idle timeout is exceeded but hard timeout is not reached.
            profile_loader: Optional async callback returning the rendered
                profile Markdown snapshot to freeze at episode start.
            advance_extracted_on_close: When ``True``, advances
                ``last_extracted_id`` to the closed episode's ``end_msg_id``;
                pass ``False`` when an asynchronous extractor callback owns
                extraction cursor advancement.

        Returns:
            :class:`EpisodeResolution` with the updated :class:`SessionRecord`,
            ``new_episode`` flag, decision ``reason``, optional continuity
            probability, and optional :class:`ClosedEpisode`.

        Raises:
            ValueError: If ``batch_messages`` is empty.
        """
        if not batch_messages:
            raise ValueError("batch_messages must not be empty.")

        batch_ts = max(int(m.timestamp) for m in batch_messages)
        batch_first_ts = min(int(m.timestamp) for m in batch_messages)
        batch_min_id = min(int(m.db_id) for m in batch_messages)

        session = await self.get_or_create(session_key, now=batch_ts)

        # Case 1: Brand-new / unengaged session.
        if session.last_message_at == 0:
            cluster_start_id = await self._find_cluster_start_id(
                group_id=group_id,
                user_id=user_id,
                floor_id=session.window_start_id,
                batch_min_id=batch_min_id,
                batch_first_ts=batch_first_ts,
            )
            snapshot = (
                (await profile_loader()).strip() if profile_loader is not None else ""
            )
            next_episode_id = max(1, session.episode_id)
            next_extracted = max(
                session.last_extracted_id,
                max(0, cluster_start_id - 1),
            )
            updated = SessionRecord(
                session_key=session_key,
                episode_id=next_episode_id,
                episode_start_id=cluster_start_id,
                window_start_id=cluster_start_id,
                last_message_at=batch_ts,
                last_extracted_id=next_extracted,
                profile_snapshot=snapshot,
                updated_at=batch_ts,
            )
            await self._db.execute(
                """
                UPDATE sessions
                SET episode_id = ?,
                    episode_start_id = ?,
                    window_start_id = ?,
                    last_message_at = ?,
                    last_extracted_id = ?,
                    profile_snapshot = ?,
                    updated_at = ?
                WHERE session_key = ?
                """,
                (
                    updated.episode_id,
                    updated.episode_start_id,
                    updated.window_start_id,
                    updated.last_message_at,
                    updated.last_extracted_id,
                    updated.profile_snapshot,
                    updated.updated_at,
                    session_key,
                ),
            )
            return EpisodeResolution(
                session=updated,
                new_episode=False,
                reason="initial",
                continuity_probability=None,
                closed_episode=None,
            )

        is_group = group_id is not None
        idle_timeout_sec = (
            self._config.group_idle_timeout_minutes
            if is_group
            else self._config.private_idle_timeout_minutes
        ) * 60
        hard_timeout_sec = (
            self._config.group_hard_timeout_hours
            if is_group
            else self._config.private_hard_timeout_hours
        ) * 3600
        gap = max(0, batch_ts - session.last_message_at)

        # Case 2: Within idle timeout -> continue current episode immediately.
        if gap < idle_timeout_sec:
            next_last_msg = max(session.last_message_at, batch_ts)
            next_updated = max(session.updated_at, batch_ts)
            await self._db.execute(
                """
                UPDATE sessions
                SET last_message_at = ?, updated_at = ?
                WHERE session_key = ?
                """,
                (next_last_msg, next_updated, session_key),
            )
            updated = SessionRecord(
                session_key=session.session_key,
                episode_id=session.episode_id,
                episode_start_id=session.episode_start_id,
                window_start_id=session.window_start_id,
                last_message_at=next_last_msg,
                last_extracted_id=session.last_extracted_id,
                profile_snapshot=session.profile_snapshot,
                updated_at=next_updated,
            )
            return EpisodeResolution(
                session=updated,
                new_episode=False,
                reason="within_idle",
                continuity_probability=None,
                closed_episode=None,
            )

        # Case 3: Idle timeout exceeded -> evaluate hard timeout, reply_to, Jev.
        cluster_start_id = await self._find_cluster_start_id(
            group_id=group_id,
            user_id=user_id,
            floor_id=session.window_start_id,
            batch_min_id=batch_min_id,
            batch_first_ts=batch_first_ts,
        )

        should_continue = False
        reason = ""
        prob: float | None = None

        if gap >= hard_timeout_sec:
            should_continue = False
            reason = "hard_timeout"
        else:
            # Step B: In-window reply_to shortcut (strictly before new cluster).
            for msg in batch_messages:
                raw_reply_id = getattr(msg, "reply_to_message_id", None)
                if not isinstance(raw_reply_id, str) or not raw_reply_id.strip():
                    continue
                target_id = await self._message_store.resolve_platform_message_id(
                    platform_message_id=raw_reply_id,
                    group_id=group_id,
                    user_id=user_id,
                )
                if (
                    target_id is not None
                    and session.window_start_id <= target_id < cluster_start_id
                ):
                    should_continue = True
                    reason = "in_window_reply"
                    break

            # Step C: Jev continuity check against prior episode tail.
            if not should_continue:
                tail = await self._fetch_tail_messages(
                    group_id=group_id,
                    user_id=user_id,
                    start_id=session.window_start_id,
                    end_id=cluster_start_id - 1,
                    limit=self._config.continuity_tail_messages,
                )
                if not tail:
                    should_continue = False
                    reason = "hard_timeout"
                elif jev_client is not None:
                    state = {
                        "chat_type": "group" if is_group else "private",
                        "idle_minutes": round(gap / 60.0, 1),
                        "previous_tail": [
                            self._format_message(r, is_group=is_group) for r in tail
                        ],
                        "new_messages": [
                            line
                            for line in self.format_inbound_batch(
                                batch_messages,
                                is_group=is_group,
                                include_datetime_reminder=False,
                            ).splitlines()
                            if line
                        ],
                    }
                    cont = await jev_client.evaluate_continuity(
                        state,
                        threshold=self._config.continuity_threshold,
                    )
                    prob = cont.probability
                    if cont.degraded:
                        should_continue = False
                        reason = "jev_degraded_new_episode"
                    elif cont.should_continue:
                        should_continue = True
                        reason = "jev_continue"
                    else:
                        should_continue = False
                        reason = "jev_new_episode"
                else:
                    should_continue = False
                    reason = "jev_degraded_new_episode"

        if should_continue:
            next_last_msg = max(session.last_message_at, batch_ts)
            next_updated = max(session.updated_at, batch_ts)
            await self._db.execute(
                """
                UPDATE sessions
                SET last_message_at = ?, updated_at = ?
                WHERE session_key = ?
                """,
                (next_last_msg, next_updated, session_key),
            )
            updated = SessionRecord(
                session_key=session.session_key,
                episode_id=session.episode_id,
                episode_start_id=session.episode_start_id,
                window_start_id=session.window_start_id,
                last_message_at=next_last_msg,
                last_extracted_id=session.last_extracted_id,
                profile_snapshot=session.profile_snapshot,
                updated_at=next_updated,
            )
            return EpisodeResolution(
                session=updated,
                new_episode=False,
                reason=reason,
                continuity_probability=prob,
                closed_episode=None,
            )

        # Start a new episode and close the previous one if non-empty.
        bounds = await self._fetch_episode_bounds(
            group_id=group_id,
            user_id=user_id,
            start_id=session.episode_start_id,
            end_id=cluster_start_id - 1,
        )
        closed_episode: ClosedEpisode | None = None
        if bounds is not None:
            min_id, max_id = bounds
            closed_episode = ClosedEpisode(
                session_key=session_key,
                group_id=group_id,
                user_id=user_id,
                episode_id=session.episode_id,
                start_msg_id=min_id,
                end_msg_id=max_id,
                closed_at=batch_ts,
            )

        snapshot = (
            (await profile_loader()).strip() if profile_loader is not None else ""
        )
        next_extracted = (
            max(session.last_extracted_id, closed_episode.end_msg_id)
            if (advance_extracted_on_close and closed_episode is not None)
            else session.last_extracted_id
        )
        next_last_msg = max(session.last_message_at, batch_ts)
        next_updated = max(session.updated_at, batch_ts)
        updated = SessionRecord(
            session_key=session_key,
            episode_id=session.episode_id + 1,
            episode_start_id=cluster_start_id,
            window_start_id=cluster_start_id,
            last_message_at=next_last_msg,
            last_extracted_id=next_extracted,
            profile_snapshot=snapshot,
            updated_at=next_updated,
        )
        await self._db.execute(
            """
            UPDATE sessions
            SET episode_id = ?,
                episode_start_id = ?,
                window_start_id = ?,
                last_message_at = ?,
                last_extracted_id = ?,
                profile_snapshot = ?,
                updated_at = ?
            WHERE session_key = ?
            """,
            (
                updated.episode_id,
                updated.episode_start_id,
                updated.window_start_id,
                updated.last_message_at,
                updated.last_extracted_id,
                updated.profile_snapshot,
                updated.updated_at,
                session_key,
            ),
        )
        return EpisodeResolution(
            session=updated,
            new_episode=True,
            reason=reason,
            continuity_probability=prob,
            closed_episode=closed_episode,
        )

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

    async def _fetch_range_messages(
        self,
        *,
        group_id: str | None,
        user_id: str,
        start_id: int,
        up_to_id: int | None,
    ) -> list[MessageRecord]:
        """Fetch all messages in ``[start_id, up_to_id]`` ordered ascending."""
        if group_id is not None:
            if up_to_id is None:
                sql = (
                    f"SELECT {_MESSAGE_COLUMNS} FROM messages "
                    "WHERE group_id = ? AND id >= ? "
                    "ORDER BY id ASC"
                )
                params: tuple[Any, ...] = (str(group_id), start_id)
            else:
                sql = (
                    f"SELECT {_MESSAGE_COLUMNS} FROM messages "
                    "WHERE group_id = ? AND id >= ? AND id <= ? "
                    "ORDER BY id ASC"
                )
                params = (str(group_id), start_id, int(up_to_id))
        else:
            if up_to_id is None:
                sql = (
                    f"SELECT {_MESSAGE_COLUMNS} FROM messages "
                    "WHERE group_id IS NULL AND user_id = ? AND id >= ? "
                    "ORDER BY id ASC"
                )
                params = (str(user_id), start_id)
            else:
                sql = (
                    f"SELECT {_MESSAGE_COLUMNS} FROM messages "
                    "WHERE group_id IS NULL AND user_id = ? AND id >= ? AND id <= ? "
                    "ORDER BY id ASC"
                )
                params = (str(user_id), start_id, int(up_to_id))

        rows = await self._db.fetchall(sql, params)
        return [_row_to_message(row) for row in rows]

    def _format_message(self, rec: MessageRecord, *, is_group: bool) -> str:
        if is_group and rec.role == "user":
            hh_mm = datetime.fromtimestamp(rec.created_at, tz=self._tz).strftime(
                "%H:%M"
            )
            return f"[{rec.sender_name}({rec.user_id}) {hh_mm}]: {rec.content}"
        return rec.content

    def format_inbound_batch(
        self,
        messages: Sequence[Any],
        *,
        is_group: bool,
        include_datetime_reminder: bool = True,
    ) -> str:
        """Format an inbound trigger batch as a fallback ``user`` turn."""
        lines: list[str] = []
        for msg in messages:
            content = str(msg.content)
            if not content.strip():
                continue
            if is_group:
                hh_mm = datetime.fromtimestamp(
                    int(msg.timestamp), tz=self._tz
                ).strftime("%H:%M")
                lines.append(f"[{msg.sender_name}({msg.user_id}) {hh_mm}]: {content}")
            else:
                lines.append(content)
        if not lines:
            return ""
        text = "\n".join(lines)
        if include_datetime_reminder and self._config.datetime_system_prompt:
            batch_ts = max(int(m.timestamp) for m in messages)
            reminder = format_datetime_reminder(
                batch_ts,
                self._tz,
                self._config.timezone,
            )
            return f"{text}\n{reminder}"
        return text

    async def build_context(
        self,
        *,
        session_key: str,
        group_id: str | None,
        user_id: str,
        system_prompt: str,
        up_to_id: int | None = None,
        min_keep_id: int | None = None,
        now: int | None = None,
    ) -> list[dict[str, Any]]:
        """Project a session's active message window into OpenAI chat messages.

        Performs prefix-stable intra-episode block rollover when the active
        window exceeds ``config.window_max_messages`` (advancing only
        ``window_start_id`` while preserving ``episode_start_id`` and
        ``profile_snapshot``), joins the frozen ``profile_snapshot`` into the
        ``system`` prompt, skips empty/whitespace rows and leading assistant
        turns, and coalesces adjacent same-role turns into strict alternating
        ``user`` / ``assistant`` messages.

        Args:
            session_key: Canonical session identifier.
            group_id: Group QQ ID for group chat, or ``None`` for private chat.
            user_id: Peer QQ ID (used for private chat message lookup).
            system_prompt: Base persona system prompt text.
            up_to_id: Optional inclusive upper bound on ``messages.id``.
            min_keep_id: Optional inclusive lower bound that rollover must
                never discard (e.g. ``min(m.db_id for m in batch)``).
            now: Optional Unix timestamp in seconds for session creation or
                rollover updates.

        Returns:
            OpenAI-formatted ``messages`` list starting with ``role="system"``.
        """
        session = await self.get_or_create(session_key, now=now)
        effective_start_id = (
            min(session.window_start_id, min_keep_id)
            if min_keep_id is not None
            else session.window_start_id
        )
        fetch_limit = self._config.window_max_messages + 1
        active = await self._fetch_active_messages(
            group_id=group_id,
            user_id=user_id,
            window_start_id=effective_start_id,
            up_to_id=up_to_id,
            limit=fetch_limit,
        )
        if (
            min_keep_id is not None
            and len(active) == fetch_limit
            and active[0].id > min_keep_id
        ):
            active = await self._fetch_range_messages(
                group_id=group_id,
                user_id=user_id,
                start_id=min_keep_id,
                up_to_id=up_to_id,
            )

        if len(active) > self._config.window_max_messages:
            keep_idx = len(active) - self._config.window_keep_messages
            if min_keep_id is not None:
                for idx, rec in enumerate(active):
                    if rec.id >= min_keep_id:
                        keep_idx = min(keep_idx, idx)
                        break

            kept = active[keep_idx:]
            while kept and (kept[0].role == "assistant" or not kept[0].content.strip()):
                kept = kept[1:]

            if kept:
                new_start_id = kept[0].id
                if new_start_id > session.window_start_id:
                    await self.update_window(
                        session_key,
                        window_start_id=new_start_id,
                        now=now,
                    )
                active = kept
            else:
                last_user_idx: int | None = None
                for idx in range(len(active) - 1, -1, -1):
                    if active[idx].role == "user" and active[idx].content.strip():
                        last_user_idx = idx
                        break

                if last_user_idx is not None:
                    kept = active[last_user_idx:]
                    new_start_id = kept[0].id
                    if new_start_id > session.window_start_id:
                        await self.update_window(
                            session_key,
                            window_start_id=new_start_id,
                            now=now,
                        )
                    active = kept
                else:
                    next_start_id = active[-1].id + 1
                    if next_start_id > session.window_start_id:
                        await self.update_window(
                            session_key,
                            window_start_id=next_start_id,
                            now=now,
                        )
                    active = []

        sys_parts: list[str] = []
        base_prompt = system_prompt.strip()
        if base_prompt:
            sys_parts.append(base_prompt)
        snapshot = session.profile_snapshot.strip()
        if snapshot:
            sys_parts.append(snapshot)
        sys_text = "\n\n".join(sys_parts)

        messages: list[dict[str, Any]] = [{"role": "system", "content": sys_text}]
        user_block_timestamps: dict[int, int] = {}

        is_group = group_id is not None
        for rec in active:
            if not rec.content.strip():
                continue
            if len(messages) == 1 and rec.role == "assistant":
                continue
            formatted = self._format_message(rec, is_group=is_group)
            if messages[-1]["role"] == rec.role:
                messages[-1]["content"] = f"{messages[-1]['content']}\n{formatted}"
            else:
                messages.append({"role": rec.role, "content": formatted})
            if rec.role == "user":
                user_block_timestamps[len(messages) - 1] = int(rec.created_at)

        if self._config.datetime_system_prompt:
            for idx, last_user_ts in user_block_timestamps.items():
                reminder = format_datetime_reminder(
                    last_user_ts,
                    self._tz,
                    self._config.timezone,
                )
                messages[idx]["content"] = f"{messages[idx]['content']}\n{reminder}"

        return messages
