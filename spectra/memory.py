"""Long-term memory store with SQLite persistence, privacy isolation, and reranking.

Owns the ``memory`` migration module with four tables:
``group_profiles``, ``user_profiles``, ``memory_items``, and ``group_jargon``.
Enforces strict SQL-level privacy boundaries between private chats and groups,
and never holds SQLite transactions open across network calls.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

import aiosqlite

from spectra.config import MemoryConfig
from spectra.db import Database
from spectra.embedding import (
    EmbeddingClient,
    EmbeddingResult,
    cosine_similarity,
    unpack_vector,
)
from spectra.jev import JevClient

logger = logging.getLogger(__name__)

MemoryKind = Literal["fact", "preference", "event", "promise"]
MemoryVisibility = Literal["public", "group_only", "private"]
MemoryStatus = Literal["active", "superseded", "deleted"]

_VALID_MEMORY_KINDS: frozenset[str] = frozenset(
    {"fact", "preference", "event", "promise"}
)
_VALID_VISIBILITIES: frozenset[str] = frozenset({"public", "group_only", "private"})

_ASCII_WORD_CHAR_RE = re.compile(r"^[A-Za-z0-9_]$")
_ASCII_TOKEN_RE = re.compile(r"[a-z0-9_+#]{2,}")
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]{2,}")

_MIGRATIONS: tuple[str, ...] = (
    """
    CREATE TABLE group_profiles (
        group_id TEXT PRIMARY KEY CHECK (length(trim(group_id)) > 0),
        interaction_style TEXT NOT NULL DEFAULT '',
        frequent_topics TEXT NOT NULL DEFAULT '',
        taboos_and_boundaries TEXT NOT NULL DEFAULT '',
        member_notes TEXT NOT NULL DEFAULT '',
        updated_at INTEGER NOT NULL
    );

    CREATE TABLE user_profiles (
        user_id TEXT PRIMARY KEY CHECK (length(trim(user_id)) > 0),
        aliases TEXT NOT NULL DEFAULT '[]'
            CHECK (json_valid(aliases) AND json_type(aliases) = 'array'),
        interaction_style TEXT NOT NULL DEFAULT '',
        topic_preferences TEXT NOT NULL DEFAULT '',
        relationship TEXT NOT NULL DEFAULT '',
        recent_focus TEXT NOT NULL DEFAULT '',
        private_notes TEXT NOT NULL DEFAULT '',
        updated_at INTEGER NOT NULL
    );

    CREATE TABLE memory_items (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        kind TEXT NOT NULL CHECK (kind IN ('fact', 'preference', 'event', 'promise')),
        content TEXT NOT NULL CHECK (length(trim(content)) > 0),
        owner_user_id TEXT,
        source_group_id TEXT,
        visibility TEXT NOT NULL
            CHECK (visibility IN ('public', 'group_only', 'private')),
        embedding BLOB,
        embedding_model TEXT,
        status TEXT NOT NULL DEFAULT 'active'
            CHECK (status IN ('active', 'superseded', 'deleted')),
        superseded_by INTEGER REFERENCES memory_items(id) ON DELETE SET NULL,
        confidence REAL NOT NULL DEFAULT 0.7
            CHECK (confidence >= 0.0 AND confidence <= 1.0),
        evidence_count INTEGER NOT NULL DEFAULT 1 CHECK (evidence_count >= 1),
        source_msg_start_id INTEGER,
        source_msg_end_id INTEGER,
        event_time TEXT,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL,
        last_accessed_at INTEGER NOT NULL DEFAULT 0,
        CHECK (owner_user_id IS NULL OR length(trim(owner_user_id)) > 0),
        CHECK (source_group_id IS NULL OR length(trim(source_group_id)) > 0),
        CHECK (
            visibility != 'private'
            OR (owner_user_id IS NOT NULL AND source_group_id IS NULL)
        ),
        CHECK (visibility != 'group_only' OR source_group_id IS NOT NULL),
        CHECK (owner_user_id IS NOT NULL OR source_group_id IS NOT NULL),
        CHECK (superseded_by IS NULL OR superseded_by != id)
    );

    CREATE INDEX idx_memory_items_active_group
        ON memory_items(source_group_id, visibility)
        WHERE status = 'active';

    CREATE INDEX idx_memory_items_active_owner
        ON memory_items(owner_user_id, visibility)
        WHERE status = 'active';

    CREATE TABLE group_jargon (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        group_id TEXT NOT NULL CHECK (length(trim(group_id)) > 0),
        term TEXT NOT NULL COLLATE NOCASE CHECK (length(trim(term)) > 0),
        meaning TEXT NOT NULL CHECK (length(trim(meaning)) > 0),
        confidence REAL NOT NULL DEFAULT 0.7
            CHECK (confidence >= 0.0 AND confidence <= 1.0),
        evidence_count INTEGER NOT NULL DEFAULT 1 CHECK (evidence_count >= 1),
        updated_at INTEGER NOT NULL,
        UNIQUE (group_id, term)
    );

    CREATE INDEX idx_group_jargon_group_conf
        ON group_jargon(group_id, confidence DESC, evidence_count DESC);
    """,
)

_GROUP_PROFILE_COLUMNS = (
    "group_id, interaction_style, frequent_topics, "
    "taboos_and_boundaries, member_notes, updated_at"
)
_USER_PROFILE_COLUMNS = (
    "user_id, aliases, interaction_style, topic_preferences, "
    "relationship, recent_focus, private_notes, updated_at"
)
_PUBLIC_USER_PROFILE_COLUMNS = (
    "user_id, aliases, interaction_style, topic_preferences, "
    "relationship, recent_focus, updated_at"
)
_MEMORY_ITEM_COLUMNS = (
    "id, kind, content, owner_user_id, source_group_id, visibility, "
    "embedding, embedding_model, status, superseded_by, confidence, "
    "evidence_count, source_msg_start_id, source_msg_end_id, event_time, "
    "created_at, updated_at, last_accessed_at"
)
_JARGON_COLUMNS = "id, group_id, term, meaning, confidence, evidence_count, updated_at"


@dataclass(slots=True, frozen=True)
class GroupProfile:
    """Immutable row from ``group_profiles``."""

    group_id: str
    interaction_style: str
    frequent_topics: str
    taboos_and_boundaries: str
    member_notes: str
    updated_at: int


@dataclass(slots=True, frozen=True)
class UserProfile:
    """Immutable row from ``user_profiles`` including private DM notes."""

    user_id: str
    aliases: tuple[str, ...]
    interaction_style: str
    topic_preferences: str
    relationship: str
    recent_focus: str
    private_notes: str
    updated_at: int


@dataclass(slots=True, frozen=True)
class PublicUserProfile:
    """Public projection of ``user_profiles`` without ``private_notes``."""

    user_id: str
    aliases: tuple[str, ...]
    interaction_style: str
    topic_preferences: str
    relationship: str
    recent_focus: str
    updated_at: int


@dataclass(slots=True, frozen=True)
class MemoryItem:
    """Immutable row from ``memory_items``."""

    id: int
    kind: MemoryKind
    content: str
    owner_user_id: str | None
    source_group_id: str | None
    visibility: MemoryVisibility
    embedding: bytes | None
    embedding_model: str | None
    status: MemoryStatus
    superseded_by: int | None
    confidence: float
    evidence_count: int
    source_msg_start_id: int | None
    source_msg_end_id: int | None
    event_time: str | None
    created_at: int
    updated_at: int
    last_accessed_at: int


@dataclass(slots=True, frozen=True)
class MemorySearchHit:
    """Single memory item returned by :meth:`MemoryStore.search_memories`."""

    item: MemoryItem
    vector_score: float | None
    rerank_score: float | None


@dataclass(slots=True, frozen=True)
class JargonEntry:
    """Immutable row from ``group_jargon``."""

    id: int
    group_id: str
    term: str
    meaning: str
    confidence: float
    evidence_count: int
    updated_at: int


@dataclass(slots=True, frozen=True)
class _Stage1Candidate:
    item: MemoryItem
    vector_score: float | None
    sort_key: tuple[int, float, float, int, int]


def _resolve_timestamp(now: int | None) -> int:
    return int(time.time()) if now is None else int(now)


def _require_non_empty(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _normalize_optional_id(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Expected string or None, got {type(value).__name__}.")
    cleaned = value.strip()
    return cleaned if cleaned else None


def _validate_unit_float(value: float, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or not (0.0 <= float(value) <= 1.0)
    ):
        raise ValueError(
            f"{field_name} must be a finite float in [0.0, 1.0], got {value!r}."
        )
    return float(value)


def _validate_non_negative_float(value: float, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{field_name} must be a finite float >= 0.0, got {value!r}.")
    return float(value)


def _validate_positive_int(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be an integer >= 1, got {value!r}.")
    return value


def _validate_non_negative_int(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be an integer >= 0, got {value!r}.")
    return value


def _validate_optional_int(value: int | None, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer or None, got {value!r}.")
    return value


def _dedupe_aliases(raw_aliases: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for entry in raw_aliases:
        cleaned = str(entry).strip()
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result


def _parse_aliases_json(raw: Any) -> tuple[str, ...]:
    if not isinstance(raw, str):
        return ()
    try:
        parsed = json.loads(raw)
    except Exception:
        return ()
    if not isinstance(parsed, list):
        return ()
    return tuple(_dedupe_aliases([str(item) for item in parsed]))


def _normalize_speaker_ids(active_speaker_ids: Sequence[str]) -> list[str]:
    if isinstance(active_speaker_ids, (str, bytes, bytearray)):
        raise TypeError(
            "active_speaker_ids must be a sequence of user IDs, not a single str."
        )
    seen: set[str] = set()
    result: list[str] = []
    for raw in active_speaker_ids:
        cleaned = _normalize_optional_id(raw)
        if cleaned is not None and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result


def _validate_memory_fields(
    *,
    kind: MemoryKind,
    content: str,
    visibility: MemoryVisibility,
    owner_user_id: str | None,
    source_group_id: str | None,
    confidence: float,
    evidence_count: int,
    source_msg_start_id: int | None,
    source_msg_end_id: int | None,
    event_time: str | None,
) -> tuple[
    MemoryKind,
    str,
    MemoryVisibility,
    str | None,
    str | None,
    float,
    int,
    int | None,
    int | None,
    str | None,
]:
    if kind not in _VALID_MEMORY_KINDS:
        raise ValueError(f"Invalid memory kind: {kind!r}")
    norm_content = _require_non_empty(content, "content")
    if visibility not in _VALID_VISIBILITIES:
        raise ValueError(f"Invalid memory visibility: {visibility!r}")

    norm_owner = _normalize_optional_id(owner_user_id)
    norm_group = _normalize_optional_id(source_group_id)
    norm_event_time = _normalize_optional_id(event_time)

    if visibility == "private" and (norm_owner is None or norm_group is not None):
        raise ValueError(
            "Private memories require non-empty owner_user_id and source_group_id=None."
        )
    if visibility == "group_only" and norm_group is None:
        raise ValueError("Group-only memories require non-empty source_group_id.")
    if norm_owner is None and norm_group is None:
        raise ValueError(
            "Memory item requires at least one of owner_user_id or source_group_id."
        )

    norm_conf = _validate_unit_float(confidence, "confidence")
    norm_evidence = _validate_positive_int(evidence_count, "evidence_count")
    norm_start = _validate_optional_int(source_msg_start_id, "source_msg_start_id")
    norm_end = _validate_optional_int(source_msg_end_id, "source_msg_end_id")
    return (
        kind,
        norm_content,
        visibility,
        norm_owner,
        norm_group,
        norm_conf,
        norm_evidence,
        norm_start,
        norm_end,
        norm_event_time,
    )


def _row_to_group_profile(row: aiosqlite.Row) -> GroupProfile:
    return GroupProfile(
        group_id=str(row["group_id"]),
        interaction_style=str(row["interaction_style"]),
        frequent_topics=str(row["frequent_topics"]),
        taboos_and_boundaries=str(row["taboos_and_boundaries"]),
        member_notes=str(row["member_notes"]),
        updated_at=int(row["updated_at"]),
    )


def _row_to_user_profile(row: aiosqlite.Row) -> UserProfile:
    return UserProfile(
        user_id=str(row["user_id"]),
        aliases=_parse_aliases_json(row["aliases"]),
        interaction_style=str(row["interaction_style"]),
        topic_preferences=str(row["topic_preferences"]),
        relationship=str(row["relationship"]),
        recent_focus=str(row["recent_focus"]),
        private_notes=str(row["private_notes"]),
        updated_at=int(row["updated_at"]),
    )


def _row_to_public_user_profile(row: aiosqlite.Row) -> PublicUserProfile:
    return PublicUserProfile(
        user_id=str(row["user_id"]),
        aliases=_parse_aliases_json(row["aliases"]),
        interaction_style=str(row["interaction_style"]),
        topic_preferences=str(row["topic_preferences"]),
        relationship=str(row["relationship"]),
        recent_focus=str(row["recent_focus"]),
        updated_at=int(row["updated_at"]),
    )


def _row_to_memory_item(row: aiosqlite.Row) -> MemoryItem:
    raw_emb = row["embedding"]
    emb_bytes = bytes(raw_emb) if raw_emb is not None else None
    return MemoryItem(
        id=int(row["id"]),
        kind=row["kind"],
        content=str(row["content"]),
        owner_user_id=row["owner_user_id"],
        source_group_id=row["source_group_id"],
        visibility=row["visibility"],
        embedding=emb_bytes,
        embedding_model=row["embedding_model"],
        status=row["status"],
        superseded_by=(
            int(row["superseded_by"]) if row["superseded_by"] is not None else None
        ),
        confidence=float(row["confidence"]),
        evidence_count=int(row["evidence_count"]),
        source_msg_start_id=(
            int(row["source_msg_start_id"])
            if row["source_msg_start_id"] is not None
            else None
        ),
        source_msg_end_id=(
            int(row["source_msg_end_id"])
            if row["source_msg_end_id"] is not None
            else None
        ),
        event_time=row["event_time"],
        created_at=int(row["created_at"]),
        updated_at=int(row["updated_at"]),
        last_accessed_at=int(row["last_accessed_at"]),
    )


def _row_to_jargon_entry(row: aiosqlite.Row) -> JargonEntry:
    return JargonEntry(
        id=int(row["id"]),
        group_id=str(row["group_id"]),
        term=str(row["term"]),
        meaning=str(row["meaning"]),
        confidence=float(row["confidence"]),
        evidence_count=int(row["evidence_count"]),
        updated_at=int(row["updated_at"]),
    )


def _extract_lexical_tokens(query_cf: str) -> list[str]:
    seen: set[str] = set()
    tokens: list[str] = []

    def _add(tok: str) -> None:
        if tok not in seen:
            seen.add(tok)
            tokens.append(tok)

    for word in _ASCII_TOKEN_RE.findall(query_cf):
        _add(word)
    for run in _CJK_RUN_RE.findall(query_cf):
        for i in range(len(run) - 1):
            _add(run[i : i + 2])
    return tokens


def _lexical_match_score(query: str, content: str) -> float:
    """Compute a lightweight case-insensitive lexical relevance score in [0, 1]."""
    query_cf = query.strip().casefold()
    if not query_cf:
        return 0.0
    content_cf = content.casefold()
    full_hit = len(query_cf) >= 2 and query_cf in content_cf

    tokens = _extract_lexical_tokens(query_cf)
    if not tokens:
        if full_hit:
            return 1.0
        return 1.0 if len(query_cf) == 1 and query_cf in content_cf else 0.0

    hit_count = sum(1 for tok in tokens if tok in content_cf)
    ratio = hit_count / len(tokens)
    if full_hit:
        return 1.0
    return ratio


def _compile_jargon_pattern(term: str) -> re.Pattern[str]:
    prefix = r"(?<![A-Za-z0-9_])" if _ASCII_WORD_CHAR_RE.match(term[0]) else ""
    suffix = r"(?![A-Za-z0-9_+#])" if _ASCII_WORD_CHAR_RE.match(term[-1]) else ""
    return re.compile(f"{prefix}{re.escape(term)}{suffix}", re.IGNORECASE | re.ASCII)


def _build_visible_active_query(
    *,
    group_id: str | None,
    user_id: str | None,
    active_speaker_ids: Sequence[str] = (),
) -> tuple[str, list[Any]]:
    """Build SQL query and params enforcing strict active-memory privacy rules."""
    if group_id is not None:
        norm_group = _require_non_empty(group_id, "group_id")
        speakers = _normalize_speaker_ids(active_speaker_ids)
        if speakers:
            placeholders = ", ".join("?" for _ in speakers)
            sql = f"""
                SELECT {_MEMORY_ITEM_COLUMNS}
                FROM memory_items
                WHERE id IN (
                    SELECT id FROM memory_items
                    WHERE status = 'active'
                      AND source_group_id = ?
                      AND visibility IN ('group_only', 'public')
                    UNION
                    SELECT id FROM memory_items
                    WHERE status = 'active'
                      AND visibility = 'public'
                      AND owner_user_id IN ({placeholders})
                )
            """
            return sql, [norm_group, *speakers]
        sql = f"""
            SELECT {_MEMORY_ITEM_COLUMNS}
            FROM memory_items
            WHERE status = 'active'
              AND source_group_id = ?
              AND visibility IN ('group_only', 'public')
        """
        return sql, [norm_group]

    norm_user = _normalize_optional_id(user_id)
    if norm_user is None:
        raise ValueError("user_id must be non-empty when group_id is None.")
    sql = f"""
        SELECT {_MEMORY_ITEM_COLUMNS}
        FROM memory_items
        WHERE status = 'active'
          AND owner_user_id = ?
          AND visibility IN ('private', 'public')
    """
    return sql, [norm_user]


class MemoryStore:
    """Async store for profiles, long-term memories, and group jargon.

    Args:
        db: Connected :class:`spectra.db.Database` instance.
        config: :class:`spectra.config.MemoryConfig` controlling recall and
            jargon thresholds.
        embedding_client: Optional :class:`spectra.embedding.EmbeddingClient`
            used to embed memory content and search queries outside SQLite
            transactions.
        jev_client: Optional :class:`spectra.jev.JevClient` used for stage-2
            relevance reranking outside SQLite transactions.
    """

    def __init__(
        self,
        db: Database,
        config: MemoryConfig,
        *,
        embedding_client: EmbeddingClient | None = None,
        jev_client: JevClient | None = None,
    ) -> None:
        self._db = db
        self._config = config
        self._embedding_client = embedding_client
        self._jev_client = jev_client

    async def init(self) -> None:
        """Apply schema migrations for the ``memory`` module."""
        await self._db.migrate("memory", _MIGRATIONS)

    # ------------------------------------------------------------------
    # Group profiles
    # ------------------------------------------------------------------

    async def get_group_profile(self, group_id: str) -> GroupProfile | None:
        """Fetch the profile for ``group_id``, or ``None`` if absent."""
        norm_gid = _normalize_optional_id(group_id)
        if norm_gid is None:
            return None
        row = await self._db.fetchone(
            f"SELECT {_GROUP_PROFILE_COLUMNS} FROM group_profiles WHERE group_id = ?",
            (norm_gid,),
        )
        return _row_to_group_profile(row) if row is not None else None

    async def patch_group_profile(
        self,
        group_id: str,
        *,
        interaction_style: str | None = None,
        frequent_topics: str | None = None,
        taboos_and_boundaries: str | None = None,
        member_notes: str | None = None,
        now: int | None = None,
    ) -> GroupProfile:
        """Create or partially update a group profile.

        Arguments left as ``None`` preserve existing column values; string
        arguments replace them (empty string clears a field).
        """
        norm_gid = _require_non_empty(group_id, "group_id")
        ts = _resolve_timestamp(now)

        async with self._db.transaction() as tx:
            row = await tx.fetchone(
                f"""
                SELECT {_GROUP_PROFILE_COLUMNS}
                FROM group_profiles
                WHERE group_id = ?
                """,
                (norm_gid,),
            )
            existing = _row_to_group_profile(row) if row is not None else None

            next_style = (
                (existing.interaction_style if existing else "")
                if interaction_style is None
                else interaction_style.strip()
            )
            next_topics = (
                (existing.frequent_topics if existing else "")
                if frequent_topics is None
                else frequent_topics.strip()
            )
            next_taboos = (
                (existing.taboos_and_boundaries if existing else "")
                if taboos_and_boundaries is None
                else taboos_and_boundaries.strip()
            )
            next_notes = (
                (existing.member_notes if existing else "")
                if member_notes is None
                else member_notes.strip()
            )

            await tx.execute(
                """
                INSERT INTO group_profiles (
                    group_id,
                    interaction_style,
                    frequent_topics,
                    taboos_and_boundaries,
                    member_notes,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(group_id) DO UPDATE SET
                    interaction_style = excluded.interaction_style,
                    frequent_topics = excluded.frequent_topics,
                    taboos_and_boundaries = excluded.taboos_and_boundaries,
                    member_notes = excluded.member_notes,
                    updated_at = excluded.updated_at
                """,
                (
                    norm_gid,
                    next_style,
                    next_topics,
                    next_taboos,
                    next_notes,
                    ts,
                ),
            )

        return GroupProfile(
            group_id=norm_gid,
            interaction_style=next_style,
            frequent_topics=next_topics,
            taboos_and_boundaries=next_taboos,
            member_notes=next_notes,
            updated_at=ts,
        )

    async def render_group_profile(self, group_id: str) -> str:
        """Render a compact Chinese Markdown section for the group profile."""
        profile = await self.get_group_profile(group_id)
        if profile is None:
            return ""
        lines: list[str] = []
        if profile.interaction_style:
            lines.append(f"- 相处风格：{profile.interaction_style}")
        if profile.frequent_topics:
            lines.append(f"- 常见话题：{profile.frequent_topics}")
        if profile.taboos_and_boundaries:
            lines.append(f"- 禁忌与边界：{profile.taboos_and_boundaries}")
        if profile.member_notes:
            lines.append(f"- 成员备注：{profile.member_notes}")
        if not lines:
            return ""
        return "## 当前群聊画像\n" + "\n".join(lines)

    # ------------------------------------------------------------------
    # User profiles
    # ------------------------------------------------------------------

    async def get_user_profile(self, user_id: str) -> UserProfile | None:
        """Fetch full user profile (including ``private_notes``) for DM use."""
        norm_uid = _normalize_optional_id(user_id)
        if norm_uid is None:
            return None
        row = await self._db.fetchone(
            f"SELECT {_USER_PROFILE_COLUMNS} FROM user_profiles WHERE user_id = ?",
            (norm_uid,),
        )
        return _row_to_user_profile(row) if row is not None else None

    async def get_public_user_profile(
        self,
        user_id: str,
    ) -> PublicUserProfile | None:
        """Fetch public-only user profile; SQL explicitly excludes ``private_notes``."""
        norm_uid = _normalize_optional_id(user_id)
        if norm_uid is None:
            return None
        row = await self._db.fetchone(
            f"""
            SELECT {_PUBLIC_USER_PROFILE_COLUMNS}
            FROM user_profiles
            WHERE user_id = ?
            """,
            (norm_uid,),
        )
        return _row_to_public_user_profile(row) if row is not None else None

    async def patch_user_profile(
        self,
        user_id: str,
        *,
        aliases: Sequence[str] | None = None,
        add_aliases: Sequence[str] = (),
        interaction_style: str | None = None,
        topic_preferences: str | None = None,
        relationship: str | None = None,
        recent_focus: str | None = None,
        private_notes: str | None = None,
        now: int | None = None,
    ) -> UserProfile:
        """Create or partially update a user profile with alias deduplication."""
        norm_uid = _require_non_empty(user_id, "user_id")
        if aliases is not None and isinstance(aliases, (str, bytes, bytearray)):
            raise TypeError("aliases must be a sequence of strings, not a single str.")
        if isinstance(add_aliases, (str, bytes, bytearray)):
            raise TypeError(
                "add_aliases must be a sequence of strings, not a single str."
            )
        ts = _resolve_timestamp(now)

        async with self._db.transaction() as tx:
            row = await tx.fetchone(
                f"""
                SELECT {_USER_PROFILE_COLUMNS}
                FROM user_profiles
                WHERE user_id = ?
                """,
                (norm_uid,),
            )
            existing = _row_to_user_profile(row) if row is not None else None

            base_aliases: list[str] = (
                list(aliases)
                if aliases is not None
                else list(existing.aliases if existing else ())
            )
            if add_aliases:
                base_aliases.extend(add_aliases)
            merged_aliases = _dedupe_aliases(base_aliases)

            next_style = (
                (existing.interaction_style if existing else "")
                if interaction_style is None
                else interaction_style.strip()
            )
            next_topics = (
                (existing.topic_preferences if existing else "")
                if topic_preferences is None
                else topic_preferences.strip()
            )
            next_relationship = (
                (existing.relationship if existing else "")
                if relationship is None
                else relationship.strip()
            )
            next_focus = (
                (existing.recent_focus if existing else "")
                if recent_focus is None
                else recent_focus.strip()
            )
            next_private = (
                (existing.private_notes if existing else "")
                if private_notes is None
                else private_notes.strip()
            )

            await tx.execute(
                """
                INSERT INTO user_profiles (
                    user_id,
                    aliases,
                    interaction_style,
                    topic_preferences,
                    relationship,
                    recent_focus,
                    private_notes,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    aliases = excluded.aliases,
                    interaction_style = excluded.interaction_style,
                    topic_preferences = excluded.topic_preferences,
                    relationship = excluded.relationship,
                    recent_focus = excluded.recent_focus,
                    private_notes = excluded.private_notes,
                    updated_at = excluded.updated_at
                """,
                (
                    norm_uid,
                    json.dumps(merged_aliases, ensure_ascii=False),
                    next_style,
                    next_topics,
                    next_relationship,
                    next_focus,
                    next_private,
                    ts,
                ),
            )

        return UserProfile(
            user_id=norm_uid,
            aliases=tuple(merged_aliases),
            interaction_style=next_style,
            topic_preferences=next_topics,
            relationship=next_relationship,
            recent_focus=next_focus,
            private_notes=next_private,
            updated_at=ts,
        )

    async def render_private_profile(self, user_id: str) -> str:
        """Render a compact Chinese Markdown section for a DM user profile."""
        profile = await self.get_user_profile(user_id)
        if profile is None:
            return ""
        lines: list[str] = []
        if profile.aliases:
            lines.append(f"- 昵称/别名：{'、'.join(profile.aliases)}")
        if profile.interaction_style:
            lines.append(f"- 相处风格：{profile.interaction_style}")
        if profile.topic_preferences:
            lines.append(f"- 话题偏好：{profile.topic_preferences}")
        if profile.relationship:
            lines.append(f"- 关系定位：{profile.relationship}")
        if profile.recent_focus:
            lines.append(f"- 近期关注：{profile.recent_focus}")
        if profile.private_notes:
            lines.append(f"- 私聊备注：{profile.private_notes}")
        if not lines:
            return ""
        return "## 当前私聊对象画像\n" + "\n".join(lines)

    async def render_public_user_hint(self, user_id: str) -> str:
        """Render a public user hint using :meth:`get_public_user_profile` only."""
        profile = await self.get_public_user_profile(user_id)
        if profile is None:
            return ""
        parts: list[str] = []
        if profile.aliases:
            parts.append(f"别名：{'、'.join(profile.aliases)}")
        if profile.interaction_style:
            parts.append(f"相处风格：{profile.interaction_style}")
        if profile.topic_preferences:
            parts.append(f"话题偏好：{profile.topic_preferences}")
        if profile.relationship:
            parts.append(f"关系：{profile.relationship}")
        if profile.recent_focus:
            parts.append(f"近期关注：{profile.recent_focus}")
        if not parts:
            return ""
        return f"- 用户 {profile.user_id}：{'；'.join(parts)}"

    # ------------------------------------------------------------------
    # Memory items
    # ------------------------------------------------------------------

    async def add_memory(
        self,
        *,
        kind: MemoryKind,
        content: str,
        visibility: MemoryVisibility,
        owner_user_id: str | None = None,
        source_group_id: str | None = None,
        confidence: float = 0.7,
        evidence_count: int = 1,
        source_msg_start_id: int | None = None,
        source_msg_end_id: int | None = None,
        event_time: str | None = None,
        embedding: EmbeddingResult | None = None,
        now: int | None = None,
    ) -> int:
        """Insert a new active memory item, computing its embedding outside any tx."""
        (
            norm_kind,
            norm_content,
            norm_vis,
            norm_owner,
            norm_group,
            norm_conf,
            norm_evidence,
            norm_start,
            norm_end,
            norm_event_time,
        ) = _validate_memory_fields(
            kind=kind,
            content=content,
            visibility=visibility,
            owner_user_id=owner_user_id,
            source_group_id=source_group_id,
            confidence=confidence,
            evidence_count=evidence_count,
            source_msg_start_id=source_msg_start_id,
            source_msg_end_id=source_msg_end_id,
            event_time=event_time,
        )

        resolved_embedding = embedding
        if resolved_embedding is None and self._embedding_client is not None:
            resolved_embedding = await self._embedding_client.embed_one(norm_content)

        emb_blob = resolved_embedding.blob if resolved_embedding is not None else None
        emb_model = (
            resolved_embedding.model_tag if resolved_embedding is not None else None
        )
        ts = _resolve_timestamp(now)

        result = await self._db.execute(
            """
            INSERT INTO memory_items (
                kind,
                content,
                owner_user_id,
                source_group_id,
                visibility,
                embedding,
                embedding_model,
                status,
                superseded_by,
                confidence,
                evidence_count,
                source_msg_start_id,
                source_msg_end_id,
                event_time,
                created_at,
                updated_at,
                last_accessed_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, 'active', NULL, ?, ?, ?, ?, ?, ?, ?, 0)
            """,
            (
                norm_kind,
                norm_content,
                norm_owner,
                norm_group,
                norm_vis,
                emb_blob,
                emb_model,
                norm_conf,
                norm_evidence,
                norm_start,
                norm_end,
                norm_event_time,
                ts,
                ts,
            ),
        )
        return result.lastrowid

    async def get_memory(self, memory_id: int) -> MemoryItem | None:
        """Fetch a single memory item by primary key, regardless of status."""
        row = await self._db.fetchone(
            f"SELECT {_MEMORY_ITEM_COLUMNS} FROM memory_items WHERE id = ?",
            (int(memory_id),),
        )
        return _row_to_memory_item(row) if row is not None else None

    async def list_active_memories(
        self,
        *,
        group_id: str | None = None,
        user_id: str | None = None,
        active_speaker_ids: Sequence[str] = (),
        limit: int = 50,
    ) -> list[MemoryItem]:
        """List active memories visible in the given group or private scope."""
        norm_limit = _validate_non_negative_int(limit, "limit")
        base_sql, params = _build_visible_active_query(
            group_id=group_id,
            user_id=user_id,
            active_speaker_ids=active_speaker_ids,
        )
        if norm_limit == 0:
            return []
        rows = await self._db.fetchall(
            f"{base_sql}\nORDER BY updated_at DESC, id DESC\nLIMIT ?",
            [*params, norm_limit],
        )
        return [_row_to_memory_item(row) for row in rows]

    async def reinforce_memory(
        self,
        memory_id: int,
        *,
        confidence_boost: float = 0.05,
        source_msg_end_id: int | None = None,
        now: int | None = None,
    ) -> bool:
        """Increment ``evidence_count`` and boost confidence of an active memory."""
        boost = _validate_non_negative_float(confidence_boost, "confidence_boost")
        norm_end = _validate_optional_int(source_msg_end_id, "source_msg_end_id")
        ts = _resolve_timestamp(now)
        result = await self._db.execute(
            """
            UPDATE memory_items
            SET evidence_count = evidence_count + 1,
                confidence = MIN(1.0, confidence + ?),
                source_msg_end_id = COALESCE(?, source_msg_end_id),
                updated_at = ?
            WHERE id = ? AND status = 'active'
            """,
            (boost, norm_end, ts, int(memory_id)),
        )
        return result.rowcount > 0

    async def supersede_memory(
        self,
        old_memory_id: int,
        *,
        kind: MemoryKind,
        content: str,
        visibility: MemoryVisibility,
        owner_user_id: str | None = None,
        source_group_id: str | None = None,
        confidence: float = 0.7,
        source_msg_start_id: int | None = None,
        source_msg_end_id: int | None = None,
        event_time: str | None = None,
        embedding: EmbeddingResult | None = None,
        now: int | None = None,
    ) -> int:
        """Atomically replace an active memory item with a newer version.

        Computes any required embedding outside the SQLite transaction, then
        inserts the new row and marks ``old_memory_id`` as ``superseded`` in
        one atomic ``BEGIN IMMEDIATE`` transaction.
        """
        (
            norm_kind,
            norm_content,
            norm_vis,
            norm_owner,
            norm_group,
            norm_conf,
            _,
            norm_start,
            norm_end,
            norm_event_time,
        ) = _validate_memory_fields(
            kind=kind,
            content=content,
            visibility=visibility,
            owner_user_id=owner_user_id,
            source_group_id=source_group_id,
            confidence=confidence,
            evidence_count=1,
            source_msg_start_id=source_msg_start_id,
            source_msg_end_id=source_msg_end_id,
            event_time=event_time,
        )
        old_id = int(old_memory_id)

        # Pre-check outside tx so we avoid paying an embedding API call for an
        # obviously missing/inactive row; the authoritative check still runs
        # inside the transaction below.
        pre_row = await self._db.fetchone(
            "SELECT status FROM memory_items WHERE id = ?",
            (old_id,),
        )
        if pre_row is None:
            raise ValueError(f"Memory item {old_id} does not exist.")
        if str(pre_row["status"]) != "active":
            raise ValueError(
                f"Memory item {old_id} is not active (status={pre_row['status']!r})."
            )

        resolved_embedding = embedding
        if resolved_embedding is None and self._embedding_client is not None:
            resolved_embedding = await self._embedding_client.embed_one(norm_content)

        emb_blob = resolved_embedding.blob if resolved_embedding is not None else None
        emb_model = (
            resolved_embedding.model_tag if resolved_embedding is not None else None
        )
        ts = _resolve_timestamp(now)

        async with self._db.transaction() as tx:
            row = await tx.fetchone(
                "SELECT status FROM memory_items WHERE id = ?",
                (old_id,),
            )
            if row is None:
                raise ValueError(f"Memory item {old_id} does not exist.")
            if str(row["status"]) != "active":
                raise ValueError(
                    f"Memory item {old_id} is not active (status={row['status']!r})."
                )

            insert_res = await tx.execute(
                """
                INSERT INTO memory_items (
                    kind,
                    content,
                    owner_user_id,
                    source_group_id,
                    visibility,
                    embedding,
                    embedding_model,
                    status,
                    superseded_by,
                    confidence,
                    evidence_count,
                    source_msg_start_id,
                    source_msg_end_id,
                    event_time,
                    created_at,
                    updated_at,
                    last_accessed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, 'active', NULL, ?, 1, ?, ?, ?, ?, ?, 0)
                """,
                (
                    norm_kind,
                    norm_content,
                    norm_owner,
                    norm_group,
                    norm_vis,
                    emb_blob,
                    emb_model,
                    norm_conf,
                    norm_start,
                    norm_end,
                    norm_event_time,
                    ts,
                    ts,
                ),
            )
            new_id = insert_res.lastrowid
            update_res = await tx.execute(
                """
                UPDATE memory_items
                SET status = 'superseded',
                    superseded_by = ?,
                    updated_at = ?
                WHERE id = ? AND status = 'active'
                """,
                (new_id, ts, old_id),
            )
            if update_res.rowcount != 1:
                raise RuntimeError(
                    f"Failed to mark memory item {old_id} as superseded."
                )
            return new_id

    async def delete_memory(
        self,
        memory_id: int,
        *,
        now: int | None = None,
    ) -> bool:
        """Soft-delete a memory item by setting ``status = 'deleted'``."""
        ts = _resolve_timestamp(now)
        result = await self._db.execute(
            """
            UPDATE memory_items
            SET status = 'deleted',
                updated_at = ?
            WHERE id = ? AND status != 'deleted'
            """,
            (ts, int(memory_id)),
        )
        return result.rowcount > 0

    async def search_memories(
        self,
        query: str,
        *,
        group_id: str | None,
        user_id: str | None = None,
        active_speaker_ids: Sequence[str] = (),
        limit: int | None = None,
        now: int | None = None,
    ) -> list[MemorySearchHit]:
        """Search visible active memories using vector/lexical recall + Jev rerank.

        Never performs HTTP requests inside a SQLite transaction. Updates
        ``last_accessed_at`` for the returned memory items.
        """
        if not self._config.enabled:
            return []
        cleaned_query = query.strip() if isinstance(query, str) else ""
        if not cleaned_query:
            return []

        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int)
        ):
            raise ValueError(f"limit must be an int or None, got {limit!r}.")
        effective_limit = self._config.max_recall_items if limit is None else limit

        base_sql, params = _build_visible_active_query(
            group_id=group_id,
            user_id=user_id,
            active_speaker_ids=active_speaker_ids,
        )
        if effective_limit <= 0:
            return []

        rows = await self._db.fetchall(
            f"{base_sql}\nORDER BY updated_at DESC, id DESC",
            params,
        )
        if not rows:
            return []
        visible_items = [_row_to_memory_item(row) for row in rows]

        # Stage 1: Vector similarity + lexical fallback (outside any DB tx).
        q_emb: EmbeddingResult | None = None
        if self._embedding_client is not None:
            q_emb = await self._embedding_client.embed_one(cleaned_query)

        stage1: list[_Stage1Candidate] = []
        if q_emb is not None:
            q_vec = unpack_vector(q_emb.blob)
            expected_bytes = q_emb.dimension * 4
            for item in visible_items:
                if (
                    item.embedding is not None
                    and item.embedding_model == q_emb.model_tag
                    and len(item.embedding) == expected_bytes
                ):
                    try:
                        item_vec = unpack_vector(item.embedding)
                        sim = cosine_similarity(q_vec, item_vec)
                    except (TypeError, ValueError) as exc:
                        logger.warning(
                            "Invalid stored embedding on memory %d; falling back to "
                            "lexical matching: %s",
                            item.id,
                            exc,
                        )
                    else:
                        if sim >= self._config.min_vector_similarity:
                            stage1.append(
                                _Stage1Candidate(
                                    item=item,
                                    vector_score=sim,
                                    sort_key=(
                                        1,
                                        sim,
                                        item.confidence,
                                        item.updated_at,
                                        item.id,
                                    ),
                                )
                            )
                        continue

                lex = _lexical_match_score(cleaned_query, item.content)
                if lex > 0.0:
                    stage1.append(
                        _Stage1Candidate(
                            item=item,
                            vector_score=None,
                            sort_key=(
                                0,
                                lex + 0.05 * item.confidence,
                                item.confidence,
                                item.updated_at,
                                item.id,
                            ),
                        )
                    )
        else:
            for item in visible_items:
                lex = _lexical_match_score(cleaned_query, item.content)
                if lex > 0.0:
                    stage1.append(
                        _Stage1Candidate(
                            item=item,
                            vector_score=None,
                            sort_key=(
                                0,
                                lex + 0.05 * item.confidence,
                                item.confidence,
                                item.updated_at,
                                item.id,
                            ),
                        )
                    )

        if not stage1:
            return []

        stage1.sort(key=lambda c: c.sort_key, reverse=True)
        stage1 = stage1[: self._config.vector_top_k]

        # Stage 2: Single-shot Jev relevance rerank (outside any DB tx).
        hits: list[MemorySearchHit]
        if self._jev_client is not None:
            rerank = await self._jev_client.rerank_memories(
                query=cleaned_query,
                candidates=[c.item.content for c in stage1],
                threshold=self._config.rerank_threshold,
            )
            if not rerank.degraded:
                hits = [
                    MemorySearchHit(
                        item=stage1[idx].item,
                        vector_score=stage1[idx].vector_score,
                        rerank_score=rerank.scores[idx],
                    )
                    for idx in rerank.ranked_indices
                ]
            else:
                hits = [
                    MemorySearchHit(
                        item=c.item,
                        vector_score=c.vector_score,
                        rerank_score=None,
                    )
                    for c in stage1
                ]
        else:
            hits = [
                MemorySearchHit(
                    item=c.item,
                    vector_score=c.vector_score,
                    rerank_score=None,
                )
                for c in stage1
            ]

        hits = hits[:effective_limit]
        if not hits:
            return []

        ts = _resolve_timestamp(now)
        hit_ids = [hit.item.id for hit in hits]
        placeholders = ", ".join("?" for _ in hit_ids)
        await self._db.execute(
            f"""
            UPDATE memory_items
            SET last_accessed_at = ?
            WHERE id IN ({placeholders})
            """,
            [ts, *hit_ids],
        )
        return [
            MemorySearchHit(
                item=replace(hit.item, last_accessed_at=ts),
                vector_score=hit.vector_score,
                rerank_score=hit.rerank_score,
            )
            for hit in hits
        ]

    # ------------------------------------------------------------------
    # Group jargon
    # ------------------------------------------------------------------

    async def upsert_jargon(
        self,
        group_id: str,
        term: str,
        meaning: str,
        *,
        confidence: float = 0.7,
        confidence_boost: float = 0.05,
        now: int | None = None,
    ) -> int:
        """Insert or reinforce a group-scoped jargon/meme term (case-insensitive)."""
        norm_gid = _require_non_empty(group_id, "group_id")
        norm_term = _require_non_empty(term, "term")
        norm_meaning = _require_non_empty(meaning, "meaning")
        norm_conf = _validate_unit_float(confidence, "confidence")
        boost = _validate_non_negative_float(confidence_boost, "confidence_boost")
        ts = _resolve_timestamp(now)

        async with self._db.transaction() as tx:
            await tx.execute(
                """
                INSERT INTO group_jargon (
                    group_id, term, meaning, confidence, evidence_count, updated_at
                )
                VALUES (?, ?, ?, ?, 1, ?)
                ON CONFLICT(group_id, term) DO UPDATE SET
                    meaning = excluded.meaning,
                    evidence_count = group_jargon.evidence_count + 1,
                    confidence = MIN(
                        1.0,
                        MAX(group_jargon.confidence, excluded.confidence) + ?
                    ),
                    updated_at = excluded.updated_at
                """,
                (norm_gid, norm_term, norm_meaning, norm_conf, ts, boost),
            )
            row = await tx.fetchone(
                "SELECT id FROM group_jargon WHERE group_id = ? AND term = ?",
                (norm_gid, norm_term),
            )
            if row is None:
                raise RuntimeError(
                    f"Failed to retrieve upserted jargon ID for ({norm_gid!r}, "
                    f"{norm_term!r})."
                )
            return int(row["id"])

    async def list_group_jargon(
        self,
        group_id: str,
        *,
        min_confidence: float = 0.0,
        limit: int = 100,
    ) -> list[JargonEntry]:
        """List jargon entries for a group filtered by ``min_confidence``."""
        norm_min = _validate_unit_float(min_confidence, "min_confidence")
        norm_limit = _validate_non_negative_int(limit, "limit")
        norm_gid = _normalize_optional_id(group_id)
        if norm_gid is None or norm_limit == 0:
            return []
        rows = await self._db.fetchall(
            f"""
            SELECT {_JARGON_COLUMNS}
            FROM group_jargon
            WHERE group_id = ? AND confidence >= ?
            ORDER BY confidence DESC, evidence_count DESC, id ASC
            LIMIT ?
            """,
            (norm_gid, norm_min, norm_limit),
        )
        return [_row_to_jargon_entry(row) for row in rows]

    async def match_jargon_in_text(
        self,
        group_id: str,
        text: str,
    ) -> list[JargonEntry]:
        """Match high-confidence group jargon terms appearing in ``text``.

        Uses ASCII-safe boundary lookarounds (avoiding ``\\b`` so CJK neighbors
        like ``"今晚go吗"`` match ``"go"``, while ``"good"`` and ``"C++"`` for
        ``"C"`` do not false-match).
        """
        if not isinstance(text, str) or not text.strip():
            return []
        norm_gid = _normalize_optional_id(group_id)
        if norm_gid is None:
            return []

        rows = await self._db.fetchall(
            f"""
            SELECT {_JARGON_COLUMNS}
            FROM group_jargon
            WHERE group_id = ? AND confidence >= ?
            ORDER BY confidence DESC, evidence_count DESC, id ASC
            """,
            (norm_gid, self._config.jargon_min_confidence),
        )
        if not rows:
            return []

        matched: list[JargonEntry] = []
        max_matches = self._config.jargon_max_matches
        for row in rows:
            entry = _row_to_jargon_entry(row)
            if _compile_jargon_pattern(entry.term).search(text) is not None:
                matched.append(entry)
                if len(matched) >= max_matches:
                    break
        return matched
