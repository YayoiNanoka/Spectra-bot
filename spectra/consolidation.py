"""Background async episode memory consolidation and safe retention cleanup.

Processes closed short episodes asynchronously without blocking live replies:
loads scope-isolated existing context outside SQLite transactions, invokes the
LLM for structured JSON extraction, enforces strict provenance and privacy
guardrails in Python before writing to :class:`spectra.memory.MemoryStore`,
advances ``sessions.last_extracted_id``, and runs throttled retention cleanup.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Self
from zoneinfo import ZoneInfo

from spectra.config import AppConfig
from spectra.llm import LLMClient
from spectra.memory import (
    GroupProfile,
    JargonEntry,
    MemoryItem,
    MemoryKind,
    MemoryStore,
    MemoryVisibility,
    PublicUserProfile,
    UserProfile,
)
from spectra.messages import MessageRecord, MessageStore
from spectra.sessions import ClosedEpisode, SessionStore

logger = logging.getLogger(__name__)

_VALID_MEMORY_KINDS: frozenset[str] = frozenset(
    {"fact", "preference", "event", "promise"}
)
_JSON_FENCE_RE = re.compile(
    r"^\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*$",
    re.DOTALL | re.IGNORECASE,
)

_GROUP_SYSTEM_PROMPT = """你是 Spectra-bot 的后台长期记忆整理器（群聊模式）。
你的任务是阅读一段已结束的群聊对话片段，并结合已有群聊画像、发言人公开画像、现有活跃记忆与现有群黑话，提取值得长期沉淀的增量更新。

你必须**只输出一个合法 JSON 对象**，不要输出任何额外说明文字。JSON 结构严格如下：
{
  "group_profile_patch": null 或 {
    "interaction_style": string|null,
    "frequent_topics": string|null,
    "taboos_and_boundaries": string|null,
    "member_notes": string|null
  },
  "user_profile_patches": [
    {
      "user_id": "实际在本片段发言的人类成员 QQ 号",
      "add_aliases": ["新观察到的昵称或外号"],
      "interaction_style": string|null,
      "topic_preferences": string|null,
      "relationship": string|null,
      "recent_focus": string|null,
      "private_notes": null
    }
  ],
  "memory_ops": [
    {
      "action": "add",
      "kind": "fact|preference|event|promise",
      "content": "简洁、独立可理解的中文记忆陈述",
      "owner_user_id": "关联发言人 QQ 号或 null",
      "visibility": "public|group_only",
      "confidence": 0.7,
      "event_time": "YYYY-MM-DD 或 null"
    },
    {"action": "reinforce", "target_id": 123},
    {
      "action": "supersede",
      "target_id": 123,
      "kind": "fact|preference|event|promise",
      "content": "更新后的完整中文记忆陈述",
      "owner_user_id": "关联发言人 QQ 号或 null",
      "visibility": "public|group_only",
      "confidence": 0.8,
      "event_time": "YYYY-MM-DD 或 null"
    }
  ],
  "jargon_ops": [
    {"term": "群黑话或梗词", "meaning": "简明释义", "confidence": 0.75}
  ]
}

关键约束：
1. 所有 profile patch 字段为 null 表示保留原值不变；仅在有明确新增或修正时输出字符串。
2. 群聊模式下严禁填写 private_notes（必须为 null），且 user_profile_patches 与
   owner_user_id 只能引用本片段中实际发言的人类用户。
3. visibility 只能为 "public"（跨群成立的个人公开事实/偏好且有 owner_user_id）
   或 "group_only"（本群语境下的约定/事件/群内事实）。
4. reinforce 与 supersede 的 target_id 必须严格来自 existing_memories 中的 id。
5. 若对话出现“今天/昨天/明天/下周五”等相对时间，必须依据消息前的
   [YYYY-MM-DD HH:MM] 换算为绝对日期 YYYY-MM-DD 填入 event_time（否则填 null）。
6. 宁缺毋滥：无长期价值的一次性寒暄不要写入记忆或黑话，各数组可为空列表。"""

_PRIVATE_SYSTEM_PROMPT = """你是 Spectra-bot 的后台长期记忆整理器（私聊模式）。
你的任务是阅读一段已结束的一对一私聊对话片段，并结合已有用户画像与现有活跃记忆，提取值得长期沉淀的增量更新。

你必须**只输出一个合法 JSON 对象**，不要输出任何额外说明文字。JSON 结构严格如下：
{
  "group_profile_patch": null,
  "user_profile_patches": [
    {
      "user_id": "当前私聊对象的 QQ 号",
      "add_aliases": [],
      "interaction_style": null,
      "topic_preferences": null,
      "relationship": null,
      "recent_focus": null,
      "private_notes": string|null
    }
  ],
  "memory_ops": [
    {
      "action": "add",
      "kind": "fact|preference|event|promise",
      "content": "简洁、独立可理解的中文记忆陈述",
      "owner_user_id": "当前私聊对象的 QQ 号",
      "visibility": "private",
      "confidence": 0.75,
      "event_time": "YYYY-MM-DD 或 null"
    },
    {"action": "reinforce", "target_id": 123},
    {
      "action": "supersede",
      "target_id": 123,
      "kind": "fact|preference|event|promise",
      "content": "更新后的完整中文记忆陈述",
      "owner_user_id": "当前私聊对象的 QQ 号",
      "visibility": "private",
      "confidence": 0.85,
      "event_time": "YYYY-MM-DD 或 null"
    }
  ],
  "jargon_ops": []
}

关键约束：
1. 私聊模式下 group_profile_patch 必须为 null，jargon_ops 必须为空列表 []。
2. 为防止私聊内容泄露到群聊公开画像，私聊模式下 add_aliases 必须为空列表 []，且
   interaction_style、topic_preferences、relationship、recent_focus 必须为 null；
   所有在私聊中观察到的昵称/称呼、性格风格、话题偏好、关系变化、近期关注与私密心事，
   必须统一整理进 private_notes（为 null 表示保留原值不变）。
3. 所有新增或替换记忆均视为该私聊对象的私有记忆（visibility="private"）。
4. reinforce 与 supersede 的 target_id 必须严格来自 existing_memories 中的 id。
5. 若对话出现“今天/昨天/明天/下周五”等相对时间，必须依据消息前的
   [YYYY-MM-DD HH:MM] 换算为绝对日期 YYYY-MM-DD 填入 event_time（否则填 null）。
6. 宁缺毋滥：无长期价值的一次性寒暄不要写入记忆。"""


@dataclass(slots=True, frozen=True)
class ConsolidationResult:
    """Summary counters produced by consolidating a single closed episode."""

    session_key: str
    episode_id: int
    group_profile_updated: bool
    user_profiles_updated: int
    memories_added: int
    memories_reinforced: int
    memories_superseded: int
    jargon_upserted: int
    degraded: bool = False


def _zero_result(
    episode: ClosedEpisode,
    *,
    degraded: bool = False,
) -> ConsolidationResult:
    return ConsolidationResult(
        session_key=episode.session_key,
        episode_id=episode.episode_id,
        group_profile_updated=False,
        user_profiles_updated=0,
        memories_added=0,
        memories_reinforced=0,
        memories_superseded=0,
        jargon_upserted=0,
        degraded=degraded,
    )


def _strip_json_fences(raw: str) -> str:
    """Strip optional outer Markdown ```json ... ``` fences from LLM output."""
    text = raw.strip()
    match = _JSON_FENCE_RE.match(text)
    if match is not None:
        return match.group(1).strip()
    return text


def _parse_json_object(raw: str) -> dict[str, Any] | None:
    """Parse a JSON object from ``raw``, stripping optional Markdown fences."""
    cleaned = _strip_json_fences(raw)
    if not cleaned:
        return None
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _optional_str(mapping: Mapping[str, Any], key: str) -> str | None:
    """Return ``mapping[key]`` when it is a ``str``; ``None`` otherwise."""
    value = mapping.get(key)
    if value is None:
        return None
    if isinstance(value, str):
        return value
    logger.warning(
        "Ignoring non-string value for consolidation field %r: %r",
        key,
        type(value).__name__,
    )
    return None


def _coerce_confidence(value: Any, *, default: float = 0.7) -> float:
    """Coerce numeric confidence into ``[0.0, 1.0]``, falling back to ``default``."""
    if value is None or isinstance(value, bool) or not isinstance(value, int | float):
        return default
    num = float(value)
    if not math.isfinite(num):
        return default
    return max(0.0, min(1.0, num))


def _coerce_target_id(value: Any) -> int | None:
    """Parse a positive integer memory target ID, rejecting booleans and floats."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            parsed = int(stripped)
            return parsed if parsed > 0 else None
    return None


def _coerce_event_time(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _extract_alias_list(raw: Any) -> list[str]:
    """Extract clean non-empty string aliases from a list field."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        logger.warning(
            "Ignoring non-list add_aliases in consolidation patch: %s",
            type(raw).__name__,
        )
        return []
    aliases: list[str] = []
    for item in raw:
        if isinstance(item, str) and item.strip():
            aliases.append(item.strip())
    return aliases


def _serialize_group_profile(profile: GroupProfile | None) -> dict[str, Any] | None:
    if profile is None:
        return None
    return {
        "group_id": profile.group_id,
        "interaction_style": profile.interaction_style,
        "frequent_topics": profile.frequent_topics,
        "taboos_and_boundaries": profile.taboos_and_boundaries,
        "member_notes": profile.member_notes,
    }


def _serialize_public_user_profile(
    profile: PublicUserProfile,
) -> dict[str, Any]:
    return {
        "user_id": profile.user_id,
        "aliases": list(profile.aliases),
        "interaction_style": profile.interaction_style,
        "topic_preferences": profile.topic_preferences,
        "relationship": profile.relationship,
        "recent_focus": profile.recent_focus,
    }


def _serialize_private_user_profile(
    profile: UserProfile | None,
) -> dict[str, Any] | None:
    if profile is None:
        return None
    return {
        "user_id": profile.user_id,
        "aliases": list(profile.aliases),
        "interaction_style": profile.interaction_style,
        "topic_preferences": profile.topic_preferences,
        "relationship": profile.relationship,
        "recent_focus": profile.recent_focus,
        "private_notes": profile.private_notes,
    }


def _serialize_memory_item(item: MemoryItem) -> dict[str, Any]:
    return {
        "id": item.id,
        "kind": item.kind,
        "content": item.content,
        "owner_user_id": item.owner_user_id,
        "visibility": item.visibility,
        "confidence": round(item.confidence, 4),
        "event_time": item.event_time,
    }


def _serialize_jargon(entry: JargonEntry) -> dict[str, Any]:
    return {
        "term": entry.term,
        "meaning": entry.meaning,
        "confidence": round(entry.confidence, 4),
        "evidence_count": entry.evidence_count,
    }


class MemoryConsolidator:
    """Async background consolidator for closed chat episodes.

    Serializes work per ``session_key``, executes LLM extraction and database
    writes asynchronously off the live reply critical path, enforces strict
    provenance and privacy boundaries in Python, advances
    ``sessions.last_extracted_id``, and runs throttled retention cleanup.

    Args:
        config: Root :class:`spectra.config.AppConfig`.
        message_store: Initialized :class:`spectra.messages.MessageStore`.
        session_store: Initialized :class:`spectra.sessions.SessionStore`.
        memory_store: Initialized :class:`spectra.memory.MemoryStore`.
        llm_client: Active :class:`spectra.llm.LLMClient`.
        clock_fn: Injectable monotonic clock in seconds (for cleanup throttle).
        time_fn: Injectable wall-clock epoch function in seconds (for retention).
    """

    def __init__(
        self,
        *,
        config: AppConfig,
        message_store: MessageStore,
        session_store: SessionStore,
        memory_store: MemoryStore,
        llm_client: LLMClient,
        clock_fn: Callable[[], float] = time.monotonic,
        time_fn: Callable[[], float] = time.time,
    ) -> None:
        self._config = config
        self._message_store = message_store
        self._session_store = session_store
        self._memory_store = memory_store
        self._llm_client = llm_client
        self._clock_fn = clock_fn
        self._time_fn = time_fn
        self._tz = ZoneInfo(config.session.timezone)

        self._session_locks: dict[str, asyncio.Lock] = {}
        self._session_lock_refs: dict[str, int] = {}
        self._session_tails: dict[str, asyncio.Task[ConsolidationResult | None]] = {}
        self._tasks: set[asyncio.Task[ConsolidationResult | None]] = set()
        self._cleanup_lock = asyncio.Lock()
        self._last_cleanup_monotonic: float | None = None
        self._closed = False

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    def _acquire_session_lock_entry(self, session_key: str) -> asyncio.Lock:
        lock = self._session_locks.get(session_key)
        if lock is None:
            lock = asyncio.Lock()
            self._session_locks[session_key] = lock
            self._session_lock_refs[session_key] = 0
        self._session_lock_refs[session_key] += 1
        return lock

    def _release_session_lock_entry(self, session_key: str) -> None:
        remaining = self._session_lock_refs.get(session_key, 0) - 1
        if remaining <= 0:
            self._session_lock_refs.pop(session_key, None)
            self._session_locks.pop(session_key, None)
        else:
            self._session_lock_refs[session_key] = remaining

    def enqueue(self, episode: ClosedEpisode) -> None:
        """Schedule episode consolidation in the background without blocking.

        Consecutive episodes for the same ``session_key`` execute strictly in
        enqueue order. Returns immediately so online reply handling adds zero
        consolidation latency.
        """
        if self._closed:
            logger.debug(
                "Skipping episode consolidation after close() (session=%s, ep=%d).",
                episode.session_key,
                episode.episode_id,
            )
            return
        if (
            not self._config.memory.enabled
            or not self._config.memory.consolidation_enabled
        ):
            return

        prev_task = self._session_tails.get(episode.session_key)
        task: asyncio.Task[ConsolidationResult | None] = asyncio.create_task(
            self._run_enqueued_episode(episode, prev_task),
            name=f"spectra-consolidate-{episode.session_key}-ep{episode.episode_id}",
        )
        self._tasks.add(task)
        self._session_tails[episode.session_key] = task

        def _cleanup_task(done_task: asyncio.Task[ConsolidationResult | None]) -> None:
            self._tasks.discard(done_task)
            if self._session_tails.get(episode.session_key) is done_task:
                self._session_tails.pop(episode.session_key, None)

        task.add_done_callback(_cleanup_task)

    async def handle_episode_closed(self, episode: ClosedEpisode) -> None:
        """Async callback adapter forwarding ``episode`` to :meth:`enqueue`."""
        self.enqueue(episode)

    async def _run_enqueued_episode(
        self,
        episode: ClosedEpisode,
        prev_task: asyncio.Task[ConsolidationResult | None] | None,
    ) -> ConsolidationResult | None:
        if prev_task is not None and not prev_task.done():
            prev_done = asyncio.Event()
            prev_task.add_done_callback(lambda _: prev_done.set())
            await prev_done.wait()
        try:
            return await self.consolidate_episode(episode)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "Unhandled error during background consolidation "
                "(session=%s, episode=%d).",
                episode.session_key,
                episode.episode_id,
            )
            return None

    async def flush(self) -> None:
        """Await completion of all currently queued and running background tasks."""
        while self._tasks:
            pending = list(self._tasks)
            await asyncio.gather(*pending, return_exceptions=True)

    async def close(self, *, drain: bool = False) -> None:
        """Idempotently stop accepting new work and clean up background tasks.

        Args:
            drain: When ``True``, waits for in-flight tasks to finish cleanly;
                when ``False`` (default), cancels pending/running tasks and
                awaits their cancellation so shutdown leaves zero dangling tasks.
        """
        if self._closed:
            return
        self._closed = True
        if drain:
            await self.flush()
        else:
            pending = [task for task in self._tasks if not task.done()]
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
        self._tasks.clear()
        self._session_tails.clear()

    def _format_transcript(
        self,
        records: list[MessageRecord],
        *,
        is_group: bool,
    ) -> str:
        lines: list[str] = []
        for rec in records:
            ts_str = datetime.fromtimestamp(rec.created_at, tz=self._tz).strftime(
                "%Y-%m-%d %H:%M"
            )
            if rec.role == "assistant":
                lines.append(f"[{ts_str}] Bot: {rec.content}")
            elif is_group:
                lines.append(
                    f"[{ts_str}] {rec.sender_name}({rec.user_id}): {rec.content}"
                )
            else:
                lines.append(f"[{ts_str}] User({rec.user_id}): {rec.content}")
        return "\n".join(lines)

    def _build_llm_messages(
        self,
        *,
        episode: ClosedEpisode,
        is_group: bool,
        transcript: str,
        human_speaker_ids: list[str],
        group_profile: GroupProfile | None,
        public_user_profiles: dict[str, PublicUserProfile],
        private_user_profile: UserProfile | None,
        existing_memories: list[MemoryItem],
        existing_jargon: list[JargonEntry],
    ) -> list[dict[str, Any]]:
        if is_group:
            system_prompt = _GROUP_SYSTEM_PROMPT
            context_payload: dict[str, Any] = {
                "chat_type": "group",
                "session_key": episode.session_key,
                "group_id": episode.group_id,
                "episode_id": episode.episode_id,
                "timezone": self._config.session.timezone,
                "active_human_speaker_ids": human_speaker_ids,
                "existing_group_profile": _serialize_group_profile(group_profile),
                "existing_public_user_profiles": [
                    _serialize_public_user_profile(public_user_profiles[uid])
                    for uid in human_speaker_ids
                    if uid in public_user_profiles
                ],
                "existing_memories": [
                    _serialize_memory_item(m) for m in existing_memories
                ],
                "existing_group_jargon": [
                    _serialize_jargon(j) for j in existing_jargon
                ],
            }
        else:
            system_prompt = _PRIVATE_SYSTEM_PROMPT
            context_payload = {
                "chat_type": "private",
                "session_key": episode.session_key,
                "user_id": episode.user_id,
                "episode_id": episode.episode_id,
                "timezone": self._config.session.timezone,
                "existing_user_profile": _serialize_private_user_profile(
                    private_user_profile
                ),
                "existing_memories": [
                    _serialize_memory_item(m) for m in existing_memories
                ],
            }

        context_json = json.dumps(context_payload, ensure_ascii=False, indent=2)
        user_prompt = (
            "## 当前作用域已有上下文（JSON）\n"
            f"{context_json}\n\n"
            "## 本次已结束对话片段原文\n"
            f"{transcript}\n\n"
            "请严格按照系统提示的 JSON Schema 输出增量更新结果（只输出 JSON）。"
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    async def maybe_cleanup_expired_messages(self, *, force: bool = False) -> int:
        """Delete expired messages safely behind the session protection watermark.

        Throttled by ``config.memory.cleanup_interval_sec`` unless ``force=True``.
        """
        async with self._cleanup_lock:
            mono_now = self._clock_fn()
            if (
                not force
                and self._last_cleanup_monotonic is not None
                and (mono_now - self._last_cleanup_monotonic)
                < self._config.memory.cleanup_interval_sec
            ):
                return 0

            cutoff = (
                int(self._time_fn()) - self._config.db.message_retention_days * 86400
            )
            try:
                protected_id = await self._session_store.min_protected_message_id(
                    active_since=cutoff
                )
                deleted = await self._message_store.cleanup_expired_messages(
                    cutoff,
                    max_deletable_id=protected_id,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Expired message cleanup failed: %s", exc)
                return 0

            self._last_cleanup_monotonic = mono_now
            if deleted > 0:
                logger.info(
                    "Cleaned up %d expired message(s) older than %d (protected_id=%s).",
                    deleted,
                    cutoff,
                    protected_id,
                )
            return deleted

    async def consolidate_episode(
        self,
        episode: ClosedEpisode,
    ) -> ConsolidationResult:
        """Consolidate one closed episode, serialized per ``episode.session_key``."""
        lock = self._acquire_session_lock_entry(episode.session_key)
        try:
            async with lock:
                return await self._consolidate_episode_unlocked(episode)
        finally:
            self._release_session_lock_entry(episode.session_key)

    async def _consolidate_episode_unlocked(
        self,
        episode: ClosedEpisode,
    ) -> ConsolidationResult:
        is_group = episode.group_id is not None

        # 1. Fetch messages in the episode span and identify human speakers.
        records = await self._message_store.get_messages_in_range(
            group_id=episode.group_id,
            user_id=episode.user_id,
            start_id=episode.start_msg_id,
            end_id=episode.end_msg_id,
        )
        non_empty_records = [rec for rec in records if rec.content.strip()]
        human_records = [rec for rec in non_empty_records if rec.role == "user"]

        if not human_records:
            await self._session_store.advance_extracted_cursor(
                episode.session_key,
                episode.end_msg_id,
                now=episode.closed_at,
            )
            await self.maybe_cleanup_expired_messages()
            return _zero_result(episode, degraded=False)

        human_speaker_ids: list[str] = []
        human_speaker_aliases: dict[str, list[str]] = {}

        if is_group:
            seen_speakers: set[str] = set()
            for rec in human_records:
                uid = rec.user_id.strip()
                if not uid:
                    continue
                if uid not in seen_speakers:
                    seen_speakers.add(uid)
                    human_speaker_ids.append(uid)
                alias = rec.sender_name.strip()
                if alias and alias != uid:
                    aliases = human_speaker_aliases.setdefault(uid, [])
                    if alias.casefold() not in {a.casefold() for a in aliases}:
                        aliases.append(alias)
        else:
            peer_id = episode.user_id.strip()
            human_speaker_ids = [peer_id]

        human_speaker_set = set(human_speaker_ids)
        max_msgs = self._config.memory.consolidation_max_messages
        bounded_records = non_empty_records[-max_msgs:]
        transcript = self._format_transcript(bounded_records, is_group=is_group)

        # 2. Load existing scope-isolated context outside any DB transaction.
        group_profile: GroupProfile | None = None
        public_user_profiles: dict[str, PublicUserProfile] = {}
        private_user_profile: UserProfile | None = None
        existing_memories: list[MemoryItem] = []
        existing_jargon: list[JargonEntry] = []
        known_aliases: dict[str, set[str]] = {uid: set() for uid in human_speaker_ids}

        if is_group:
            assert episode.group_id is not None
            group_profile = await self._memory_store.get_group_profile(episode.group_id)
            for uid in human_speaker_ids:
                pub = await self._memory_store.get_public_user_profile(uid)
                if pub is not None:
                    public_user_profiles[uid] = pub
                    known_aliases[uid] = {a.casefold() for a in pub.aliases}
            existing_memories = await self._memory_store.list_active_memories(
                group_id=episode.group_id,
                active_speaker_ids=human_speaker_ids,
                limit=50,
            )
            existing_jargon = await self._memory_store.list_group_jargon(
                episode.group_id,
                limit=50,
            )
        else:
            peer_id = human_speaker_ids[0]
            private_user_profile = await self._memory_store.get_user_profile(peer_id)
            existing_memories = await self._memory_store.list_active_memories(
                group_id=None,
                user_id=peer_id,
                limit=50,
            )

        visible_existing_ids: set[int] = {item.id for item in existing_memories}

        # 3. Call LLM outside any DB transaction.
        llm_messages = self._build_llm_messages(
            episode=episode,
            is_group=is_group,
            transcript=transcript,
            human_speaker_ids=human_speaker_ids,
            group_profile=group_profile,
            public_user_profiles=public_user_profiles,
            private_user_profile=private_user_profile,
            existing_memories=existing_memories,
            existing_jargon=existing_jargon,
        )
        try:
            response = await self._llm_client.complete(
                llm_messages,
                effort="default",
                session_key=f"spectra:consolidate:{episode.session_key}",
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Episode consolidation LLM call failed (session=%s, episode=%d): %s",
                episode.session_key,
                episode.episode_id,
                exc,
            )
            return _zero_result(episode, degraded=True)

        payload = _parse_json_object(response.content)
        if payload is None:
            logger.warning(
                "Episode consolidation returned unparseable JSON root "
                "(session=%s, episode=%d).",
                episode.session_key,
                episode.episode_id,
            )
            return _zero_result(episode, degraded=True)

        # 4. Apply parsed updates with strict provenance & privacy enforcement.
        group_profile_updated = False
        updated_user_ids: set[str] = set()
        memories_added = 0
        memories_reinforced = 0
        memories_superseded = 0
        jargon_upserted = 0

        # 4a. Group profile patch (group episodes only).
        if is_group:
            assert episode.group_id is not None
            raw_gp = payload.get("group_profile_patch")
            if isinstance(raw_gp, Mapping):
                gp_style = _optional_str(raw_gp, "interaction_style")
                gp_topics = _optional_str(raw_gp, "frequent_topics")
                gp_taboos = _optional_str(raw_gp, "taboos_and_boundaries")
                gp_members = _optional_str(raw_gp, "member_notes")
                if any(
                    v is not None for v in (gp_style, gp_topics, gp_taboos, gp_members)
                ):
                    try:
                        await self._memory_store.patch_group_profile(
                            episode.group_id,
                            interaction_style=gp_style,
                            frequent_topics=gp_topics,
                            taboos_and_boundaries=gp_taboos,
                            member_notes=gp_members,
                            now=episode.closed_at,
                        )
                        group_profile_updated = True
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        logger.warning(
                            "Failed to apply group_profile_patch (group=%s): %s",
                            episode.group_id,
                            exc,
                        )
            elif raw_gp is not None:
                logger.warning(
                    "Ignoring malformed group_profile_patch (type=%s).",
                    type(raw_gp).__name__,
                )

        # 4b. User profile patches + automatic sender_name alias indexing.
        raw_user_patches = payload.get("user_profile_patches")
        if raw_user_patches is not None and not isinstance(raw_user_patches, list):
            logger.warning(
                "Ignoring malformed user_profile_patches (type=%s).",
                type(raw_user_patches).__name__,
            )
            raw_user_patches = []

        for raw_patch in raw_user_patches or ():
            if not isinstance(raw_patch, Mapping):
                logger.warning("Skipping non-object entry in user_profile_patches.")
                continue
            raw_uid = raw_patch.get("user_id")
            if not isinstance(raw_uid, str) or not raw_uid.strip():
                logger.warning("Skipping user_profile_patch missing valid user_id.")
                continue
            uid = raw_uid.strip()
            if uid not in human_speaker_set:
                logger.warning(
                    "Rejected user_profile_patch for non-participant user_id=%r "
                    "(session=%s).",
                    uid,
                    episode.session_key,
                )
                continue

            if is_group:
                llm_aliases = _extract_alias_list(raw_patch.get("add_aliases"))
                auto_aliases = human_speaker_aliases.get(uid, [])
                combined_aliases: tuple[str, ...] | list[str] = [
                    *llm_aliases,
                    *auto_aliases,
                ]
                new_aliases = [
                    a
                    for a in combined_aliases
                    if a.casefold() not in known_aliases[uid]
                ]

                u_style = _optional_str(raw_patch, "interaction_style")
                u_topics = _optional_str(raw_patch, "topic_preferences")
                u_rel = _optional_str(raw_patch, "relationship")
                u_focus = _optional_str(raw_patch, "recent_focus")
                if raw_patch.get("private_notes") is not None:
                    logger.warning(
                        "Stripped forbidden private_notes from group "
                        "user_profile_patch (group=%s, user=%s).",
                        episode.group_id,
                        uid,
                    )
                u_private = None
            else:
                stripped_public_fields: list[str] = []
                raw_add_aliases = raw_patch.get("add_aliases")
                if raw_add_aliases is not None and raw_add_aliases != []:
                    stripped_public_fields.append("add_aliases")
                for pub_field in (
                    "interaction_style",
                    "topic_preferences",
                    "relationship",
                    "recent_focus",
                ):
                    if raw_patch.get(pub_field) is not None:
                        stripped_public_fields.append(pub_field)
                if stripped_public_fields:
                    logger.warning(
                        "Stripped forbidden public profile field(s) %s from private "
                        "user_profile_patch (session=%s, user=%s).",
                        ", ".join(stripped_public_fields),
                        episode.session_key,
                        uid,
                    )
                combined_aliases = ()
                new_aliases = []
                u_style = None
                u_topics = None
                u_rel = None
                u_focus = None
                u_private = _optional_str(raw_patch, "private_notes")

            has_field_update = any(
                v is not None for v in (u_style, u_topics, u_rel, u_focus, u_private)
            )
            if not has_field_update and not new_aliases:
                continue

            try:
                updated_prof = await self._memory_store.patch_user_profile(
                    uid,
                    add_aliases=combined_aliases,
                    interaction_style=u_style,
                    topic_preferences=u_topics,
                    relationship=u_rel,
                    recent_focus=u_focus,
                    private_notes=u_private,
                    now=episode.closed_at,
                )
                if is_group:
                    known_aliases[uid] = {a.casefold() for a in updated_prof.aliases}
                updated_user_ids.add(uid)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "Failed to apply user_profile_patch (user=%s): %s",
                    uid,
                    exc,
                )

        # Ensure every group human speaker's observed sender_name alias stays indexed.
        if is_group:
            for uid in human_speaker_ids:
                auto_aliases = human_speaker_aliases.get(uid, [])
                missing_aliases = [
                    a for a in auto_aliases if a.casefold() not in known_aliases[uid]
                ]
                if not missing_aliases:
                    continue
                try:
                    updated_prof = await self._memory_store.patch_user_profile(
                        uid,
                        add_aliases=missing_aliases,
                        private_notes=None,
                        now=episode.closed_at,
                    )
                    known_aliases[uid] = {a.casefold() for a in updated_prof.aliases}
                    updated_user_ids.add(uid)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning(
                        "Failed to index speaker aliases (user=%s): %s",
                        uid,
                        exc,
                    )

        # 4c. Memory operations with strict target_id & visibility enforcement.
        raw_mem_ops = payload.get("memory_ops")
        if raw_mem_ops is not None and not isinstance(raw_mem_ops, list):
            logger.warning(
                "Ignoring malformed memory_ops (type=%s).",
                type(raw_mem_ops).__name__,
            )
            raw_mem_ops = []

        active_visible_ids = set(visible_existing_ids)
        for raw_op in raw_mem_ops or ():
            if not isinstance(raw_op, Mapping):
                logger.warning("Skipping non-object entry in memory_ops.")
                continue
            action = raw_op.get("action")
            if action == "reinforce":
                target_id = _coerce_target_id(raw_op.get("target_id"))
                if target_id is None or target_id not in active_visible_ids:
                    logger.warning(
                        "Rejected memory reinforce with out-of-scope target_id=%r "
                        "(session=%s).",
                        raw_op.get("target_id"),
                        episode.session_key,
                    )
                    continue
                try:
                    if await self._memory_store.reinforce_memory(
                        target_id,
                        source_msg_end_id=episode.end_msg_id,
                        now=episode.closed_at,
                    ):
                        memories_reinforced += 1
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning(
                        "Failed to reinforce memory %d: %s",
                        target_id,
                        exc,
                    )
                continue

            if action not in ("add", "supersede"):
                logger.warning("Skipping unknown memory_op action=%r.", action)
                continue

            target_id = None
            if action == "supersede":
                target_id = _coerce_target_id(raw_op.get("target_id"))
                if target_id is None or target_id not in active_visible_ids:
                    logger.warning(
                        "Rejected memory supersede with out-of-scope target_id=%r "
                        "(session=%s).",
                        raw_op.get("target_id"),
                        episode.session_key,
                    )
                    continue

            raw_kind = raw_op.get("kind")
            if not isinstance(raw_kind, str) or raw_kind not in _VALID_MEMORY_KINDS:
                logger.warning("Skipping memory_op with invalid kind=%r.", raw_kind)
                continue
            kind: MemoryKind = raw_kind  # type: ignore[assignment]

            raw_content = raw_op.get("content")
            if not isinstance(raw_content, str) or not raw_content.strip():
                logger.warning("Skipping memory_op with empty content.")
                continue
            content = raw_content.strip()

            confidence = _coerce_confidence(raw_op.get("confidence"), default=0.7)
            event_time = _coerce_event_time(raw_op.get("event_time"))

            if is_group:
                assert episode.group_id is not None
                source_group_id: str | None = episode.group_id
                raw_owner = raw_op.get("owner_user_id")
                if (
                    isinstance(raw_owner, str)
                    and raw_owner.strip() in human_speaker_set
                ):
                    owner_user_id: str | None = raw_owner.strip()
                else:
                    if raw_owner is not None:
                        logger.warning(
                            "Cleared non-participant owner_user_id=%r on group "
                            "memory_op (group=%s).",
                            raw_owner,
                            episode.group_id,
                        )
                    owner_user_id = None

                raw_vis = raw_op.get("visibility")
                if raw_vis == "public" and owner_user_id is not None:
                    visibility: MemoryVisibility = "public"
                else:
                    visibility = "group_only"
            else:
                visibility = "private"
                owner_user_id = human_speaker_ids[0]
                source_group_id = None

            try:
                if action == "add":
                    await self._memory_store.add_memory(
                        kind=kind,
                        content=content,
                        visibility=visibility,
                        owner_user_id=owner_user_id,
                        source_group_id=source_group_id,
                        confidence=confidence,
                        source_msg_start_id=episode.start_msg_id,
                        source_msg_end_id=episode.end_msg_id,
                        event_time=event_time,
                        now=episode.closed_at,
                    )
                    memories_added += 1
                else:
                    assert target_id is not None
                    await self._memory_store.supersede_memory(
                        target_id,
                        kind=kind,
                        content=content,
                        visibility=visibility,
                        owner_user_id=owner_user_id,
                        source_group_id=source_group_id,
                        confidence=confidence,
                        source_msg_start_id=episode.start_msg_id,
                        source_msg_end_id=episode.end_msg_id,
                        event_time=event_time,
                        now=episode.closed_at,
                    )
                    active_visible_ids.discard(target_id)
                    memories_superseded += 1
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "Failed to apply memory_op action=%r (session=%s): %s",
                    action,
                    episode.session_key,
                    exc,
                )

        # 4d. Group jargon operations (group episodes only; ignored in private chat).
        if is_group:
            assert episode.group_id is not None
            raw_jargon_ops = payload.get("jargon_ops")
            if raw_jargon_ops is not None and not isinstance(raw_jargon_ops, list):
                logger.warning(
                    "Ignoring malformed jargon_ops (type=%s).",
                    type(raw_jargon_ops).__name__,
                )
                raw_jargon_ops = []

            for raw_jargon in raw_jargon_ops or ():
                if not isinstance(raw_jargon, Mapping):
                    logger.warning("Skipping non-object entry in jargon_ops.")
                    continue
                raw_term = raw_jargon.get("term")
                raw_meaning = raw_jargon.get("meaning")
                if (
                    not isinstance(raw_term, str)
                    or not raw_term.strip()
                    or not isinstance(raw_meaning, str)
                    or not raw_meaning.strip()
                ):
                    logger.warning("Skipping jargon_op missing valid term/meaning.")
                    continue
                j_conf = _coerce_confidence(raw_jargon.get("confidence"), default=0.7)
                try:
                    await self._memory_store.upsert_jargon(
                        episode.group_id,
                        raw_term.strip(),
                        raw_meaning.strip(),
                        confidence=j_conf,
                        now=episode.closed_at,
                    )
                    jargon_upserted += 1
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning(
                        "Failed to upsert jargon %r (group=%s): %s",
                        raw_term,
                        episode.group_id,
                        exc,
                    )

        # 5. Advance extracted cursor and run throttled retention cleanup.
        await self._session_store.advance_extracted_cursor(
            episode.session_key,
            episode.end_msg_id,
            now=episode.closed_at,
        )
        await self.maybe_cleanup_expired_messages()

        return ConsolidationResult(
            session_key=episode.session_key,
            episode_id=episode.episode_id,
            group_profile_updated=group_profile_updated,
            user_profiles_updated=len(updated_user_ids),
            memories_added=memories_added,
            memories_reinforced=memories_reinforced,
            memories_superseded=memories_superseded,
            jargon_upserted=jargon_upserted,
            degraded=False,
        )
