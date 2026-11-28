"""Unit and integration tests for spectra.consolidation and recall_memory tool."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
from openai import AsyncOpenAI
from tenacity import wait_none

from spectra.config import (
    AppConfig,
    DatabaseConfig,
    EmbeddingConfig,
    LLMConfig,
    MemoryConfig,
    SessionConfig,
)
from spectra.consolidation import ConsolidationResult, MemoryConsolidator
from spectra.db import Database
from spectra.embedding import EmbeddingClient
from spectra.llm import LLMClient
from spectra.memory import (
    MemoryStore,
    build_recall_memory_tool,
    format_jargon_reminder,
)
from spectra.messages import MessageStore
from spectra.sessions import ClosedEpisode, SessionStore


def _chat_completion_json(content: str, model: str = "gpt-4o") -> dict[str, Any]:
    return {
        "id": "chatcmpl-consolidate",
        "object": "chat.completion",
        "created": 1700000000,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        },
    }


def _make_llm_client(
    cfg: LLMConfig,
    handler: Callable[[httpx.Request], httpx.Response],
) -> tuple[LLMClient, httpx.AsyncClient, AsyncOpenAI]:
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    sdk_client = AsyncOpenAI(
        api_key=cfg.api_key or "sk-test",
        base_url=cfg.base_url,
        http_client=http_client,
        max_retries=0,
    )
    return (
        LLMClient(cfg, openai_client=sdk_client, retry_wait=wait_none()),
        http_client,
        sdk_client,
    )


def _make_embedding_client(
    vectors: dict[str, list[float]],
    default_vec: list[float] | None = None,
) -> tuple[EmbeddingClient, httpx.AsyncClient]:
    fallback = default_vec or [1.0, 0.0, 0.0]

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        inputs = body["input"]
        if isinstance(inputs, str):
            inputs = [inputs]
        data = [
            {
                "index": idx,
                "embedding": vectors.get(text, fallback),
            }
            for idx, text in enumerate(inputs)
        ]
        return httpx.Response(200, json={"data": data})

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    cfg = EmbeddingConfig(
        base_url="https://embed.example.invalid/v1",
        api_key="sk-embed",
        model="text-embedding-v4",
        max_retries=0,
    )
    return EmbeddingClient(cfg, http_client=http_client), http_client


def _make_config(
    *,
    max_messages: int = 120,
    cleanup_interval_sec: float = 3600.0,
    retention_days: int = 7,
    timezone: str = "Asia/Shanghai",
) -> AppConfig:
    return AppConfig(
        db=DatabaseConfig(message_retention_days=retention_days),
        llm=LLMConfig(
            base_url="https://llm.example.invalid/v1",
            api_key="sk-llm-test",
            model="gpt-4o",
            max_retries=1,
        ),
        session=SessionConfig(timezone=timezone),
        memory=MemoryConfig(
            enabled=True,
            consolidation_enabled=True,
            consolidation_max_messages=max_messages,
            cleanup_interval_sec=cleanup_interval_sec,
        ),
    )


async def test_group_episode_consolidation_full_pipeline_and_privacy_guardrails() -> (
    None
):
    cfg = _make_config(timezone="Asia/Shanghai")
    llm_requests: list[tuple[dict[str, Any], dict[str, str]]] = []

    llm_output_box: list[str] = []

    def llm_handler(request: httpx.Request) -> httpx.Response:
        llm_requests.append((json.loads(request.content), dict(request.headers)))
        return httpx.Response(
            200,
            json=_chat_completion_json(content=llm_output_box[0]),
        )

    embed_client, embed_http = _make_embedding_client(
        {
            "小明改喝美式咖啡了": [1.0, 0.0, 0.0],
            "群里约好 2026-05-08 周五晚八点开黑": [0.0, 1.0, 0.0],
            "小红打算下个月学 Rust": [0.0, 0.0, 1.0],
        }
    )
    llm_client, llm_http, sdk_client = _make_llm_client(cfg.llm, llm_handler)

    try:
        async with Database(":memory:") as db:
            msg_store = MessageStore(db)
            await msg_store.init()
            sess_store = SessionStore(db, msg_store, cfg.session)
            await sess_store.init()
            mem_store = MemoryStore(db, cfg.memory, embedding_client=embed_client)
            await mem_store.init()

            # Pre-seed private_notes (must never enter group prompt or be overwritten).
            await mem_store.patch_user_profile(
                "u1",
                interaction_style="旧风格",
                private_notes="绝密私聊心事_DO_NOT_LEAK",
                now=100,
            )
            # Pre-seed existing active memories in group g1.
            m_reinforce = await mem_store.add_memory(
                kind="fact",
                content="小明在用 Linux 写 Python",
                visibility="public",
                owner_user_id="u1",
                source_group_id="g1",
                confidence=0.70,
                now=100,
            )
            m_supersede = await mem_store.add_memory(
                kind="preference",
                content="小明以前喜欢喝拿铁",
                visibility="public",
                owner_user_id="u1",
                source_group_id="g1",
                confidence=0.75,
                now=100,
            )
            # Pre-seed unreachable private DM memory for u1 (excluded from group).
            m_private = await mem_store.add_memory(
                kind="fact",
                content="小明私聊秘密：准备跳槽",
                visibility="private",
                owner_user_id="u1",
                source_group_id=None,
                now=100,
            )

            ts_base = int(datetime(2026, 5, 4, 12, 0, 0, tzinfo=UTC).timestamp())
            id1 = await msg_store.append_message(
                group_id="g1",
                user_id="u1",
                sender_name="小明卡片",
                role="user",
                content="我最近改喝美式了，还是天天用 Linux 写 Python",
                created_at=ts_base,
            )
            await msg_store.append_message(
                group_id="g1",
                user_id="99999",
                sender_name="Spectra",
                role="assistant",
                content="美式提神呀！",
                created_at=ts_base + 30,
            )
            id3 = await msg_store.append_message(
                group_id="g1",
                user_id="u2",
                sender_name="小红",
                role="user",
                content="这周五晚八点大家一起冲吗，别当鸽王啊，我下个月也想学 Rust",
                created_at=ts_base + 60,
            )

            # Fenced JSON attempting to write private_notes & private memory.
            llm_output_box.append(
                "```json\n"
                + json.dumps(
                    {
                        "group_profile_patch": {
                            "interaction_style": "轻松互怼、技术闲聊",
                            "frequent_topics": "Python、Rust、开黑",
                            "taboos_and_boundaries": None,
                            "member_notes": "小明爱喝咖啡",
                        },
                        "user_profile_patches": [
                            {
                                "user_id": "u1",
                                "add_aliases": ["明明"],
                                "interaction_style": "直爽幽默",
                                "topic_preferences": "Linux、Python",
                                "relationship": None,
                                "recent_focus": "美式咖啡",
                                "private_notes": "试图从群聊覆盖私聊备注_BLOCKED",
                            }
                        ],
                        "memory_ops": [
                            {"action": "reinforce", "target_id": m_reinforce},
                            {
                                "action": "supersede",
                                "target_id": m_supersede,
                                "kind": "preference",
                                "content": "小明改喝美式咖啡了",
                                "owner_user_id": "u1",
                                "visibility": "public",
                                "confidence": 0.9,
                                "event_time": None,
                            },
                            {
                                "action": "add",
                                "kind": "promise",
                                "content": "群里约好 2026-05-08 周五晚八点开黑",
                                "owner_user_id": None,
                                # Must clamp to group_only when owner is None.
                                "visibility": "public",
                                "confidence": 0.85,
                                "event_time": "2026-05-08",
                            },
                            {
                                "action": "add",
                                "kind": "fact",
                                "content": "小红打算下个月学 Rust",
                                "owner_user_id": "u2",
                                # Must clamp to group_only in group chat.
                                "visibility": "private",
                                "confidence": 0.8,
                                "event_time": None,
                            },
                        ],
                        "jargon_ops": [
                            {
                                "term": "鸽王",
                                "meaning": "经常放大家鸽子的群友",
                                "confidence": 0.82,
                            }
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n```"
            )

            consolidator = MemoryConsolidator(
                config=cfg,
                message_store=msg_store,
                session_store=sess_store,
                memory_store=mem_store,
                llm_client=llm_client,
            )
            episode = ClosedEpisode(
                session_key="group:g1",
                group_id="g1",
                user_id="u1",
                episode_id=1,
                start_msg_id=id1,
                end_msg_id=id3,
                closed_at=ts_base + 3600,
            )

            result = await consolidator.consolidate_episode(episode)
            assert result == ConsolidationResult(
                session_key="group:g1",
                episode_id=1,
                group_profile_updated=True,
                user_profiles_updated=2,  # u1 patched + u2 auto-aliased ("小红")
                memories_added=2,
                memories_reinforced=1,
                memories_superseded=1,
                jargon_upserted=1,
                degraded=False,
            )

            # Verify LLM prompt format, date anchoring, and zero private leakage.
            assert len(llm_requests) == 1
            prompt_text = llm_requests[0][0]["messages"][1]["content"]
            assert "[2026-05-04 20:00] 小明卡片(u1):" in prompt_text
            assert "[2026-05-04 20:00] Bot: 美式提神呀！" in prompt_text
            assert "[2026-05-04 20:01] 小红(u2):" in prompt_text
            assert "绝密私聊心事_DO_NOT_LEAK" not in prompt_text
            assert "小明私聊秘密" not in prompt_text
            assert f'"id": {m_private}' not in prompt_text

            # Verify group profile updated.
            gp = await mem_store.get_group_profile("g1")
            assert gp is not None
            assert gp.interaction_style == "轻松互怼、技术闲聊"
            assert gp.frequent_topics == "Python、Rust、开黑"
            assert gp.member_notes == "小明爱喝咖啡"

            # Verify u1 profile updated, aliases include both LLM + sender_name,
            # and pre-existing private_notes was NEVER overwritten!
            u1_prof = await mem_store.get_user_profile("u1")
            assert u1_prof is not None
            assert set(u1_prof.aliases) == {"明明", "小明卡片"}
            assert u1_prof.interaction_style == "直爽幽默"
            assert u1_prof.private_notes == "绝密私聊心事_DO_NOT_LEAK"

            # Verify u2 got automatic sender_name alias indexing.
            u2_prof = await mem_store.get_user_profile("u2")
            assert u2_prof is not None
            assert u2_prof.aliases == ("小红",)

            # Verify reinforced, superseded, and added memories + embeddings.
            reinf = await mem_store.get_memory(m_reinforce)
            assert reinf is not None
            assert reinf.evidence_count == 2
            assert reinf.source_msg_end_id == id3

            old_sup = await mem_store.get_memory(m_supersede)
            assert old_sup is not None
            assert old_sup.status == "superseded"
            assert old_sup.superseded_by is not None
            new_sup = await mem_store.get_memory(old_sup.superseded_by)
            assert new_sup is not None
            assert new_sup.content == "小明改喝美式咖啡了"
            assert new_sup.visibility == "public"
            assert new_sup.source_group_id == "g1"
            assert new_sup.embedding is not None

            active_g1 = await mem_store.list_active_memories(
                group_id="g1",
                active_speaker_ids=["u1", "u2"],
            )
            by_content = {m.content: m for m in active_g1}
            assert (
                by_content["群里约好 2026-05-08 周五晚八点开黑"].visibility
                == "group_only"
            )
            assert (
                by_content["群里约好 2026-05-08 周五晚八点开黑"].event_time
                == "2026-05-08"
            )
            assert by_content["小红打算下个月学 Rust"].visibility == "group_only"
            assert by_content["小红打算下个月学 Rust"].owner_user_id == "u2"
            assert by_content["小红打算下个月学 Rust"].source_group_id == "g1"

            # Verify group jargon upserted.
            jargon = await mem_store.list_group_jargon("g1")
            assert len(jargon) == 1
            assert jargon[0].term == "鸽王"
            assert jargon[0].meaning == "经常放大家鸽子的群友"

            # Verify extracted cursor advanced.
            sess = await sess_store.get_or_create("group:g1")
            assert sess.last_extracted_id == id3
    finally:
        await embed_client.close()
        await embed_http.aclose()
        await sdk_client.close()
        await llm_http.aclose()


async def test_private_episode_consolidation_private_notes_and_no_jargon() -> None:
    cfg = _make_config(timezone="UTC")
    llm_requests: list[dict[str, Any]] = []

    def llm_handler(request: httpx.Request) -> httpx.Response:
        llm_requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json=_chat_completion_json(
                content=json.dumps(
                    {
                        # Must be completely ignored in private chat:
                        "group_profile_patch": {
                            "interaction_style": "不应写入任何群画像",
                        },
                        "user_profile_patches": [
                            {
                                "user_id": "u_dm",
                                "add_aliases": ["阿强"],
                                "recent_focus": "准备跳槽面试",
                                "private_notes": "最近压力比较大，私聊时多鼓励他",
                            },
                            {
                                # Outsider user_id must be rejected in private chat too!
                                "user_id": "u_other",
                                "private_notes": "越权写入",
                            },
                        ],
                        "memory_ops": [
                            {
                                "action": "add",
                                "kind": "event",
                                "content": "阿强约了 2026-06-10 参加终面",
                                # Must force visibility="private" and owner=u_dm.
                                "owner_user_id": "u_other",
                                "visibility": "public",
                                "confidence": 0.9,
                                "event_time": "2026-06-10",
                            }
                        ],
                        # Must be completely ignored in private chat:
                        "jargon_ops": [
                            {
                                "term": "私聊黑话",
                                "meaning": "不应该被写入",
                                "confidence": 0.95,
                            }
                        ],
                    },
                    ensure_ascii=False,
                )
            ),
        )

    llm_client, llm_http, sdk_client = _make_llm_client(cfg.llm, llm_handler)
    try:
        async with Database(":memory:") as db:
            msg_store = MessageStore(db)
            await msg_store.init()
            sess_store = SessionStore(db, msg_store, cfg.session)
            await sess_store.init()
            mem_store = MemoryStore(db, cfg.memory)
            await mem_store.init()

            ts = int(datetime(2026, 6, 8, 9, 30, 0, tzinfo=UTC).timestamp())
            id1 = await msg_store.append_message(
                group_id=None,
                user_id="u_dm",
                sender_name="大强",
                role="user",
                content="后天有终面，有点紧张",
                created_at=ts,
            )
            id2 = await msg_store.append_message(
                group_id=None,
                user_id="u_dm",
                sender_name="Spectra",
                role="assistant",
                content="别紧张，放轻松一定没问题的！",
                created_at=ts + 15,
            )

            consolidator = MemoryConsolidator(
                config=cfg,
                message_store=msg_store,
                session_store=sess_store,
                memory_store=mem_store,
                llm_client=llm_client,
            )
            ep = ClosedEpisode(
                session_key="private:u_dm",
                group_id=None,
                user_id="u_dm",
                episode_id=1,
                start_msg_id=id1,
                end_msg_id=id2,
                closed_at=ts + 7200,
            )
            res = await consolidator.consolidate_episode(ep)
            assert res == ConsolidationResult(
                session_key="private:u_dm",
                episode_id=1,
                group_profile_updated=False,
                user_profiles_updated=1,
                memories_added=1,
                memories_reinforced=0,
                memories_superseded=0,
                jargon_upserted=0,
                degraded=False,
            )

            prompt_text = llm_requests[0]["messages"][1]["content"]
            assert "[2026-06-08 09:30] User(u_dm): 后天有终面，有点紧张" in prompt_text
            assert "existing_group_jargon" not in prompt_text

            prof = await mem_store.get_user_profile("u_dm")
            assert prof is not None
            assert set(prof.aliases) == {"阿强", "大强"}
            assert prof.private_notes == "最近压力比较大，私聊时多鼓励他"
            assert await mem_store.get_user_profile("u_other") is None

            # Verify memory is strictly private, owned by u_dm, source_group_id=None.
            dm_mems = await mem_store.list_active_memories(
                group_id=None, user_id="u_dm"
            )
            assert len(dm_mems) == 1
            assert dm_mems[0].visibility == "private"
            assert dm_mems[0].owner_user_id == "u_dm"
            assert dm_mems[0].source_group_id is None
            assert dm_mems[0].event_time == "2026-06-10"

            # Verify zero group_profiles and zero group_jargon rows exist.
            gp_cnt = await db.fetchone("SELECT COUNT(*) AS cnt FROM group_profiles")
            jg_cnt = await db.fetchone("SELECT COUNT(*) AS cnt FROM group_jargon")
            assert gp_cnt is not None and int(gp_cnt["cnt"]) == 0
            assert jg_cnt is not None and int(jg_cnt["cnt"]) == 0

            sess = await sess_store.get_or_create("private:u_dm")
            assert sess.last_extracted_id == id2
    finally:
        await sdk_client.close()
        await llm_http.aclose()


async def test_consolidation_anti_tampering_malformed_ops_and_safe_degradation() -> (
    None
):
    cfg = _make_config(timezone="UTC")
    responses: list[httpx.Response] = []

    def llm_handler(request: httpx.Request) -> httpx.Response:
        return responses.pop(0)

    llm_client, llm_http, sdk_client = _make_llm_client(cfg.llm, llm_handler)
    try:
        async with Database(":memory:") as db:
            msg_store = MessageStore(db)
            await msg_store.init()
            sess_store = SessionStore(db, msg_store, cfg.session)
            await sess_store.init()
            mem_store = MemoryStore(db, cfg.memory)
            await mem_store.init()

            # Out-of-scope memories: victim private memory + other group memory.
            other_priv_id = await mem_store.add_memory(
                kind="fact",
                content="别人私聊秘密",
                visibility="private",
                owner_user_id="victim",
                source_group_id=None,
                now=100,
            )
            other_grp_id = await mem_store.add_memory(
                kind="fact",
                content="隔壁群的机密约定",
                visibility="group_only",
                owner_user_id="u1",
                source_group_id="g_other",
                now=100,
            )

            id1 = await msg_store.append_message(
                group_id="g1",
                user_id="u1",
                sender_name="u1",  # equals user_id -> no auto-alias update
                role="user",
                content="聊聊今天写代码的事",
                created_at=1700000000,
            )
            ep = ClosedEpisode(
                session_key="group:g1",
                group_id="g1",
                user_id="u1",
                episode_id=1,
                start_msg_id=id1,
                end_msg_id=id1,
                closed_at=1700003600,
            )
            consolidator = MemoryConsolidator(
                config=cfg,
                message_store=msg_store,
                session_store=sess_store,
                memory_store=mem_store,
                llm_client=llm_client,
            )

            # 1. LLM HTTP 400 error -> degraded=True, cursor NOT advanced.
            responses.append(
                httpx.Response(400, json={"error": {"message": "bad request"}})
            )
            res_err = await consolidator.consolidate_episode(ep)
            assert res_err.degraded is True
            assert (await sess_store.get_or_create("group:g1")).last_extracted_id == 0

            # 2. Malformed JSON root -> degraded=True, cursor NOT advanced.
            responses.append(
                httpx.Response(200, json=_chat_completion_json("not valid json {{{"))
            )
            res_bad = await consolidator.consolidate_episode(ep)
            assert res_bad.degraded is True
            assert (await sess_store.get_or_create("group:g1")).last_extracted_id == 0

            # 3. Anti-tampering + partial malformed ops mixed with 1 valid op.
            responses.append(
                httpx.Response(
                    200,
                    json=_chat_completion_json(
                        json.dumps(
                            {
                                "group_profile_patch": None,
                                "user_profile_patches": [
                                    # Non-participant user_id -> rejected!
                                    {
                                        "user_id": "outsider_999",
                                        "interaction_style": "不应该写入",
                                    },
                                    "not-a-dict",
                                ],
                                "memory_ops": [
                                    # Cross-scope reinforce -> rejected.
                                    {"action": "reinforce", "target_id": other_priv_id},
                                    # Cross-scope supersede -> rejected.
                                    {
                                        "action": "supersede",
                                        "target_id": other_grp_id,
                                        "kind": "fact",
                                        "content": "篡改隔壁群记忆",
                                        "visibility": "group_only",
                                    },
                                    # Malformed op (invalid kind) -> skipped!
                                    {
                                        "action": "add",
                                        "kind": "invalid_kind",
                                        "content": "无效类型",
                                    },
                                    # Non-participant owner -> cleared & group_only.
                                    {
                                        "action": "add",
                                        "kind": "fact",
                                        "content": "群里今天讨论了代码重构",
                                        "owner_user_id": "outsider_999",
                                        "visibility": "public",
                                        "confidence": 0.75,
                                    },
                                ],
                                "jargon_ops": [
                                    {"term": "", "meaning": "空词条跳过"},
                                    "malformed-jargon",
                                ],
                            },
                            ensure_ascii=False,
                        )
                    ),
                )
            )
            res_ok = await consolidator.consolidate_episode(ep)
            assert res_ok.degraded is False
            assert res_ok.user_profiles_updated == 0
            assert res_ok.memories_added == 1
            assert res_ok.memories_reinforced == 0
            assert res_ok.memories_superseded == 0
            assert res_ok.jargon_upserted == 0

            # Outsider profile must not exist; out-of-scope memories untouched.
            assert await mem_store.get_user_profile("outsider_999") is None
            victim_mem = await mem_store.get_memory(other_priv_id)
            assert victim_mem is not None and victim_mem.evidence_count == 1
            other_grp_mem = await mem_store.get_memory(other_grp_id)
            assert other_grp_mem is not None and other_grp_mem.status == "active"

            # Valid op landed with owner_user_id=None and visibility="group_only".
            g1_mems = await mem_store.list_active_memories(group_id="g1")
            assert len(g1_mems) == 1
            assert g1_mems[0].content == "群里今天讨论了代码重构"
            assert g1_mems[0].owner_user_id is None
            assert g1_mems[0].visibility == "group_only"
            assert (await sess_store.get_or_create("group:g1")).last_extracted_id == id1
    finally:
        await sdk_client.close()
        await llm_http.aclose()


async def test_consolidation_empty_episode_fifo_serialization_cleanup_and_close() -> (
    None
):
    cfg = _make_config(
        max_messages=10,
        cleanup_interval_sec=60.0,
        retention_days=7,
        timezone="UTC",
    )
    execution_order: list[int] = []
    ep1_started = asyncio.Event()
    ep1_release = asyncio.Event()

    async def slow_complete(
        messages: Any,
        *,
        effort: Any = "default",
        tools: Any = (),
        session_key: str | None = None,
    ) -> Any:
        user_content = messages[1]["content"]
        if "EPISODE_ONE_MARKER" in user_content:
            execution_order.append(1)
            ep1_started.set()
            await ep1_release.wait()
        elif "EPISODE_TWO_MARKER" in user_content:
            execution_order.append(2)

        class _Resp:
            content = json.dumps(
                {
                    "group_profile_patch": None,
                    "user_profile_patches": [],
                    "memory_ops": [],
                    "jargon_ops": [],
                }
            )

        return _Resp()

    class _StubLLM:
        complete = staticmethod(slow_complete)

    mono_clock = [1000.0]
    wall_clock = [2_000_000_000.0]

    async with Database(":memory:") as db:
        msg_store = MessageStore(db)
        await msg_store.init()
        sess_store = SessionStore(db, msg_store, cfg.session)
        await sess_store.init()
        mem_store = MemoryStore(db, cfg.memory)
        await mem_store.init()

        # Seed an old expired message (10 days old) + active recent messages.
        old_ts = int(wall_clock[0]) - 10 * 86400
        recent_ts = int(wall_clock[0]) - 100
        old_id = await msg_store.append_message(
            group_id="g1",
            user_id="u1",
            sender_name="u1",
            role="user",
            content="十天前的过期消息",
            created_at=old_ts,
        )
        ep1_id = await msg_store.append_message(
            group_id="g1",
            user_id="u1",
            sender_name="u1",
            role="user",
            content="EPISODE_ONE_MARKER",
            created_at=recent_ts,
        )
        ep2_id = await msg_store.append_message(
            group_id="g1",
            user_id="u1",
            sender_name="u1",
            role="user",
            content="EPISODE_TWO_MARKER",
            created_at=recent_ts + 10,
        )
        bot_only_id = await msg_store.append_message(
            group_id="g1",
            user_id="99999",
            sender_name="Spectra",
            role="assistant",
            content="仅机器人消息",
            created_at=recent_ts + 20,
        )
        # Active session window starts at ep1_id, last_extracted_id starts at 0.
        await sess_store.update_window(
            "group:g1", window_start_id=ep1_id, now=recent_ts
        )

        consolidator = MemoryConsolidator(
            config=cfg,
            message_store=msg_store,
            session_store=sess_store,
            memory_store=mem_store,
            llm_client=_StubLLM(),  # type: ignore[arg-type]
            clock_fn=lambda: mono_clock[0],
            time_fn=lambda: wall_clock[0],
        )

        # Enqueue ep1 then ep2 immediately -> non-blocking & strictly serialized!
        await consolidator.handle_episode_closed(
            ClosedEpisode(
                session_key="group:g1",
                group_id="g1",
                user_id="u1",
                episode_id=1,
                start_msg_id=ep1_id,
                end_msg_id=ep1_id,
                closed_at=recent_ts + 5,
            )
        )
        consolidator.enqueue(
            ClosedEpisode(
                session_key="group:g1",
                group_id="g1",
                user_id="u1",
                episode_id=2,
                start_msg_id=ep2_id,
                end_msg_id=ep2_id,
                closed_at=recent_ts + 15,
            )
        )

        async with asyncio.timeout(2.0):
            await ep1_started.wait()
        # ep2 must NOT have started while ep1 is still running on same session_key!
        await asyncio.sleep(0.01)
        assert execution_order == [1]

        ep1_release.set()
        await consolidator.flush()
        assert execution_order == [1, 2]

        # Expired old_id (id < min_protected_message_id=ep1_id) must have been deleted!
        remaining = await msg_store.get_group_messages("g1", limit=20)
        assert all(r.id != old_id for r in remaining)

        # Assistant-only episode advances cursor with zero LLM call.
        res_empty = await consolidator.consolidate_episode(
            ClosedEpisode(
                session_key="group:g1",
                group_id="g1",
                user_id="u1",
                episode_id=3,
                start_msg_id=bot_only_id,
                end_msg_id=bot_only_id,
                closed_at=recent_ts + 25,
            )
        )
        assert res_empty.degraded is False
        assert (
            await sess_store.get_or_create("group:g1")
        ).last_extracted_id == bot_only_id

        # Idempotent close works cleanly.
        await consolidator.close()
        await consolidator.close()


async def test_consolidation_close_cancels_inflight_task_cleanly() -> None:
    cfg = _make_config(timezone="UTC")
    started = asyncio.Event()
    cancelled = asyncio.Event()

    class _BlockingLLM:
        async def complete(self, *args: Any, **kwargs: Any) -> Any:
            started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                cancelled.set()
                raise

    async with Database(":memory:") as db:
        msg_store = MessageStore(db)
        await msg_store.init()
        sess_store = SessionStore(db, msg_store, cfg.session)
        await sess_store.init()
        mem_store = MemoryStore(db, cfg.memory)
        await mem_store.init()

        msg_id = await msg_store.append_message(
            group_id="g1",
            user_id="u1",
            sender_name="u1",
            role="user",
            content="阻塞测试",
            created_at=1700000000,
        )
        consolidator = MemoryConsolidator(
            config=cfg,
            message_store=msg_store,
            session_store=sess_store,
            memory_store=mem_store,
            llm_client=_BlockingLLM(),  # type: ignore[arg-type]
        )
        consolidator.enqueue(
            ClosedEpisode(
                session_key="group:g1",
                group_id="g1",
                user_id="u1",
                episode_id=1,
                start_msg_id=msg_id,
                end_msg_id=msg_id,
                closed_at=1700003600,
            )
        )
        async with asyncio.timeout(2.0):
            await started.wait()

        # CancelledError must propagate out of consolidate_episode (separate session).
        msg_id_g2 = await msg_store.append_message(
            group_id="g2",
            user_id="u1",
            sender_name="u1",
            role="user",
            content="取消传播测试",
            created_at=1700000010,
        )
        with pytest.raises(asyncio.CancelledError):
            started.clear()
            task = asyncio.create_task(
                consolidator.consolidate_episode(
                    ClosedEpisode(
                        session_key="group:g2",
                        group_id="g2",
                        user_id="u1",
                        episode_id=1,
                        start_msg_id=msg_id_g2,
                        end_msg_id=msg_id_g2,
                        closed_at=1700003700,
                    )
                )
            )
            async with asyncio.timeout(2.0):
                await started.wait()
            task.cancel()
            await task

        await consolidator.close(drain=False)
        assert cancelled.is_set()
        assert not consolidator._tasks


async def test_build_recall_memory_tool_scope_locking_and_validation() -> None:
    async with Database(":memory:") as db:
        store = MemoryStore(db, MemoryConfig())
        await store.init()

        await store.add_memory(
            kind="preference",
            content="小明最喜欢手冲咖啡",
            visibility="public",
            owner_user_id="u1",
            source_group_id="g_other",
            confidence=0.9,
            now=100,
        )
        await store.add_memory(
            kind="fact",
            content="本群每周五咖啡拼单",
            visibility="group_only",
            owner_user_id=None,
            source_group_id="g1",
            confidence=0.85,
            now=100,
        )
        await store.add_memory(
            kind="fact",
            content="小明私聊秘密咖啡账单",
            visibility="private",
            owner_user_id="u1",
            source_group_id=None,
            confidence=0.95,
            now=100,
        )

        group_tool = build_recall_memory_tool(
            store,
            group_id="g1",
            user_id="u1",
            active_speaker_ids=("u1",),
        )
        assert group_tool.name == "recall_memory"
        # Schema MUST ONLY expose query and limit — never scope parameters!
        assert set(group_tool.parameters["properties"].keys()) == {"query", "limit"}
        assert group_tool.parameters["required"] == ["query"]

        # Blank query & bool limit return error strings instead of raising.
        assert (await group_tool.handler(query="   ")).startswith("Error:")
        assert (await group_tool.handler(query="咖啡", limit=True)).startswith("Error:")

        # Spoofed scope kwargs cannot leak private memory in group chat.
        raw_res = await group_tool.handler(
            query="咖啡",
            limit=99,  # clamped to 20
            group_id=None,
            user_id="u1",
        )
        parsed = json.loads(raw_res)
        contents = [m["content"] for m in parsed["memories"]]
        assert "本群每周五咖啡拼单" in contents
        assert "小明最喜欢手冲咖啡" in contents
        assert "小明私聊秘密咖啡账单" not in contents

        # Clamping limit=0 -> 1 returns at most 1 item.
        raw_one = await group_tool.handler(query="咖啡", limit=0)
        assert len(json.loads(raw_one)["memories"]) == 1

        # Private tool sees private + cross-group public of u1, not group_only.
        dm_tool = build_recall_memory_tool(store, group_id=None, user_id="u1")
        dm_parsed = json.loads(await dm_tool.handler(query="咖啡"))
        dm_contents = [m["content"] for m in dm_parsed["memories"]]
        assert "小明私聊秘密咖啡账单" in dm_contents
        assert "小明最喜欢手冲咖啡" in dm_contents
        assert "本群每周五咖啡拼单" not in dm_contents

        # format_jargon_reminder returns empty string when no entries match.
        assert format_jargon_reminder([]) == ""
