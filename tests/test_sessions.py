"""Unit tests for spectra.sessions."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest

from spectra.config import JevConfig, SessionConfig
from spectra.db import Database
from spectra.jev import JevClient
from spectra.messages import MessageStore
from spectra.napcat import InboundMessage
from spectra.sessions import SessionStore


async def _make_stores(
    config: SessionConfig | None = None,
) -> tuple[Database, MessageStore, SessionStore]:
    db = Database(":memory:")
    await db.connect()
    msg_store = MessageStore(db)
    await msg_store.init()
    session_store = SessionStore(db, msg_store, config or SessionConfig())
    await session_store.init()
    return db, msg_store, session_store


def _make_inbound(
    *,
    db_id: int,
    group_id: str | None,
    user_id: str,
    sender_name: str,
    content: str,
    timestamp: int,
    message_id: str = "",
    reply_to_message_id: str | None = None,
) -> InboundMessage:
    return InboundMessage(
        db_id=db_id,
        message_id=message_id or str(db_id),
        self_id="99999",
        group_id=group_id,
        user_id=user_id,
        sender_name=sender_name,
        content=content,
        is_mentioned_bot=True,
        starts_with_at_other=False,
        timestamp=timestamp,
        reply_to_message_id=reply_to_message_id,
    )


async def test_sessions_migration_schema_and_idempotency() -> None:
    db, _, session_store = await _make_stores()
    try:
        # Second init must be a no-op.
        await session_store.init()
        cols = await db.fetchall("PRAGMA table_info(sessions)")
        col_names = [str(row["name"]) for row in cols]
        assert col_names == [
            "session_key",
            "episode_id",
            "episode_start_id",
            "window_start_id",
            "last_message_at",
            "last_extracted_id",
            "profile_snapshot",
            "updated_at",
        ]
    finally:
        await db.close()


async def test_get_or_create_touch_record_turn_update_window_and_extracted() -> None:
    db, _, session_store = await _make_stores()
    try:
        # Concurrent get_or_create should be race-free.
        recs = await asyncio.gather(
            session_store.get_or_create("group:100", now=1000),
            session_store.get_or_create("group:100", now=1000),
            session_store.get_or_create("group:100", now=1000),
        )
        assert all(r == recs[0] for r in recs)
        assert recs[0].episode_id == 1
        assert recs[0].episode_start_id == 0
        assert recs[0].window_start_id == 0
        assert recs[0].last_message_at == 0
        assert recs[0].last_extracted_id == 0
        assert recs[0].profile_snapshot == ""
        assert recs[0].updated_at == 1000

        await session_store.update_window("group:100", window_start_id=25, now=1100)
        rec = await session_store.get_or_create("group:100")
        assert rec.window_start_id == 25
        assert rec.updated_at == 1100

        await session_store.record_turn("group:100", message_at=1300)
        rec = await session_store.get_or_create("group:100")
        assert rec.last_message_at == 1300
        assert rec.updated_at == 1300
        # Monotonic last_message_at must not regress.
        await session_store.record_turn("group:100", message_at=1200)
        rec = await session_store.get_or_create("group:100")
        assert rec.last_message_at == 1300

        # Monotonic last_extracted_id must not regress.
        await session_store.advance_extracted_cursor("group:100", 20, now=1400)
        await session_store.advance_extracted_cursor("group:100", 10, now=1450)
        rec = await session_store.get_or_create("group:100")
        assert rec.last_extracted_id == 20
        assert rec.updated_at == 1450

        # touch updates updated_at without clobbering fields.
        await session_store.touch("group:100", now=1500)
        rec = await session_store.get_or_create("group:100")
        assert rec.window_start_id == 25
        assert rec.last_message_at == 1300
        assert rec.last_extracted_id == 20
        assert rec.updated_at == 1500

        with pytest.raises(ValueError, match="non-negative"):
            await session_store.update_window("group:100", window_start_id=-1)
        with pytest.raises(ValueError, match="non-negative"):
            await session_store.advance_extracted_cursor("group:100", -1)
    finally:
        await db.close()


async def test_min_protected_message_id_active_since_and_retention_safety() -> None:
    db, msg_store, session_store = await _make_stores()
    try:
        assert await session_store.min_protected_message_id() is None

        ids = [
            await msg_store.append_message(
                group_id="100",
                user_id="1",
                sender_name="Alice",
                role="user",
                content=f"msg-{i}",
                created_at=100 + i,
            )
            for i in range(1, 11)
        ]

        # Dormant session at low cursor (old updated_at=200).
        await session_store.update_window(
            "group:dormant", window_start_id=ids[1], now=200
        )
        await session_store.advance_extracted_cursor("group:dormant", ids[0], now=200)

        # Active session at higher cursor (episode_start_id protects rolled-over rows).
        await session_store.update_window("group:100", window_start_id=ids[7], now=5000)
        await db.execute(
            "UPDATE sessions SET episode_start_id = ?, last_extracted_id = ?, "
            "last_message_at = ? WHERE session_key = ?",
            (ids[4], ids[6], 5000, "group:100"),
        )

        # Without active_since, dormant session protects from ids[0] (= 1).
        assert await session_store.min_protected_message_id() == ids[0]

        # With active_since=1000, dormant session is excluded; active session protects
        # min(window_start_id=ids[7], last_extracted_id=ids[6], episode_start_id=ids[4])
        # == ids[4] (= 5).
        protected = await session_store.min_protected_message_id(active_since=1000)
        assert protected == ids[4]

        # When active_since is past all sessions, returns None (not deadlocked at 0).
        assert await session_store.min_protected_message_id(active_since=9999) is None

        deleted = await msg_store.cleanup_expired_messages(
            before_created_at=999,
            max_deletable_id=protected,
        )
        assert deleted == 4
        remaining = await msg_store.get_group_messages("100", limit=20)
        assert [r.id for r in remaining] == ids[4:]
    finally:
        await db.close()


async def test_build_context_group_formatting_timezone_coalescing_and_snapshot() -> (
    None
):
    # 2026-01-01 01:05:00 UTC == 2026-01-01 09:05 in Asia/Shanghai
    ts_utc = int(datetime(2026, 1, 1, 1, 5, 0, tzinfo=UTC).timestamp())
    db, msg_store, session_store = await _make_stores(
        SessionConfig(timezone="Asia/Shanghai")
    )
    try:
        # Leading assistant message should be skipped.
        await msg_store.append_message(
            group_id="500",
            user_id="99999",
            sender_name="Spectra",
            role="assistant",
            content="旧的孤立回复",
            created_at=ts_utc - 60,
        )
        await msg_store.append_message(
            group_id="500",
            user_id="1001",
            sender_name="小明",
            role="user",
            content="早上好",
            created_at=ts_utc,
        )
        await msg_store.append_message(
            group_id="500",
            user_id="1002",
            sender_name="小红",
            role="user",
            content="今天吃啥",
            created_at=ts_utc + 60,
        )
        await msg_store.append_message(
            group_id="500",
            user_id="99999",
            sender_name="Spectra",
            role="assistant",
            content="早呀",
            created_at=ts_utc + 120,
        )
        await msg_store.append_message(
            group_id="500",
            user_id="99999",
            sender_name="Spectra",
            role="assistant",
            content="推荐牛肉面",
            created_at=ts_utc + 125,
        )
        last_included_id = await msg_store.append_message(
            group_id="500",
            user_id="1001",
            sender_name="小明",
            role="user",
            content="好主意",
            created_at=ts_utc + 180,
        )
        # Inserted after up_to_id -> must be excluded.
        await msg_store.append_message(
            group_id="500",
            user_id="1003",
            sender_name="路人",
            role="user",
            content="这条不应该进上下文",
            created_at=ts_utc + 240,
        )

        await session_store.get_or_create("group:500", now=ts_utc)
        await db.execute(
            "UPDATE sessions SET profile_snapshot = ? WHERE session_key = ?",
            ("## 当前群聊画像\n- 相处风格：轻松技术群", "group:500"),
        )

        ctx = await session_store.build_context(
            session_key="group:500",
            group_id="500",
            user_id="1001",
            system_prompt="基础人设",
            up_to_id=last_included_id,
        )

        assert [m["role"] for m in ctx] == ["system", "user", "assistant", "user"]
        assert (
            ctx[0]["content"] == "基础人设\n\n## 当前群聊画像\n- 相处风格：轻松技术群"
        )
        assert ctx[1]["content"] == (
            "[小明(1001) 09:05]: 早上好\n[小红(1002) 09:06]: 今天吃啥"
        )
        assert ctx[2]["content"] == "早呀\n推荐牛肉面"
        assert ctx[3]["content"] == "[小明(1001) 09:08]: 好主意"
    finally:
        await db.close()


async def test_build_context_private_raw_format_and_empty_snapshot() -> None:
    db, msg_store, session_store = await _make_stores(SessionConfig(timezone="UTC"))
    try:
        await msg_store.append_message(
            group_id=None,
            user_id="2001",
            sender_name="Bob",
            role="user",
            content="你好",
            created_at=1000,
        )
        await msg_store.append_message(
            group_id=None,
            user_id="2001",
            sender_name="Bob",
            role="user",
            content="在吗",
            created_at=1005,
        )
        await msg_store.append_message(
            group_id=None,
            user_id="2001",
            sender_name="Spectra",
            role="assistant",
            content="在的！",
            created_at=1010,
        )
        await msg_store.append_message(
            group_id=None,
            user_id="2001",
            sender_name="Bob",
            role="user",
            content="帮我看下代码",
            created_at=1020,
        )

        ctx = await session_store.build_context(
            session_key="private:2001",
            group_id=None,
            user_id="2001",
            system_prompt="  私聊人设  ",
        )

        assert [m["role"] for m in ctx] == ["system", "user", "assistant", "user"]
        assert ctx[0]["content"] == "私聊人设"
        assert ctx[1]["content"] == "你好\n在吗"
        assert ctx[2]["content"] == "在的！"
        assert ctx[3]["content"] == "帮我看下代码"
    finally:
        await db.close()


async def test_prefix_stable_block_rollover_preserves_episode_start_and_snapshot() -> (
    None
):
    cfg = SessionConfig(window_max_messages=6, window_keep_messages=3, timezone="UTC")
    db, msg_store, session_store = await _make_stores(cfg)
    try:
        roles_6: list[str] = [
            "user",
            "assistant",
            "user",
            "assistant",
            "user",
            "assistant",
        ]
        ids: list[int] = []
        for idx, role in enumerate(roles_6, start=1):
            msg_id = await msg_store.append_message(
                group_id=None,
                user_id="3001",
                sender_name="U" if role == "user" else "Spectra",
                role=role,  # type: ignore[arg-type]
                content=f"m{idx}",
                created_at=1000 + idx,
            )
            ids.append(msg_id)

        # Initialize episode at ids[0] with frozen profile snapshot.
        async def _profile() -> str:
            return "## 当前私聊对象画像\n- 相处风格：简洁直接"

        init_res = await session_store.resolve_episode(
            session_key="private:3001",
            group_id=None,
            user_id="3001",
            batch_messages=[
                _make_inbound(
                    db_id=ids[0],
                    group_id=None,
                    user_id="3001",
                    sender_name="U",
                    content="m1",
                    timestamp=1001,
                )
            ],
            profile_loader=_profile,
        )
        assert init_res.session.episode_start_id == ids[0]
        assert init_res.session.window_start_id == ids[0]

        ctx6 = await session_store.build_context(
            session_key="private:3001",
            group_id=None,
            user_id="3001",
            system_prompt="sys",
            now=2000,
        )
        assert len(ctx6) == 7

        # Insert 7th message (user m7) -> triggers rollover to ids[4] (m5),
        # while preserving episode_start_id=ids[0] and profile_snapshot!
        m7_id = await msg_store.append_message(
            group_id=None,
            user_id="3001",
            sender_name="U",
            role="user",
            content="m7",
            created_at=1007,
        )
        ids.append(m7_id)

        ctx7 = await session_store.build_context(
            session_key="private:3001",
            group_id=None,
            user_id="3001",
            system_prompt="sys",
            now=2100,
        )
        session_after = await session_store.get_or_create("private:3001")
        assert session_after.window_start_id == ids[4]
        assert session_after.episode_start_id == ids[0]
        assert (
            session_after.profile_snapshot
            == "## 当前私聊对象画像\n- 相处风格：简洁直接"
        )
        assert [m["content"] for m in ctx7[1:]] == ["m5", "m6", "m7"]

        # Subsequent hard timeout cut must emit the FULL [episode_start_id, m7] span!
        m8_id = await msg_store.append_message(
            group_id=None,
            user_id="3001",
            sender_name="U",
            role="user",
            content="次日新话题",
            created_at=1007 + 25 * 3600,
        )
        cut_res = await session_store.resolve_episode(
            session_key="private:3001",
            group_id=None,
            user_id="3001",
            batch_messages=[
                _make_inbound(
                    db_id=m8_id,
                    group_id=None,
                    user_id="3001",
                    sender_name="U",
                    content="次日新话题",
                    timestamp=1007 + 25 * 3600,
                )
            ],
        )
        assert cut_res.new_episode is True
        assert cut_res.reason == "hard_timeout"
        assert cut_res.closed_episode is not None
        assert cut_res.closed_episode.episode_id == 1
        assert cut_res.closed_episode.start_msg_id == ids[0]
        assert cut_res.closed_episode.end_msg_id == m7_id
    finally:
        await db.close()


async def test_default_60_to_30_prefix_stable_rollover() -> None:
    db, msg_store, session_store = await _make_stores(SessionConfig(timezone="UTC"))
    try:
        ids: list[int] = []
        for i in range(1, 61):
            role = "user" if i % 2 == 1 else "assistant"
            ids.append(
                await msg_store.append_message(
                    group_id=None,
                    user_id="4001",
                    sender_name="U" if role == "user" else "Spectra",
                    role=role,  # type: ignore[arg-type]
                    content=f"msg-{i}",
                    created_at=1000 + i,
                )
            )

        ctx_60 = await session_store.build_context(
            session_key="private:4001",
            group_id=None,
            user_id="4001",
            system_prompt="sys",
            now=2000,
        )
        assert (await session_store.get_or_create("private:4001")).window_start_id == 0
        assert len(ctx_60) == 61

        ids.append(
            await msg_store.append_message(
                group_id=None,
                user_id="4001",
                sender_name="U",
                role="user",
                content="msg-61",
                created_at=1061,
            )
        )
        ctx_61 = await session_store.build_context(
            session_key="private:4001",
            group_id=None,
            user_id="4001",
            system_prompt="sys",
            now=2100,
        )
        expected_anchor = ids[32]
        assert (
            await session_store.get_or_create("private:4001")
        ).window_start_id == expected_anchor
        assert ctx_61[1] == {"role": "user", "content": "msg-33"}
        assert ctx_61[-1] == {"role": "user", "content": "msg-61"}

        for i in range(62, 72):
            role = "user" if i % 2 == 1 else "assistant"
            await msg_store.append_message(
                group_id=None,
                user_id="4001",
                sender_name="U" if role == "user" else "Spectra",
                role=role,  # type: ignore[arg-type]
                content=f"msg-{i}",
                created_at=1000 + i,
            )
        await session_store.build_context(
            session_key="private:4001",
            group_id=None,
            user_id="4001",
            system_prompt="sys",
            now=2200,
        )
        assert (
            await session_store.get_or_create("private:4001")
        ).window_start_id == expected_anchor
    finally:
        await db.close()


async def test_rollover_all_assistant_tail_and_all_assistant_session() -> None:
    cfg = SessionConfig(window_max_messages=3, window_keep_messages=1, timezone="UTC")
    db, msg_store, session_store = await _make_stores(cfg)
    try:
        ids_a: list[int] = []
        for idx, role in enumerate(["user", "assistant", "user", "assistant"], start=1):
            ids_a.append(
                await msg_store.append_message(
                    group_id=None,
                    user_id="5001",
                    sender_name="U" if role == "user" else "Spectra",
                    role=role,  # type: ignore[arg-type]
                    content=f"m{idx}",
                    created_at=1000 + idx,
                )
            )
        ctx_a = await session_store.build_context(
            session_key="private:5001",
            group_id=None,
            user_id="5001",
            system_prompt="sys",
            now=2000,
        )
        assert [m["role"] for m in ctx_a] == ["system", "user", "assistant"]
        assert [m["content"] for m in ctx_a[1:]] == ["m3", "m4"]
        assert (
            await session_store.get_or_create("private:5001")
        ).window_start_id == ids_a[2]

        ids_b: list[int] = []
        for idx in range(1, 5):
            ids_b.append(
                await msg_store.append_message(
                    group_id=None,
                    user_id="5002",
                    sender_name="Spectra",
                    role="assistant",
                    content=f"a{idx}",
                    created_at=1000 + idx,
                )
            )
        ctx_b = await session_store.build_context(
            session_key="private:5002",
            group_id=None,
            user_id="5002",
            system_prompt="sys",
            now=2000,
        )
        assert ctx_b == [{"role": "system", "content": "sys"}]
        assert (
            await session_store.get_or_create("private:5002")
        ).window_start_id == ids_b[-1] + 1
    finally:
        await db.close()


async def test_min_keep_id_preserves_multi_message_trigger_batch() -> None:
    cfg = SessionConfig(window_max_messages=6, window_keep_messages=2, timezone="UTC")
    db, msg_store, session_store = await _make_stores(cfg)
    try:
        ids: list[int] = []
        for idx in range(1, 10):
            ids.append(
                await msg_store.append_message(
                    group_id=None,
                    user_id="6001",
                    sender_name="U",
                    role="user",
                    content=f"m{idx}",
                    created_at=1000 + idx,
                )
            )
        ctx = await session_store.build_context(
            session_key="private:6001",
            group_id=None,
            user_id="6001",
            system_prompt="sys",
            up_to_id=ids[-1],
            min_keep_id=ids[5],
            now=2000,
        )
        assert [m["role"] for m in ctx] == ["system", "user"]
        assert ctx[1]["content"] == "m6\nm7\nm8\nm9"
        assert (
            await session_store.get_or_create("private:6001")
        ).window_start_id == ids[5]

        burst_ids: list[int] = []
        for idx in range(1, 9):
            burst_ids.append(
                await msg_store.append_message(
                    group_id=None,
                    user_id="6002",
                    sender_name="U",
                    role="user",
                    content=f"burst-{idx}",
                    created_at=2000 + idx,
                )
            )
        ctx_burst = await session_store.build_context(
            session_key="private:6002",
            group_id=None,
            user_id="6002",
            system_prompt="sys",
            up_to_id=burst_ids[-1],
            min_keep_id=burst_ids[0],
            now=2500,
        )
        assert ctx_burst[1]["content"] == "\n".join(f"burst-{i}" for i in range(1, 9))
    finally:
        await db.close()


async def test_whitespace_rows_skipped_without_breaking_alternation() -> None:
    db, msg_store, session_store = await _make_stores(SessionConfig(timezone="UTC"))
    try:
        for role, content in (
            ("assistant", "孤立开头"),
            ("user", "   \n\t  "),
            ("assistant", "紧随空白用户的assistant"),
            ("user", "第一句"),
            ("assistant", "   "),
            ("user", "第二句"),
            ("assistant", "有效回复"),
        ):
            await msg_store.append_message(
                group_id=None,
                user_id="7001",
                sender_name="U" if role == "user" else "Spectra",
                role=role,  # type: ignore[arg-type]
                content=content,
                created_at=1000,
            )

        ctx = await session_store.build_context(
            session_key="private:7001",
            group_id=None,
            user_id="7001",
            system_prompt="sys",
        )
        assert ctx == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "第一句\n第二句"},
            {"role": "assistant", "content": "有效回复"},
        ]
    finally:
        await db.close()


async def test_resolve_episode_lead_in_cluster_and_intra_cluster_quote_invariant() -> (
    None
):
    """Group silent 2h -> A sends unmentioned setup at 14:00:00 -> B @Bots quoting A
    at 14:00:30:
    1. A's message is absorbed into the new episode (cluster_start_id == a_id);
    2. B quoting A (id >= cluster_start_id) is NOT misclassified as in_window_reply;
    3. Jev says new topic -> ClosedEpisode ends strictly before A's message!
    """
    cfg = SessionConfig(
        timezone="UTC",
        group_idle_timeout_minutes=30,
        group_hard_timeout_hours=6,
        lead_in_gap_sec=300,
        lead_in_max_messages=12,
    )
    db, msg_store, session_store = await _make_stores(cfg)
    jev_calls: list[dict[str, Any]] = []

    def jev_handler(request: httpx.Request) -> httpx.Response:
        jev_calls.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "model": "typesafe/jev-1.14",
                "usage": {"input_tokens": 50},
                "answers": {"should_continue": {"noul": 0.15}},
            },
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(jev_handler))
    jev = JevClient(JevConfig(api_key="sk-jev-test"), http_client=http)
    try:
        # Episode 1 at 12:00:00.
        t_ep1 = 1700000000
        ep1_u = await msg_store.append_message(
            group_id="8001",
            user_id="u1",
            sender_name="Alice",
            role="user",
            content="@Bot 午饭吃什么",
            created_at=t_ep1,
            platform_message_id="p-ep1-u",
        )
        ep1_a = await msg_store.append_message(
            group_id="8001",
            user_id="99999",
            sender_name="Spectra",
            role="assistant",
            content="吃牛肉面吧！",
            created_at=t_ep1 + 10,
            platform_message_id="p-ep1-a",
        )
        res1 = await session_store.resolve_episode(
            session_key="group:8001",
            group_id="8001",
            user_id="u1",
            batch_messages=[
                _make_inbound(
                    db_id=ep1_u,
                    group_id="8001",
                    user_id="u1",
                    sender_name="Alice",
                    content="@Bot 午饭吃什么",
                    timestamp=t_ep1,
                    message_id="p-ep1-u",
                )
            ],
            jev_client=jev,
        )
        assert res1.reason == "initial"
        assert res1.new_episode is False
        await session_store.record_turn("group:8001", message_at=t_ep1 + 10)

        # 2 hours later (14:00:00): A sends unmentioned lead-in setup.
        t_lead = t_ep1 + 2 * 3600
        lead_a = await msg_store.append_message(
            group_id="8001",
            user_id="u1",
            sender_name="Alice",
            role="user",
            content="我刚跑 CI 报错了，pytest 卡在 timeout",
            created_at=t_lead,
            platform_message_id="p-lead-a",
        )
        # 30s later (14:00:30): B @Bots AND quotes A's lead-in message p-lead-a!
        b_msg = await msg_store.append_message(
            group_id="8001",
            user_id="u2",
            sender_name="Bob",
            role="user",
            content="[回复 Alice: 我刚跑 CI 报错了] @Bot 帮看看这个报错",
            created_at=t_lead + 30,
            platform_message_id="p-b-msg",
            reply_to_message_id="p-lead-a",
        )

        res2 = await session_store.resolve_episode(
            session_key="group:8001",
            group_id="8001",
            user_id="u2",
            batch_messages=[
                _make_inbound(
                    db_id=b_msg,
                    group_id="8001",
                    user_id="u2",
                    sender_name="Bob",
                    content="[回复 Alice: 我刚跑 CI 报错了] @Bot 帮看看这个报错",
                    timestamp=t_lead + 30,
                    message_id="p-b-msg",
                    reply_to_message_id="p-lead-a",
                )
            ],
            jev_client=jev,
            advance_extracted_on_close=False,
        )
        # Intra-cluster quote must NOT trigger in_window_reply -> fell through to Jev!
        assert len(jev_calls) == 1
        assert res2.new_episode is True
        assert res2.reason == "jev_new_episode"
        assert res2.continuity_probability == pytest.approx(0.15)
        assert res2.session.episode_id == 2
        assert res2.session.episode_start_id == lead_a
        assert res2.session.window_start_id == lead_a
        # advance_extracted_on_close=False preserves last_extracted_id for async worker.
        assert res2.session.last_extracted_id == ep1_u - 1
        # ClosedEpisode ends strictly before A's lead-in message!
        assert res2.closed_episode is not None
        assert res2.closed_episode.episode_id == 1
        assert res2.closed_episode.start_msg_id == ep1_u
        assert res2.closed_episode.end_msg_id == ep1_a

        # Episode 2 context includes A's lead-in and B's message, not episode 1!
        ctx2 = await session_store.build_context(
            session_key="group:8001",
            group_id="8001",
            user_id="u2",
            system_prompt="sys",
            up_to_id=b_msg,
            min_keep_id=b_msg,
        )
        assert len(ctx2) == 2
        assert "午饭吃什么" not in ctx2[1]["content"]
        assert "我刚跑 CI 报错了" in ctx2[1]["content"]
        assert "@Bot 帮看看这个报错" in ctx2[1]["content"]
    finally:
        await http.aclose()
        await db.close()


async def test_resolve_episode_in_window_reply_vs_out_of_window_and_jev_continue() -> (
    None
):
    cfg = SessionConfig(
        timezone="UTC",
        group_idle_timeout_minutes=30,
        group_hard_timeout_hours=6,
    )
    db, msg_store, session_store = await _make_stores(cfg)
    jev_calls = 0

    def jev_handler(_: httpx.Request) -> httpx.Response:
        nonlocal jev_calls
        jev_calls += 1
        return httpx.Response(
            200,
            json={
                "model": "typesafe/jev-1.14",
                "answers": {"should_continue": {"noul": 0.82}},
            },
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(jev_handler))
    jev = JevClient(JevConfig(api_key="sk-jev-test"), http_client=http)
    try:
        # Very old pre-window message (will be outside window_start_id).
        old_id = await msg_store.append_message(
            group_id="8002",
            user_id="u1",
            sender_name="Alice",
            role="user",
            content="很早以前的消息",
            created_at=1700000000,
            platform_message_id="p-very-old",
        )
        # Current episode starts at ep1_u (window_start_id = ep1_u > old_id).
        t1 = 1700010000
        ep1_u = await msg_store.append_message(
            group_id="8002",
            user_id="u1",
            sender_name="Alice",
            role="user",
            content="开始新话题",
            created_at=t1,
            platform_message_id="p-ep1-u",
        )
        # Multi-bubble bot reply with bubble #2 ID "p-bubble-2".
        await msg_store.append_message(
            group_id="8002",
            user_id="99999",
            sender_name="Spectra",
            role="assistant",
            content="第一句。第二句。",
            created_at=t1 + 5,
            platform_message_id="p-bubble-1",
            extra_platform_ids=["p-bubble-1", "p-bubble-2"],
        )
        await session_store.resolve_episode(
            session_key="group:8002",
            group_id="8002",
            user_id="u1",
            batch_messages=[
                _make_inbound(
                    db_id=ep1_u,
                    group_id="8002",
                    user_id="u1",
                    sender_name="Alice",
                    content="开始新话题",
                    timestamp=t1,
                )
            ],
            jev_client=jev,
        )
        assert (
            await session_store.get_or_create("group:8002")
        ).window_start_id > old_id

        # 1) After 45 min silence, quoting bubble #2 triggers in_window_reply
        #    with zero Jev calls.
        t2 = t1 + 45 * 60
        m2 = await msg_store.append_message(
            group_id="8002",
            user_id="u2",
            sender_name="Bob",
            role="user",
            content="同意你第二句",
            created_at=t2,
            platform_message_id="p-m2",
            reply_to_message_id="p-bubble-2",
        )
        res_reply = await session_store.resolve_episode(
            session_key="group:8002",
            group_id="8002",
            user_id="u2",
            batch_messages=[
                _make_inbound(
                    db_id=m2,
                    group_id="8002",
                    user_id="u2",
                    sender_name="Bob",
                    content="同意你第二句",
                    timestamp=t2,
                    reply_to_message_id="p-bubble-2",
                )
            ],
            jev_client=jev,
        )
        assert res_reply.new_episode is False
        assert res_reply.reason == "in_window_reply"
        assert jev_calls == 0

        # 2) After another 45 min silence, quoting out-of-window p-very-old
        #    (< window_start_id) falls through to Jev -> jev_continue.
        t3 = t2 + 45 * 60
        m3 = await msg_store.append_message(
            group_id="8002",
            user_id="u1",
            sender_name="Alice",
            role="user",
            content="顺便补充一句",
            created_at=t3,
            platform_message_id="p-m3",
            reply_to_message_id="p-very-old",
        )
        res_jev = await session_store.resolve_episode(
            session_key="group:8002",
            group_id="8002",
            user_id="u1",
            batch_messages=[
                _make_inbound(
                    db_id=m3,
                    group_id="8002",
                    user_id="u1",
                    sender_name="Alice",
                    content="顺便补充一句",
                    timestamp=t3,
                    reply_to_message_id="p-very-old",
                )
            ],
            jev_client=jev,
        )
        assert jev_calls == 1
        assert res_jev.new_episode is False
        assert res_jev.reason == "jev_continue"
        assert res_jev.continuity_probability == pytest.approx(0.82)

        # 3) After 45 min silence with jev_client=None -> jev_degraded_new_episode.
        t4 = t3 + 45 * 60
        m4 = await msg_store.append_message(
            group_id="8002",
            user_id="u1",
            sender_name="Alice",
            role="user",
            content="聊点别的",
            created_at=t4,
        )
        res_deg = await session_store.resolve_episode(
            session_key="group:8002",
            group_id="8002",
            user_id="u1",
            batch_messages=[
                _make_inbound(
                    db_id=m4,
                    group_id="8002",
                    user_id="u1",
                    sender_name="Alice",
                    content="聊点别的",
                    timestamp=t4,
                )
            ],
            jev_client=None,
        )
        assert res_deg.new_episode is True
        assert res_deg.reason == "jev_degraded_new_episode"
        assert res_deg.session.episode_id == 2
        assert res_deg.session.last_extracted_id == m3
    finally:
        await http.aclose()
        await db.close()
