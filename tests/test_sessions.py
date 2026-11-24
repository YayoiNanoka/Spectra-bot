"""Unit tests for spectra.sessions."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from spectra.config import SessionConfig
from spectra.db import Database
from spectra.messages import MessageStore
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


async def test_sessions_migration_schema_and_idempotency() -> None:
    db, _, session_store = await _make_stores()
    try:
        # Second init must be a no-op.
        await session_store.init()
        cols = await db.fetchall("PRAGMA table_info(sessions)")
        col_names = [str(row["name"]) for row in cols]
        assert col_names == [
            "session_key",
            "window_start_id",
            "summary",
            "last_extracted_id",
            "group_impression",
            "updated_at",
        ]
    finally:
        await db.close()


async def test_get_or_create_touch_and_update_window() -> None:
    db, _, session_store = await _make_stores()
    try:
        # Concurrent get_or_create should be race-free.
        recs = await asyncio.gather(
            session_store.get_or_create("group:100", now=1000),
            session_store.get_or_create("group:100", now=1000),
            session_store.get_or_create("group:100", now=1000),
        )
        assert all(r == recs[0] for r in recs)
        assert recs[0].window_start_id == 0
        assert recs[0].summary == ""
        assert recs[0].last_extracted_id == 0
        assert recs[0].group_impression == ""
        assert recs[0].updated_at == 1000

        await session_store.update_window(
            "group:100",
            window_start_id=25,
            summary="近期摘要",
            now=1100,
        )
        rec = await session_store.get_or_create("group:100")
        assert rec.window_start_id == 25
        assert rec.summary == "近期摘要"
        assert rec.updated_at == 1100

        # Passing summary=None preserves existing summary.
        await session_store.update_window("group:100", window_start_id=40, now=1200)
        rec = await session_store.get_or_create("group:100")
        assert rec.window_start_id == 40
        assert rec.summary == "近期摘要"
        assert rec.updated_at == 1200

        # touch updates updated_at without clobbering fields.
        await session_store.touch("group:100", now=1500)
        rec = await session_store.get_or_create("group:100")
        assert rec.window_start_id == 40
        assert rec.summary == "近期摘要"
        assert rec.updated_at == 1500

        with pytest.raises(ValueError, match="non-negative"):
            await session_store.update_window("group:100", window_start_id=-1)
    finally:
        await db.close()


async def test_min_protected_message_id_and_retention_safety() -> None:
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

        await session_store.update_window("group:100", window_start_id=ids[5], now=200)
        await db.execute(
            "UPDATE sessions SET last_extracted_id = ? WHERE session_key = ?",
            (ids[3], "group:100"),
        )
        await session_store.update_window("private:2", window_start_id=ids[7], now=200)
        await db.execute(
            "UPDATE sessions SET last_extracted_id = ? WHERE session_key = ?",
            (ids[8], "private:2"),
        )

        # min(ids[5]=6, ids[3]=4, ids[7]=8, ids[8]=9) == 4.
        protected = await session_store.min_protected_message_id()
        assert protected == ids[3]

        deleted = await msg_store.cleanup_expired_messages(
            before_created_at=999,
            max_deletable_id=protected,
        )
        assert deleted == 3
        remaining = await msg_store.get_group_messages("100", limit=20)
        assert [r.id for r in remaining] == ids[3:]
    finally:
        await db.close()


async def test_build_context_group_formatting_timezone_coalescing_and_sections() -> (
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
            "UPDATE sessions SET group_impression = ?, summary = ? "
            "WHERE session_key = ?",
            ("轻松技术群", "大家在讨论午饭", "group:500"),
        )

        ctx = await session_store.build_context(
            session_key="group:500",
            group_id="500",
            user_id="1001",
            system_prompt="基础人设",
            up_to_id=last_included_id,
        )

        assert [m["role"] for m in ctx] == ["system", "user", "assistant", "user"]
        assert ctx[0]["content"] == (
            "基础人设\n\n"
            "## 当前群聊氛围与印象\n轻松技术群\n\n"
            "## 近期对话前情摘要\n大家在讨论午饭"
        )
        assert ctx[1]["content"] == (
            "[小明(1001) 09:05]: 早上好\n[小红(1002) 09:06]: 今天吃啥"
        )
        assert ctx[2]["content"] == "早呀\n推荐牛肉面"
        assert ctx[3]["content"] == "[小明(1001) 09:08]: 好主意"
    finally:
        await db.close()


async def test_build_context_private_raw_format_and_no_group_impression() -> None:
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

        await session_store.get_or_create("private:2001", now=1000)
        await db.execute(
            "UPDATE sessions SET group_impression = ?, summary = ? "
            "WHERE session_key = ?",
            ("不应该出现在私聊", "之前聊过Python", "private:2001"),
        )

        ctx = await session_store.build_context(
            session_key="private:2001",
            group_id=None,
            user_id="2001",
            system_prompt="私聊人设",
        )

        assert [m["role"] for m in ctx] == ["system", "user", "assistant", "user"]
        assert "当前群聊氛围与印象" not in ctx[0]["content"]
        assert ctx[0]["content"] == "私聊人设\n\n## 近期对话前情摘要\n之前聊过Python"
        assert ctx[1]["content"] == "你好\n在吗"
        assert ctx[2]["content"] == "在的！"
        assert ctx[3]["content"] == "帮我看下代码"
    finally:
        await db.close()


async def test_prefix_stable_block_rollover_and_assistant_boundary() -> None:
    cfg = SessionConfig(window_max_messages=6, window_keep_messages=3, timezone="UTC")
    db, msg_store, session_store = await _make_stores(cfg)
    try:
        # Insert 6 messages: [u1, a2, u3, a4, u5, a6] -> no rollover yet.
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

        ctx6 = await session_store.build_context(
            session_key="private:3001",
            group_id=None,
            user_id="3001",
            system_prompt="sys",
            now=2000,
        )
        session_before = await session_store.get_or_create("private:3001")
        assert session_before.window_start_id == 0
        assert len(ctx6) == 7  # system + 6 alternating messages

        # Insert 7th message (user m7):
        # Active rows = [m1(u), m2(a), m3(u), m4(a), m5(u), m6(a), m7(u)] (7 > 6)
        # Last 3 = [m5(u), m6(a), m7(u)] -> rolls window_start_id to ids[4] (m5).
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
        assert [m["content"] for m in ctx7[1:]] == ["m5", "m6", "m7"]

        # Append m8(a) and m9(u): active count is 5 <= 6, so cursor stays fixed.
        for idx, role in ((8, "assistant"), (9, "user")):
            ids.append(
                await msg_store.append_message(
                    group_id=None,
                    user_id="3001",
                    sender_name="U" if role == "user" else "Spectra",
                    role=role,  # type: ignore[arg-type]
                    content=f"m{idx}",
                    created_at=1000 + idx,
                )
            )
        ctx9 = await session_store.build_context(
            session_key="private:3001",
            group_id=None,
            user_id="3001",
            system_prompt="sys",
            now=2200,
        )
        session_stable = await session_store.get_or_create("private:3001")
        assert session_stable.window_start_id == ids[4]
        assert [m["content"] for m in ctx9[1:]] == ["m5", "m6", "m7", "m8", "m9"]

        # Test a separate session where active[-3:] starts with an assistant row.
        for idx, role in enumerate(
            [
                "user",
                "assistant",
                "user",
                "assistant",
                "assistant",
                "user",
                "assistant",
            ],
            start=1,
        ):
            await msg_store.append_message(
                group_id=None,
                user_id="3002",
                sender_name="U" if role == "user" else "Spectra",
                role=role,  # type: ignore[arg-type]
                content=f"b{idx}",
                created_at=2000 + idx,
            )
        # Last 3 are [b5(assistant), b6(user), b7(assistant)].
        # Rollover must advance past leading assistant b5 and anchor on b6(user)!
        ctx_b = await session_store.build_context(
            session_key="private:3002",
            group_id=None,
            user_id="3002",
            system_prompt="sys",
            now=3000,
        )
        assert [m["role"] for m in ctx_b] == ["system", "user", "assistant"]
        assert [m["content"] for m in ctx_b[1:]] == ["b6", "b7"]
        rec_b = await session_store.get_or_create("private:3002")
        all_b = await msg_store.get_private_messages("3002", limit=20)
        b6_id = next(r.id for r in all_b if r.content == "b6")
        assert rec_b.window_start_id == b6_id
    finally:
        await db.close()


async def test_default_60_to_30_prefix_stable_rollover() -> None:
    db, msg_store, session_store = await _make_stores(SessionConfig(timezone="UTC"))
    try:
        ids: list[int] = []
        # Append 60 alternating messages: odd=user, even=assistant.
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
        assert len(ctx_60) == 61  # system + 60 alternating messages

        # 61st message triggers 60 -> 30 rollover:
        # active[-30:] = [msg-32(assistant), msg-33(user), ..., msg-61(user)].
        # Leading assistant msg-32 is skipped so cursor anchors on msg-33(user)!
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
        expected_anchor = ids[32]  # msg-33 (1-based index 33 -> 0-based 32)
        assert (
            await session_store.get_or_create("private:4001")
        ).window_start_id == expected_anchor
        assert ctx_61[1] == {"role": "user", "content": "msg-33"}
        assert ctx_61[-1] == {"role": "user", "content": "msg-61"}

        # Messages 62..90 stay <= 60 active rows, so window_start_id remains fixed.
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
        # Case A: [u1, a2, u3, a4] with keep=1 -> kept=[a4] is all-assistant.
        # Backtrack to latest user u3, anchor on u3.id, and keep [u3, a4].
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

        # Case B: [a1, a2, a3, a4] has zero user rows.
        # Must advance window_start_id to a4.id + 1 and return [system].
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
        # 5 old messages + 4-message trigger burst (> window_keep_messages=2).
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
        # Trigger batch is [m6, m7, m8, m9] -> min_keep_id = ids[5].
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

        # Even if a burst exceeds window_max_messages, min_keep_id keeps the batch.
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
