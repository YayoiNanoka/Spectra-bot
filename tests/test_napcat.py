"""Unit tests for spectra.napcat."""

from __future__ import annotations

import json
from typing import Any

import pytest

from spectra.config import AccessConfig, AppConfig, NapCatConfig
from spectra.db import Database
from spectra.messages import MessageStore
from spectra.napcat import InboundMessage, NapCatAdapter


class _FakeCaptioner:
    def __init__(self, mapping: dict[str, str | None]) -> None:
        self._mapping = mapping
        self.calls: list[str] = []

    async def caption(self, source: str) -> str | None:
        self.calls.append(source)
        return self._mapping.get(source)


class _ActionRecorder:
    def __init__(self, responses: dict[str, Any] | None = None) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._responses = responses or {}

    async def __call__(self, action: str, **params: Any) -> Any:
        self.calls.append((action, params))
        handler = self._responses.get(action)
        if callable(handler):
            return handler(params)
        if isinstance(handler, Exception):
            raise handler
        return handler


@pytest.fixture
async def store() -> MessageStore:
    db = Database(":memory:")
    await db.connect()
    msg_store = MessageStore(db)
    await msg_store.init()
    try:
        yield msg_store
    finally:
        await db.close()


def _make_adapter(
    store: MessageStore,
    *,
    config: AppConfig | None = None,
    captioner: Any = None,
    on_message: Any = None,
    action_responses: dict[str, Any] | None = None,
) -> tuple[NapCatAdapter, _ActionRecorder, list[float]]:
    sleeps: list[float] = []

    async def fake_sleep(sec: float) -> None:
        sleeps.append(sec)

    adapter = NapCatAdapter(
        config=config or AppConfig(),
        message_store=store,
        captioner=captioner,
        on_message=on_message,
        sleep_fn=fake_sleep,
    )
    recorder = _ActionRecorder(action_responses)
    adapter.bot.call_action = recorder  # type: ignore[method-assign]
    return adapter, recorder, sleeps


async def test_inbound_at_bot_and_callback_and_persistence(store: MessageStore) -> None:
    received: list[InboundMessage] = []

    async def callback(msg: InboundMessage) -> None:
        received.append(msg)

    adapter, _, _ = _make_adapter(store, on_message=callback)
    event = {
        "self_id": 99999,
        "group_id": 123456,
        "user_id": 10001,
        "message_id": 555,
        "time": 1700000001,
        "sender": {"card": "AliceCard", "nickname": "AliceNick"},
        "message": [
            {"type": "at", "data": {"qq": "99999"}},
            {"type": "text", "data": {"text": " 你好呀"}},
        ],
    }

    inbound = await adapter.handle_event(event)
    assert inbound is not None
    assert inbound.is_mentioned_bot is True
    assert inbound.starts_with_at_other is False
    assert inbound.sender_name == "AliceCard"
    assert inbound.content == "@Bot 你好呀"
    assert len(received) == 1
    assert received[0] == inbound

    rows = await store.get_group_messages("123456")
    assert len(rows) == 1
    assert rows[0].id == inbound.db_id
    assert rows[0].role == "user"
    assert rows[0].content == "@Bot 你好呀"


async def test_inbound_at_all_does_not_wake_and_at_other_sets_flag(
    store: MessageStore,
) -> None:
    member_calls = 0

    def member_handler(params: dict[str, Any]) -> dict[str, Any]:
        nonlocal member_calls
        member_calls += 1
        return {"card": "BobCard", "nickname": "BobNick"}

    adapter, _, _ = _make_adapter(
        store,
        action_responses={"get_group_member_info": member_handler},
    )

    at_all_event = {
        "self_id": 99999,
        "group_id": 123456,
        "user_id": 10001,
        "message_id": 556,
        "time": 1700000002,
        "sender": {"nickname": "Alice"},
        "message": [
            {"type": "at", "data": {"qq": "all"}},
            {"type": "text", "data": {"text": " 通知大家"}},
        ],
    }
    res_all = await adapter.handle_event(at_all_event)
    assert res_all is not None
    assert res_all.is_mentioned_bot is False
    assert res_all.starts_with_at_other is False
    assert res_all.content == "@全体成员 通知大家"

    at_other_event = {
        "self_id": 99999,
        "group_id": 123456,
        "user_id": 10001,
        "message_id": 557,
        "time": 1700000003,
        "sender": {"nickname": "Alice"},
        "message": [
            {"type": "text", "data": {"text": "   "}},
            {"type": "at", "data": {"qq": "20002"}},
            {"type": "text", "data": {"text": " 吃了吗"}},
        ],
    }
    res_other = await adapter.handle_event(at_other_event)
    assert res_other is not None
    assert res_other.is_mentioned_bot is False
    assert res_other.starts_with_at_other is True
    assert res_other.content == "@BobCard(20002) 吃了吗"
    assert member_calls == 1

    # Second @20002 hits in-memory member cache without another API call.
    res_other_2 = await adapter.handle_event(at_other_event)
    assert res_other_2 is not None
    assert res_other_2.content == "@BobCard(20002) 吃了吗"
    assert member_calls == 1


async def test_inbound_reply_one_level_and_quoting_bot_sets_mentioned(
    store: MessageStore,
) -> None:
    def get_msg_handler(params: dict[str, Any]) -> dict[str, Any]:
        assert params["message_id"] == 777
        return {
            "user_id": 99999,
            "sender": {"user_id": 99999, "nickname": "Spectra"},
            "message": [
                # Nested reply inside quoted msg must NOT trigger recursive get_msg!
                {"type": "reply", "data": {"id": "666"}},
                {"type": "text", "data": {"text": "之前说的那件事"}},
                {"type": "image", "data": {"url": "https://img.example/a.jpg"}},
            ],
        }

    adapter, recorder, _ = _make_adapter(
        store,
        action_responses={"get_msg": get_msg_handler},
    )
    event = {
        "self_id": 99999,
        "group_id": 123456,
        "user_id": 10001,
        "message_id": 778,
        "time": 1700000010,
        "sender": {"nickname": "Alice"},
        "message": [
            {"type": "reply", "data": {"id": "777"}},
            {"type": "text", "data": {"text": "好的收到"}},
        ],
    }

    inbound = await adapter.handle_event(event)
    assert inbound is not None
    assert inbound.is_mentioned_bot is True
    assert inbound.starts_with_at_other is False
    assert "[回复 Spectra: [回复] 之前说的那件事 [图片]]" in inbound.content
    assert inbound.content.endswith("好的收到")
    # Only ONE get_msg call (no recursion into id=666).
    assert [action for action, _ in recorder.calls] == ["get_msg"]


async def test_inbound_json_card_forward_image_caption_and_media_segments(
    store: MessageStore,
) -> None:
    captioner = _FakeCaptioner(
        {
            "https://cdn.example/cat.png": "一只橘猫",
            "https://cdn.example/fail.png": None,
        }
    )

    def get_forward_handler(params: dict[str, Any]) -> dict[str, Any]:
        assert params["id"] == "fwd-1"
        return {
            "messages": [
                {
                    "sender": {"nickname": "Alice"},
                    "content": [{"type": "text", "data": {"text": "第一条"}}],
                },
                {
                    "sender": {"nickname": "Bob"},
                    "content": [
                        {"type": "text", "data": {"text": "第二条"}},
                        {"type": "forward", "data": {"id": "nested-fwd"}},
                    ],
                },
            ]
        }

    adapter, recorder, _ = _make_adapter(
        store,
        captioner=captioner,
        action_responses={"get_forward_msg": get_forward_handler},
    )

    nested_card = json.dumps(
        {
            "meta": {
                "detail_1": {
                    "title": "GitHub",
                    "desc": "Spectra-bot 仓库",
                    "qqdocurl": "https://github.com/example/spectra-bot",
                }
            }
        }
    )
    event = {
        "self_id": 99999,
        "group_id": 123456,
        "user_id": 10001,
        "message_id": 800,
        "time": 1700000020,
        "sender": {"nickname": "Alice"},
        "message": [
            {"type": "image", "data": {"url": "https://cdn.example/cat.png"}},
            {"type": "image", "data": {"url": "https://cdn.example/fail.png"}},
            {"type": "json", "data": {"data": nested_card}},
            {"type": "forward", "data": {"id": "fwd-1"}},
            {"type": "face", "data": {"id": "14"}},
            {"type": "record", "data": {"file": "a.amr"}},
            {"type": "video", "data": {"file": "b.mp4"}},
            {"type": "file", "data": {"name": "report.pdf"}},
        ],
    }

    inbound = await adapter.handle_event(event)
    assert inbound is not None
    assert "[图片: 一只橘猫]" in inbound.content
    assert " [图片] " in inbound.content
    assert (
        "[分享卡片: 标题: GitHub; 简介: Spectra-bot 仓库; 链接: https://github.com/example/spectra-bot]"
        in inbound.content
    )
    assert "[合并转发: Alice: 第一条 | Bob: 第二条 [合并转发]]" in inbound.content
    assert "[表情]" in inbound.content
    assert "[语音]" in inbound.content
    assert "[视频]" in inbound.content
    assert "[文件: report.pdf]" in inbound.content
    # Only 1-level get_forward_msg call (no recursion into nested-fwd).
    assert [action for action, _ in recorder.calls] == ["get_forward_msg"]


async def test_inbound_access_filtering_and_callback_exception_isolation(
    store: MessageStore,
) -> None:
    async def broken_callback(_msg: InboundMessage) -> None:
        raise RuntimeError("boom")

    cfg = AppConfig(
        access=AccessConfig(
            group_mode="blacklist",
            group_blacklist=["999000"],
        )
    )
    adapter, _, _ = _make_adapter(store, config=cfg, on_message=broken_callback)

    # Blocked group is ignored and not persisted.
    blocked = await adapter.handle_event(
        {
            "self_id": 99999,
            "group_id": 999000,
            "user_id": 10001,
            "message_id": 1,
            "message": [{"type": "text", "data": {"text": "blocked"}}],
        }
    )
    assert blocked is None
    assert await store.get_group_messages("999000") == []

    # Default ignored user (QQ Guanjia) is ignored.
    guanjia = await adapter.handle_event(
        {
            "self_id": 99999,
            "group_id": 111,
            "user_id": 2854196310,
            "message_id": 2,
            "message": [{"type": "text", "data": {"text": "bot spam"}}],
        }
    )
    assert guanjia is None

    # Allowed message survives callback exception and is persisted.
    ok = await adapter.handle_event(
        {
            "self_id": 99999,
            "group_id": 111,
            "user_id": 10001,
            "message_id": 3,
            "message": [{"type": "text", "data": {"text": "hello"}}],
        }
    )
    assert ok is not None
    assert len(await store.get_group_messages("111")) == 1


async def test_outbound_segmented_reply_only_quotes_first_and_persists_one_row(
    store: MessageStore,
) -> None:
    adapter, recorder, sleeps = _make_adapter(store)

    text = "第一句话。第二句话！第三句话？"
    db_id = await adapter.send_reply(
        text,
        group_id="123456",
        user_id="10001",
        reply_to_msg_id="888",
        self_id="99999",
    )
    assert db_id is not None

    # Sent in 3 segments with 2 inter-segment sleeps.
    assert len(recorder.calls) == 3
    assert len(sleeps) == 2
    assert all(action == "send_group_msg" for action, _ in recorder.calls)

    seg0 = recorder.calls[0][1]["message"]
    seg1 = recorder.calls[1][1]["message"]
    seg2 = recorder.calls[2][1]["message"]

    assert seg0 == [
        {"type": "reply", "data": {"id": "888"}},
        {"type": "text", "data": {"text": "第一句话。"}},
    ]
    assert seg1 == [{"type": "text", "data": {"text": "第二句话！"}}]
    assert seg2 == [{"type": "text", "data": {"text": "第三句话？"}}]

    # Only ONE merged assistant row persisted in SQLite!
    rows = await store.get_group_messages("123456")
    assert len(rows) == 1
    assert rows[0].id == db_id
    assert rows[0].role == "assistant"
    assert rows[0].user_id == "99999"
    assert rows[0].sender_name == "Spectra"
    assert rows[0].content == text


async def test_outbound_unsplit_over_threshold_and_codeblock(
    store: MessageStore,
) -> None:
    adapter, recorder, sleeps = _make_adapter(store)

    # > 150 chars stays unsplit.
    long_text = "这是一段超过阈值长度的句子。" * 20
    assert 150 < len(long_text) <= 1500
    await adapter.send_reply(
        long_text,
        group_id="123456",
        user_id="10001",
        reply_to_msg_id="900",
        self_id="99999",
    )
    assert len(recorder.calls) == 1
    assert recorder.calls[0][0] == "send_group_msg"
    assert sleeps == []

    # Codeblock <= 150 chars stays unsplit.
    recorder.calls.clear()
    code_text = "看代码。\n```python\nprint('hi')\n```"
    await adapter.send_reply(
        code_text,
        group_id="123456",
        user_id="10001",
        self_id="99999",
    )
    assert len(recorder.calls) == 1
    assert recorder.calls[0][0] == "send_group_msg"
    assert sleeps == []


async def test_outbound_forward_threshold_and_fallback(store: MessageStore) -> None:
    adapter, recorder, _ = _make_adapter(store)
    huge_text = "超长文本内容。" * 300
    assert len(huge_text) > 1500

    db_id = await adapter.send_reply(
        huge_text,
        group_id="123456",
        user_id="10001",
        self_id="99999",
    )
    assert db_id is not None
    assert len(recorder.calls) == 1
    action, params = recorder.calls[0]
    assert action == "send_group_forward_msg"
    assert params["group_id"] == 123456
    assert params["messages"][0]["data"]["uin"] == "99999"
    assert params["messages"][0]["data"]["name"] == "Spectra"

    # Forward API failure falls back to normal single message.
    adapter_fb, recorder_fb, _ = _make_adapter(
        store,
        action_responses={"send_group_forward_msg": RuntimeError("forward failed")},
    )
    await adapter_fb.send_reply(
        huge_text,
        group_id="123456",
        user_id="10001",
        reply_to_msg_id="901",
        self_id="99999",
    )
    assert [a for a, _ in recorder_fb.calls] == [
        "send_group_forward_msg",
        "send_group_msg",
    ]


async def test_outbound_private_reply_persists_peer_user_id(
    store: MessageStore,
) -> None:
    cfg = AppConfig(napcat=NapCatConfig(enable_segmented_reply=False))
    adapter, recorder, _ = _make_adapter(store, config=cfg)

    # First receive private user message.
    inbound = await adapter.handle_event(
        {
            "self_id": 99999,
            "user_id": 424242,
            "message_id": 100,
            "time": 1700000100,
            "sender": {"nickname": "PeerUser"},
            "message": [{"type": "text", "data": {"text": "你好私聊"}}],
        }
    )
    assert inbound is not None

    # Then reply in private chat.
    reply_id = await adapter.send_reply(
        "你好呀，有什么可以帮你？",
        group_id=None,
        user_id="424242",
        reply_to_msg_id="100",
        self_id="99999",
    )
    assert reply_id is not None
    assert recorder.calls[0][0] == "send_private_msg"
    assert recorder.calls[0][1]["user_id"] == 424242

    # get_private_messages(peer_user_id) returns both user and assistant messages!
    history = await store.get_private_messages("424242", limit=10)
    assert [(m.role, m.user_id, m.sender_name, m.content) for m in history] == [
        ("user", "424242", "PeerUser", "你好私聊"),
        ("assistant", "424242", "Spectra", "你好呀，有什么可以帮你？"),
    ]


async def test_outbound_empty_text_returns_none(store: MessageStore) -> None:
    adapter, recorder, _ = _make_adapter(store)
    assert await adapter.send_reply("   \n ", group_id="1", user_id="2") is None
    assert recorder.calls == []


async def test_member_name_fallback_and_shutdown_closes_ws_clients(
    store: MessageStore,
) -> None:
    adapter, _, _ = _make_adapter(
        store,
        action_responses={"get_group_member_info": RuntimeError("api down")},
    )
    inbound = await adapter.handle_event(
        {
            "self_id": 99999,
            "group_id": 123456,
            "user_id": 10001,
            "message_id": 999,
            "sender": {"nickname": "Alice"},
            "message": [
                {"type": "at", "data": {"qq": "30003"}},
                {"type": "text", "data": {"text": " ping"}},
            ],
        }
    )
    assert inbound is not None
    assert inbound.starts_with_at_other is True
    assert inbound.content == "@30003(30003) ping"

    closed: list[tuple[int, str]] = []

    class _FakeWS:
        async def close(self, code: int = 1000, reason: str = "") -> None:
            closed.append((code, reason))

    ws_api = _FakeWS()
    ws_event = _FakeWS()
    adapter.bot._wsr_api_clients = {99999: ws_api}  # type: ignore[attr-defined]
    adapter.bot._wsr_event_clients = {ws_event}  # type: ignore[attr-defined]

    await adapter.shutdown()
    assert len(closed) == 2
    assert adapter.bot._wsr_api_clients == {}  # type: ignore[attr-defined]
    assert adapter.bot._wsr_event_clients == set()  # type: ignore[attr-defined]


async def test_private_temp_session_with_group_id_stays_private(
    store: MessageStore,
) -> None:
    adapter, _, _ = _make_adapter(store)
    inbound = await adapter.handle_event(
        {
            "self_id": 99999,
            "message_type": "private",
            "sub_type": "group",
            "group_id": 777888,
            "user_id": 555666,
            "message_id": 1001,
            "time": 1700000200,
            "sender": {"nickname": "TempPeer"},
            "message": [
                {"type": "text", "data": {"text": "私聊临时会话"}},
                {"type": "file", "data": {"file_name": "notes.txt"}},
            ],
        }
    )
    assert inbound is not None
    assert inbound.group_id is None
    assert "[文件: notes.txt]" in inbound.content
    assert await store.get_group_messages("777888") == []
    private_rows = await store.get_private_messages("555666")
    assert len(private_rows) == 1
    assert private_rows[0].group_id is None


async def test_disallowed_event_runs_zero_api_or_caption_calls(
    store: MessageStore,
) -> None:
    captioner = _FakeCaptioner({"https://cdn.example/secret.png": "secret"})
    cfg = AppConfig(
        access=AccessConfig(group_mode="blacklist", group_blacklist=["888000"])
    )
    adapter, recorder, _ = _make_adapter(store, config=cfg, captioner=captioner)

    res = await adapter.handle_event(
        {
            "self_id": 99999,
            "message_type": "group",
            "group_id": 888000,
            "user_id": 10001,
            "message_id": 1002,
            "message": [
                {"type": "reply", "data": {"id": "100"}},
                {"type": "at", "data": {"qq": "20002"}},
                {"type": "image", "data": {"url": "https://cdn.example/secret.png"}},
                {"type": "forward", "data": {"id": "fwd-99"}},
            ],
        }
    )
    assert res is None
    assert recorder.calls == []
    assert captioner.calls == []


async def test_leading_reply_and_unrecognized_segments_starts_with_at_other(
    store: MessageStore,
) -> None:
    def get_msg_handler(params: dict[str, Any]) -> dict[str, Any]:
        if params["message_id"] == 10:
            return {
                "user_id": 20002,
                "sender": {"user_id": 20002, "nickname": "Bob"},
                "message": [{"type": "text", "data": {"text": "Bob said"}}],
            }
        return {
            "user_id": 99999,
            "sender": {"user_id": 99999, "nickname": "Spectra"},
            "message": [{"type": "text", "data": {"text": "Bot said"}}],
        }

    adapter, _, _ = _make_adapter(
        store,
        action_responses={
            "get_msg": get_msg_handler,
            "get_group_member_info": {"card": "Bob", "nickname": "Bob"},
        },
    )

    # Leading [reply] + unrecognized segment + @other -> starts_with_at_other=True.
    ev_other = await adapter.handle_event(
        {
            "self_id": 99999,
            "message_type": "group",
            "group_id": 123,
            "user_id": 10001,
            "message_id": 11,
            "sender": {"nickname": "Alice"},
            "message": [
                {"type": "reply", "data": {"id": "10"}},
                {"type": "anonymous", "data": {"ignore": 1}},
                {"type": "at", "data": {"qq": "20002"}},
                {"type": "text", "data": {"text": " 同意"}},
            ],
        }
    )
    assert ev_other is not None
    assert ev_other.starts_with_at_other is True
    assert ev_other.is_mentioned_bot is False

    # Leading [reply to Bot] + @Bot: starts_with_at_other=False, is_mentioned_bot=True.
    ev_bot = await adapter.handle_event(
        {
            "self_id": 99999,
            "message_type": "group",
            "group_id": 123,
            "user_id": 10001,
            "message_id": 12,
            "sender": {"nickname": "Alice"},
            "message": [
                {"type": "reply", "data": {"id": "20"}},
                {"type": "at", "data": {"qq": "99999"}},
                {"type": "text", "data": {"text": " 收到"}},
            ],
        }
    )
    assert ev_bot is not None
    assert ev_bot.starts_with_at_other is False
    assert ev_bot.is_mentioned_bot is True


async def test_reply_non_int_id_and_get_msg_failure_degrade_cleanly(
    store: MessageStore,
) -> None:
    adapter, recorder, _ = _make_adapter(
        store,
        action_responses={"get_msg": RuntimeError("ActionFailed")},
    )
    inbound = await adapter.handle_event(
        {
            "self_id": 99999,
            "message_type": "group",
            "group_id": 123,
            "user_id": 10001,
            "message_id": 13,
            "sender": {"nickname": "Alice"},
            "message": [
                {"type": "reply", "data": {"id": "non-int-id"}},
                {"type": "text", "data": {"text": "还在吗"}},
            ],
        }
    )
    assert inbound is not None
    assert inbound.content == "[回复] 还在吗"
    assert recorder.calls == [
        ("get_msg", {"message_id": "non-int-id", "self_id": 99999})
    ]


async def test_outbound_send_failure_does_not_persist_or_sleep(
    store: MessageStore,
) -> None:
    adapter, _, sleeps = _make_adapter(
        store,
        action_responses={"send_group_msg": RuntimeError("network error")},
    )
    with pytest.raises(RuntimeError, match="network error"):
        await adapter.send_reply(
            "第一句。第二句。",
            group_id="123456",
            user_id="10001",
            self_id="99999",
        )
    assert sleeps == []
    assert await store.get_group_messages("123456") == []


async def test_outbound_segmented_regex_with_capturing_groups(
    store: MessageStore,
) -> None:
    cfg = AppConfig(
        napcat=NapCatConfig(
            segmented_regex=r"([^。！？]+)([。！？]?)",
        )
    )
    adapter, recorder, sleeps = _make_adapter(store, config=cfg)
    db_id = await adapter.send_reply(
        "第一句。第二句！",
        group_id="123456",
        user_id="10001",
        self_id="99999",
    )
    assert db_id is not None
    assert len(recorder.calls) == 2
    assert len(sleeps) == 1
    assert recorder.calls[0][1]["message"] == [
        {"type": "text", "data": {"text": "第一句。"}}
    ]
    assert recorder.calls[1][1]["message"] == [
        {"type": "text", "data": {"text": "第二句！"}}
    ]


async def test_shutdown_signaled_before_run_exits_immediately(
    store: MessageStore,
) -> None:
    adapter, _, _ = _make_adapter(store)
    observed_trigger_set = False

    async def fake_run_task(*, host: str, port: int, shutdown_trigger: Any) -> None:
        nonlocal observed_trigger_set
        await shutdown_trigger()
        observed_trigger_set = True

    adapter.bot.run_task = fake_run_task  # type: ignore[method-assign]
    await adapter.shutdown()
    await adapter.run()
    assert observed_trigger_set is True
