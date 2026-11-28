"""Offline joint end-to-end integration tests wiring db + config + messages +
vision + llm + napcat with real production instances.

Only the two outermost network transport edges are replaced in-memory:
1. OneBot v11 WebSocket RPC edge (``adapter._bot.call_action``).
2. HTTP & OpenAI SDK wire transport edge (``httpx.MockTransport`` injected into
   real ``httpx.AsyncClient`` and ``openai.AsyncOpenAI`` clients).
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from openai import AsyncOpenAI
from PIL import Image
from tenacity import wait_none

from spectra.config import (
    AccessConfig,
    AppConfig,
    DatabaseConfig,
    LLMConfig,
    NapCatConfig,
    SegmentedReplyConfig,
    VisionConfig,
    load_config,
    save_config,
)
from spectra.db import Database
from spectra.llm import AgentResult, LLMClient, TokenUsage, ToolSpec
from spectra.messages import MessageRecord, MessageStore
from spectra.napcat import InboundMessage, NapCatAdapter
from spectra.vision import MODEL_IMAGE_MAX_BYTES, ImageCaptioner


def _make_animated_gif(
    frame_colors: list[tuple[int, int, int]],
    size: tuple[int, int] = (80, 60),
) -> bytes:
    """Create a real multi-frame animated GIF in memory via Pillow."""
    frames = [Image.new("RGB", size, color) for color in frame_colors]
    try:
        buf = io.BytesIO()
        frames[0].save(
            buf,
            format="GIF",
            save_all=True,
            append_images=frames[1:],
            duration=100,
            loop=0,
        )
        return buf.getvalue()
    finally:
        for frame in frames:
            frame.close()


def _make_static_png(
    size: tuple[int, int] = (48, 48),
    color: tuple[int, int, int] = (30, 144, 255),
) -> bytes:
    """Create a real static PNG in memory via Pillow."""
    with Image.new("RGB", size, color) as img:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()


def _chat_completion_json(
    *,
    content: Any,
    model: str,
    finish_reason: str = "stop",
    tool_calls: list[dict[str, Any]] | None = None,
    reasoning_content: str | None = None,
    prompt_tokens: int = 100,
    cached_tokens: int | None = None,
    completion_tokens: int = 20,
) -> dict[str, Any]:
    """Build an OpenAI-compatible ``/v1/chat/completions`` JSON response body."""
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    if reasoning_content is not None:
        message["reasoning_content"] = reasoning_content

    usage: dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    if cached_tokens is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached_tokens}

    return {
        "id": "chatcmpl-e2e",
        "object": "chat.completion",
        "created": 1700000000,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage,
    }


def _make_sdk_client(
    *,
    api_key: str,
    base_url: str,
    handler: Callable[[httpx.Request], httpx.Response],
) -> tuple[AsyncOpenAI, httpx.AsyncClient]:
    """Wire a real ``AsyncOpenAI`` client over an in-memory ``httpx.MockTransport``."""
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    sdk_client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=http_client,
        max_retries=0,
    )
    return sdk_client, http_client


class _OneBotRpcStub:
    """In-memory OneBot v11 ``call_action`` boundary stub and call recorder."""

    def __init__(
        self,
        *,
        messages_by_id: dict[int, dict[str, Any]] | None = None,
        members_by_key: dict[tuple[int, int], dict[str, Any]] | None = None,
    ) -> None:
        self._messages_by_id = messages_by_id or {}
        self._members_by_key = members_by_key or {}
        self._next_msg_id = 90000
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def __call__(self, action: str, **params: Any) -> Any:
        self.calls.append((action, params))
        if action == "get_msg":
            msg_id = int(params["message_id"])
            return self._messages_by_id.get(msg_id)
        if action == "get_group_member_info":
            key = (int(params["group_id"]), int(params["user_id"]))
            return self._members_by_key.get(key)
        if action in {
            "send_group_msg",
            "send_private_msg",
            "send_group_forward_msg",
            "send_private_forward_msg",
        }:
            self._next_msg_id += 1
            return {"message_id": self._next_msg_id}
        raise AssertionError(f"Unexpected OneBot action: {action!r} ({params!r})")


async def test_e2e_group_rich_message_gif_vision_cache_llm_tool_loop_segmented_reply(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scenario 1: Group rich message -> 3x3 GIF montage & cache -> DB -> LLM tool loop
    (high effort + reasoning + 503 retry) -> segmented reply -> DB -> cache hit."""
    db_path = tmp_path / "spectra_e2e.db"
    config_path = tmp_path / "config.json"

    initial_config = AppConfig(
        db=DatabaseConfig(path=db_path),
        access=AccessConfig(group_mode="blacklist", private_mode="blacklist"),
        napcat=NapCatConfig(
            bot_name="Spectra",
            forward_threshold=1500,
            segmented_reply=SegmentedReplyConfig(
                enable=True,
                interval_method="random",
                interval="0.8,1.6",
                words_count_threshold=150,
                split_mode="regex",
            ),
        ),
        vision=VisionConfig(
            enabled=True,
            base_url="https://vision.example.test/v1",
            api_key="sk-vision-e2e",
            model="gpt-4o-mini-vision",
            max_size=1280,
        ),
        llm=LLMConfig(
            base_url="https://llm.example.test/v1",
            api_key="sk-llm-e2e",
            model="gpt-4o-reasoner",
            fallback_model="gpt-4o-fallback",
            default_effort="",
            high_effort="high",
            max_retries=3,
            max_tool_steps=6,
        ),
    )
    save_config(initial_config, config_path)
    config = load_config(config_path)
    assert config == initial_config
    assert config.db.path == db_path

    frame_w, frame_h = 80, 60
    frame_colors = [
        (220, 40, 40),
        (40, 200, 40),
        (40, 60, 220),
        (220, 200, 40),
        (200, 40, 200),
        (40, 200, 220),
    ]
    gif_bytes = _make_animated_gif(frame_colors, size=(frame_w, frame_h))
    gif_sha256 = hashlib.sha256(gif_bytes).hexdigest()
    with Image.open(io.BytesIO(gif_bytes)) as opened_gif:
        assert getattr(opened_gif, "n_frames", 1) == 6
        assert opened_gif.size == (frame_w, frame_h)

    gif_url_primary = "https://img.example.test/dance.gif"
    gif_url_mirror = "https://cdn2.example.test/copy.gif"
    image_download_urls: list[str] = []

    def image_http_handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        image_download_urls.append(url)
        if url in {gif_url_primary, gif_url_mirror}:
            return httpx.Response(
                200,
                content=gif_bytes,
                headers={
                    "content-type": "image/gif",
                    "content-length": str(len(gif_bytes)),
                },
            )
        return httpx.Response(404, text="not found")

    vision_requests: list[dict[str, Any]] = []

    def vision_http_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["authorization"] == f"Bearer {config.vision.api_key}"
        body = json.loads(request.content)
        vision_requests.append(body)
        return httpx.Response(
            200,
            json=_chat_completion_json(
                content="  [一只橘猫随着节拍开心跳舞]\n  ",
                model=body["model"],
            ),
        )

    expected_reasoning = "用户询问发布排期，先调用 lookup_release_schedule 工具查询。"
    expected_final_reply = (
        "已经查到 Spectra 的发布排期啦！"
        "v0.2 计划在本周五晚上八点发布。"
        "届时记得提前备份本地配置哦！"
    )
    llm_requests: list[dict[str, Any]] = []

    def llm_http_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["authorization"] == f"Bearer {config.llm.api_key}"
        body = json.loads(request.content)
        attempt_idx = len(llm_requests)
        llm_requests.append(body)

        if attempt_idx == 0:
            return httpx.Response(
                503,
                json={
                    "error": {
                        "message": "temporary upstream overload",
                        "type": "server_error",
                    }
                },
            )
        if attempt_idx == 1:
            return httpx.Response(
                200,
                json=_chat_completion_json(
                    content=None,
                    model=body["model"],
                    finish_reason="tool_calls",
                    reasoning_content=expected_reasoning,
                    tool_calls=[
                        {
                            "id": "call_rel_001",
                            "type": "function",
                            "function": {
                                "name": "lookup_release_schedule",
                                "arguments": json.dumps({"product": "Spectra"}),
                            },
                        }
                    ],
                    prompt_tokens=180,
                    cached_tokens=120,
                    completion_tokens=35,
                ),
            )
        if attempt_idx == 2:
            return httpx.Response(
                200,
                json=_chat_completion_json(
                    content=expected_final_reply,
                    model=body["model"],
                    finish_reason="stop",
                    prompt_tokens=260,
                    cached_tokens=200,
                    completion_tokens=42,
                ),
            )
        raise AssertionError(f"Unexpected LLM HTTP call #{attempt_idx + 1}")

    image_http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(image_http_handler)
    )
    vision_sdk_client, vision_http_client = _make_sdk_client(
        api_key=config.vision.api_key,
        base_url=config.vision.base_url,
        handler=vision_http_handler,
    )
    llm_sdk_client, llm_http_client = _make_sdk_client(
        api_key=config.llm.api_key,
        base_url=config.llm.base_url,
        handler=llm_http_handler,
    )

    db = Database(config.db.path)
    captioner: ImageCaptioner | None = None
    llm_client: LLMClient | None = None
    try:
        await db.connect()
        wal_row = await db.fetchone("PRAGMA journal_mode")
        assert wal_row is not None
        assert str(wal_row[0]).lower() == "wal"

        message_store = MessageStore(db)
        await message_store.init()

        captioner = ImageCaptioner(
            db=db,
            config=config.vision,
            http_client=image_http_client,
            openai_client=vision_sdk_client,
            retry_wait=wait_none(),
        )
        await captioner.init()

        migration_rows = await db.fetchall(
            "SELECT module, version FROM _migrations ORDER BY module"
        )
        assert [(str(r["module"]), int(r["version"])) for r in migration_rows] == [
            ("image_cache", 1),
            ("messages", 1),
        ]

        llm_client = LLMClient(
            config=config.llm,
            openai_client=llm_sdk_client,
            retry_wait=wait_none(),
        )

        tool_invocations: list[dict[str, Any]] = []

        async def lookup_release_schedule(product: str) -> dict[str, Any]:
            tool_invocations.append({"product": product})
            return {
                "product": product,
                "version": "v0.2",
                "release_time": "周五 20:00",
                "notes": "上线前请备份本地配置",
            }

        release_tool = ToolSpec(
            name="lookup_release_schedule",
            description="查询指定产品的发布排期和版本计划",
            parameters={
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "产品名称",
                    }
                },
                "required": ["product"],
            },
            handler=lookup_release_schedule,
        )

        loaded_histories: list[list[MessageRecord]] = []
        agent_results: list[AgentResult] = []
        callback_errors: list[BaseException] = []
        recorded_sleeps: list[float] = []

        async def record_sleep(seconds: float) -> None:
            recorded_sleeps.append(seconds)

        async def on_message(msg: InboundMessage) -> None:
            try:
                if msg.group_id is None or not msg.is_mentioned_bot:
                    return
                history = await message_store.get_group_messages(msg.group_id, limit=20)
                loaded_histories.append(history)

                chat_messages: list[dict[str, Any]] = [
                    {
                        "role": "system",
                        "content": "你是 Spectra 群聊助手，回答简洁准确。",
                    }
                ]
                for record in history:
                    if record.role == "assistant":
                        chat_messages.append(
                            {"role": "assistant", "content": record.content}
                        )
                    else:
                        chat_messages.append(
                            {
                                "role": "user",
                                "content": (
                                    f"[{record.sender_name}({record.user_id})]: "
                                    f"{record.content}"
                                ),
                            }
                        )

                assert llm_client is not None
                result = await llm_client.run_tool_loop(
                    chat_messages,
                    tools=[release_tool],
                    effort="high",
                )
                agent_results.append(result)
                await adapter.send_reply(
                    result.content,
                    group_id=msg.group_id,
                    user_id=msg.user_id,
                    reply_to_msg_id=msg.message_id,
                    self_id=msg.self_id,
                )
            except BaseException as exc:
                callback_errors.append(exc)
                raise

        adapter = NapCatAdapter(
            config=config,
            message_store=message_store,
            captioner=captioner,
            on_message=on_message,
            sleep_fn=record_sleep,
        )
        rpc_stub = _OneBotRpcStub(
            messages_by_id={
                7001: {
                    "message_id": 7001,
                    "user_id": 20001,
                    "sender": {"card": "AliceCard", "nickname": "AliceNick"},
                    "message": [
                        {"type": "text", "data": {"text": "周五发版前记得核对排期"}}
                    ],
                }
            },
            members_by_key={
                (500001, 20001): {
                    "group_id": 500001,
                    "user_id": 20001,
                    "card": "AliceCard",
                    "nickname": "AliceNick",
                }
            },
        )
        monkeypatch.setattr(adapter._bot, "call_action", rpc_stub)

        # 1. Seed normal group message from User A (does NOT mention bot).
        event_a = {
            "self_id": 900001,
            "message_type": "group",
            "group_id": 500001,
            "user_id": 20001,
            "message_id": 7001,
            "time": 1700000100,
            "sender": {"card": "AliceCard", "nickname": "AliceNick"},
            "message": [
                {"type": "text", "data": {"text": "周五发版前记得核对排期"}},
            ],
        }
        inbound_a = await adapter.handle_event(event_a)
        assert not callback_errors
        assert inbound_a is not None
        assert inbound_a.is_mentioned_bot is False
        assert inbound_a.group_id == "500001"
        assert inbound_a.user_id == "20001"
        assert llm_requests == []
        assert vision_requests == []
        assert rpc_stub.calls == []

        rows_after_a = await message_store.get_group_messages("500001")
        assert len(rows_after_a) == 1
        assert rows_after_a[0].role == "user"
        assert rows_after_a[0].user_id == "20001"
        assert rows_after_a[0].sender_name == "AliceCard"
        assert rows_after_a[0].content == "周五发版前记得核对排期"

        # 2. Rich group message from User B: reply + @Bot + @UserA + GIF + question.
        event_b = {
            "self_id": 900001,
            "message_type": "group",
            "group_id": 500001,
            "user_id": 20002,
            "message_id": 7002,
            "time": 1700000110,
            "sender": {"card": "BobCard", "nickname": "BobNick"},
            "message": [
                {"type": "reply", "data": {"id": "7001"}},
                {"type": "at", "data": {"qq": "900001"}},
                {"type": "at", "data": {"qq": "20001"}},
                {"type": "image", "data": {"url": gif_url_primary}},
                {"type": "text", "data": {"text": "帮我查一下 Spectra 的发布排期"}},
            ],
        }
        inbound_b = await adapter.handle_event(event_b)
        assert not callback_errors
        assert inbound_b is not None
        assert inbound_b.is_mentioned_bot is True
        assert inbound_b.starts_with_at_other is False

        # Verify vision HTTP request carried a valid 3x3 JPEG montage < 512 KiB
        # along with the montage prompt note.
        assert image_download_urls == [gif_url_primary]
        assert len(vision_requests) == 1
        assert vision_requests[0]["model"] == config.vision.model
        vision_content = vision_requests[0]["messages"][0]["content"]
        text_parts = [p["text"] for p in vision_content if p["type"] == "text"]
        image_parts = [
            p["image_url"]["url"] for p in vision_content if p["type"] == "image_url"
        ]
        assert len(text_parts) == 1
        assert config.vision.prompt in text_parts[0]
        assert "3×3" in text_parts[0]
        assert len(image_parts) == 1
        assert image_parts[0].startswith("data:image/jpeg;base64,")

        montage_b64 = image_parts[0].removeprefix("data:image/jpeg;base64,")
        montage_bytes = base64.b64decode(montage_b64)
        assert len(montage_bytes) < MODEL_IMAGE_MAX_BYTES
        with Image.open(io.BytesIO(montage_bytes)) as montage_img:
            assert montage_img.format == "JPEG"
            assert montage_img.size == (3 * frame_w, 3 * frame_h)
            with montage_img.convert("RGB") as rgb:
                cell0 = rgb.getpixel((frame_w // 2, frame_h // 2))
                cell5 = rgb.getpixel(
                    (2 * frame_w + frame_w // 2, frame_h + frame_h // 2)
                )
                cell8_unused = rgb.getpixel(
                    (2 * frame_w + frame_w // 2, 2 * frame_h + frame_h // 2)
                )
            assert cell0[0] > 180 and cell0[1] < 80 and cell0[2] < 80
            assert cell5[0] < 80 and cell5[1] > 160 and cell5[2] > 180
            assert all(ch > 240 for ch in cell8_unused)

        # Verify OneBot RPC resolved both quoted message and @UserA group card.
        get_msg_calls = [
            params for action, params in rpc_stub.calls if action == "get_msg"
        ]
        member_calls = [
            params
            for action, params in rpc_stub.calls
            if action == "get_group_member_info"
        ]
        assert len(get_msg_calls) == 1
        assert get_msg_calls[0]["message_id"] == 7001
        assert len(member_calls) == 1
        assert member_calls[0]["group_id"] == 500001
        assert member_calls[0]["user_id"] == 20001

        # Verify image_cache row stores SHA-256 and normalized caption.
        cache_row = await db.fetchone(
            "SELECT sha256, caption, updated_at FROM image_cache WHERE sha256 = ?",
            (gif_sha256,),
        )
        assert cache_row is not None
        assert str(cache_row["sha256"]) == gif_sha256
        assert str(cache_row["caption"]) == "一只橘猫随着节拍开心跳舞"
        assert int(cache_row["updated_at"]) > 0

        # Verify InboundMessage & persisted messages row for User B.
        expected_b_fragments = (
            "[回复 AliceCard: 周五发版前记得核对排期]",
            "@Bot",
            "@AliceCard(20001)",
            "[图片: 一只橘猫随着节拍开心跳舞]",
            "帮我查一下 Spectra 的发布排期",
        )
        for fragment in expected_b_fragments:
            assert fragment in inbound_b.content

        # Verify group history loaded inside on_message was in id ASC order.
        assert len(loaded_histories) == 1
        assert [r.id for r in loaded_histories[0]] == [inbound_a.db_id, inbound_b.db_id]
        assert loaded_histories[0][0].id < loaded_histories[0][1].id
        assert loaded_histories[0][1].content == inbound_b.content

        # Verify LLM HTTP transport: 503 retry + Step 1 tool_calls + Step 2 reply.
        assert len(llm_requests) == 3
        assert llm_requests[0] == llm_requests[1]
        assert llm_requests[0]["model"] == "gpt-4o-reasoner"
        assert llm_requests[0]["reasoning_effort"] == "high"
        assert llm_requests[0]["tool_choice"] == "auto"
        assert llm_requests[0]["tools"][0]["function"]["name"] == (
            "lookup_release_schedule"
        )
        assert llm_requests[0]["messages"][1]["content"].startswith(
            "[AliceCard(20001)]: "
        )
        assert llm_requests[0]["messages"][2]["content"].startswith(
            "[BobCard(20002)]: "
        )
        assert tool_invocations == [{"product": "Spectra"}]

        # Step 2 request must preserve reasoning_content, tool_calls, tool output,
        # and reasoning_effort="high".
        step2_req = llm_requests[2]
        assert step2_req["reasoning_effort"] == "high"
        step2_msgs = step2_req["messages"]
        assert step2_msgs[-2]["role"] == "assistant"
        assert step2_msgs[-2]["content"] is None
        assert step2_msgs[-2]["reasoning_content"] == expected_reasoning
        assert step2_msgs[-2]["tool_calls"][0]["id"] == "call_rel_001"
        assert step2_msgs[-1]["role"] == "tool"
        assert step2_msgs[-1]["tool_call_id"] == "call_rel_001"
        tool_output = json.loads(step2_msgs[-1]["content"])
        assert tool_output["version"] == "v0.2"
        assert tool_output["release_time"] == "周五 20:00"

        # Verify AgentResult usage accumulation including cached_tokens.
        assert len(agent_results) == 1
        agent_result = agent_results[0]
        assert agent_result.steps == 2
        assert agent_result.content == expected_final_reply
        assert agent_result.usage == TokenUsage(
            input_other=(180 - 120) + (260 - 200),
            input_cached=120 + 200,
            output=35 + 42,
        )

        # Verify segmented reply: 3 send_group_msg calls, reply segment ONLY on #0,
        # 2 sleep delays between segments 1 and 2, and 1 merged assistant row in DB.
        send_group_calls = [
            params for action, params in rpc_stub.calls if action == "send_group_msg"
        ]
        assert len(send_group_calls) == 3
        assert send_group_calls[0]["group_id"] == 500001
        assert send_group_calls[0]["self_id"] == 900001
        assert send_group_calls[0]["message"] == [
            {"type": "reply", "data": {"id": "7002"}},
            {"type": "text", "data": {"text": "已经查到 Spectra 的发布排期啦！"}},
        ]
        assert send_group_calls[1]["message"] == [
            {"type": "text", "data": {"text": "v0.2 计划在本周五晚上八点发布。"}},
        ]
        assert send_group_calls[2]["message"] == [
            {"type": "text", "data": {"text": "届时记得提前备份本地配置哦！"}},
        ]
        for follow_up in send_group_calls[1:]:
            assert all(seg["type"] != "reply" for seg in follow_up["message"])

        assert len(recorded_sleeps) == 2
        assert all(0.8 <= delay <= 1.6 for delay in recorded_sleeps)

        rows_after_b = await message_store.get_group_messages("500001")
        assert len(rows_after_b) == 3
        assert [r.role for r in rows_after_b] == ["user", "user", "assistant"]
        user_b_row = rows_after_b[1]
        assert user_b_row.id == inbound_b.db_id
        assert user_b_row.group_id == "500001"
        assert user_b_row.user_id == "20002"
        assert user_b_row.sender_name == "BobCard"
        assert user_b_row.content == inbound_b.content
        assistant_row = rows_after_b[2]
        assert assistant_row.group_id == "500001"
        assert assistant_row.user_id == "900001"
        assert assistant_row.sender_name == "Spectra"
        assert assistant_row.content == expected_final_reply
        # Verify all 3 segmented reply bubble IDs (90001, 90002, 90003) resolve
        # back to the single persisted merged assistant row.
        assert assistant_row.platform_message_id == "90001"
        for bubble_pid in ("90001", "90002", "90003"):
            assert (
                await message_store.resolve_platform_message_id(
                    platform_message_id=bubble_pid,
                    group_id="500001",
                    user_id="20002",
                )
                == assistant_row.id
            )
        assert db_path.is_file() and db_path.stat().st_size > 0

        # 3. Same GIF bytes from a different URL must hit SQLite image_cache
        #    with zero additional vision /chat/completions calls.
        event_c = {
            "self_id": 900001,
            "message_type": "group",
            "group_id": 500001,
            "user_id": 20001,
            "message_id": 7003,
            "time": 1700000150,
            "sender": {"card": "AliceCard", "nickname": "AliceNick"},
            "message": [
                {"type": "image", "data": {"url": gif_url_mirror}},
                {"type": "text", "data": {"text": "表情包复读一下"}},
            ],
        }
        inbound_c = await adapter.handle_event(event_c)
        assert not callback_errors
        assert inbound_c is not None
        assert "[图片: 一只橘猫随着节拍开心跳舞]" in inbound_c.content
        assert "表情包复读一下" in inbound_c.content
        assert image_download_urls == [gif_url_primary, gif_url_mirror]
        assert len(vision_requests) == 1
        assert len(llm_requests) == 3

        cache_count_row = await db.fetchone("SELECT COUNT(*) AS cnt FROM image_cache")
        assert cache_count_row is not None
        assert int(cache_count_row["cnt"]) == 1
    finally:
        if llm_client is not None:
            await llm_client.close()
        if captioner is not None:
            await captioner.close()
        await llm_sdk_client.close()
        await llm_http_client.aclose()
        await vision_sdk_client.close()
        await vision_http_client.aclose()
        await image_http_client.aclose()
        await db.close()


async def test_e2e_private_multiturn_temp_session_forward_and_blacklist_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scenario 2: Blacklist gate -> multi-turn private DM with fallback ->
    temp-session group_id stripping -> forward_threshold long reply -> DB checks."""
    db_path = tmp_path / "spectra_e2e_private.db"
    config_path = tmp_path / "config.json"

    initial_config = AppConfig(
        db=DatabaseConfig(path=db_path),
        access=AccessConfig(
            group_mode="blacklist",
            group_blacklist=["999000"],
            private_mode="blacklist",
            private_blacklist=["444000"],
        ),
        napcat=NapCatConfig(
            bot_name="Spectra",
            forward_threshold=120,
            segmented_reply=SegmentedReplyConfig(
                enable=True,
                interval_method="random",
                interval="0.5,1.0",
                words_count_threshold=100,
                split_mode="regex",
            ),
        ),
        vision=VisionConfig(
            enabled=True,
            base_url="https://vision.example.test/v1",
            api_key="sk-vision-e2e",
            model="gpt-4o-mini-vision",
        ),
        llm=LLMConfig(
            base_url="https://llm.example.test/v1",
            api_key="sk-llm-e2e",
            model="gpt-4o-primary",
            fallback_model="gpt-4o-fallback",
            default_effort="",
            high_effort="high",
            max_retries=2,
            max_tool_steps=5,
        ),
    )
    save_config(initial_config, config_path)
    config = load_config(config_path)
    assert config == initial_config

    png_bytes = _make_static_png()
    image_download_urls: list[str] = []

    def image_http_handler(request: httpx.Request) -> httpx.Response:
        image_download_urls.append(str(request.url))
        return httpx.Response(
            200,
            content=png_bytes,
            headers={
                "content-type": "image/png",
                "content-length": str(len(png_bytes)),
            },
        )

    vision_requests: list[dict[str, Any]] = []

    def vision_http_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        vision_requests.append(body)
        return httpx.Response(
            200,
            json=_chat_completion_json(content="蓝色方块图", model=body["model"]),
        )

    turn1_reply = "记下啦，项目代号是 Orion，稍后随时来问我。"
    turn2_long_reply = (
        "关于项目 Orion 的离线联调检查清单如下："
        "1. 确认 SQLite WAL 模式已开启且 _migrations 版本记录完整；"
        "2. 确认私聊临时会话的 group_id 已被剥离且不污染任何群聊记录；"
        "3. 确认多轮私聊上下文按时间正序送入模型且出站合并转发后"
        "仍仅持久化一条完整 assistant 记录；"
        "4. 确认黑名单群与黑名单用户均在入站门卫被零副作用拦截。"
    )
    assert len(turn2_long_reply) > config.napcat.forward_threshold

    llm_requests: list[dict[str, Any]] = []

    def llm_http_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["authorization"] == f"Bearer {config.llm.api_key}"
        body = json.loads(request.content)
        call_idx = len(llm_requests)
        llm_requests.append(body)

        # Turn 1: primary model exhausts max_retries=2 with 502, then fallback succeeds.
        if call_idx in {0, 1}:
            return httpx.Response(
                502,
                json={"error": {"message": "bad gateway", "type": "server_error"}},
            )
        if call_idx == 2:
            return httpx.Response(
                200,
                json=_chat_completion_json(
                    content=turn1_reply,
                    model=body["model"],
                    prompt_tokens=60,
                    cached_tokens=0,
                    completion_tokens=18,
                ),
            )
        # Turn 2: primary model returns long reply exceeding forward_threshold.
        if call_idx == 3:
            return httpx.Response(
                200,
                json=_chat_completion_json(
                    content=turn2_long_reply,
                    model=body["model"],
                    prompt_tokens=140,
                    cached_tokens=80,
                    completion_tokens=95,
                ),
            )
        raise AssertionError(f"Unexpected LLM HTTP call #{call_idx + 1}")

    image_http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(image_http_handler)
    )
    vision_sdk_client, vision_http_client = _make_sdk_client(
        api_key=config.vision.api_key,
        base_url=config.vision.base_url,
        handler=vision_http_handler,
    )
    llm_sdk_client, llm_http_client = _make_sdk_client(
        api_key=config.llm.api_key,
        base_url=config.llm.base_url,
        handler=llm_http_handler,
    )

    db = Database(config.db.path)
    captioner: ImageCaptioner | None = None
    llm_client: LLMClient | None = None
    try:
        await db.connect()
        wal_row = await db.fetchone("PRAGMA journal_mode")
        assert wal_row is not None
        assert str(wal_row[0]).lower() == "wal"
        message_store = MessageStore(db)
        await message_store.init()

        captioner = ImageCaptioner(
            db=db,
            config=config.vision,
            http_client=image_http_client,
            openai_client=vision_sdk_client,
            retry_wait=wait_none(),
        )
        await captioner.init()

        llm_client = LLMClient(
            config=config.llm,
            openai_client=llm_sdk_client,
            retry_wait=wait_none(),
        )

        triggered_messages: list[InboundMessage] = []
        callback_errors: list[BaseException] = []
        recorded_sleeps: list[float] = []

        async def record_sleep(seconds: float) -> None:
            recorded_sleeps.append(seconds)

        async def on_message(msg: InboundMessage) -> None:
            try:
                triggered_messages.append(msg)
                if msg.group_id is not None:
                    return
                history = await message_store.get_private_messages(
                    msg.user_id, limit=20
                )
                chat_messages: list[dict[str, Any]] = [
                    {
                        "role": "system",
                        "content": "你是 Spectra 私聊助手，保持上下文连贯。",
                    }
                ]
                for record in history:
                    chat_messages.append(
                        {"role": record.role, "content": record.content}
                    )

                assert llm_client is not None
                result = await llm_client.run_tool_loop(
                    chat_messages,
                    effort="default",
                )
                await adapter.send_reply(
                    result.content,
                    group_id=None,
                    user_id=msg.user_id,
                    reply_to_msg_id=msg.message_id,
                    self_id=msg.self_id,
                )
            except BaseException as exc:
                callback_errors.append(exc)
                raise

        adapter = NapCatAdapter(
            config=config,
            message_store=message_store,
            captioner=captioner,
            on_message=on_message,
            sleep_fn=record_sleep,
        )
        rpc_stub = _OneBotRpcStub(
            messages_by_id={
                8001: {
                    "message_id": 8001,
                    "user_id": 333001,
                    "sender": {"nickname": "Charlie"},
                    "message": [{"type": "text", "data": {"text": "should not fetch"}}],
                },
                8002: {
                    "message_id": 8002,
                    "user_id": 444000,
                    "sender": {"nickname": "BlockedUser"},
                    "message": [{"type": "text", "data": {"text": "should not fetch"}}],
                },
            },
            members_by_key={
                (999000, 20001): {
                    "group_id": 999000,
                    "user_id": 20001,
                    "card": "ShouldNotResolve",
                }
            },
        )
        monkeypatch.setattr(adapter._bot, "call_action", rpc_stub)

        # 1. Blacklist gate check: blocked group ("999000") & blocked user ("444000").
        blocked_group_event = {
            "self_id": 900001,
            "message_type": "group",
            "group_id": 999000,
            "user_id": 333001,
            "message_id": 8010,
            "time": 1700000180,
            "sender": {"card": "Charlie"},
            "message": [
                {"type": "reply", "data": {"id": "8001"}},
                {"type": "at", "data": {"qq": "900001"}},
                {"type": "at", "data": {"qq": "20001"}},
                {
                    "type": "image",
                    "data": {"url": "https://img.example.test/blocked-group.png"},
                },
                {"type": "text", "data": {"text": "黑名单群消息"}},
            ],
        }
        blocked_private_event = {
            "self_id": 900001,
            "message_type": "private",
            "sub_type": "friend",
            "user_id": 444000,
            "message_id": 8011,
            "time": 1700000190,
            "sender": {"nickname": "BlockedUser"},
            "message": [
                {"type": "reply", "data": {"id": "8002"}},
                {"type": "at", "data": {"qq": "900001"}},
                {
                    "type": "image",
                    "data": {"url": "https://img.example.test/blocked-private.png"},
                },
                {"type": "text", "data": {"text": "黑名单私聊消息"}},
            ],
        }

        assert await adapter.handle_event(blocked_group_event) is None
        assert await adapter.handle_event(blocked_private_event) is None
        assert not callback_errors
        assert triggered_messages == []
        assert rpc_stub.calls == []
        assert image_download_urls == []
        assert vision_requests == []
        assert llm_requests == []

        msg_count_row = await db.fetchone("SELECT COUNT(*) AS cnt FROM messages")
        cache_count_row = await db.fetchone("SELECT COUNT(*) AS cnt FROM image_cache")
        assert msg_count_row is not None and int(msg_count_row["cnt"]) == 0
        assert cache_count_row is not None and int(cache_count_row["cnt"]) == 0

        # 2. Private DM Turn 1 (normal friend message).
        turn1_user_text = "你好，帮我记住项目代号是 Orion"
        event_dm1 = {
            "self_id": 900001,
            "message_type": "private",
            "sub_type": "friend",
            "user_id": 333001,
            "message_id": 8101,
            "time": 1700000200,
            "sender": {"nickname": "Charlie"},
            "message": [{"type": "text", "data": {"text": turn1_user_text}}],
        }
        inbound_1 = await adapter.handle_event(event_dm1)
        assert not callback_errors
        assert inbound_1 is not None
        assert inbound_1.group_id is None
        assert inbound_1.user_id == "333001"
        assert len(triggered_messages) == 1

        # Primary failed twice (502), then fallback_model succeeded.
        assert [req["model"] for req in llm_requests] == [
            "gpt-4o-primary",
            "gpt-4o-primary",
            "gpt-4o-fallback",
        ]
        assert "reasoning_effort" not in llm_requests[0]
        assert llm_requests[2]["messages"] == [
            {"role": "system", "content": "你是 Spectra 私聊助手，保持上下文连贯。"},
            {"role": "user", "content": turn1_user_text},
        ]

        send_private_calls = [
            params for action, params in rpc_stub.calls if action == "send_private_msg"
        ]
        assert len(send_private_calls) == 1
        assert send_private_calls[0]["user_id"] == 333001
        assert send_private_calls[0]["self_id"] == 900001
        assert send_private_calls[0]["message"] == [
            {"type": "reply", "data": {"id": "8101"}},
            {"type": "text", "data": {"text": turn1_reply}},
        ]

        # 3. Private DM Turn 2: temp-session private message carrying group_id=888777!
        turn2_user_text = "请结合刚才记下的代号，给我一份详细的联调检查清单"
        event_dm2 = {
            "self_id": 900001,
            "message_type": "private",
            "sub_type": "group",
            "group_id": 888777,
            "user_id": 333001,
            "message_id": 8102,
            "time": 1700000230,
            "sender": {"nickname": "Charlie"},
            "message": [{"type": "text", "data": {"text": turn2_user_text}}],
        }
        inbound_2 = await adapter.handle_event(event_dm2)
        assert not callback_errors
        assert inbound_2 is not None
        assert inbound_2.group_id is None
        assert inbound_2.user_id == "333001"
        assert len(triggered_messages) == 2

        # Turn 2 OpenAI payload must contain Turn 1 user, Turn 1 assistant, Turn 2 user
        # in chronological order.
        assert len(llm_requests) == 4
        assert llm_requests[3]["model"] == "gpt-4o-primary"
        assert llm_requests[3]["messages"] == [
            {"role": "system", "content": "你是 Spectra 私聊助手，保持上下文连贯。"},
            {"role": "user", "content": turn1_user_text},
            {"role": "assistant", "content": turn1_reply},
            {"role": "user", "content": turn2_user_text},
        ]

        # Turn 2 long reply exceeds forward_threshold -> send_private_forward_msg.
        forward_calls = [
            params
            for action, params in rpc_stub.calls
            if action == "send_private_forward_msg"
        ]
        assert len(forward_calls) == 1
        assert forward_calls[0]["user_id"] == 333001
        assert forward_calls[0]["self_id"] == 900001
        assert forward_calls[0]["messages"] == [
            {
                "type": "node",
                "data": {
                    "name": "Spectra",
                    "uin": "900001",
                    "content": [{"type": "text", "data": {"text": turn2_long_reply}}],
                },
            }
        ]
        assert all(
            action not in {"send_group_msg", "send_group_forward_msg"}
            for action, _ in rpc_stub.calls
        )
        assert recorded_sleeps == []

        # Verify all 4 persisted private rows have group_id=None and user_id="333001",
        # and temp-session group_id "888777" has zero persisted rows.
        private_rows = await message_store.get_private_messages("333001", limit=20)
        assert len(private_rows) == 4
        assert [r.role for r in private_rows] == [
            "user",
            "assistant",
            "user",
            "assistant",
        ]
        assert all(r.group_id is None for r in private_rows)
        assert all(r.user_id == "333001" for r in private_rows)
        assert [r.content for r in private_rows] == [
            turn1_user_text,
            turn1_reply,
            turn2_user_text,
            turn2_long_reply,
        ]
        assert private_rows[1].sender_name == "Spectra"
        assert private_rows[3].sender_name == "Spectra"

        assert await message_store.get_group_messages("888777") == []
        total_rows = await db.fetchone("SELECT COUNT(*) AS cnt FROM messages")
        assert total_rows is not None and int(total_rows["cnt"]) == 4
    finally:
        if llm_client is not None:
            await llm_client.close()
        if captioner is not None:
            await captioner.close()
        await llm_sdk_client.close()
        await llm_http_client.aclose()
        await vision_sdk_client.close()
        await vision_http_client.aclose()
        await image_http_client.aclose()
        await db.close()


async def test_e2e_multi_episode_consolidation_jargon_and_recall_lifecycle(
    tmp_path: Path,
) -> None:
    """Scenario 3: Full 3-episode SpectraBot lifecycle:
    Episode 1 chat -> hard timeout cut -> Episode 2 starts & triggers background
    consolidation -> consolidator.flush() extracts profile + embedded memory + jargon ->
    Episode 3 freezes updated profile snapshot, injects matched jargon footnote,
    recalls Episode 1 atomic memory via recall_memory (vector + Jev rerank), and closes
    with zero ResourceWarnings.
    """
    import asyncio

    from spectra.bot import SpectraBot
    from spectra.config import (
        EmbeddingConfig,
        GateConfig,
        JevConfig,
        MemoryConfig,
        SessionConfig,
        TraceConfig,
    )
    from spectra.embedding import EmbeddingClient
    from spectra.jev import JevClient

    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "default.md").write_text("你是 Spectra 群聊伙伴。", encoding="utf-8")

    cfg = AppConfig(
        db=DatabaseConfig(path=tmp_path / "e2e_memory.db"),
        napcat=NapCatConfig(
            bot_name="Spectra",
            segmented_reply=SegmentedReplyConfig(enable=False),
        ),
        vision=VisionConfig(enabled=False),
        llm=LLMConfig(
            base_url="https://llm.example.test/v1",
            api_key="sk-llm-e2e",
            model="gpt-4o",
            max_retries=1,
        ),
        embedding=EmbeddingConfig(
            base_url="https://embed.example.test/v1",
            api_key="sk-emb-e2e",
            model="text-embedding-v4",
            max_retries=0,
        ),
        jev=JevConfig(
            enabled=True,
            base_url="https://jev.example.test/v1/systemone",
            api_key="sk-jev-e2e",
            model="jev-1.14",
        ),
        gate=GateConfig(
            proactive_mode="on",
            private_debounce_sec=0.0,
            group_debounce_sec=0.0,
            max_batch_wait_sec=0.0,
        ),
        session=SessionConfig(
            prompts_dir=prompts_dir,
            group_idle_timeout_minutes=30,
            group_hard_timeout_hours=2,
            timezone="Asia/Shanghai",
        ),
        memory=MemoryConfig(
            enabled=True,
            consolidation_enabled=True,
            min_vector_similarity=0.5,
            rerank_threshold=0.5,
        ),
        trace=TraceConfig(enabled=True, path=tmp_path / "traces.jsonl"),
    )

    def embed_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        inputs = body["input"]
        if isinstance(inputs, str):
            inputs = [inputs]
        data = []
        for idx, text in enumerate(inputs):
            vec = (
                [1.0, 0.0, 0.0]
                if ("抹茶" in text or "饮料" in text)
                else [0.0, 1.0, 0.0]
            )
            data.append({"index": idx, "embedding": vec})
        return httpx.Response(200, json={"data": data})

    jev_calls: list[dict[str, Any]] = []

    def jev_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        jev_calls.append(body)
        questions = body["questions"]
        if "m0" in questions:
            return httpx.Response(
                200,
                json={
                    "model": "jev-1.14",
                    "usage": {"input_tokens": 30},
                    "answers": {k: {"noul": 0.92} for k in questions},
                },
            )
        return httpx.Response(
            200,
            json={
                "model": "jev-1.14",
                "usage": {"input_tokens": 25},
                "answers": {
                    "effort": {
                        "choice": "default",
                        "probabilities": {"default": 0.9, "high": 0.1},
                    }
                },
            },
        )

    chat_requests: list[dict[str, Any]] = []
    consolidation_requests: list[dict[str, Any]] = []

    def llm_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        sys_text = body["messages"][0]["content"]
        if "后台长期记忆整理器" in sys_text:
            consolidation_requests.append(body)
            if "抹茶拿铁" in body["messages"][1]["content"]:
                return httpx.Response(
                    200,
                    json=_chat_completion_json(
                        model="gpt-4o",
                        content=json.dumps(
                            {
                                "group_profile_patch": {
                                    "interaction_style": "轻松吐槽、爱聊奶茶咖啡",
                                    "frequent_topics": "饮品、加班吐槽",
                                },
                                "user_profile_patches": [
                                    {
                                        "user_id": "10001",
                                        "interaction_style": "爱分享饮品",
                                    }
                                ],
                                "memory_ops": [
                                    {
                                        "action": "add",
                                        "kind": "preference",
                                        "content": "小明最爱喝冰抹茶拿铁",
                                        "owner_user_id": "10001",
                                        "visibility": "public",
                                        "confidence": 0.9,
                                        "event_time": None,
                                    }
                                ],
                                "jargon_ops": [
                                    {
                                        "term": "摸鱼茶话会",
                                        "meaning": (
                                            "下午三点群里一起点奶茶聊天的固定节目"
                                        ),
                                        "confidence": 0.88,
                                    }
                                ],
                            },
                            ensure_ascii=False,
                        ),
                    ),
                )
            return httpx.Response(
                200,
                json=_chat_completion_json(
                    model="gpt-4o",
                    content=json.dumps(
                        {
                            "group_profile_patch": None,
                            "user_profile_patches": [],
                            "memory_ops": [],
                            "jargon_ops": [],
                        }
                    ),
                ),
            )

        chat_requests.append(body)
        turn = len(chat_requests)
        if turn == 1:
            return httpx.Response(
                200,
                json=_chat_completion_json(
                    model="gpt-4o", content="冰抹茶拿铁确实好喝！"
                ),
            )
        if turn == 2:
            return httpx.Response(
                200,
                json=_chat_completion_json(
                    model="gpt-4o", content="新话题来啦，聊聊看！"
                ),
            )
        if turn == 3:
            return httpx.Response(
                200,
                json=_chat_completion_json(
                    model="gpt-4o",
                    content="",
                    tool_calls=[
                        {
                            "id": "call_recall_ep3",
                            "type": "function",
                            "function": {
                                "name": "recall_memory",
                                "arguments": json.dumps({"query": "小明饮料抹茶"}),
                            },
                        }
                    ],
                ),
            )
        return httpx.Response(
            200,
            json=_chat_completion_json(
                model="gpt-4o",
                content="下午摸鱼茶话会给你点冰抹茶拿铁准没错！",
            ),
        )

    embed_http = httpx.AsyncClient(transport=httpx.MockTransport(embed_handler))
    embed_client = EmbeddingClient(cfg.embedding, http_client=embed_http)
    jev_http = httpx.AsyncClient(transport=httpx.MockTransport(jev_handler))
    jev_client = JevClient(cfg.jev, http_client=jev_http)
    llm_sdk, llm_http = _make_sdk_client(
        api_key=cfg.llm.api_key,
        base_url=cfg.llm.base_url,
        handler=llm_handler,
    )
    llm_client = LLMClient(cfg.llm, openai_client=llm_sdk, retry_wait=wait_none())

    async def zero_sleep(_: float) -> None:
        await asyncio.sleep(0)

    rpc_stub = _OneBotRpcStub()
    trace_count = 0
    trace_changed = asyncio.Event()

    bot = SpectraBot(
        cfg,
        jev_client=jev_client,
        llm_client=llm_client,
        embedding_client=embed_client,
        sleep_fn=zero_sleep,
    )
    try:
        async with bot:
            assert bot.adapter is not None
            assert bot.trace_logger is not None
            assert bot.consolidator is not None
            assert bot.message_store is not None
            assert bot.session_store is not None
            bot.adapter.bot.call_action = rpc_stub  # type: ignore[method-assign]

            real_write = bot.trace_logger.write

            async def wrapped_write(entry: Any) -> None:
                nonlocal trace_count
                await real_write(entry)
                trace_count += 1
                trace_changed.set()

            bot.trace_logger.write = wrapped_write  # type: ignore[method-assign]

            async def wait_traces(expected: int) -> None:
                async with asyncio.timeout(3.0):
                    while trace_count < expected:
                        trace_changed.clear()
                        await trace_changed.wait()

            # Episode 1: Chat about matcha latte and group tea party.
            t0 = 1715000000
            await bot.adapter.handle_event(
                {
                    "post_type": "message",
                    "message_type": "group",
                    "self_id": 99999,
                    "group_id": 80001,
                    "user_id": 10001,
                    "message_id": 5001,
                    "time": t0,
                    "sender": {"card": "小明"},
                    "message": [
                        {"type": "at", "data": {"qq": "99999"}},
                        {
                            "type": "text",
                            "data": {"text": " 我最爱喝冰抹茶拿铁，下午搞摸鱼茶话会吧"},
                        },
                    ],
                }
            )
            await wait_traces(1)

            # Episode 2 (3h later > 2h hard timeout): closes Ep 1 & consolidates.
            t1 = t0 + 3 * 3600
            await bot.adapter.handle_event(
                {
                    "post_type": "message",
                    "message_type": "group",
                    "self_id": 99999,
                    "group_id": 80001,
                    "user_id": 10001,
                    "message_id": 5002,
                    "time": t1,
                    "sender": {"card": "小明"},
                    "message": [
                        {"type": "at", "data": {"qq": "99999"}},
                        {"type": "text", "data": {"text": " 换个新话题聊聊"}},
                    ],
                }
            )
            await wait_traces(2)

            # Deterministically wait for Episode 1 background consolidation to complete.
            await bot.consolidator.flush()
            assert len(consolidation_requests) == 1
            sess_after_ep1 = await bot.session_store.get_or_create("group:80001")
            assert sess_after_ep1.last_extracted_id > 0

            # Episode 3 (another 3h later): freezes updated profile snapshot, injects
            # matched jargon footnote, and invokes recall_memory tool!
            t2 = t1 + 3 * 3600
            await bot.adapter.handle_event(
                {
                    "post_type": "message",
                    "message_type": "group",
                    "self_id": 99999,
                    "group_id": 80001,
                    "user_id": 10001,
                    "message_id": 5003,
                    "time": t2,
                    "sender": {"card": "小明"},
                    "message": [
                        {"type": "at", "data": {"qq": "99999"}},
                        {
                            "type": "text",
                            "data": {"text": " 今天摸鱼茶话会帮我推荐下我爱喝的饮料"},
                        },
                    ],
                }
            )
            await wait_traces(3)
            await bot.consolidator.flush()

            # Verify Episode 3 LLM request #1:
            # 1) System prompt has updated group profile (no <system_reminder>).
            ep3_req1 = chat_requests[2]
            ep3_sys = ep3_req1["messages"][0]["content"]
            assert "## 当前群聊画像" in ep3_sys
            assert "轻松吐槽、爱聊奶茶咖啡" in ep3_sys
            assert "<system_reminder>" not in ep3_sys

            # 2) Last user turn contains ephemeral matched jargon footnote.
            ep3_user = ep3_req1["messages"][-1]["content"]
            assert "<system_reminder>" in ep3_user
            assert "摸鱼茶话会：下午三点群里一起点奶茶聊天的固定节目" in ep3_user

            # 3) Episode 3 req #2 carries recall_memory output with Ep 1 memory.
            ep3_req2 = chat_requests[3]
            tool_msg = ep3_req2["messages"][-1]
            assert tool_msg["role"] == "tool"
            recalled = json.loads(tool_msg["content"])["memories"]
            assert any(m["content"] == "小明最爱喝冰抹茶拿铁" for m in recalled)

            # 4) Verify Jev rerank ran during recall_memory.
            assert any("m0" in call["questions"] for call in jev_calls)

            # 5) Verify <system_reminder> was NEVER persisted into SQLite messages.
            db_rows = await bot.message_store.get_group_messages("80001", limit=50)
            assert all("<system_reminder>" not in r.content for r in db_rows)
    finally:
        await llm_client.close()
        await llm_sdk.close()
        await llm_http.aclose()
        await jev_client.close()
        await jev_http.aclose()
        await embed_client.close()
        await embed_http.aclose()
