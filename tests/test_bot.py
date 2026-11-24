"""Unit and full-stack offline E2E tests for spectra.bot and spectra.__main__."""

from __future__ import annotations

import asyncio
import io
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from openai import AsyncOpenAI
from PIL import Image
from tenacity import wait_none

from spectra.__main__ import build_parser, main
from spectra.bot import SpectraBot
from spectra.config import (
    AppConfig,
    DatabaseConfig,
    GateConfig,
    JevConfig,
    LLMConfig,
    NapCatConfig,
    SegmentedReplyConfig,
    SessionConfig,
    TraceConfig,
    VisionConfig,
    load_config,
)
from spectra.db import Database
from spectra.jev import JevClient
from spectra.llm import LLMClient, ToolSpec
from spectra.vision import ImageCaptioner


def _chat_completion_json(
    *,
    content: Any,
    model: str = "gpt-4o",
    finish_reason: str = "stop",
    tool_calls: list[dict[str, Any]] | None = None,
    prompt_tokens: int = 120,
    cached_tokens: int = 40,
    completion_tokens: int = 25,
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-test",
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
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_tokens_details": {"cached_tokens": cached_tokens},
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


def _make_jev_client(
    cfg: JevConfig,
    handler: Callable[[httpx.Request], httpx.Response],
) -> tuple[JevClient, httpx.AsyncClient]:
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return JevClient(cfg, http_client=http_client), http_client


class _TraceWaiter:
    """Event-driven observer wrapping ``TraceLogger.write`` for deterministic tests."""

    def __init__(self, bot: SpectraBot) -> None:
        assert bot.trace_logger is not None
        self._real_write = bot.trace_logger.write
        self._count = 0
        self._changed = asyncio.Event()
        bot.trace_logger.write = self._wrapped_write  # type: ignore[method-assign]

    async def _wrapped_write(self, entry: Any) -> None:
        await self._real_write(entry)
        self._count += 1
        self._changed.set()

    async def wait_count(self, expected: int, *, max_wait_sec: float = 2.0) -> None:
        async with asyncio.timeout(max_wait_sec):
            while self._count < expected:
                self._changed.clear()
                await self._changed.wait()


async def _zero_sleep(_: float) -> None:
    await asyncio.sleep(0)


def _make_static_png(
    size: tuple[int, int] = (32, 32),
    color: tuple[int, int, int] = (100, 180, 255),
) -> bytes:
    with Image.new("RGB", size, color) as img:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()


def _make_app_config(
    tmp_path: Path,
    *,
    proactive_mode: str = "on",
    jev_api_key: str = "sk-jev-test",
    vision_enabled: bool = False,
    segmented_enable: bool = False,
) -> AppConfig:
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir(exist_ok=True)
    return AppConfig(
        db=DatabaseConfig(path=tmp_path / "bot.db"),
        napcat=NapCatConfig(
            bot_name="Spectra",
            segmented_reply=SegmentedReplyConfig(enable=segmented_enable),
        ),
        vision=VisionConfig(
            enabled=vision_enabled,
            base_url="https://vision.example.invalid/v1",
            api_key="sk-vision-test" if vision_enabled else "",
            model="gpt-4o-mini",
        ),
        llm=LLMConfig(
            base_url="https://llm.example.invalid/v1",
            api_key="sk-llm-test",
            model="gpt-4o",
            default_effort="",
            high_effort="high",
            max_retries=1,
        ),
        jev=JevConfig(
            enabled=True,
            base_url="https://jev.example.invalid/v1/systemone",
            api_key=jev_api_key,
            model="jev-1.14",
        ),
        gate=GateConfig(
            proactive_mode=proactive_mode,  # type: ignore[arg-type]
            private_debounce_sec=0.0,
            group_debounce_sec=0.0,
            max_batch_wait_sec=0.0,
        ),
        session=SessionConfig(
            prompts_dir=prompts_dir,
            window_max_messages=60,
            window_keep_messages=30,
            timezone="Asia/Shanghai",
        ),
        trace=TraceConfig(
            enabled=True,
            path=tmp_path / "data" / "traces.jsonl",
        ),
    )


async def test_bot_fullstack_e2e_group_mention_tool_loop_and_trace(
    tmp_path: Path,
) -> None:
    cfg = _make_app_config(tmp_path, vision_enabled=True, segmented_enable=True)
    (cfg.session.prompts_dir / "group_50001.md").write_text(
        "你是群50001里的技术搭子 Spectra。",
        encoding="utf-8",
    )

    png_bytes = _make_static_png()

    def media_handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://img.example.invalid/sky.png"
        return httpx.Response(
            200,
            content=png_bytes,
            headers={"content-type": "image/png"},
        )

    def vision_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_chat_completion_json(content="晴朗蓝天白云", model="gpt-4o-mini"),
        )

    jev_requests: list[dict[str, Any]] = []

    def jev_handler(request: httpx.Request) -> httpx.Response:
        jev_requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "jev-1.14",
                "usage": {"input_tokens": 55},
                "answers": {
                    "effort": {
                        "choice": "high",
                        "probabilities": {"default": 0.15, "high": 0.85},
                    }
                },
            },
        )

    llm_requests: list[dict[str, Any]] = []
    llm_call_idx = 0

    def llm_handler(request: httpx.Request) -> httpx.Response:
        nonlocal llm_call_idx
        llm_requests.append(json.loads(request.content))
        llm_call_idx += 1
        if llm_call_idx == 1:
            return httpx.Response(
                200,
                json=_chat_completion_json(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_weather_1",
                            "type": "function",
                            "function": {
                                "name": "lookup_weather",
                                "arguments": json.dumps({"city": "上海"}),
                            },
                        }
                    ],
                    prompt_tokens=100,
                    cached_tokens=30,
                    completion_tokens=15,
                ),
            )
        return httpx.Response(
            200,
            json=_chat_completion_json(
                content="上海今天晴！22度适合出门走走。",
                prompt_tokens=180,
                cached_tokens=90,
                completion_tokens=20,
            ),
        )

    db = Database(cfg.db.path)
    media_http = httpx.AsyncClient(transport=httpx.MockTransport(media_handler))
    vision_http = httpx.AsyncClient(transport=httpx.MockTransport(vision_handler))
    vision_sdk = AsyncOpenAI(
        api_key="sk-vision-test",
        base_url=cfg.vision.base_url,
        http_client=vision_http,
        max_retries=0,
    )
    captioner = ImageCaptioner(
        db,
        cfg.vision,
        http_client=media_http,
        openai_client=vision_sdk,
        retry_wait=wait_none(),
    )
    jev_client, jev_http = _make_jev_client(cfg.jev, jev_handler)
    llm_client, llm_http, sdk_client = _make_llm_client(cfg.llm, llm_handler)

    async def lookup_weather(city: str) -> dict[str, Any]:
        return {"city": city, "weather": "晴", "temp_c": 22}

    weather_tool = ToolSpec(
        name="lookup_weather",
        description="查询城市天气",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
        handler=lookup_weather,
    )

    rpc_calls: list[tuple[str, dict[str, Any]]] = []

    async def fake_call_action(action: str, **params: Any) -> dict[str, Any]:
        rpc_calls.append((action, params))
        return {"message_id": 9001}

    bot = SpectraBot(
        cfg,
        db=db,
        captioner=captioner,
        jev_client=jev_client,
        llm_client=llm_client,
        tools=(weather_tool,),
        sleep_fn=_zero_sleep,
    )
    try:
        async with bot:
            assert bot.adapter is not None
            assert bot.captioner is captioner
            bot.adapter.bot.call_action = fake_call_action  # type: ignore[method-assign]
            waiter = _TraceWaiter(bot)

            ts = int(datetime(2026, 1, 1, 2, 30, 0, tzinfo=UTC).timestamp())
            inbound = await bot.adapter.handle_event(
                {
                    "post_type": "message",
                    "message_type": "group",
                    "self_id": 99999,
                    "group_id": 50001,
                    "user_id": 10001,
                    "message_id": 777,
                    "time": ts,
                    "sender": {"card": "小明", "nickname": "xiaoming"},
                    "message": [
                        {"type": "at", "data": {"qq": "99999"}},
                        {"type": "text", "data": {"text": " 帮我查下上海天气"}},
                        {
                            "type": "image",
                            "data": {"url": "https://img.example.invalid/sky.png"},
                        },
                    ],
                }
            )
            assert inbound is not None

            await waiter.wait_count(1)

            # Verify Jev received effort-only request with captioned image.
            assert len(jev_requests) == 1
            assert "should_reply" not in jev_requests[0]["questions"]
            assert "effort" in jev_requests[0]["questions"]
            assert "[图片: 晴朗蓝天白云]" in jev_requests[0]["state"]["latest_batch"][0]

            # Verify LLM received group persona, formatted [昵称(QQ号) HH:MM]: message,
            # and high reasoning_effort routed from Jev.
            assert len(llm_requests) == 2
            first_req = llm_requests[0]
            assert first_req["reasoning_effort"] == "high"
            assert first_req["messages"][0] == {
                "role": "system",
                "content": "你是群50001里的技术搭子 Spectra。",
            }
            assert first_req["messages"][1] == {
                "role": "user",
                "content": (
                    "[小明(10001) 10:30]: @Bot 帮我查下上海天气 [图片: 晴朗蓝天白云]"
                ),
            }

            # Verify segmented outbound group reply: 2 segments, only first quotes 777.
            assert len(rpc_calls) == 2
            assert rpc_calls[0][0] == "send_group_msg"
            assert rpc_calls[0][1]["group_id"] == 50001
            assert rpc_calls[0][1]["message"] == [
                {"type": "reply", "data": {"id": "777"}},
                {"type": "text", "data": {"text": "上海今天晴！"}},
            ]
            assert rpc_calls[1][0] == "send_group_msg"
            assert rpc_calls[1][1]["group_id"] == 50001
            assert rpc_calls[1][1]["message"] == [
                {"type": "text", "data": {"text": "22度适合出门走走。"}},
            ]

            # Verify merged assistant row persisted, image cached, and session touched.
            assert bot.message_store is not None
            assert bot.session_store is not None
            rows = await bot.message_store.get_group_messages("50001", limit=10)
            assert [r.role for r in rows] == ["user", "assistant"]
            assert rows[1].content == "上海今天晴！22度适合出门走走。"
            cached = await db.fetchall("SELECT caption FROM image_cache")
            assert [str(r["caption"]) for r in cached] == ["晴朗蓝天白云"]
            session = await bot.session_store.get_or_create("group:50001")
            assert session.updated_at > 0

            # Verify exactly 1 unified JSONL trace line was written.
            trace_lines = cfg.trace.path.read_text(encoding="utf-8").splitlines()
            assert len(trace_lines) == 1
            trace = json.loads(trace_lines[0])
            assert trace["session_key"] == "group:50001"
            assert trace["reason"] == "mentioned"
            assert trace["should_reply"] is True
            assert trace["effort"] == "high"
            assert trace["jev_model"] == "jev-1.14"
            assert trace["jev_high_effort_probability"] == 0.85
            assert trace["llm_model"] == "gpt-4o"
            assert trace["steps"] == 2
            assert trace["tool_calls"] == ["lookup_weather"]
            assert trace["input_other_tokens"] == 160  # (100-30) + (180-90)
            assert trace["input_cached_tokens"] == 120  # 30 + 90
            assert trace["output_tokens"] == 35  # 15 + 20
            assert trace["reply_chars"] == len("上海今天晴！22度适合出门走走。")
            assert trace["reply_sent"] is True
            assert trace["error"] is None
    finally:
        await captioner.close()
        await vision_sdk.close()
        await vision_http.aclose()
        await media_http.aclose()
        await sdk_client.close()
        await llm_http.aclose()
        await jev_http.aclose()
        await db.close()


async def test_bot_private_chat_no_quote_and_multi_turn_context(
    tmp_path: Path,
) -> None:
    cfg = _make_app_config(tmp_path, jev_api_key="")
    (cfg.session.prompts_dir / "private_20002.md").write_text(
        "私聊专属人设",
        encoding="utf-8",
    )

    llm_requests: list[dict[str, Any]] = []
    replies = iter(["你好呀，有什么事？", "当然记得，你刚打了招呼！"])

    def llm_handler(request: httpx.Request) -> httpx.Response:
        llm_requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json=_chat_completion_json(content=next(replies)),
        )

    llm_client, llm_http, sdk_client = _make_llm_client(cfg.llm, llm_handler)
    rpc_calls: list[tuple[str, dict[str, Any]]] = []

    async def fake_call_action(action: str, **params: Any) -> dict[str, Any]:
        rpc_calls.append((action, params))
        return {"message_id": 9100}

    bot = SpectraBot(cfg, llm_client=llm_client, sleep_fn=_zero_sleep)
    try:
        async with bot:
            assert bot.adapter is not None
            bot.adapter.bot.call_action = fake_call_action  # type: ignore[method-assign]
            waiter = _TraceWaiter(bot)

            await bot.adapter.handle_event(
                {
                    "post_type": "message",
                    "message_type": "private",
                    "self_id": 99999,
                    "user_id": 20002,
                    "message_id": 801,
                    "time": 1700000100,
                    "sender": {"nickname": "小红"},
                    "message": [{"type": "text", "data": {"text": "哈喽"}}],
                }
            )
            await waiter.wait_count(1)

            # Private reply must NOT include a quote reply segment.
            assert rpc_calls[0][0] == "send_private_msg"
            assert rpc_calls[0][1]["message"] == [
                {"type": "text", "data": {"text": "你好呀，有什么事？"}}
            ]

            await bot.adapter.handle_event(
                {
                    "post_type": "message",
                    "message_type": "private",
                    "self_id": 99999,
                    "user_id": 20002,
                    "message_id": 802,
                    "time": 1700000120,
                    "sender": {"nickname": "小红"},
                    "message": [
                        {"type": "text", "data": {"text": "你还记得我刚说啥吗"}}
                    ],
                }
            )
            await waiter.wait_count(2)

            assert len(llm_requests) == 2
            assert llm_requests[1]["messages"] == [
                {"role": "system", "content": "私聊专属人设"},
                {"role": "user", "content": "哈喽"},
                {"role": "assistant", "content": "你好呀，有什么事？"},
                {"role": "user", "content": "你还记得我刚说啥吗"},
            ]
    finally:
        await sdk_client.close()
        await llm_http.aclose()


async def test_bot_shadow_and_filtered_decisions_write_single_trace_without_llm(
    tmp_path: Path,
) -> None:
    cfg = _make_app_config(tmp_path, proactive_mode="shadow")

    def jev_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "jev-1.14",
                "usage": {"input_tokens": 42},
                "answers": {
                    "should_reply": {"noul": 0.91},
                    "effort": {"choice": "default", "probabilities": {"high": 0.1}},
                },
            },
        )

    llm_calls = 0

    def llm_handler(request: httpx.Request) -> httpx.Response:
        nonlocal llm_calls
        llm_calls += 1
        return httpx.Response(200, json=_chat_completion_json(content="should not run"))

    jev_client, jev_http = _make_jev_client(cfg.jev, jev_handler)
    llm_client, llm_http, sdk_client = _make_llm_client(cfg.llm, llm_handler)
    rpc_calls: list[tuple[str, dict[str, Any]]] = []

    async def fake_call_action(action: str, **params: Any) -> dict[str, Any]:
        rpc_calls.append((action, params))
        return {}

    bot = SpectraBot(
        cfg,
        jev_client=jev_client,
        llm_client=llm_client,
        sleep_fn=_zero_sleep,
    )
    try:
        async with bot:
            assert bot.adapter is not None
            bot.adapter.bot.call_action = fake_call_action  # type: ignore[method-assign]
            waiter = _TraceWaiter(bot)

            # 1. Unmentioned open chatter -> shadow_jev (no LLM, no send).
            await bot.adapter.handle_event(
                {
                    "post_type": "message",
                    "message_type": "group",
                    "self_id": 99999,
                    "group_id": 60001,
                    "user_id": 10001,
                    "message_id": 901,
                    "time": 1700000200,
                    "sender": {"nickname": "甲"},
                    "message": [{"type": "text", "data": {"text": "今晚吃火锅吗"}}],
                }
            )
            await waiter.wait_count(1)

            # 2. Starts with @other -> filtered_at_other (no Jev, no LLM, no send).
            await bot.adapter.handle_event(
                {
                    "post_type": "message",
                    "message_type": "group",
                    "self_id": 99999,
                    "group_id": 60001,
                    "user_id": 10001,
                    "message_id": 902,
                    "time": 1700000210,
                    "sender": {"nickname": "甲"},
                    "message": [
                        {"type": "at", "data": {"qq": "10002"}},
                        {"type": "text", "data": {"text": " 出来打游戏"}},
                    ],
                }
            )
            await waiter.wait_count(2)

            assert llm_calls == 0
            # Only get_group_member_info may be called for @10002; no send_group_msg.
            assert not any(action.startswith("send_") for action, _ in rpc_calls)

            lines = [
                json.loads(line)
                for line in cfg.trace.path.read_text(encoding="utf-8").splitlines()
            ]
            assert len(lines) == 2
            assert lines[0]["reason"] == "shadow_jev"
            assert lines[0]["shadow"] is True
            assert lines[0]["should_reply"] is False
            assert lines[0]["jev_reply_probability"] == 0.91
            assert lines[0]["llm_model"] is None
            assert lines[0]["reply_sent"] is False

            assert lines[1]["reason"] == "filtered_at_other"
            assert lines[1]["should_reply"] is False
            assert lines[1]["jev_model"] is None
            assert lines[1]["reply_sent"] is False
    finally:
        await sdk_client.close()
        await llm_http.aclose()
        await jev_http.aclose()


async def test_bot_empty_reply_and_llm_error_trace_and_recovery(
    tmp_path: Path,
) -> None:
    cfg = _make_app_config(tmp_path, jev_api_key="")
    call_count = 0

    def llm_handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(200, json=_chat_completion_json(content="   \n "))
        if call_count == 2:
            return httpx.Response(400, json={"error": {"message": "bad request"}})
        return httpx.Response(200, json=_chat_completion_json(content="恢复正常啦"))

    llm_client, llm_http, sdk_client = _make_llm_client(cfg.llm, llm_handler)
    rpc_calls: list[tuple[str, dict[str, Any]]] = []

    async def fake_call_action(action: str, **params: Any) -> dict[str, Any]:
        rpc_calls.append((action, params))
        return {}

    bot = SpectraBot(cfg, llm_client=llm_client, sleep_fn=_zero_sleep)
    try:
        async with bot:
            assert bot.adapter is not None
            bot.adapter.bot.call_action = fake_call_action  # type: ignore[method-assign]
            waiter = _TraceWaiter(bot)

            for msg_id, text in ((1, "第一句"), (2, "第二句"), (3, "第三句")):
                await bot.adapter.handle_event(
                    {
                        "post_type": "message",
                        "message_type": "private",
                        "self_id": 99999,
                        "user_id": 30003,
                        "message_id": msg_id,
                        "time": 1700000300 + msg_id,
                        "sender": {"nickname": "丙"},
                        "message": [{"type": "text", "data": {"text": text}}],
                    }
                )
                await waiter.wait_count(msg_id)

            assert len(rpc_calls) == 1
            assert rpc_calls[0][1]["message"] == [
                {"type": "text", "data": {"text": "恢复正常啦"}}
            ]

            traces = [
                json.loads(line)
                for line in cfg.trace.path.read_text(encoding="utf-8").splitlines()
            ]
            assert len(traces) == 3
            assert traces[0]["reply_sent"] is False
            assert traces[0]["reply_chars"] == 0
            assert traces[0]["error"] is None

            assert traces[1]["reply_sent"] is False
            assert traces[1]["llm_model"] == "gpt-4o"
            assert traces[1]["llm_latency_ms"] is not None
            assert traces[1]["error"] is not None
            assert "bad request" in traces[1]["error"]

            assert traces[2]["reply_sent"] is True
            assert traces[2]["error"] is None
    finally:
        await sdk_client.close()
        await llm_http.aclose()


async def test_bot_startup_failure_cleans_up_resources(tmp_path: Path) -> None:
    cfg = _make_app_config(tmp_path)

    class FailingCaptioner:
        async def init(self) -> None:
            raise RuntimeError("captioner init boom")

        async def close(self) -> None:
            pass

    bot = SpectraBot(cfg, captioner=FailingCaptioner())  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="captioner init boom"):
        await bot.start()

    # Owned DB must have been closed on startup rollback.
    assert bot.db is not None
    assert bot.db._conn is None


async def test_bot_reverse_order_idempotent_shutdown_and_run_signal(
    tmp_path: Path,
) -> None:
    cfg = _make_app_config(tmp_path, vision_enabled=True)
    order: list[str] = []

    class FakeAdapter:
        def __init__(self) -> None:
            self._stop = asyncio.Event()

        async def run(self) -> None:
            await self._stop.wait()

        async def shutdown(self) -> None:
            order.append("adapter")
            self._stop.set()

    bot = SpectraBot(cfg, adapter=FakeAdapter())  # type: ignore[arg-type]
    await bot.start()

    assert bot.gate is not None
    assert bot.captioner is not None
    assert bot.jev_client is not None
    assert bot.llm_client is not None
    assert bot.db is not None

    real_gate_close = bot.gate.close
    real_cap_close = bot.captioner.close
    real_jev_close = bot.jev_client.close
    real_llm_close = bot.llm_client.close
    real_db_close = bot.db.close

    async def wrap(name: str, fn: Callable[[], Any]) -> None:
        order.append(name)
        await fn()

    bot.gate.close = lambda: wrap("gate", real_gate_close)  # type: ignore[method-assign]
    bot.captioner.close = lambda: wrap("captioner", real_cap_close)  # type: ignore[method-assign]
    bot.jev_client.close = lambda: wrap("jev_client", real_jev_close)  # type: ignore[method-assign]
    bot.llm_client.close = lambda: wrap("llm_client", real_llm_close)  # type: ignore[method-assign]
    bot.db.close = lambda: wrap("db", real_db_close)  # type: ignore[method-assign]

    run_task = asyncio.create_task(bot.run())
    await asyncio.sleep(0.01)
    bot.request_shutdown()
    await run_task

    # Calling close() again must be a no-op.
    await bot.close()

    assert order == [
        "adapter",
        "adapter",
        "gate",
        "captioner",
        "jev_client",
        "llm_client",
        "db",
    ]


def test_cli_init_config_and_refuse_overwrite(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg_path = tmp_path / "sub" / "config.json"

    rc = main(["--init-config", "--config", str(cfg_path)])
    assert rc == 0
    assert cfg_path.is_file()
    assert load_config(cfg_path) == AppConfig()
    out = capsys.readouterr().out
    assert str(cfg_path) in out

    # Must refuse to overwrite existing config file.
    rc2 = main(["--init-config", "--config", str(cfg_path)])
    assert rc2 == 1
    err = capsys.readouterr().err
    assert "Refusing to overwrite" in err

    # Must also refuse to clobber a dangling symlink.
    dangling = tmp_path / "dangling_config.json"
    dangling.symlink_to(tmp_path / "nonexistent_target.json")
    rc3 = main(["--init-config", "--config", str(dangling)])
    assert rc3 == 1
    assert dangling.is_symlink()
    assert not (tmp_path / "nonexistent_target.json").exists()


async def test_bot_midflight_close_cancels_cleanly_and_respects_ownership(
    tmp_path: Path,
) -> None:
    cfg = _make_app_config(tmp_path, jev_api_key="", vision_enabled=True)
    db = Database(cfg.db.path)
    captioner = ImageCaptioner(db, cfg.vision)
    jev_client = JevClient(cfg.jev)

    started_llm = asyncio.Event()
    cancelled_llm = asyncio.Event()

    class BlockingLLMClient:
        closed = False

        async def run_tool_loop(self, *args: Any, **kwargs: Any) -> Any:
            started_llm.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                cancelled_llm.set()
                raise

        async def close(self) -> None:
            self.closed = True

    llm_client = BlockingLLMClient()
    bot = SpectraBot(
        cfg,
        db=db,
        captioner=captioner,
        jev_client=jev_client,
        llm_client=llm_client,  # type: ignore[arg-type]
        sleep_fn=_zero_sleep,
    )
    try:
        await bot.start()
        assert bot.adapter is not None

        await bot.adapter.handle_event(
            {
                "post_type": "message",
                "message_type": "private",
                "self_id": 99999,
                "user_id": 40004,
                "message_id": 123,
                "time": 1700000400,
                "sender": {"nickname": "丁"},
                "message": [{"type": "text", "data": {"text": "在吗"}}],
            }
        )
        async with asyncio.timeout(2.0):
            await started_llm.wait()

        # Closing while LLM is mid-flight must cancel cleanly and skip trace write.
        await bot.close()
        assert cancelled_llm.is_set()
        assert not cfg.trace.path.exists()

        # Injected dependencies must NOT be closed by SpectraBot.close().
        assert db._conn is not None
        assert llm_client.closed is False
        assert jev_client._closed is False
    finally:
        await captioner.close()
        await jev_client.close()
        await db.close()


async def test_bot_fallback_user_turn_when_db_rows_missing(tmp_path: Path) -> None:
    cfg = _make_app_config(tmp_path, jev_api_key="")
    seen_messages: list[list[dict[str, Any]]] = []

    def llm_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen_messages.append(body["messages"])
        return httpx.Response(200, json=_chat_completion_json(content="收到！"))

    llm_client, llm_http, sdk_client = _make_llm_client(cfg.llm, llm_handler)
    rpc_calls: list[tuple[str, dict[str, Any]]] = []

    async def fake_call_action(action: str, **params: Any) -> dict[str, Any]:
        rpc_calls.append((action, params))
        return {}

    bot = SpectraBot(cfg, llm_client=llm_client, sleep_fn=_zero_sleep)
    try:
        async with bot:
            assert bot.adapter is not None
            assert bot.db is not None
            bot.adapter.bot.call_action = fake_call_action  # type: ignore[method-assign]
            waiter = _TraceWaiter(bot)

            # Delete message rows right before build_context runs to simulate race.
            assert bot.session_store is not None
            real_build = bot.session_store.build_context

            async def deleting_build(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
                await bot.db.execute("DELETE FROM messages")
                return await real_build(*args, **kwargs)

            bot.session_store.build_context = deleting_build  # type: ignore[method-assign]

            ts = int(datetime(2026, 1, 1, 2, 30, 0, tzinfo=UTC).timestamp())
            await bot.adapter.handle_event(
                {
                    "post_type": "message",
                    "message_type": "group",
                    "self_id": 99999,
                    "group_id": 70001,
                    "user_id": 10001,
                    "message_id": 555,
                    "time": ts,
                    "sender": {"card": "小明"},
                    "message": [
                        {"type": "at", "data": {"qq": "99999"}},
                        {"type": "text", "data": {"text": " 测试兜底"}},
                    ],
                }
            )
            await waiter.wait_count(1)

            assert len(seen_messages) == 1
            assert [m["role"] for m in seen_messages[0]] == ["system", "user"]
            assert (
                seen_messages[0][1]["content"] == "[小明(10001) 10:30]: @Bot 测试兜底"
            )
            assert len(rpc_calls) == 1
    finally:
        await sdk_client.close()
        await llm_http.aclose()


def test_cli_parser_defaults() -> None:
    args = build_parser().parse_args([])
    assert args.config == Path("config.json")
    assert args.init_config is False
    assert args.log_level == "INFO"

    args2 = build_parser().parse_args(
        ["--config", "custom.json", "--log-level", "debug"]
    )
    assert args2.config == Path("custom.json")
    assert args2.log_level == "DEBUG"
