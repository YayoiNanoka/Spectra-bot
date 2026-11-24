"""Unit tests for spectra.llm."""

from __future__ import annotations

import asyncio
import copy
import json
from collections.abc import Callable
from typing import Any

import httpx
import openai
import pytest
from openai import AsyncOpenAI
from tenacity import wait_none

from spectra.config import LLMConfig
from spectra.llm import (
    EmptyLLMResponseError,
    LLMClient,
    LLMError,
    LLMResponse,
    MaxToolStepsExceededError,
    TokenUsage,
    ToolCall,
    ToolSpec,
    _normalize_content,
    sanitize_messages,
)


def _chat_completion_json(
    *,
    content: Any = "hello",
    model: str = "gpt-4o",
    finish_reason: str | None = "stop",
    tool_calls: list[dict[str, Any]] | None = None,
    reasoning_content: str | None = None,
    prompt_tokens: int = 10,
    cached_tokens: int | None = None,
    completion_tokens: int = 5,
    include_usage: bool = True,
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    if reasoning_content is not None:
        message["reasoning_content"] = reasoning_content

    payload: dict[str, Any] = {
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
    }
    if include_usage:
        usage: dict[str, Any] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        if cached_tokens is not None:
            usage["prompt_tokens_details"] = {"cached_tokens": cached_tokens}
        payload["usage"] = usage
    return payload


def _make_mock_openai(
    handler: Callable[[httpx.Request], httpx.Response],
) -> tuple[AsyncOpenAI, httpx.AsyncClient]:
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    sdk_client = AsyncOpenAI(
        api_key="sk-test",
        base_url="https://api.example.invalid/v1",
        http_client=http_client,
        max_retries=0,
    )
    return sdk_client, http_client


async def test_effort_routing_omits_empty_default_and_sends_configured() -> None:
    captured_bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured_bodies.append(body)
        return httpx.Response(200, json=_chat_completion_json(content="ok"))

    sdk_client, http_client = _make_mock_openai(handler)
    try:
        cfg_default_empty = LLMConfig(
            api_key="sk-test",
            model="gpt-4o",
            default_effort="",
            high_effort="high",
        )
        llm = LLMClient(
            cfg_default_empty, openai_client=sdk_client, retry_wait=wait_none()
        )
        await llm.complete([{"role": "user", "content": "hi"}], effort="default")
        await llm.complete([{"role": "user", "content": "think"}], effort="high")

        assert "reasoning_effort" not in captured_bodies[0]
        assert captured_bodies[1]["reasoning_effort"] == "high"
        assert "tools" not in captured_bodies[0]
        assert "tool_choice" not in captured_bodies[0]

        cfg_configured_default = LLMConfig(
            api_key="sk-test",
            model="gpt-4o",
            default_effort="low",
            high_effort="",
        )
        llm2 = LLMClient(
            cfg_configured_default, openai_client=sdk_client, retry_wait=wait_none()
        )
        await llm2.complete([{"role": "user", "content": "hi"}], effort="default")
        await llm2.complete([{"role": "user", "content": "hi"}], effort="high")

        assert captured_bodies[2]["reasoning_effort"] == "low"
        assert "reasoning_effort" not in captured_bodies[3]

        with pytest.raises(ValueError, match="Unsupported effort"):
            await llm2.complete(
                [{"role": "user", "content": "hi"}],
                effort="ultra",  # type: ignore[arg-type]
            )
    finally:
        await http_client.aclose()


async def test_token_usage_cached_parsing_clamping_and_addition() -> None:
    responses = [
        _chat_completion_json(
            content="first",
            prompt_tokens=100,
            cached_tokens=40,
            completion_tokens=25,
        ),
        _chat_completion_json(
            content="second",
            prompt_tokens=20,
            cached_tokens=50,  # exceeds prompt_tokens -> input_other clamped to 0
            completion_tokens=10,
        ),
        _chat_completion_json(content="third", include_usage=False),
    ]
    idx = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal idx
        payload = responses[idx]
        idx += 1
        return httpx.Response(200, json=payload)

    sdk_client, http_client = _make_mock_openai(handler)
    try:
        llm = LLMClient(
            LLMConfig(api_key="sk-test"),
            openai_client=sdk_client,
            retry_wait=wait_none(),
        )
        r1 = await llm.complete([{"role": "user", "content": "1"}])
        r2 = await llm.complete([{"role": "user", "content": "2"}])
        r3 = await llm.complete([{"role": "user", "content": "3"}])

        assert r1.usage == TokenUsage(input_other=60, input_cached=40, output=25)
        assert r1.usage.total_input == 100
        assert r1.usage.total == 125

        assert r2.usage == TokenUsage(input_other=0, input_cached=50, output=10)
        assert r3.usage == TokenUsage(input_other=0, input_cached=0, output=0)

        combined = r1.usage + r2.usage + r3.usage
        assert combined == TokenUsage(input_other=60, input_cached=90, output=35)
        assert combined.total_input == 150
        assert combined.total == 185
        assert TokenUsage().__add__("bad") is NotImplemented  # type: ignore[arg-type]
    finally:
        await http_client.aclose()


def test_content_normalization_and_to_assistant_message() -> None:
    assert _normalize_content(None) == ""
    assert _normalize_content("  hello world  ") == "hello world"
    assert (
        _normalize_content(
            [
                {"type": "text", "text": "part1 "},
                {"type": "image_url", "image_url": {"url": "x"}},
                {"type": "text", "text": "part2"},
            ]
        )
        == "part1 part2"
    )
    assert _normalize_content([]) == ""
    assert _normalize_content({"type": "text", "text": "from dict"}) == "from dict"
    assert _normalize_content({"other": "ignored"}) == ""
    assert _normalize_content('[{"type": "text", "text": "json list"}]') == "json list"

    tc = ToolCall(id="call_1", name="search", arguments='{"q": "hi"}')
    resp_tool = LLMResponse(
        content="",
        tool_calls=(tc,),
        reasoning_content="need to search",
    )
    assert resp_tool.to_assistant_message() == {
        "role": "assistant",
        "content": None,
        "reasoning_content": "need to search",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "search", "arguments": '{"q": "hi"}'},
            }
        ],
    }

    resp_plain = LLMResponse(content="done", reasoning_content="")
    assert resp_plain.to_assistant_message() == {
        "role": "assistant",
        "content": "done",
    }


async def test_empty_choices_raises_empty_llm_response_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-empty",
                "object": "chat.completion",
                "created": 1700000000,
                "model": "gpt-4o",
                "choices": [],
            },
        )

    sdk_client, http_client = _make_mock_openai(handler)
    try:
        llm = LLMClient(
            LLMConfig(api_key="sk-test"),
            openai_client=sdk_client,
            retry_wait=wait_none(),
        )
        with pytest.raises(EmptyLLMResponseError):
            await llm.complete([{"role": "user", "content": "hi"}])
    finally:
        await http_client.aclose()


def test_sanitize_messages_non_mutation_and_astrbot_rules() -> None:
    original: list[dict[str, Any]] = [
        {"role": "system", "content": "sys"},
        {"role": "tool", "tool_call_id": "orphan_start", "content": "drop me"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": ""},
        {"role": "assistant", "content": None},
        {"role": "assistant", "content": []},
        {"role": "assistant", "content": None, "reasoning_content": "silent thought"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "calling two tools",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "t1", "arguments": "{}"},
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "t2", "arguments": "{}"},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "res1"},
        {"role": "tool", "tool_call_id": "call_1", "content": "dup1"},
        {"role": "tool", "tool_call_id": "call_unknown", "content": "orphan"},
        {"role": "tool", "tool_call_id": "call_2", "content": "res2"},
        {"role": "user", "content": "next"},
        {"role": "tool", "tool_call_id": "call_2", "content": "late orphan"},
    ]
    snapshot = copy.deepcopy(original)

    sanitized = sanitize_messages(original)

    assert original == snapshot
    assert sanitized == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "", "reasoning_content": "silent thought"},
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "calling two tools",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "t1", "arguments": "{}"},
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "t2", "arguments": "{}"},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "res1"},
        {"role": "tool", "tool_call_id": "call_2", "content": "res2"},
        {"role": "user", "content": "next"},
    ]


async def test_retry_on_429_503_timeout_and_no_retry_on_400_401() -> None:
    # 1) Retries on 429, 503, and httpx timeout then succeeds on attempt 4.
    attempts = 0

    def flaky_handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, json={"error": {"message": "rate limit"}})
        if attempts == 2:
            return httpx.Response(503, json={"error": {"message": "unavailable"}})
        if attempts == 3:
            raise httpx.ReadTimeout("timed out", request=request)
        return httpx.Response(200, json=_chat_completion_json(content="recovered"))

    sdk_client, http_client = _make_mock_openai(flaky_handler)
    try:
        llm = LLMClient(
            LLMConfig(api_key="sk-test", max_retries=5),
            openai_client=sdk_client,
            retry_wait=wait_none(),
        )
        resp = await llm.complete([{"role": "user", "content": "retry me"}])
        assert resp.content == "recovered"
        assert attempts == 4
    finally:
        await http_client.aclose()

    # 2) Immediate failure without retry on 400 and 401.
    for status_code, expected_exc in (
        (400, openai.BadRequestError),
        (401, openai.AuthenticationError),
    ):
        call_count = 0

        def fail_fast_handler(
            _request: httpx.Request,
            code: int = status_code,
        ) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(code, json={"error": {"message": "bad request"}})

        sdk_client, http_client = _make_mock_openai(fail_fast_handler)
        try:
            llm = LLMClient(
                LLMConfig(api_key="sk-test", max_retries=5),
                openai_client=sdk_client,
                retry_wait=wait_none(),
            )
            with pytest.raises(expected_exc):
                await llm.complete([{"role": "user", "content": "fail"}])
            assert call_count == 1
        finally:
            await http_client.aclose()


async def test_fallback_model_switches_after_primary_exhausts_retries() -> None:
    requests_seen: list[dict[str, Any]] = []
    fallback_attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal fallback_attempts
        body = json.loads(request.content)
        requests_seen.append(body)
        if body["model"] == "gpt-4o":
            return httpx.Response(503, json={"error": {"message": "primary down"}})
        fallback_attempts += 1
        if fallback_attempts == 1:
            return httpx.Response(429, json={"error": {"message": "fallback 429"}})
        return httpx.Response(
            200,
            json=_chat_completion_json(content="from fallback", model=body["model"]),
        )

    ping_tool = ToolSpec(
        name="ping",
        description="Ping with no args",
        parameters={"type": "object", "properties": {}},
        handler=lambda: "pong",
    )

    sdk_client, http_client = _make_mock_openai(handler)
    try:
        llm = LLMClient(
            LLMConfig(
                api_key="sk-test",
                model="gpt-4o",
                fallback_model="gemini-2.5-flash",
                max_retries=2,
            ),
            openai_client=sdk_client,
            retry_wait=wait_none(),
        )
        resp = await llm.complete(
            [{"role": "user", "content": "hi"}],
            tools=[ping_tool],
        )
        assert resp.content == "from fallback"
        assert resp.model == "gemini-2.5-flash"
        # 2 retries on primary + 2 attempts on fallback
        assert [r["model"] for r in requests_seen] == [
            "gpt-4o",
            "gpt-4o",
            "gemini-2.5-flash",
            "gemini-2.5-flash",
        ]
        # Primary gpt-4o included empty parameters; fallback gemini omitted them.
        assert "parameters" in requests_seen[0]["tools"][0]["function"]
        assert "parameters" not in requests_seen[2]["tools"][0]["function"]
    finally:
        await http_client.aclose()


async def test_lifecycle_api_key_validation_and_ownership() -> None:
    # Missing API key without custom client raises LLMError before any request/fallback.
    llm_no_key = LLMClient(
        LLMConfig(api_key="   ", model="gpt-4o", fallback_model="gpt-4o-mini"),
        retry_wait=wait_none(),
    )
    with pytest.raises(LLMError, match="api_key"):
        await llm_no_key.complete([{"role": "user", "content": "hi"}])

    # Externally injected client is NOT closed by LLMClient.close().
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_chat_completion_json(content="ok"))

    sdk_client, http_client = _make_mock_openai(handler)
    try:
        async with LLMClient(
            LLMConfig(api_key=""),
            openai_client=sdk_client,
            retry_wait=wait_none(),
        ) as llm:
            res = await llm.complete([{"role": "user", "content": "hi"}])
            assert res.content == "ok"

        assert not http_client.is_closed
        # Calling after close raises LLMError.
        with pytest.raises(LLMError, match="closed"):
            await llm.complete([{"role": "user", "content": "again"}])
        # Second close is idempotent.
        await llm.close()
    finally:
        await http_client.aclose()

    # Owned client IS closed by async with exit.
    async with LLMClient(LLMConfig(api_key="sk-owned")) as owned_llm:
        internal = owned_llm._get_openai_client()
        assert internal.max_retries == 0
    assert internal.is_closed()


def test_gemini_tool_schema_omits_empty_parameters() -> None:
    empty_tool = ToolSpec(
        name="get_time",
        description="Get current time",
        parameters={"type": "object", "properties": {}},
        handler=lambda: "12:00",
    )
    param_tool = ToolSpec(
        name="get_weather",
        description="Get weather",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
        handler=lambda city: f"{city}: sunny",
    )

    gpt_empty = empty_tool.to_openai_tool(model="gpt-4o")
    assert gpt_empty == {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get current time",
            "parameters": {"type": "object", "properties": {}},
        },
    }

    gemini_empty = empty_tool.to_openai_tool(model="google/Gemini-2.5-Pro")
    assert gemini_empty == {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get current time",
        },
    }

    gemini_param = param_tool.to_openai_tool(model="gemini-2.5-flash")
    assert "parameters" in gemini_param["function"]
    assert gemini_param["function"]["parameters"]["required"] == ["city"]


async def test_run_tool_loop_multistep_sync_async_parallel_and_reasoning() -> None:
    captured_requests: list[dict[str, Any]] = []
    call_order: list[str] = []

    def sync_add(a: int, b: int) -> dict[str, Any]:
        call_order.append(f"add({a},{b})")
        return {"结果": a + b}

    async def async_lookup(key: str) -> list[str]:
        call_order.append(f"lookup({key})")
        await asyncio.sleep(0)
        return [f"val:{key}"]

    def sync_ping() -> int:
        call_order.append("ping")
        return 42

    step_responses = [
        _chat_completion_json(
            content="",
            reasoning_content="Let me run add and lookup in parallel.",
            finish_reason="tool_calls",
            prompt_tokens=100,
            cached_tokens=20,
            completion_tokens=30,
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "add", "arguments": '{"a": 2, "b": 3}'},
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"key": "k1"}'},
                },
            ],
        ),
        _chat_completion_json(
            content="",
            finish_reason="tool_calls",
            prompt_tokens=150,
            cached_tokens=50,
            completion_tokens=15,
            tool_calls=[
                {
                    "id": "call_3",
                    "type": "function",
                    "function": {"name": "ping", "arguments": ""},
                }
            ],
        ),
        _chat_completion_json(
            content="Sum is 5, lookup is val:k1, ping is 42.",
            finish_reason="stop",
            prompt_tokens=200,
            cached_tokens=100,
            completion_tokens=20,
        ),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured_requests.append(body)
        return httpx.Response(200, json=step_responses[len(captured_requests) - 1])

    tools = [
        ToolSpec(
            name="add",
            description="Add two integers",
            parameters={
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            },
            handler=sync_add,
        ),
        ToolSpec(
            name="lookup",
            description="Lookup a key",
            parameters={"type": "object", "properties": {"key": {"type": "string"}}},
            handler=async_lookup,
        ),
        ToolSpec(
            name="ping",
            description="No-arg ping",
            parameters={"type": "object", "properties": {}},
            handler=sync_ping,
        ),
    ]

    initial_messages = [{"role": "user", "content": "compute everything"}]
    snapshot = copy.deepcopy(initial_messages)

    sdk_client, http_client = _make_mock_openai(handler)
    try:
        llm = LLMClient(
            LLMConfig(api_key="sk-test"),
            openai_client=sdk_client,
            retry_wait=wait_none(),
        )
        result = await llm.run_tool_loop(initial_messages, tools=tools, effort="high")

        # Caller's messages were not mutated.
        assert initial_messages == snapshot
        assert result.content == "Sum is 5, lookup is val:k1, ping is 42."
        assert result.steps == 3
        assert call_order == ["add(2,3)", "lookup(k1)", "ping"]
        assert result.usage == TokenUsage(
            input_other=80 + 100 + 100,
            input_cached=20 + 50 + 100,
            output=30 + 15 + 20,
        )
        assert result.usage.total == 450 + 65

        # Reasoning content and tool outputs preserved into step 2 request.
        step2_msgs = captured_requests[1]["messages"]
        assert step2_msgs[1]["content"] is None
        assert (
            step2_msgs[1]["reasoning_content"]
            == "Let me run add and lookup in parallel."
        )
        assert step2_msgs[2] == {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": '{"结果": 5}',
        }
        assert step2_msgs[3] == {
            "role": "tool",
            "tool_call_id": "call_2",
            "content": '["val:k1"]',
        }
        assert captured_requests[2]["messages"][5] == {
            "role": "tool",
            "tool_call_id": "call_3",
            "content": "42",
        }
        assert all(req["reasoning_effort"] == "high" for req in captured_requests)
    finally:
        await http_client.aclose()


async def test_run_tool_loop_recovers_from_tool_errors_and_bad_json() -> None:
    captured_requests: list[dict[str, Any]] = []

    def boom_tool(x: str) -> str:
        raise RuntimeError(f"exploded on {x}")

    step_responses = [
        _chat_completion_json(
            content="",
            finish_reason="tool_calls",
            tool_calls=[
                {
                    "id": "call_unknown",
                    "type": "function",
                    "function": {"name": "no_such_tool", "arguments": "{}"},
                },
                {
                    "id": "call_bad_json",
                    "type": "function",
                    "function": {"name": "boom", "arguments": "{not json"},
                },
                {
                    "id": "call_non_obj",
                    "type": "function",
                    "function": {"name": "boom", "arguments": "[1, 2, 3]"},
                },
                {
                    "id": "call_runtime",
                    "type": "function",
                    "function": {"name": "boom", "arguments": '{"x": "test"}'},
                },
            ],
        ),
        _chat_completion_json(content="Recovered cleanly from all errors."),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured_requests.append(body)
        return httpx.Response(200, json=step_responses[len(captured_requests) - 1])

    tools = [
        ToolSpec(
            name="boom",
            description="Always fails",
            parameters={"type": "object", "properties": {"x": {"type": "string"}}},
            handler=boom_tool,
        )
    ]

    sdk_client, http_client = _make_mock_openai(handler)
    try:
        llm = LLMClient(
            LLMConfig(api_key="sk-test"),
            openai_client=sdk_client,
            retry_wait=wait_none(),
        )
        result = await llm.run_tool_loop(
            [{"role": "user", "content": "try tools"}],
            tools=tools,
        )
        assert result.content == "Recovered cleanly from all errors."
        assert result.steps == 2

        tool_msgs = [m for m in captured_requests[1]["messages"] if m["role"] == "tool"]
        assert len(tool_msgs) == 4
        assert "Unknown tool 'no_such_tool'" in tool_msgs[0]["content"]
        assert "must be a JSON object" in tool_msgs[1]["content"]
        assert "must be a JSON object" in tool_msgs[2]["content"]
        assert (
            "Error executing tool 'boom': exploded on test" in tool_msgs[3]["content"]
        )
    finally:
        await http_client.aclose()


async def test_run_tool_loop_propagates_cancelled_error() -> None:
    async def cancelling_tool() -> str:
        raise asyncio.CancelledError

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_chat_completion_json(
                content="",
                finish_reason="tool_calls",
                tool_calls=[
                    {
                        "id": "call_cancel",
                        "type": "function",
                        "function": {"name": "cancel_me", "arguments": "{}"},
                    }
                ],
            ),
        )

    sdk_client, http_client = _make_mock_openai(handler)
    try:
        llm = LLMClient(
            LLMConfig(api_key="sk-test"),
            openai_client=sdk_client,
            retry_wait=wait_none(),
        )
        tool = ToolSpec(
            name="cancel_me",
            description="Cancels current task",
            parameters={"type": "object", "properties": {}},
            handler=cancelling_tool,
        )
        with pytest.raises(asyncio.CancelledError):
            await llm.run_tool_loop([{"role": "user", "content": "go"}], tools=[tool])
    finally:
        await http_client.aclose()


async def test_run_tool_loop_max_steps_exceeded_and_validation() -> None:
    call_count = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(
            200,
            json=_chat_completion_json(
                content="",
                finish_reason="tool_calls",
                tool_calls=[
                    {
                        "id": f"call_{call_count}",
                        "type": "function",
                        "function": {"name": "loop_tool", "arguments": "{}"},
                    }
                ],
            ),
        )

    loop_tool = ToolSpec(
        name="loop_tool",
        description="Loops forever",
        parameters={"type": "object", "properties": {}},
        handler=lambda: "keep going",
    )

    sdk_client, http_client = _make_mock_openai(handler)
    try:
        llm = LLMClient(
            LLMConfig(api_key="sk-test", max_tool_steps=3),
            openai_client=sdk_client,
            retry_wait=wait_none(),
        )
        with pytest.raises(MaxToolStepsExceededError):
            await llm.run_tool_loop(
                [{"role": "user", "content": "loop"}],
                tools=[loop_tool],
            )
        assert call_count == 3

        call_count = 0
        with pytest.raises(MaxToolStepsExceededError):
            await llm.run_tool_loop(
                [{"role": "user", "content": "loop"}],
                tools=[loop_tool],
                max_steps=2,
            )
        assert call_count == 2

        with pytest.raises(ValueError, match="max_steps"):
            await llm.run_tool_loop(
                [{"role": "user", "content": "x"}],
                tools=[loop_tool],
                max_steps=0,
            )

        with pytest.raises(ValueError, match="Duplicate tool name"):
            await llm.run_tool_loop(
                [{"role": "user", "content": "x"}],
                tools=[loop_tool, loop_tool],
            )
    finally:
        await http_client.aclose()
