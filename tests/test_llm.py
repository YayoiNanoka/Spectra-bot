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


def test_sanitize_messages_reconciles_partial_and_unanswered_tool_calls() -> None:
    messages: list[dict[str, Any]] = [
        # 1) Partial tool results + tool result interrupted by user message.
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "a", "arguments": "{}"},
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "b", "arguments": "{}"},
                },
                {
                    "id": "call_3",
                    "type": "function",
                    "function": {"name": "c", "arguments": "{}"},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "ok1"},
        {"role": "user", "content": "interrupt"},
        {"role": "tool", "tool_call_id": "call_2", "content": "late2"},
        # 2) Completely unanswered assistant(tool_calls) variants:
        # 2a) With non-empty text -> drop tool_calls, keep text.
        {
            "role": "assistant",
            "content": "Let me look that up.",
            "tool_calls": [
                {
                    "id": "unanswered_text",
                    "type": "function",
                    "function": {"name": "x", "arguments": "{}"},
                }
            ],
        },
        # 2b) Reasoning-only -> drop tool_calls, set content="", keep reasoning.
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "thinking before cancelled tool",
            "tool_calls": [
                {
                    "id": "unanswered_reasoning",
                    "type": "function",
                    "function": {"name": "x", "arguments": "{}"},
                }
            ],
        },
        # 2c) Empty content & no reasoning -> drop entire message.
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "unanswered_empty",
                    "type": "function",
                    "function": {"name": "x", "arguments": "{}"},
                }
            ],
        },
        # 3) Malformed / duplicate / empty tool_call IDs + non-list tool_calls.
        {
            "role": "assistant",
            "content": "still valid text",
            "tool_calls": "not-a-list",
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                "not-a-dict",
                {"id": "", "type": "function", "function": {"name": "bad"}},
                {"id": None, "type": "function", "function": {"name": "bad2"}},
                {
                    "id": "dup_id",
                    "type": "function",
                    "function": {"name": "first", "arguments": "{}"},
                },
                {
                    "id": "dup_id",
                    "type": "function",
                    "function": {"name": "second", "arguments": "{}"},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "dup_id", "content": "ok_dup"},
    ]
    snapshot = copy.deepcopy(messages)

    sanitized = sanitize_messages(messages)

    assert messages == snapshot
    assert sanitized == [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "a", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "ok1"},
        {"role": "user", "content": "interrupt"},
        {"role": "assistant", "content": "Let me look that up."},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "thinking before cancelled tool",
        },
        {"role": "assistant", "content": "still valid text"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "dup_id",
                    "type": "function",
                    "function": {"name": "first", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "dup_id", "content": "ok_dup"},
    ]


async def test_parallel_tool_calls_with_none_or_duplicate_ids_and_null_args() -> None:
    captured_requests: list[dict[str, Any]] = []

    step_responses = [
        _chat_completion_json(
            content="",
            finish_reason="tool_calls",
            tool_calls=[
                {
                    "id": None,
                    "type": "function",
                    "function": {"name": "echo", "arguments": '{"msg": "one"}'},
                },
                {
                    "id": None,
                    "type": "function",
                    "function": {"name": "echo", "arguments": '{"msg": "two"}'},
                },
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "echo", "arguments": "null"},
                },
            ],
        ),
        _chat_completion_json(content="All three handled."),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured_requests.append(body)
        return httpx.Response(200, json=step_responses[len(captured_requests) - 1])

    echo_tool = ToolSpec(
        name="echo",
        description="Echo message",
        parameters={"type": "object", "properties": {"msg": {"type": "string"}}},
        handler=lambda msg: f"echo:{msg}",
    )

    sdk_client, http_client = _make_mock_openai(handler)
    try:
        llm = LLMClient(
            LLMConfig(api_key="sk-test"),
            openai_client=sdk_client,
            retry_wait=wait_none(),
        )
        result = await llm.run_tool_loop(
            [{"role": "user", "content": "run parallel"}],
            tools=[echo_tool],
        )
        assert result.content == "All three handled."
        assert result.steps == 2

        step2_msgs = captured_requests[1]["messages"]
        assistant_tcs = step2_msgs[1]["tool_calls"]
        tool_msgs = [m for m in step2_msgs if m["role"] == "tool"]
        assert len(assistant_tcs) == 3
        assert len(tool_msgs) == 3
        ids = [tc["id"] for tc in assistant_tcs]
        assert len(set(ids)) == 3
        assert all(tc_id and tc_id != "None" for tc_id in ids)
        assert [m["tool_call_id"] for m in tool_msgs] == ids
        assert tool_msgs[0]["content"] == "echo:one"
        assert tool_msgs[1]["content"] == "echo:two"
        assert "must be a JSON object" in tool_msgs[2]["content"]
    finally:
        await http_client.aclose()


def test_normalize_content_unhashable_literal_eval_and_null_cached_tokens() -> None:
    assert _normalize_content("[{[1]: 2}]") == "[{[1]: 2}]"

    from spectra.llm import _extract_usage, _parse_completion

    usage = _extract_usage(
        {
            "prompt_tokens": 50,
            "completion_tokens": 12,
            "prompt_tokens_details": {"cached_tokens": None},
        }
    )
    assert usage == TokenUsage(input_other=50, input_cached=0, output=12)

    class _ChoiceWithoutMessage:
        message = None
        finish_reason = "stop"

    class _CompletionMissingMessage:
        choices = [_ChoiceWithoutMessage()]
        model = "gpt-4o"
        usage = None

    with pytest.raises(EmptyLLMResponseError):
        _parse_completion(_CompletionMissingMessage(), "gpt-4o")


async def test_auth_error_skips_fallback_model_validation_and_early_close() -> None:
    # 1) Empty/whitespace model raises LLMError before any HTTP call.
    requests_seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests_seen.append(body["model"])
        return httpx.Response(401, json={"error": {"message": "invalid key"}})

    sdk_client, http_client = _make_mock_openai(handler)
    try:
        llm_empty_model = LLMClient(
            LLMConfig(
                api_key="sk-test",
                model="   ",
                fallback_model="gpt-4o-mini",
            ),
            openai_client=sdk_client,
            retry_wait=wait_none(),
        )
        with pytest.raises(LLMError, match="model"):
            await llm_empty_model.complete([{"role": "user", "content": "hi"}])
        assert requests_seen == []

        # 2) HTTP 401 AuthenticationError does NOT fall back to fallback_model.
        llm_401 = LLMClient(
            LLMConfig(
                api_key="sk-test",
                model="gpt-4o",
                fallback_model="gpt-4o-mini",
                max_retries=3,
            ),
            openai_client=sdk_client,
            retry_wait=wait_none(),
        )
        with pytest.raises(openai.AuthenticationError):
            await llm_401.complete([{"role": "user", "content": "hi"}])
        assert requests_seen == ["gpt-4o"]

        # 3) Whitespace-padded model equal to fallback_model does not redundantly retry.
        requests_seen.clear()

        def err503_handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            requests_seen.append(body["model"])
            return httpx.Response(503, json={"error": {"message": "down"}})

        sdk_503, http_503 = _make_mock_openai(err503_handler)
        try:
            llm_same = LLMClient(
                LLMConfig(
                    api_key="sk-test",
                    model="  gpt-4o  ",
                    fallback_model="gpt-4o",
                    max_retries=1,
                ),
                openai_client=sdk_503,
                retry_wait=wait_none(),
            )
            with pytest.raises(openai.InternalServerError):
                await llm_same.complete([{"role": "user", "content": "hi"}])
            assert requests_seen == ["gpt-4o"]
        finally:
            await http_503.aclose()
    finally:
        await http_client.aclose()

    # 4) Closing owned client before lazy init and calling close() twice is safe.
    uninitialized_llm = LLMClient(LLMConfig(api_key="sk-owned"))
    await uninitialized_llm.close()
    await uninitialized_llm.close()
    with pytest.raises(LLMError, match="closed"):
        await uninitialized_llm.complete([{"role": "user", "content": "hi"}])


def test_sanitize_messages_reasoning_key_bidirectional() -> None:
    msgs: list[dict[str, Any]] = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": None, "reasoning_content": "thought A"},
        {"role": "assistant", "content": None, "reasoning": "thought B"},
        {"role": "assistant", "content": "", "reasoning": ""},
        {"role": "assistant", "content": "hello", "reasoning_content": ""},
    ]

    or_sanitized = sanitize_messages(msgs, reasoning_key="reasoning")
    assert or_sanitized == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "reasoning": "thought A"},
        {"role": "assistant", "content": "", "reasoning": "thought B"},
        {"role": "assistant", "content": "hello"},
    ]

    std_sanitized = sanitize_messages(msgs, reasoning_key="reasoning_content")
    assert std_sanitized == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "reasoning_content": "thought A"},
        {"role": "assistant", "content": "", "reasoning_content": "thought B"},
        {"role": "assistant", "content": "hello"},
    ]


async def test_openrouter_reasoning_key_headers_and_tool_loop_roundtrip() -> None:
    captured_bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured_bodies.append(body)
        if len(captured_bodies) == 1:
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-or-1",
                    "object": "chat.completion",
                    "created": 1700000000,
                    "model": "deepseek/deepseek-r1",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "reasoning": "openrouter reasoning trace",
                                "tool_calls": [
                                    {
                                        "id": "call_or_1",
                                        "type": "function",
                                        "function": {
                                            "name": "ping",
                                            "arguments": "{}",
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                },
            )
        return httpx.Response(
            200,
            json=_chat_completion_json(
                content="final ok", model="deepseek/deepseek-r1"
            ),
        )

    cfg = LLMConfig(
        base_url="https://openrouter.ai/api/v1",
        api_key="sk-or-test",
        model="deepseek/deepseek-r1",
        extra_body={"provider": {"sort": "throughput"}},
    )
    # Verify owned AsyncOpenAI receives OpenRouter default headers and User-Agent.
    async with LLMClient(cfg) as owned_llm:
        internal = owned_llm._get_openai_client()
        assert internal.default_headers["User-Agent"] == "spectra-bot/0.1.0"
        assert (
            internal.default_headers["HTTP-Referer"]
            == "https://github.com/YayoiNanoka/Spectra-bot"
        )
        assert internal.default_headers["X-OpenRouter-Title"] == "Spectra-bot"
        assert owned_llm.compat.reasoning_key == "reasoning"

    # Verify tool-loop extracts `reasoning`, echoes it, and merges extra_body.
    sdk_client, http_client = _make_mock_openai(handler)
    try:
        llm = LLMClient(cfg, openai_client=sdk_client, retry_wait=wait_none())
        tool = ToolSpec(
            name="ping",
            description="Ping",
            parameters={"type": "object", "properties": {}},
            handler=lambda: "pong",
        )
        res = await llm.run_tool_loop(
            [{"role": "user", "content": "ping please"}],
            tools=[tool],
        )
        assert res.content == "final ok"
        assert res.messages[1]["reasoning"] == "openrouter reasoning trace"
        assert "reasoning_content" not in res.messages[1]
        assert captured_bodies[0]["provider"] == {"sort": "throughput"}
        step2_assistant = captured_bodies[1]["messages"][1]
        assert step2_assistant["reasoning"] == "openrouter reasoning trace"
        assert "reasoning_content" not in step2_assistant
    finally:
        await http_client.aclose()


async def test_non_string_reasoning_metadata_ignored_on_wire() -> None:
    captured_bodies: list[dict[str, Any]] = []
    call_idx = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_idx
        captured_bodies.append(json.loads(request.content))
        call_idx += 1
        if call_idx == 1:
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-meta-1",
                    "object": "chat.completion",
                    "created": 1700000000,
                    "model": "deepseek/deepseek-r1",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "reasoning": {"effort": "low", "summary": None},
                                "tool_calls": [
                                    {
                                        "id": "call_m1",
                                        "type": "function",
                                        "function": {"name": "ping", "arguments": "{}"},
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                },
            )
        if call_idx == 2:
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-meta-2",
                    "object": "chat.completion",
                    "created": 1700000000,
                    "model": "deepseek/deepseek-r1",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "done",
                                "reasoning": False,
                            },
                            "finish_reason": "stop",
                        }
                    ],
                },
            )
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-meta-3",
                "object": "chat.completion",
                "created": 1700000000,
                "model": "deepseek/deepseek-r1",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "fallback ok",
                            "reasoning": {"effort": "low"},
                            "reasoning_content": "real string fallback",
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    cfg = LLMConfig(
        base_url="https://openrouter.ai/api/v1",
        api_key="sk-or-test",
        model="deepseek/deepseek-r1",
    )
    sdk_client, http_client = _make_mock_openai(handler)
    try:
        llm = LLMClient(cfg, openai_client=sdk_client, retry_wait=wait_none())
        tool = ToolSpec(
            name="ping",
            description="Ping",
            parameters={"type": "object", "properties": {}},
            handler=lambda: "pong",
        )
        res = await llm.run_tool_loop(
            [{"role": "user", "content": "check"}],
            tools=[tool],
        )
        assert res.content == "done"
        # Non-string reasoning dict must NOT be coerced into text on step 2 or result.
        assert "reasoning" not in captured_bodies[1]["messages"][1]
        assert "reasoning_content" not in captured_bodies[1]["messages"][1]
        assert "reasoning" not in res.messages[1]
        assert "reasoning" not in res.messages[-1]

        # Non-string primary key must NOT shadow a valid string fallback key.
        fallback_resp = await llm.complete([{"role": "user", "content": "check2"}])
        assert fallback_resp.reasoning_content == "real string fallback"
    finally:
        await http_client.aclose()


async def test_opencode_go_url_rewrite_and_session_header() -> None:
    captured_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        return httpx.Response(200, json=_chat_completion_json(content="ok"))

    cfg = LLMConfig(
        base_url="https://opencode.ai/zen/go/v1",
        api_key="sk-oc-test",
        model="kimi-k2.5",
    )
    async with LLMClient(cfg) as owned_llm:
        internal = owned_llm._get_openai_client()
        assert (
            str(internal.base_url).rstrip("/")
            == "https://opencode.ai/inference/go/openai/v1"
        )
        assert internal.default_headers["User-Agent"] == "spectra-bot/0.1.0"

    sdk_client, http_client = _make_mock_openai(handler)
    try:
        llm = LLMClient(cfg, openai_client=sdk_client, retry_wait=wait_none())
        await llm.complete(
            [{"role": "user", "content": "hi"}],
            session_key="group:88888",
        )
        await llm.complete([{"role": "user", "content": "hi"}])

        assert captured_requests[0].headers["x-opencode-session"] == "group:88888"
        assert captured_requests[1].headers["x-opencode-session"] == "spectra:llm"
    finally:
        await http_client.aclose()

    # Wire-level verification of case-insensitive custom_headers overrides with zero
    # duplicate header keys on a real AsyncOpenAI + httpx.MockTransport client.
    override_requests: list[httpx.Request] = []

    def override_handler(request: httpx.Request) -> httpx.Response:
        override_requests.append(request)
        return httpx.Response(200, json=_chat_completion_json(content="ok"))

    cfg_override = LLMConfig(
        base_url="https://opencode.ai/inference/go/openai/v1",
        api_key="sk-oc-test",
        model="kimi-k2.5",
        custom_headers={
            "X-OpenCode-Session": "pinned",
            "user-agent": "custom/1.0",
        },
    )
    override_http = httpx.AsyncClient(transport=httpx.MockTransport(override_handler))
    try:
        llm_override_probe = LLMClient(cfg_override)
        override_sdk = AsyncOpenAI(
            api_key="sk-oc-test",
            base_url=llm_override_probe.compat.base_url,
            default_headers=dict(llm_override_probe.compat.default_headers),
            http_client=override_http,
            max_retries=0,
        )
        llm_override = LLMClient(
            cfg_override,
            openai_client=override_sdk,
            retry_wait=wait_none(),
        )
        await llm_override.complete(
            [{"role": "user", "content": "hi"}],
            session_key="group:88888",
        )
        assert len(override_requests) == 1
        raw_headers = override_requests[0].headers.raw
        session_entries = [
            v for k, v in raw_headers if k.lower() == b"x-opencode-session"
        ]
        ua_entries = [v for k, v in raw_headers if k.lower() == b"user-agent"]
        assert session_entries == [b"pinned"]
        assert ua_entries == [b"custom/1.0"]
    finally:
        await override_http.aclose()
