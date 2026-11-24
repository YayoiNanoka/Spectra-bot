"""Unit tests for spectra.jev."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from spectra.config import JevConfig
from spectra.jev import JevClient, resolve_jev_endpoint


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "https://opencode.ai/inference/go/systemone/v1/systemone",
            "https://opencode.ai/inference/go/systemone/v1/systemone",
        ),
        (
            "  https://opencode.ai/inference/go/systemone/v1/systemone/  ",
            "https://opencode.ai/inference/go/systemone/v1/systemone",
        ),
        (
            "https://opencode.ai/zen/v1/systemone",
            "https://opencode.ai/zen/v1/systemone",
        ),
        (
            "https://openrouter.ai/api/alpha/decisions",
            "https://openrouter.ai/api/alpha/decisions",
        ),
        (
            "https://api.typesafe.ai/v1",
            "https://api.typesafe.ai/v1/systemone",
        ),
        (
            "https://opencode.ai/zen/v1/",
            "https://opencode.ai/zen/v1/systemone",
        ),
        ("   ", ""),
    ],
)
def test_resolve_jev_endpoint(raw: str, expected: str) -> None:
    assert resolve_jev_endpoint(raw) == expected


async def test_wire_payload_and_response_parsing_with_should_reply() -> None:
    captured: list[tuple[httpx.Request, dict[str, Any]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        captured.append((request, body))
        return httpx.Response(
            200,
            json={
                "model": "jev-1.14",
                "usage": {"input_tokens": 128},
                "answers": {
                    "should_reply": {"noul": 0.78},
                    "effort": {
                        "choice": "high",
                        "probabilities": {"default": 0.25, "high": 0.75},
                    },
                },
            },
        )

    cfg = JevConfig(
        base_url="https://opencode.ai/inference/go/systemone/v1/systemone",
        api_key="sk-jev-secret",
        model="jev-1.14",
        timeout_sec=1.5,
        reply_threshold=0.65,
        high_effort_threshold=0.60,
    )
    state = {
        "chat_type": "group",
        "bot_name": "Spectra",
        "recent_messages": ["[Alice]: 大家晚上好"],
        "latest_batch": ["[Bob]: 帮我查一下今天 Python 3.14 发布了没"],
    }

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = JevClient(cfg, http_client=http)
        decision = await client.evaluate(state, include_should_reply=True)

    assert len(captured) == 1
    req, body = captured[0]
    assert req.method == "POST"
    assert str(req.url) == "https://opencode.ai/inference/go/systemone/v1/systemone"
    assert req.headers["Authorization"] == "Bearer sk-jev-secret"
    assert req.headers["Content-Type"] == "application/json"

    assert body["model"] == "jev-1.14"
    assert body["state"] == state
    assert list(body["questions"].keys()) == ["should_reply", "effort"]

    q_reply = body["questions"]["should_reply"]
    assert q_reply["type"] == "noul"
    assert "recent_messages" in q_reply["instructions"]
    assert "bot_name" in q_reply["instructions"]
    assert "latest_batch" in q_reply["instructions"]
    assert list(q_reply["criteria"].keys()) == ["true", "false"]

    q_effort = body["questions"]["effort"]
    assert q_effort["type"] == "choice"
    assert list(q_effort["criteria"].keys()) == ["default", "high"]
    assert "联网搜索" in q_effort["criteria"]["high"]
    assert "调用工具" in q_effort["criteria"]["high"]

    assert decision.should_reply is True
    assert decision.effort == "high"
    assert decision.reply_probability == pytest.approx(0.78)
    assert decision.high_effort_probability == pytest.approx(0.75)
    assert decision.model == "jev-1.14"
    assert decision.input_tokens == 128
    assert decision.degraded is False


async def test_wire_payload_without_should_reply() -> None:
    captured_bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_bodies.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "model": "jev-1.14",
                "usage": {"input_tokens": 64},
                "answers": {
                    "effort": {
                        "probabilities": {"default": 0.85, "high": 0.15},
                    }
                },
            },
        )

    cfg = JevConfig(base_url="https://api.typesafe.ai/v1", api_key="sk-test")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = JevClient(cfg, http_client=http)
        decision = await client.evaluate(
            {"bot_name": "Spectra", "recent_messages": [], "latest_batch": ["hi"]},
            include_should_reply=False,
        )

    assert len(captured_bodies) == 1
    assert list(captured_bodies[0]["questions"].keys()) == ["effort"]
    assert decision.should_reply is True
    assert decision.reply_probability is None
    assert decision.effort == "default"
    assert decision.high_effort_probability == pytest.approx(0.15)
    assert decision.degraded is False


@pytest.mark.parametrize(
    ("reply_prob", "high_prob", "expected_reply", "expected_effort"),
    [
        (0.65, 0.60, True, "high"),
        (0.6499, 0.5999, False, "default"),
        (1.5, -0.2, True, "default"),
    ],
)
async def test_threshold_boundaries_and_clamping(
    reply_prob: float,
    high_prob: float,
    expected_reply: bool,
    expected_effort: str,
) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "answers": {
                    "should_reply": {"noul": reply_prob},
                    "effort": {"probabilities": {"default": 0.5, "high": high_prob}},
                }
            },
        )

    cfg = JevConfig(
        api_key="sk-test",
        reply_threshold=0.65,
        high_effort_threshold=0.60,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = JevClient(cfg, http_client=http)
        decision = await client.evaluate({}, include_should_reply=True)

    assert decision.degraded is False
    assert decision.should_reply is expected_reply
    assert decision.effort == expected_effort
    assert decision.reply_probability == pytest.approx(max(0.0, min(1.0, reply_prob)))
    assert decision.high_effort_probability == pytest.approx(
        max(0.0, min(1.0, high_prob))
    )


async def test_alternative_answer_formats() -> None:
    responses = iter(
        [
            # probability key + choice/confidence above threshold
            {
                "answers": {
                    "should_reply": {"probability": 0.70},
                    "effort": {"choice": "high", "confidence": 0.65},
                }
            },
            # bare numeric should_reply + choice/confidence below threshold
            {
                "answers": {
                    "should_reply": 0.80,
                    "effort": {"choice": "high", "confidence": 0.50},
                }
            },
            # choice="high" with no probabilities or confidence
            {
                "answers": {
                    "should_reply": {"noul": 0.90},
                    "effort": {"choice": "high"},
                }
            },
            # choice="default" with confidence
            {
                "answers": {
                    "should_reply": {"noul": 0.90},
                    "effort": {"choice": "default", "confidence": 0.95},
                }
            },
        ]
    )

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=next(responses))

    cfg = JevConfig(api_key="sk-test")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = JevClient(cfg, http_client=http)
        d1 = await client.evaluate({}, include_should_reply=True)
        d2 = await client.evaluate({}, include_should_reply=True)
        d3 = await client.evaluate({}, include_should_reply=True)
        d4 = await client.evaluate({}, include_should_reply=True)

    assert (d1.should_reply, d1.effort, d1.degraded) == (True, "high", False)
    assert (d2.should_reply, d2.effort, d2.degraded) == (True, "default", False)
    assert (d3.should_reply, d3.effort, d3.degraded) == (True, "high", False)
    assert (d4.should_reply, d4.effort, d4.degraded) == (True, "default", False)


async def test_unconfigured_client_degrades_without_http_call() -> None:
    called = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal called
        called += 1
        return httpx.Response(200, json={})

    cfg = JevConfig(api_key="")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = JevClient(cfg, http_client=http)
        d_group = await client.evaluate({}, include_should_reply=True)
        d_direct = await client.evaluate({}, include_should_reply=False)

    assert called == 0
    assert (d_group.should_reply, d_group.effort, d_group.degraded) == (
        False,
        "default",
        True,
    )
    assert (d_direct.should_reply, d_direct.effort, d_direct.degraded) == (
        True,
        "default",
        True,
    )


async def test_timeout_no_retries_and_graceful_degradation() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("timed out", request=request)

    cfg = JevConfig(api_key="sk-test", timeout_sec=0.5)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = JevClient(cfg, http_client=http)
        d_group = await client.evaluate({}, include_should_reply=True)
        d_direct = await client.evaluate({}, include_should_reply=False)

    assert calls == 2  # single-shot per evaluate(), zero retries
    assert (d_group.should_reply, d_group.effort, d_group.degraded) == (
        False,
        "default",
        True,
    )
    assert (d_direct.should_reply, d_direct.effort, d_direct.degraded) == (
        True,
        "default",
        True,
    )


@pytest.mark.parametrize(
    ("status_code", "raw_body"),
    [
        (500, b'{"error": "boom"}'),
        (429, b"rate limited"),
        (200, b"not-json"),
        (200, b"[]"),
        (200, b"{}"),
        (200, b'{"answers": {"effort": {"probabilities": {"high": 0.8}}}}'),
        (
            200,
            b'{"answers": {"should_reply": {"noul": true}, '
            b'"effort": {"probabilities": {"high": 0.2}}}}',
        ),
        (200, b'{"answers": {"should_reply": {"noul": 0.8}}}'),
        (
            200,
            b'{"answers": {"should_reply": {"noul": 0.8}, '
            b'"effort": {"probabilities": {"default": 0.2}}}}',
        ),
        (
            200,
            b'{"answers": {"should_reply": {"noul": 0.8}, '
            b'"effort": {"choice": "ultra"}}}',
        ),
        (
            200,
            b'{"answers": {"should_reply": {"noul": NaN}, '
            b'"effort": {"probabilities": {"high": 0.2}}}}',
        ),
        (
            200,
            b'{"answers": {"should_reply": {"noul": 0.8}, '
            b'"effort": {"probabilities": {"high": Infinity}}}}',
        ),
    ],
)
async def test_http_and_schema_errors_degrade_safely(
    status_code: int,
    raw_body: bytes,
) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, content=raw_body)

    cfg = JevConfig(api_key="sk-test")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = JevClient(cfg, http_client=http)
        decision = await client.evaluate({}, include_should_reply=True)

    assert decision.degraded is True
    assert decision.should_reply is False
    assert decision.effort == "default"


async def test_cancelled_error_propagates() -> None:
    started = asyncio.Event()

    async def handler(_: httpx.Request) -> httpx.Response:
        started.set()
        await asyncio.sleep(10)
        return httpx.Response(200, json={})

    cfg = JevConfig(api_key="sk-test")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = JevClient(cfg, http_client=http)
        task = asyncio.create_task(client.evaluate({}, include_should_reply=True))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_close_is_idempotent_and_respects_ownership() -> None:
    # Owned client is created lazily and closed idempotently.
    cfg = JevConfig(api_key="sk-test")
    client = JevClient(cfg)
    owned_http = client._get_http_client()
    assert owned_http.is_closed is False
    await client.close()
    assert owned_http.is_closed is True
    await client.close()  # idempotent
    # Calling evaluate after close degrades without opening a new client.
    degraded = await client.evaluate({}, include_should_reply=True)
    assert degraded.degraded is True
    assert client._http_client is None

    # Injected client is NOT closed by JevClient.close().
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json={}))
    ) as injected:
        async with JevClient(cfg, http_client=injected) as wrapped:
            assert wrapped.endpoint == cfg.base_url
        assert injected.is_closed is False
