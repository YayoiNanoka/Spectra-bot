"""Unit tests for spectra.embedding (vector codec and async EmbeddingClient)."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import struct
from typing import Any

import httpx
import pytest
from tenacity import wait_none

from spectra.config import EmbeddingConfig
from spectra.embedding import (
    EmbeddingClient,
    EmbeddingResult,
    cosine_similarity,
    format_embedding_model_tag,
    pack_normalized_vector,
    unpack_vector,
)

# ---------- Vector Codec & Math Helpers ----------


def test_format_embedding_model_tag_valid_and_invalid() -> None:
    assert format_embedding_model_tag("text-embedding-v4", 1024) == (
        "text-embedding-v4:1024"
    )
    assert format_embedding_model_tag("  qwen/qwen3-embedding-8b ", 512) == (
        "qwen/qwen3-embedding-8b:512"
    )

    for bad_model in ("", "   "):
        with pytest.raises(ValueError):
            format_embedding_model_tag(bad_model, 128)
    for bad_dim in (0, -4, True, False, 1.5):
        with pytest.raises(ValueError):
            format_embedding_model_tag("text-embedding-v4", bad_dim)  # type: ignore[arg-type]


def test_pack_unpack_and_cosine_similarity_precision() -> None:
    blob_a, dim_a = pack_normalized_vector([3.0, 4.0])
    assert dim_a == 2
    assert len(blob_a) == 8

    vec_a = unpack_vector(blob_a)
    assert len(vec_a) == 2
    assert math.isclose(vec_a[0], 0.6, rel_tol=1e-6)
    assert math.isclose(vec_a[1], 0.8, rel_tol=1e-6)
    assert math.isclose(cosine_similarity(vec_a, vec_a), 1.0, abs_tol=1e-6)

    # bytearray and memoryview inputs are supported.
    assert unpack_vector(bytearray(blob_a)) == vec_a
    assert unpack_vector(memoryview(blob_a)) == vec_a

    # Opposite and orthogonal vectors
    blob_neg, _ = pack_normalized_vector([-3.0, -4.0])
    blob_orth, _ = pack_normalized_vector([-4.0, 3.0])
    vec_neg = unpack_vector(blob_neg)
    vec_orth = unpack_vector(blob_orth)
    assert math.isclose(cosine_similarity(vec_a, vec_neg), -1.0, abs_tol=1e-6)
    assert math.isclose(cosine_similarity(vec_a, vec_orth), 0.0, abs_tol=1e-6)

    # Clamping absorbs tiny >1 / <-1 rounding overshoot
    assert cosine_similarity([1.00001, 0.0], [1.0, 0.0]) == 1.0
    assert cosine_similarity([-1.00001, 0.0], [1.0, 0.0]) == -1.0


def test_vector_codec_rejects_invalid_inputs() -> None:
    # Empty sequence
    with pytest.raises(ValueError):
        pack_normalized_vector([])

    # Non-sequence / string / bytes
    with pytest.raises(TypeError):
        pack_normalized_vector("123")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        pack_normalized_vector(b"\x00\x00\x80\x3f")  # type: ignore[arg-type]

    # bool, non-numeric, NaN, Inf, zero/near-zero norm
    for bad_vec in (
        [True, 1.0],
        [False, 1.0],
        ["1.0", 2.0],
        [None, 1.0],
        [float("nan"), 1.0],
        [float("inf"), 1.0],
        [float("-inf"), 1.0],
        [0.0, 0.0],
        [1e-15, -1e-15],
    ):
        with pytest.raises(ValueError):
            pack_normalized_vector(bad_vec)  # type: ignore[arg-type]

    # unpack_vector validation
    with pytest.raises(TypeError):
        unpack_vector("not-bytes")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        unpack_vector(b"")
    with pytest.raises(ValueError):
        unpack_vector(b"\x00\x01\x02")
    with pytest.raises(ValueError):
        unpack_vector(struct.pack("<2f", 1.0, float("nan")))
    with pytest.raises(ValueError):
        unpack_vector(struct.pack("<2f", float("inf"), 0.0))

    # cosine_similarity validation
    with pytest.raises(TypeError):
        cosine_similarity("ab", "ab")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        cosine_similarity([], [])
    with pytest.raises(ValueError):
        cosine_similarity([1.0, 0.0], [1.0])
    with pytest.raises(ValueError):
        cosine_similarity([True, 0.0], [1.0, 0.0])
    with pytest.raises(ValueError):
        cosine_similarity([1.0, 0.0], [float("nan"), 0.0])
    with pytest.raises(ValueError):
        cosine_similarity([float("inf"), 0.0], [1.0, 0.0])


# ---------- EmbeddingClient Behavior ----------


def _make_embedding_response(
    vectors: list[list[float]],
    *,
    reverse: bool = False,
) -> dict[str, Any]:
    items = [
        {"object": "embedding", "index": idx, "embedding": vec}
        for idx, vec in enumerate(vectors)
    ]
    if reverse:
        items.reverse()
    return {"object": "list", "data": items}


async def test_embed_texts_batching_blank_preservation_and_out_of_order_indices() -> (
    None
):
    recorded_payloads: list[dict[str, Any]] = []
    recorded_urls: list[str] = []
    recorded_headers: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        recorded_urls.append(str(request.url))
        recorded_headers.append(dict(request.headers))
        payload = json.loads(request.content.decode("utf-8"))
        recorded_payloads.append(payload)
        inputs: list[str] = payload["input"]
        # Return vectors in reversed order to verify index-based sorting.
        vectors = [[float(len(t)), 1.0, 0.0] for t in inputs]
        return httpx.Response(
            200,
            json=_make_embedding_response(vectors, reverse=True),
        )

    transport = httpx.MockTransport(handler)
    cfg = EmbeddingConfig(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key="sk-dashscope-test",
        model="text-embedding-v4",
        dimensions=3,
        dimensions_mode="auto",
        batch_size=10,
        extra_body={"encoding_format": "float"},
    )

    # 23 total inputs: 20 non-blank + 3 blank interspersed -> 2 batches of 10.
    texts: list[str] = []
    blank_indices = {0, 11, 22}
    non_blank_counter = 0
    for i in range(23):
        if i in blank_indices:
            texts.append("   " if i == 11 else "")
        else:
            non_blank_counter += 1
            texts.append("a" * non_blank_counter)
    assert non_blank_counter == 20

    async with httpx.AsyncClient(transport=transport) as http_client:
        client = EmbeddingClient(cfg, http_client=http_client, retry_wait=wait_none())
        results = await client.embed_texts(texts)

    assert len(results) == 23
    assert len(recorded_payloads) == 2
    assert len(recorded_payloads[0]["input"]) == 10
    assert len(recorded_payloads[1]["input"]) == 10
    assert all(
        url == "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
        for url in recorded_urls
    )
    for payload in recorded_payloads:
        assert payload["model"] == "text-embedding-v4"
        assert payload["dimensions"] == 3
        assert payload["encoding_format"] == "float"
    assert recorded_headers[0]["authorization"] == "Bearer sk-dashscope-test"

    # Verify exact 1:1 alignment of blanks and sorted vectors.
    seen_non_blank = 0
    for idx, res in enumerate(results):
        if idx in blank_indices:
            assert res is None
        else:
            seen_non_blank += 1
            assert isinstance(res, EmbeddingResult)
            assert res.dimension == 3
            assert res.model_tag == "text-embedding-v4:3"
            expected_blob, _ = pack_normalized_vector([float(seen_non_blank), 1.0, 0.0])
            assert res.blob == expected_blob


async def test_embed_texts_empty_and_all_blank_make_zero_http_calls() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"data": []})

    cfg = EmbeddingConfig(model="text-embedding-v4", api_key="sk-test")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = EmbeddingClient(cfg, http_client=http_client, retry_wait=wait_none())
        assert await client.embed_texts([]) == []
        assert await client.embed_texts(["", "   ", "\n\t"]) == [None, None, None]
        assert await client.embed_one("   ") is None
    assert calls == 0


@pytest.mark.parametrize(
    ("model", "dimensions", "mode", "expect_dimensions_field"),
    [
        ("text-embedding-v4", 512, "auto", True),
        ("qwen/qwen3-embedding-8b", 1024, "auto", True),
        ("bge-m3", 512, "auto", False),
        ("bge-m3", 512, "always", True),
        ("text-embedding-v4", 512, "never", False),
        ("text-embedding-v4", None, "always", False),
    ],
)
async def test_embed_one_dimensions_payload_rules(
    model: str,
    dimensions: int | None,
    mode: str,
    expect_dimensions_field: bool,
) -> None:
    seen_payloads: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        seen_payloads.append(payload)
        return httpx.Response(200, json=_make_embedding_response([[1.0, 0.0]]))

    cfg = EmbeddingConfig(
        base_url="http://localhost:11434",
        model=model,
        dimensions=dimensions,
        dimensions_mode=mode,  # type: ignore[arg-type]
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = EmbeddingClient(cfg, http_client=http_client, retry_wait=wait_none())
        res = await client.embed_one("hello world")

    assert res is not None
    assert len(seen_payloads) == 1
    if expect_dimensions_field:
        assert seen_payloads[0]["dimensions"] == dimensions
    else:
        assert "dimensions" not in seen_payloads[0]


async def test_embed_retries_on_429_503_and_transport_error() -> None:
    attempt = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempt
        attempt += 1
        if attempt == 1:
            return httpx.Response(429, text="rate limited")
        if attempt == 2:
            raise httpx.ConnectError("transient network drop", request=request)
        if attempt == 3:
            return httpx.Response(503, text="service unavailable")
        return httpx.Response(200, json=_make_embedding_response([[0.0, 5.0]]))

    cfg = EmbeddingConfig(
        model="text-embedding-v4",
        api_key="sk-retry-test",
        max_retries=3,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = EmbeddingClient(cfg, http_client=http_client, retry_wait=wait_none())
        res = await client.embed_one("retry me")

    assert attempt == 4
    assert res is not None
    assert unpack_vector(res.blob) == pytest.approx((0.0, 1.0))


async def test_embed_max_retries_zero_makes_single_attempt() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(503, text="still down")

    cfg = EmbeddingConfig(
        model="text-embedding-v4",
        api_key="sk-zero-retry",
        max_retries=0,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = EmbeddingClient(cfg, http_client=http_client, retry_wait=wait_none())
        res = await client.embed_one("once only")

    assert attempts == 1
    assert res is None


@pytest.mark.parametrize("status_code", [400, 401])
async def test_embed_fail_fast_on_non_retryable_4xx_and_redacts_api_key(
    status_code: int,
    caplog: pytest.LogCaptureFixture,
) -> None:
    attempts = 0
    secret_key = "sk-super-secret-embedding-key-999"

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(
            status_code,
            text=f"auth/request error echoing {secret_key} in body",
        )

    cfg = EmbeddingConfig(
        model="text-embedding-v4",
        api_key=secret_key,
        max_retries=3,
    )
    with caplog.at_level(logging.WARNING, logger="spectra.embedding"):
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            client = EmbeddingClient(
                cfg,
                http_client=http_client,
                retry_wait=wait_none(),
            )
            res = await client.embed_one("bad request")

    assert attempts == 1
    assert res is None
    log_text = "\n".join(record.getMessage() for record in caplog.records)
    assert secret_key not in log_text
    assert "[REDACTED]" in log_text


async def test_embed_redacts_api_key_in_transport_error_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret_key = "sk-transport-leak-secret-123"

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(
            f"connection error with {secret_key}",
            request=request,
        )

    cfg = EmbeddingConfig(
        model="text-embedding-v4",
        api_key=secret_key,
        max_retries=1,
    )
    with caplog.at_level(logging.WARNING, logger="spectra.embedding"):
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            client = EmbeddingClient(
                cfg,
                http_client=http_client,
                retry_wait=wait_none(),
            )
            res = await client.embed_one("transport fail")

    assert res is None
    log_text = "\n".join(record.getMessage() for record in caplog.records)
    assert secret_key not in log_text
    assert "[REDACTED]" in log_text


@pytest.mark.parametrize(
    "bad_body",
    [
        # Duplicate index
        {
            "data": [
                {"index": 0, "embedding": [1.0, 0.0]},
                {"index": 0, "embedding": [0.0, 1.0]},
            ]
        },
        # Missing index / wrong count
        {"data": [{"index": 0, "embedding": [1.0, 0.0]}]},
        # Out-of-range index
        {
            "data": [
                {"index": 0, "embedding": [1.0, 0.0]},
                {"index": 2, "embedding": [0.0, 1.0]},
            ]
        },
        # Bool index
        {
            "data": [
                {"index": False, "embedding": [1.0, 0.0]},
                {"index": True, "embedding": [0.0, 1.0]},
            ]
        },
        # Mismatched dimensions within the same batch
        {
            "data": [
                {"index": 0, "embedding": [1.0, 0.0]},
                {"index": 1, "embedding": [0.0, 1.0, 0.0]},
            ]
        },
    ],
)
async def test_embed_rejects_malformed_batch_responses(
    bad_body: dict[str, Any],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=bad_body)

    cfg = EmbeddingConfig(model="text-embedding-v4", api_key="sk-test", max_retries=0)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = EmbeddingClient(cfg, http_client=http_client, retry_wait=wait_none())
        results = await client.embed_texts(["one", "two"])

    assert results == [None, None]


async def test_embed_partial_batch_failure_isolation() -> None:
    call_num = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_num
        call_num += 1
        payload = json.loads(request.content.decode("utf-8"))
        if call_num == 2:
            return httpx.Response(400, text="bad second batch")
        vectors = [[float(call_num), 1.0] for _ in payload["input"]]
        return httpx.Response(200, json=_make_embedding_response(vectors))

    cfg = EmbeddingConfig(
        model="text-embedding-v4",
        api_key="sk-test",
        batch_size=2,
        max_retries=0,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = EmbeddingClient(cfg, http_client=http_client, retry_wait=wait_none())
        results = await client.embed_texts(["t1", "t2", "t3", "t4", "t5"])

    assert call_num == 3
    assert results[0] is not None
    assert results[1] is not None
    assert results[2] is None
    assert results[3] is None
    assert results[4] is not None


async def test_embed_unconfigured_closed_and_cancelled_behavior(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # 1. Unconfigured (default empty model) degrades safely with warning.
    unconfigured = EmbeddingClient(EmbeddingConfig())
    with caplog.at_level(logging.WARNING, logger="spectra.embedding"):
        assert await unconfigured.embed_texts(["hello"]) == [None]
    await unconfigured.close()
    await unconfigured.close()  # idempotent

    # 2. Closed client degrades safely without closing externally injected client.
    calls = 0

    def ok_handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=_make_embedding_response([[1.0, 0.0]]))

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(ok_handler)
    ) as external_http:
        client = EmbeddingClient(
            EmbeddingConfig(model="text-embedding-v4"),
            http_client=external_http,
            retry_wait=wait_none(),
        )
        async with client:
            first = await client.embed_one("before close")
            assert first is not None
        assert external_http.is_closed is False
        assert await client.embed_one("after close") is None
    assert calls == 1

    # 3. Owned client is lazily created and closed via async with / close().
    async with EmbeddingClient(
        EmbeddingConfig(base_url="http://127.0.0.1:11434", model="bge-m3")
    ) as owned_client:
        owned_http = owned_client._get_http_client()
        assert owned_http.is_closed is False
    assert owned_http.is_closed is True

    # 4. asyncio.CancelledError always propagates.
    async def cancel_handler(request: httpx.Request) -> httpx.Response:
        raise asyncio.CancelledError

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(cancel_handler)
    ) as cancel_http:
        cancel_client = EmbeddingClient(
            EmbeddingConfig(model="text-embedding-v4"),
            http_client=cancel_http,
            retry_wait=wait_none(),
        )
        with pytest.raises(asyncio.CancelledError):
            await cancel_client.embed_one("cancel me")
