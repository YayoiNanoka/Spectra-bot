"""Pure-stdlib float32 vector codec and async OpenAI-compatible embedding client.

Provides compact L2-normalized little-endian ``float32`` serialization, cosine
similarity helpers, and :class:`EmbeddingClient` with batching, retry/jitter,
API-key redaction, and safe degradation on transient or schema failures.
"""

from __future__ import annotations

import asyncio
import copy
import functools
import json
import logging
import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Self

import httpx
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)
from tenacity.wait import wait_base

from spectra.config import EmbeddingConfig
from spectra.providers import (
    EmbeddingEndpointProfile,
    resolve_embedding_profile,
    should_send_embedding_dimensions,
)

logger = logging.getLogger(__name__)

_FLOAT32_BYTES = 4
_MIN_VECTOR_NORM = 1e-12
_RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({408, 409, 429})
_ERROR_BODY_PREVIEW_CHARS = 200


@functools.lru_cache(maxsize=64)
def _float32_struct(dimension: int) -> struct.Struct:
    return struct.Struct(f"<{dimension}f")


def format_embedding_model_tag(model: str, dimension: int) -> str:
    """Format a canonical ``<model>:<dimension>`` tag for persisted embeddings.

    Args:
        model: Non-empty embedding model identifier.
        dimension: Positive integer vector dimension.

    Returns:
        Canonical model tag string in ``"{model}:{dimension}"`` format.

    Raises:
        ValueError: If ``model`` is blank or ``dimension`` is not a positive int.
    """
    if not isinstance(model, str) or not model.strip():
        raise ValueError("Embedding model name must be a non-empty string.")
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
        raise ValueError(
            f"Embedding dimension must be an integer >= 1, got {dimension!r}."
        )
    return f"{model.strip()}:{dimension}"


def pack_normalized_vector(values: Sequence[float]) -> tuple[bytes, int]:
    """L2-normalize a real vector and pack it as little-endian ``float32`` bytes.

    Args:
        values: Non-empty sequence of finite real numbers.

    Returns:
        Tuple of ``(blob, dimension)`` where ``blob`` has length ``4 * dimension``.

    Raises:
        TypeError: If ``values`` is not a numeric sequence.
        ValueError: If ``values`` is empty, contains booleans/non-numbers/non-finite
            values, has near-zero norm, or overflows ``float32``.
    """
    if isinstance(values, (str, bytes, bytearray, memoryview)) or not isinstance(
        values, Sequence
    ):
        raise TypeError(
            f"Embedding vector must be a sequence of numbers, "
            f"got {type(values).__name__}."
        )
    dim = len(values)
    if dim == 0:
        raise ValueError("Embedding vector must not be empty.")

    floats: list[float] = []
    for idx, raw in enumerate(values):
        if isinstance(raw, bool) or not isinstance(raw, int | float):
            raise ValueError(
                f"Embedding vector element at index {idx} must be a real number, "
                f"got {type(raw).__name__}."
            )
        fval = float(raw)
        if not math.isfinite(fval):
            raise ValueError(f"Embedding vector element at index {idx} must be finite.")
        floats.append(fval)

    norm = math.hypot(*floats)
    if not math.isfinite(norm) or norm <= _MIN_VECTOR_NORM:
        raise ValueError(
            f"Embedding vector norm is invalid or too close to zero: {norm!r}."
        )

    normalized = [v / norm for v in floats]
    packer = _float32_struct(dim)
    try:
        blob = packer.pack(*normalized)
    except OverflowError as exc:
        raise ValueError("Embedding vector overflowed float32 packing.") from exc

    unpacked = packer.unpack(blob)
    if not all(math.isfinite(v) for v in unpacked):
        raise ValueError("Packed float32 embedding vector contains non-finite values.")
    return blob, dim


def unpack_vector(blob: bytes | bytearray | memoryview) -> tuple[float, ...]:
    """Unpack a little-endian ``float32`` byte blob into a tuple of floats.

    Args:
        blob: Non-empty bytes-like object with byte length divisible by 4.

    Returns:
        Tuple of unpacked finite floats.

    Raises:
        TypeError: If ``blob`` is not bytes-like.
        ValueError: If byte length is zero, not a multiple of 4, or contains
            non-finite float32 values.
    """
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        raise TypeError(f"Vector blob must be bytes-like, got {type(blob).__name__}.")
    raw = blob.tobytes() if isinstance(blob, memoryview) else blob
    byte_len = len(raw)
    if byte_len == 0 or byte_len % _FLOAT32_BYTES != 0:
        raise ValueError(
            f"Vector blob byte length must be a positive multiple of 4, got {byte_len}."
        )
    dim = byte_len // _FLOAT32_BYTES
    values = _float32_struct(dim).unpack(raw)
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Vector blob contains non-finite float32 values.")
    return values


def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Compute cosine similarity between two equal-length L2-normalized vectors.

    Computes the inner product via :func:`math.sumprod` and clamps the result
    into ``[-1.0, 1.0]`` to absorb minor ``float32`` rounding drift.

    Args:
        vec_a: First normalized float sequence.
        vec_b: Second normalized float sequence of the same length.

    Returns:
        Clamped similarity score in ``[-1.0, 1.0]``.

    Raises:
        TypeError: If either input is not a sequence.
        ValueError: If lengths are zero or mismatched, elements are invalid or
            non-finite, or the dot product is non-finite.
    """
    if (
        isinstance(vec_a, (str, bytes, bytearray, memoryview))
        or isinstance(vec_b, (str, bytes, bytearray, memoryview))
        or not isinstance(vec_a, Sequence)
        or not isinstance(vec_b, Sequence)
    ):
        raise TypeError("Vectors must be sequences of real numbers.")
    if not vec_a or len(vec_a) != len(vec_b):
        raise ValueError(
            "Vectors must have equal non-zero lengths, "
            f"got {len(vec_a)} and {len(vec_b)}."
        )

    for idx, (a, b) in enumerate(zip(vec_a, vec_b, strict=True)):
        if type(a) is not float and (
            isinstance(a, bool) or not isinstance(a, int | float)
        ):
            raise ValueError(
                f"Vector element at index {idx} must be a real number, "
                f"got {type(a).__name__}."
            )
        if type(b) is not float and (
            isinstance(b, bool) or not isinstance(b, int | float)
        ):
            raise ValueError(
                f"Vector element at index {idx} must be a real number, "
                f"got {type(b).__name__}."
            )
        if not (math.isfinite(a) and math.isfinite(b)):
            raise ValueError(f"Vector elements at index {idx} must be finite.")

    dot = float(math.sumprod(vec_a, vec_b))
    if not math.isfinite(dot):
        raise ValueError("Cosine similarity result is not finite.")
    return max(-1.0, min(1.0, dot))


@dataclass(slots=True, frozen=True)
class EmbeddingResult:
    """Normalized float32 embedding blob and metadata for a single input text."""

    blob: bytes
    dimension: int
    model_tag: str


class _RetryableEmbeddingError(Exception):
    """Internal retryable error carrying an already-redacted message."""


class _NonRetryableEmbeddingError(Exception):
    """Internal fail-fast error carrying an already-redacted message."""


class EmbeddingClient:
    """Async OpenAI-compatible text embedding client with batching and retries.

    Args:
        config: Dedicated :class:`spectra.config.EmbeddingConfig` instance.
        http_client: Optional externally managed :class:`httpx.AsyncClient`.
            When omitted, an owned client is created lazily and closed by
            :meth:`close`.
        retry_wait: Optional tenacity wait strategy override (useful in tests).
    """

    def __init__(
        self,
        config: EmbeddingConfig,
        http_client: httpx.AsyncClient | None = None,
        *,
        retry_wait: wait_base | None = None,
    ) -> None:
        self._config = config
        self._http_client = http_client
        self._owns_client = http_client is None
        self._closed = False
        self._profile: EmbeddingEndpointProfile = resolve_embedding_profile(config)
        self._endpoint = self._profile.endpoint_url
        self._api_key = config.api_key.strip()
        self._retry_wait = retry_wait or wait_exponential_jitter(
            multiplier=0.2,
            max=10.0,
        )

    @property
    def profile(self) -> EmbeddingEndpointProfile:
        """Return the resolved :class:`EmbeddingEndpointProfile`."""
        return self._profile

    @property
    def endpoint(self) -> str:
        """Return the resolved embedding POST endpoint URL."""
        return self._endpoint

    def _redact(self, text: str) -> str:
        if self._api_key and self._api_key in text:
            return text.replace(self._api_key, "[REDACTED]")
        return text

    def _get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=self._config.timeout_sec)
            self._owns_client = True
        return self._http_client

    async def close(self) -> None:
        """Close the owned HTTP client idempotently."""
        self._closed = True
        if self._owns_client and self._http_client is not None:
            client = self._http_client
            self._http_client = None
            await client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    def _log_retry(self, retry_state: RetryCallState) -> None:
        error = retry_state.outcome.exception() if retry_state.outcome else None
        detail = self._redact(str(error)) if error is not None else "UnknownError"
        logger.warning(
            "Embedding request to model %r failed with retryable error; "
            "retrying (%d/%d): %s",
            self._profile.model,
            retry_state.attempt_number,
            self._config.max_retries,
            detail,
        )

    def _build_batch_body(self, batch_texts: Sequence[str]) -> bytes:
        payload: dict[str, Any] = copy.deepcopy(self._profile.extra_body)
        payload["model"] = self._profile.model
        payload["input"] = list(batch_texts)
        if self._config.dimensions is not None and should_send_embedding_dimensions(
            self._profile.model,
            self._config.dimensions_mode,
        ):
            payload["dimensions"] = self._config.dimensions
        return json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")

    def _parse_batch_response(
        self,
        data: Any,
        expected_count: int,
    ) -> list[EmbeddingResult]:
        if not isinstance(data, Mapping):
            raise _NonRetryableEmbeddingError(
                "Embedding response root is not a JSON object."
            )
        raw_items = data.get("data")
        if not isinstance(raw_items, list) or len(raw_items) != expected_count:
            got = len(raw_items) if isinstance(raw_items, list) else type(raw_items)
            raise _NonRetryableEmbeddingError(
                f"Embedding response 'data' must be a list of length "
                f"{expected_count}, got {got!r}."
            )

        ordered: list[tuple[bytes, int] | None] = [None] * expected_count
        for entry in raw_items:
            if not isinstance(entry, Mapping):
                raise _NonRetryableEmbeddingError(
                    "Embedding response entry is not a JSON object."
                )
            idx = entry.get("index")
            if (
                isinstance(idx, bool)
                or not isinstance(idx, int)
                or idx < 0
                or idx >= expected_count
            ):
                raise _NonRetryableEmbeddingError(
                    f"Embedding response entry has invalid index {idx!r}."
                )
            if ordered[idx] is not None:
                raise _NonRetryableEmbeddingError(
                    f"Embedding response contains duplicate index {idx}."
                )
            raw_embedding = entry.get("embedding")
            try:
                blob, dim = pack_normalized_vector(raw_embedding)
            except (TypeError, ValueError) as exc:
                raise _NonRetryableEmbeddingError(
                    f"Invalid embedding vector at index {idx}: {self._redact(str(exc))}"
                ) from None
            ordered[idx] = (blob, dim)

        if any(item is None for item in ordered):
            raise _NonRetryableEmbeddingError(
                "Embedding response did not cover all expected batch indices."
            )

        packed = [item for item in ordered if item is not None]
        expected_dim = packed[0][1]
        for idx, (_, dim) in enumerate(packed):
            if dim != expected_dim:
                raise _NonRetryableEmbeddingError(
                    f"Inconsistent embedding dimensions in batch: index 0 has "
                    f"{expected_dim}, index {idx} has {dim}."
                )

        model_tag = format_embedding_model_tag(self._profile.model, expected_dim)
        return [
            EmbeddingResult(blob=blob, dimension=dim, model_tag=model_tag)
            for blob, dim in packed
        ]

    async def _send_batch_once(
        self,
        client: httpx.AsyncClient,
        body_bytes: bytes,
        headers: dict[str, str],
        expected_count: int,
    ) -> list[EmbeddingResult]:
        try:
            response = await client.post(
                self._endpoint,
                content=body_bytes,
                headers=headers,
                timeout=self._config.timeout_sec,
            )
        except asyncio.CancelledError:
            raise
        except (httpx.TransportError, TimeoutError) as exc:
            raise _RetryableEmbeddingError(
                self._redact(f"{type(exc).__name__}: {exc}")
            ) from None
        except Exception as exc:
            raise _NonRetryableEmbeddingError(
                self._redact(f"{type(exc).__name__}: {exc}")
            ) from None

        status = response.status_code
        if status < 200 or status >= 300:
            preview = " ".join(response.text[:_ERROR_BODY_PREVIEW_CHARS].split())
            detail = self._redact(
                f"HTTP {status}: {preview}" if preview else f"HTTP {status}"
            )
            if status in _RETRYABLE_STATUS_CODES or status >= 500:
                raise _RetryableEmbeddingError(detail) from None
            raise _NonRetryableEmbeddingError(detail) from None

        try:
            data = response.json()
        except Exception as exc:
            raise _NonRetryableEmbeddingError(
                self._redact(f"Invalid JSON response: {exc}")
            ) from None

        return self._parse_batch_response(data, expected_count)

    async def _embed_batch(self, batch_texts: Sequence[str]) -> list[EmbeddingResult]:
        headers = dict(self._profile.headers)
        if "content-type" not in {k.lower() for k in headers}:
            headers["Content-Type"] = "application/json"

        try:
            body_bytes = self._build_batch_body(batch_texts)
        except Exception as exc:
            raise _NonRetryableEmbeddingError(
                self._redact(f"Failed to serialize embedding request payload: {exc}")
            ) from None

        client = self._get_http_client()
        retrying = AsyncRetrying(
            retry=retry_if_exception_type(_RetryableEmbeddingError),
            stop=stop_after_attempt(self._config.max_retries + 1),
            wait=self._retry_wait,
            before_sleep=self._log_retry,
            reraise=True,
        )
        async for attempt in retrying:
            with attempt:
                return await self._send_batch_once(
                    client,
                    body_bytes,
                    headers,
                    len(batch_texts),
                )
        raise AssertionError("Unreachable: tenacity AsyncRetrying exhausted.")

    async def embed_texts(self, texts: Sequence[str]) -> list[EmbeddingResult | None]:
        """Embed a sequence of texts, preserving 1:1 input indices.

        Blank or whitespace-only inputs map directly to ``None`` without issuing
        HTTP calls. Non-blank inputs are sliced by ``config.batch_size``; any
        batch failure logs a redacted warning and leaves ``None`` at that
        batch's positions without failing other batches. ``asyncio.CancelledError``
        always propagates.

        Args:
            texts: Sequence of input text strings.

        Returns:
            List of :class:`EmbeddingResult` or ``None`` aligned 1:1 with ``texts``.
        """
        if not texts:
            return []

        results: list[EmbeddingResult | None] = [None] * len(texts)
        non_blank: list[tuple[int, str]] = [
            (idx, text)
            for idx, text in enumerate(texts)
            if isinstance(text, str) and text.strip()
        ]
        if not non_blank:
            return results

        if self._closed:
            logger.warning(
                "EmbeddingClient.embed_texts called after close(); skipping."
            )
            return results

        if not self._config.is_configured or not self._endpoint:
            logger.warning("EmbeddingClient is not configured; returning None results.")
            return results

        batch_size = self._config.batch_size
        for start in range(0, len(non_blank), batch_size):
            chunk = non_blank[start : start + batch_size]
            chunk_texts = [text for _, text in chunk]
            try:
                batch_results = await self._embed_batch(chunk_texts)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "Embedding batch (%d item(s)) failed for model %r: %s",
                    len(chunk_texts),
                    self._profile.model,
                    self._redact(str(exc)),
                )
                continue

            for (orig_idx, _), emb in zip(chunk, batch_results, strict=True):
                results[orig_idx] = emb

        return results

    async def embed_one(self, text: str) -> EmbeddingResult | None:
        """Embed a single text string, returning ``None`` on blank input or error."""
        return (await self.embed_texts([text]))[0]
