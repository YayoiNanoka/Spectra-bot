"""Async System One / OpenAI Decisions router client.

Sends a single-shot HTTP POST to a System One or OpenAI Decisions compatible
endpoint with structured ``should_reply`` / ``effort`` or memory reranking
questions and degrades safely on any network, timeout, refusal, or schema error
so online chat never stalls.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Self

import httpx

from spectra.config import (
    DEFAULT_JEV_CONTINUITY_FALSE_CRITERIA,
    DEFAULT_JEV_CONTINUITY_INSTRUCTIONS,
    DEFAULT_JEV_CONTINUITY_TRUE_CRITERIA,
    DEFAULT_JEV_EFFORT_DEFAULT_CRITERIA,
    DEFAULT_JEV_EFFORT_HIGH_CRITERIA,
    DEFAULT_JEV_EFFORT_INSTRUCTIONS,
    DEFAULT_JEV_REPLY_FALSE_CRITERIA,
    DEFAULT_JEV_REPLY_INSTRUCTIONS,
    DEFAULT_JEV_REPLY_TRUE_CRITERIA,
    DEFAULT_JEV_RERANK_FALSE_CRITERIA,
    DEFAULT_JEV_RERANK_INSTRUCTIONS,
    DEFAULT_JEV_RERANK_TRUE_CRITERIA,
    JevConfig,
    JevPromptsConfig,
)
from spectra.providers import (
    DecisionEndpointProfile,
    DecisionProtocol,
    build_decision_payload,
    normalize_decision_answers,
    resolve_decision_endpoint,
    resolve_decision_profile,
)

logger = logging.getLogger(__name__)

EffortLevel = Literal["default", "high"]


def resolve_jev_endpoint(base_url: str) -> str:
    """Resolve the full decision POST endpoint URL from ``base_url``.

    Delegates to :func:`spectra.providers.resolve_decision_endpoint` using
    automatic gateway/protocol detection.
    """
    return resolve_decision_endpoint(base_url)


def _prompt_or_default(value: str | None, default: str) -> str:
    if value is None:
        return default
    cleaned = value.strip()
    return cleaned or default


def _build_questions(
    *,
    include_should_reply: bool,
    prompts: JevPromptsConfig | None = None,
) -> dict[str, Any]:
    questions: dict[str, Any] = {}
    if include_should_reply:
        questions["should_reply"] = {
            "type": "noul",
            "instructions": _prompt_or_default(
                prompts.reply_instructions if prompts is not None else None,
                DEFAULT_JEV_REPLY_INSTRUCTIONS,
            ),
            "criteria": {
                "true": _prompt_or_default(
                    prompts.reply_true_criteria if prompts is not None else None,
                    DEFAULT_JEV_REPLY_TRUE_CRITERIA,
                ),
                "false": _prompt_or_default(
                    prompts.reply_false_criteria if prompts is not None else None,
                    DEFAULT_JEV_REPLY_FALSE_CRITERIA,
                ),
            },
        }
    questions["effort"] = {
        "type": "choice",
        "instructions": _prompt_or_default(
            prompts.effort_instructions if prompts is not None else None,
            DEFAULT_JEV_EFFORT_INSTRUCTIONS,
        ),
        "criteria": {
            "default": _prompt_or_default(
                prompts.effort_default_criteria if prompts is not None else None,
                DEFAULT_JEV_EFFORT_DEFAULT_CRITERIA,
            ),
            "high": _prompt_or_default(
                prompts.effort_high_criteria if prompts is not None else None,
                DEFAULT_JEV_EFFORT_HIGH_CRITERIA,
            ),
        },
    }
    return questions


def _build_rerank_questions(
    count: int,
    *,
    prompts: JevPromptsConfig | None = None,
) -> dict[str, dict[str, Any]]:
    template = _prompt_or_default(
        prompts.rerank_instructions if prompts is not None else None,
        DEFAULT_JEV_RERANK_INSTRUCTIONS,
    )
    true_criteria = _prompt_or_default(
        prompts.rerank_true_criteria if prompts is not None else None,
        DEFAULT_JEV_RERANK_TRUE_CRITERIA,
    )
    false_criteria = _prompt_or_default(
        prompts.rerank_false_criteria if prompts is not None else None,
        DEFAULT_JEV_RERANK_FALSE_CRITERIA,
    )
    return {
        f"m{i}": {
            "type": "noul",
            "instructions": template.replace("{index}", str(i)),
            "criteria": {
                "true": true_criteria,
                "false": false_criteria,
            },
        }
        for i in range(count)
    }


def _build_continuity_questions(
    prompts: JevPromptsConfig | None = None,
) -> dict[str, dict[str, Any]]:
    return {
        "should_continue": {
            "type": "noul",
            "instructions": _prompt_or_default(
                prompts.continuity_instructions if prompts is not None else None,
                DEFAULT_JEV_CONTINUITY_INSTRUCTIONS,
            ),
            "criteria": {
                "true": _prompt_or_default(
                    prompts.continuity_true_criteria if prompts is not None else None,
                    DEFAULT_JEV_CONTINUITY_TRUE_CRITERIA,
                ),
                "false": _prompt_or_default(
                    prompts.continuity_false_criteria if prompts is not None else None,
                    DEFAULT_JEV_CONTINUITY_FALSE_CRITERIA,
                ),
            },
        }
    }


def _parse_probability(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    prob = float(value)
    if not math.isfinite(prob):
        return None
    return max(0.0, min(1.0, prob))


def _parse_predicate_answer(
    raw_answer: Any,
    *,
    protocol: DecisionProtocol,
) -> float | None:
    """Extract a clamped probability from a predicate/noul decision answer."""
    if protocol == "openai_decision":
        if not isinstance(raw_answer, Mapping) or raw_answer.get("type") != "predicate":
            return None
        return _parse_probability(raw_answer.get("probability"))

    if isinstance(raw_answer, Mapping):
        if "noul" in raw_answer:
            return _parse_probability(raw_answer["noul"])
        if "probability" in raw_answer:
            return _parse_probability(raw_answer["probability"])
        return None
    return _parse_probability(raw_answer)


def _degraded_decision(*, include_should_reply: bool) -> JevDecision:
    return JevDecision(
        should_reply=not include_should_reply,
        effort="default",
        degraded=True,
    )


def _degraded_rerank() -> JevRerankResult:
    return JevRerankResult(ranked_indices=(), scores={}, degraded=True)


def _degraded_continuity() -> JevContinuityResult:
    return JevContinuityResult(should_continue=False, degraded=True)


@dataclass(slots=True, frozen=True)
class _NormalizedDecisionResponse:
    """Internal normalized decision response payload."""

    answers: dict[str, Any]
    model: str
    input_tokens: int


@dataclass(slots=True, frozen=True)
class JevDecision:
    """Normalized outcome of a single System One / Jev evaluation call."""

    should_reply: bool
    effort: EffortLevel
    reply_probability: float | None = None
    high_effort_probability: float | None = None
    model: str = ""
    input_tokens: int = 0
    degraded: bool = False


@dataclass(slots=True, frozen=True)
class JevRerankResult:
    """Normalized outcome of a single-shot Jev memory reranking call."""

    ranked_indices: tuple[int, ...]
    scores: dict[int, float]
    model: str = ""
    input_tokens: int = 0
    degraded: bool = False


@dataclass(slots=True, frozen=True)
class JevContinuityResult:
    """Normalized outcome of a single-shot Jev episode continuity check."""

    should_continue: bool
    probability: float | None = None
    model: str = ""
    input_tokens: int = 0
    degraded: bool = False


class JevClient:
    """Async HTTP client for System One / Jev decision routing and reranking.

    Args:
        config: Dedicated :class:`spectra.config.JevConfig` instance.
        http_client: Optional externally managed :class:`httpx.AsyncClient`
            (e.g. for tests using :class:`httpx.MockTransport`). When omitted,
            a persistent client is created lazily for HTTP Keep-Alive reuse and
            closed by :meth:`close`.
    """

    def __init__(
        self,
        config: JevConfig,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._config = config
        self._http_client = http_client
        self._owns_client = http_client is None
        self._closed = False
        self._profile: DecisionEndpointProfile = resolve_decision_profile(config)
        self._endpoint = self._profile.endpoint_url

    @property
    def endpoint(self) -> str:
        """Return the resolved decision POST endpoint URL."""
        return self._endpoint

    @property
    def protocol(self) -> DecisionProtocol:
        """Return the resolved wire protocol (``system_one`` or ``openai_decision``)."""
        return self._profile.protocol

    @property
    def profile(self) -> DecisionEndpointProfile:
        """Return the full resolved :class:`DecisionEndpointProfile`."""
        return self._profile

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

    async def _post_decision(
        self,
        *,
        state: Mapping[str, Any],
        questions: Mapping[str, Mapping[str, Any]],
        operation: str,
    ) -> _NormalizedDecisionResponse | None:
        if self._closed:
            logger.warning("JevClient.%s called after close(); degrading.", operation)
            return None

        if not self._config.is_configured or not self._endpoint:
            logger.warning("JevClient is not configured; degrading %s.", operation)
            return None

        headers = dict(self._profile.headers)
        if "content-type" not in {k.lower() for k in headers}:
            headers["Content-Type"] = "application/json"

        client = self._get_http_client()
        try:
            payload = build_decision_payload(
                self._profile,
                state=state,
                questions=questions,
            )
            body_bytes = json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            response = await client.post(
                self._endpoint,
                content=body_bytes,
                headers=headers,
                timeout=self._config.timeout_sec,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Jev %s request failed (%s); degrading.", operation, exc)
            return None

        if response.status_code < 200 or response.status_code >= 300:
            logger.warning(
                "Jev %s returned HTTP %d; degrading.",
                operation,
                response.status_code,
            )
            return None

        try:
            data = response.json()
        except Exception as exc:
            logger.warning(
                "Jev %s returned invalid JSON (%s); degrading.",
                operation,
                exc,
            )
            return None

        if not isinstance(data, Mapping):
            logger.warning(
                "Jev %s response root is not a JSON object; degrading.", operation
            )
            return None

        answers = normalize_decision_answers(data, protocol=self._profile.protocol)
        if answers is None:
            logger.warning(
                "Jev %s response (%s) missing or invalid 'answers'; degrading.",
                operation,
                self._profile.protocol,
            )
            return None

        raw_model = data.get("model")
        model = raw_model if isinstance(raw_model, str) else ""

        input_tokens = 0
        usage = data.get("usage")
        if isinstance(usage, Mapping):
            raw_tokens = usage.get("input_tokens")
            if raw_tokens is None:
                raw_tokens = usage.get("prompt_tokens")
            if (
                isinstance(raw_tokens, int)
                and not isinstance(raw_tokens, bool)
                and raw_tokens >= 0
            ):
                input_tokens = raw_tokens

        return _NormalizedDecisionResponse(
            answers=answers,
            model=model,
            input_tokens=input_tokens,
        )

    async def evaluate(
        self,
        state: Mapping[str, Any],
        *,
        include_should_reply: bool,
    ) -> JevDecision:
        """Evaluate chat context via Jev with single-shot timeout and safe fallback.

        Args:
            state: Structured conversation state dictionary containing
                ``bot_name``, ``recent_messages``, and ``latest_batch``.
            include_should_reply: When ``True``, asks both ``should_reply``
                (``noul``) and ``effort`` (``choice``). When ``False``, asks
                only ``effort`` and returns ``should_reply=True``.

        Returns:
            Normalized :class:`JevDecision`. Never raises on network, timeout,
            HTTP, or payload errors (``asyncio.CancelledError`` still
            propagates).
        """
        normalized = await self._post_decision(
            state=state,
            questions=_build_questions(
                include_should_reply=include_should_reply,
                prompts=self._config.prompts,
            ),
            operation="evaluate",
        )
        if normalized is None:
            return _degraded_decision(include_should_reply=include_should_reply)
        return self._parse_evaluate_answers(
            normalized,
            include_should_reply=include_should_reply,
        )

    async def rerank_memories(
        self,
        *,
        query: str,
        candidates: Sequence[str],
        threshold: float = 0.50,
    ) -> JevRerankResult:
        """Score candidate memories against ``query`` in a single Jev decision call.

        Args:
            query: Current user message or retrieval query.
            candidates: Candidate memory texts to evaluate in parallel.
            threshold: Minimum probability in ``[0.0, 1.0]`` required for an
                index to appear in ``ranked_indices``.

        Returns:
            :class:`JevRerankResult` with ``ranked_indices`` sorted by
            ``(-score, index)``. Never raises on network, timeout, HTTP, or
            response schema errors (``asyncio.CancelledError`` always
            propagates).

        Raises:
            ValueError: If ``threshold`` is boolean, non-numeric, non-finite,
                or outside ``[0.0, 1.0]``.
        """
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, int | float)
            or not math.isfinite(float(threshold))
            or not (0.0 <= float(threshold) <= 1.0)
        ):
            raise ValueError(
                f"threshold must be a finite float in [0.0, 1.0], got {threshold!r}."
            )
        threshold_val = float(threshold)

        cleaned_query = query.strip()
        if not cleaned_query or not candidates:
            return JevRerankResult(ranked_indices=(), scores={}, degraded=False)

        candidate_list = [str(c) for c in candidates]
        state = {"query": cleaned_query, "candidates": candidate_list}
        questions = _build_rerank_questions(
            len(candidate_list),
            prompts=self._config.prompts,
        )

        normalized = await self._post_decision(
            state=state,
            questions=questions,
            operation="rerank_memories",
        )
        if normalized is None:
            return _degraded_rerank()

        scores: dict[int, float] = {}
        for idx in range(len(candidate_list)):
            key = f"m{idx}"
            if key not in normalized.answers:
                logger.warning(
                    "Jev rerank response missing %r answer; degrading.",
                    key,
                )
                return _degraded_rerank()
            raw_answer = normalized.answers[key]
            prob = _parse_predicate_answer(
                raw_answer,
                protocol=self._profile.protocol,
            )
            if prob is None:
                logger.warning(
                    "Jev rerank response has invalid %r value %r; degrading.",
                    key,
                    raw_answer,
                )
                return _degraded_rerank()
            scores[idx] = prob

        ranked = tuple(
            sorted(
                (i for i, score in scores.items() if score >= threshold_val),
                key=lambda i: (-scores[i], i),
            )
        )
        return JevRerankResult(
            ranked_indices=ranked,
            scores=scores,
            model=normalized.model,
            input_tokens=normalized.input_tokens,
            degraded=False,
        )

    async def evaluate_continuity(
        self,
        state: Mapping[str, Any],
        *,
        threshold: float = 0.55,
    ) -> JevContinuityResult:
        """Evaluate whether ``new_messages`` continues the previous episode.

        Args:
            state: Structured continuity state dictionary containing
                ``chat_type``, ``idle_minutes``, ``previous_tail``, and
                ``new_messages``.
            threshold: Minimum probability in ``[0.0, 1.0]`` required to
                continue the current episode.

        Returns:
            Normalized :class:`JevContinuityResult`. Never raises on network,
            timeout, HTTP, or response schema errors (``asyncio.CancelledError``
            always propagates).

        Raises:
            ValueError: If ``threshold`` is boolean, non-numeric, non-finite,
                or outside ``[0.0, 1.0]``.
        """
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, int | float)
            or not math.isfinite(float(threshold))
            or not (0.0 <= float(threshold) <= 1.0)
        ):
            raise ValueError(
                f"threshold must be a finite float in [0.0, 1.0], got {threshold!r}."
            )
        threshold_val = float(threshold)

        normalized = await self._post_decision(
            state=state,
            questions=_build_continuity_questions(prompts=self._config.prompts),
            operation="evaluate_continuity",
        )
        if normalized is None:
            return _degraded_continuity()

        if "should_continue" not in normalized.answers:
            logger.warning(
                "Jev continuity response missing 'should_continue' answer; degrading."
            )
            return _degraded_continuity()

        raw_answer = normalized.answers["should_continue"]
        prob = _parse_predicate_answer(
            raw_answer,
            protocol=self._profile.protocol,
        )
        if prob is None:
            logger.warning(
                "Jev continuity response has invalid 'should_continue' value %r; "
                "degrading.",
                raw_answer,
            )
            return _degraded_continuity()

        return JevContinuityResult(
            should_continue=prob >= threshold_val,
            probability=prob,
            model=normalized.model,
            input_tokens=normalized.input_tokens,
            degraded=False,
        )

    def _parse_evaluate_answers(
        self,
        normalized: _NormalizedDecisionResponse,
        *,
        include_should_reply: bool,
    ) -> JevDecision:
        answers = normalized.answers
        reply_probability: float | None = None
        if include_should_reply:
            if "should_reply" not in answers:
                logger.warning("Jev response missing 'should_reply' answer; degrading.")
                return _degraded_decision(include_should_reply=True)
            raw_sr = answers["should_reply"]
            reply_probability = _parse_predicate_answer(
                raw_sr,
                protocol=self._profile.protocol,
            )
            if reply_probability is None:
                logger.warning(
                    "Jev response has invalid 'should_reply' value %r; degrading.",
                    raw_sr,
                )
                return _degraded_decision(include_should_reply=True)
            should_reply = reply_probability >= self._config.reply_threshold
        else:
            should_reply = True

        if "effort" not in answers:
            logger.warning("Jev response missing 'effort' answer; degrading.")
            return _degraded_decision(include_should_reply=include_should_reply)

        raw_effort = answers["effort"]
        if self._profile.protocol == "openai_decision":
            if (
                not isinstance(raw_effort, Mapping)
                or raw_effort.get("type") != "choice"
                or raw_effort.get("choice") not in ("default", "high")
                or not isinstance(raw_effort.get("probabilities"), Mapping)
                or set(raw_effort["probabilities"]) != {"default", "high"}
            ):
                logger.warning(
                    "OpenAI decision 'effort' answer is malformed %r; degrading.",
                    raw_effort,
                )
                return _degraded_decision(include_should_reply=include_should_reply)

        parsed_effort = self._parse_effort(raw_effort)
        if parsed_effort is None:
            logger.warning(
                "Jev response has invalid 'effort' value %r; degrading.",
                raw_effort,
            )
            return _degraded_decision(include_should_reply=include_should_reply)

        effort, high_effort_probability = parsed_effort
        return JevDecision(
            should_reply=should_reply,
            effort=effort,
            reply_probability=reply_probability,
            high_effort_probability=high_effort_probability,
            model=normalized.model,
            input_tokens=normalized.input_tokens,
            degraded=False,
        )

    def _parse_effort(
        self,
        raw_effort: Any,
    ) -> tuple[EffortLevel, float | None] | None:
        if not isinstance(raw_effort, Mapping):
            return None

        probabilities = raw_effort.get("probabilities")
        if probabilities is not None:
            if not isinstance(probabilities, Mapping):
                return None
            high_prob = _parse_probability(probabilities.get("high"))
            if high_prob is None:
                return None
            effort: EffortLevel = (
                "high" if high_prob >= self._config.high_effort_threshold else "default"
            )
            return effort, high_prob

        choice = raw_effort.get("choice")
        if choice not in ("default", "high"):
            return None

        raw_confidence = raw_effort.get("confidence")
        if raw_confidence is not None:
            confidence = _parse_probability(raw_confidence)
            if confidence is None:
                return None
            if choice == "high":
                effort = (
                    "high"
                    if confidence >= self._config.high_effort_threshold
                    else "default"
                )
                return effort, confidence
            return "default", None

        return ("high" if choice == "high" else "default"), None
