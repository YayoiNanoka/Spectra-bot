"""Async System One / Jev decision and reasoning-effort router client.

Sends a single-shot HTTP POST to a System One compatible endpoint with
structured ``should_reply`` (``noul``) and/or ``effort`` (``choice``)
questions and degrades safely on any network, timeout, or schema error so
online chat never stalls.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Self

import httpx

from spectra.config import JevConfig

logger = logging.getLogger(__name__)

EffortLevel = Literal["default", "high"]


def resolve_jev_endpoint(base_url: str) -> str:
    """Resolve the full System One / Jev POST endpoint URL from ``base_url``.

    URLs ending with ``/systemone`` or ``/decisions`` (after trimming whitespace
    and trailing slashes) are used verbatim; all other non-empty base URLs have
    ``/systemone`` appended.
    """
    raw = base_url.strip().rstrip("/")
    if not raw:
        return ""
    if raw.endswith("/systemone") or raw.endswith("/decisions"):
        return raw
    return f"{raw}/systemone"


def _build_questions(*, include_should_reply: bool) -> dict[str, Any]:
    questions: dict[str, Any] = {}
    if include_should_reply:
        questions["should_reply"] = {
            "type": "noul",
            "instructions": (
                "结合 `recent_messages` 的上下文，"
                "判断机器人 `bot_name` 现在自然顺嘴接一句 "
                "`latest_batch` 的话是否合适、不突兀？"
            ),
            "criteria": {
                "true": (
                    "群友在抛开放话题、求助提问、玩梗吐槽、讨论公共话题，"
                    "或者正在延续刚才和机器人的对话，机器人像普通群友一样插一句很自然。"
                ),
                "false": (
                    "群友在和特定别人单聊对线、发纯表情包/哈哈/复读刷屏、"
                    "正经通知或严肃争吵，或者话没说完、机器人插嘴会显得突兀烦人。"
                ),
            },
        }
    questions["effort"] = {
        "type": "choice",
        "instructions": "回答 `latest_batch` 中的消息需要什么程度的思考深度？",
        "criteria": {
            "default": (
                "日常闲聊、玩梗接话、情绪回应、简单常识问答、打招呼，"
                "不需要多步推理或外部搜索就能直接自然回复。"
            ),
            "high": (
                "代码编写与报错排查、数学与逻辑推理、复杂技术方案分析、"
                "需要联网搜索查询外部实时信息/最新资讯/事实查证，"
                "或需要调用工具多步分析才能答准的问题。"
            ),
        },
    }
    return questions


def _parse_probability(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    prob = float(value)
    if not math.isfinite(prob):
        return None
    return max(0.0, min(1.0, prob))


def _degraded_decision(*, include_should_reply: bool) -> JevDecision:
    return JevDecision(
        should_reply=not include_should_reply,
        effort="default",
        degraded=True,
    )


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


class JevClient:
    """Async HTTP client for System One / Jev decision routing.

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
        self._endpoint = resolve_jev_endpoint(config.base_url)

    @property
    def endpoint(self) -> str:
        """Return the resolved System One POST endpoint URL."""
        return self._endpoint

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
        if self._closed:
            logger.warning("JevClient.evaluate called after close(); degrading.")
            return _degraded_decision(include_should_reply=include_should_reply)

        if not self._config.is_configured or not self._endpoint:
            logger.warning("JevClient is not configured; degrading decision.")
            return _degraded_decision(include_should_reply=include_should_reply)

        payload: dict[str, Any] = {
            "model": self._config.model.strip(),
            "state": dict(state),
            "questions": _build_questions(include_should_reply=include_should_reply),
        }
        headers = {
            "Authorization": f"Bearer {self._config.api_key.strip()}",
            "Content-Type": "application/json",
        }

        client = self._get_http_client()
        try:
            response = await client.post(
                self._endpoint,
                json=payload,
                headers=headers,
                timeout=self._config.timeout_sec,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Jev request failed (%s); degrading decision.", exc)
            return _degraded_decision(include_should_reply=include_should_reply)

        if response.status_code < 200 or response.status_code >= 300:
            logger.warning(
                "Jev returned HTTP %d; degrading decision.",
                response.status_code,
            )
            return _degraded_decision(include_should_reply=include_should_reply)

        try:
            data = response.json()
        except Exception as exc:
            logger.warning("Jev returned invalid JSON (%s); degrading decision.", exc)
            return _degraded_decision(include_should_reply=include_should_reply)

        return self._parse_response(data, include_should_reply=include_should_reply)

    def _parse_response(
        self,
        data: Any,
        *,
        include_should_reply: bool,
    ) -> JevDecision:
        if not isinstance(data, Mapping):
            logger.warning("Jev response root is not a JSON object; degrading.")
            return _degraded_decision(include_should_reply=include_should_reply)

        answers = data.get("answers")
        if not isinstance(answers, Mapping):
            logger.warning("Jev response missing 'answers' object; degrading.")
            return _degraded_decision(include_should_reply=include_should_reply)

        raw_model = data.get("model")
        model = raw_model if isinstance(raw_model, str) else ""

        input_tokens = 0
        usage = data.get("usage")
        if isinstance(usage, Mapping):
            raw_tokens = usage.get("input_tokens")
            if (
                isinstance(raw_tokens, int)
                and not isinstance(raw_tokens, bool)
                and raw_tokens >= 0
            ):
                input_tokens = raw_tokens

        reply_probability: float | None = None
        if include_should_reply:
            if "should_reply" not in answers:
                logger.warning("Jev response missing 'should_reply' answer; degrading.")
                return _degraded_decision(include_should_reply=True)
            raw_sr = answers["should_reply"]
            if isinstance(raw_sr, Mapping):
                if "noul" in raw_sr:
                    reply_probability = _parse_probability(raw_sr["noul"])
                elif "probability" in raw_sr:
                    reply_probability = _parse_probability(raw_sr["probability"])
                else:
                    reply_probability = None
            else:
                reply_probability = _parse_probability(raw_sr)
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

        parsed_effort = self._parse_effort(answers["effort"])
        if parsed_effort is None:
            logger.warning(
                "Jev response has invalid 'effort' value %r; degrading.",
                answers["effort"],
            )
            return _degraded_decision(include_should_reply=include_should_reply)

        effort, high_effort_probability = parsed_effort
        return JevDecision(
            should_reply=should_reply,
            effort=effort,
            reply_probability=reply_probability,
            high_effort_probability=high_effort_probability,
            model=model,
            input_tokens=input_tokens,
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
