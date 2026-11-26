"""Unified declarative provider & gateway compatibility layer for Spectra-bot.

Centralizes gateway-specific URL normalization, default headers, session
affinity headers, reasoning field names, Gemini tool schema quirks, and dual
decision-API protocol translation (``system_one`` and ``openai_decision``) in a
single thin module without subclass hierarchies.
"""

from __future__ import annotations

import copy
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypeGuard

import httpx

if TYPE_CHECKING:
    from spectra.config import JevConfig

DEFAULT_USER_AGENT = "spectra-bot/0.1.0"
OPENROUTER_REFERER = "https://github.com/YayoiNanoka/Spectra-bot"
OPENROUTER_TITLE = "Spectra-bot"
OPENCODE_SESSION_HEADER = "x-opencode-session"

DecisionProtocol = Literal["system_one", "openai_decision"]

_OPENROUTER_JEV_ZERO_PATCH_RE = re.compile(r"^(typesafe/jev-\d+\.\d+)\.0$")
_OPENAI_DECISION_PROB_SUM_TOL = 1e-2


def _parse_url(raw_url: str) -> httpx.URL | None:
    stripped = raw_url.strip()
    if not stripped:
        return None
    try:
        return httpx.URL(stripped)
    except httpx.InvalidURL:
        return None


def _host_matches(host: str, domain: str) -> bool:
    normalized = host.lower().rstrip(".")
    return normalized == domain or normalized.endswith(f".{domain}")


def _is_unit_number(value: Any) -> TypeGuard[int | float]:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0.0 <= value <= 1.0
    )


def build_provider_headers(
    custom_headers: Mapping[str, str] | None = None,
    *,
    gateway_headers: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build provider HTTP headers with an overridable ``User-Agent``.

    Merges ``{"User-Agent": DEFAULT_USER_AGENT}``, optional ``gateway_headers``,
    and user ``custom_headers`` in order using case-insensitive deduplication so
    keys such as ``User-Agent``/``user-agent`` or ``HTTP-Referer`` never
    duplicate. Empty or whitespace-only ``User-Agent`` overrides are ignored.

    Args:
        custom_headers: Optional user-configured headers taking highest priority.
        gateway_headers: Optional gateway-default headers merged before
            ``custom_headers``.

    Returns:
        Fresh header dictionary with string keys and values.
    """
    headers: dict[str, str] = {"User-Agent": DEFAULT_USER_AGENT}
    key_map: dict[str, str] = {"user-agent": "User-Agent"}

    def _merge(source: Mapping[str, str] | None) -> None:
        if not source:
            return
        for raw_name, raw_value in source.items():
            name = str(raw_name).strip()
            if not name:
                continue
            value = str(raw_value)
            lower = name.lower()
            if lower == "user-agent":
                if value.strip():
                    headers["User-Agent"] = value
                continue
            existing = key_map.get(lower)
            if existing is not None and existing != name:
                del headers[existing]
            headers[name] = value
            key_map[lower] = name

    _merge(gateway_headers)
    _merge(custom_headers)
    return headers


def should_omit_empty_tool_parameters(model: str) -> bool:
    """Return whether empty tool ``parameters`` should be omitted for ``model``.

    Mirrors AstrBot ``openai_source.py``: Gemini models reject function tool
    descriptors that include ``parameters`` with an empty ``properties`` dict.
    """
    return "gemini" in model.lower()


@dataclass(slots=True, frozen=True)
class OpenAICompatProfile:
    """Resolved gateway compatibility profile for an OpenAI-compatible client."""

    base_url: str
    default_headers: dict[str, str]
    session_header_name: str | None
    reasoning_key: str
    extra_body: dict[str, Any]

    def build_request_headers(
        self,
        session_key: str | None = None,
        *,
        default_session: str = "spectra:default",
    ) -> dict[str, str]:
        """Build per-request dynamic headers (such as OpenCode session affinity).

        Returns an empty dict when no session header is required or when the
        user already explicitly set the same header in ``custom_headers``.
        """
        if not self.session_header_name:
            return {}
        existing_lower = {k.lower() for k in self.default_headers}
        if self.session_header_name.lower() in existing_lower:
            return {}
        value = (
            session_key.strip()
            if session_key and session_key.strip()
            else default_session
        )
        return {self.session_header_name: value}


def resolve_openai_compat(
    *,
    base_url: str,
    model: str,
    custom_headers: Mapping[str, str] | None = None,
    extra_body: Mapping[str, Any] | None = None,
) -> OpenAICompatProfile:
    """Resolve gateway URL, headers, session header, and reasoning key rules.

    Args:
        base_url: Configured OpenAI-compatible API base URL.
        model: Configured model identifier.
        custom_headers: Optional user-supplied HTTP headers.
        extra_body: Optional extra JSON body fields merged into chat requests.

    Returns:
        Frozen :class:`OpenAICompatProfile` describing the resolved behavior.
    """
    _ = model  # Reserved for future model-specific overrides.
    resolved_url = base_url.strip()
    parsed = _parse_url(resolved_url)
    host = parsed.host if parsed is not None else ""

    gateway_headers: dict[str, str] = {}
    session_header_name: str | None = None
    reasoning_key = "reasoning_content"

    if parsed is not None and host and _host_matches(host, "opencode.ai"):
        stripped_path = parsed.path.rstrip("/")
        if stripped_path.endswith("/zen/go/v1") or stripped_path.endswith("/zen/go"):
            parsed = parsed.copy_with(path="/inference/go/openai/v1")
            resolved_url = str(parsed)
        segments = [seg for seg in parsed.path.split("/") if seg]
        if "go" in segments:
            session_header_name = OPENCODE_SESSION_HEADER
    elif parsed is not None and host and _host_matches(host, "openrouter.ai"):
        gateway_headers["HTTP-Referer"] = OPENROUTER_REFERER
        gateway_headers["X-OpenRouter-Title"] = OPENROUTER_TITLE
        reasoning_key = "reasoning"

    default_headers = build_provider_headers(
        custom_headers,
        gateway_headers=gateway_headers,
    )
    normalized_extra_body = copy.deepcopy(dict(extra_body)) if extra_body else {}

    return OpenAICompatProfile(
        base_url=resolved_url,
        default_headers=default_headers,
        session_header_name=session_header_name,
        reasoning_key=reasoning_key,
        extra_body=normalized_extra_body,
    )


@dataclass(slots=True, frozen=True)
class DecisionEndpointProfile:
    """Resolved endpoint, model ID, wire protocol, and headers for decision APIs."""

    endpoint_url: str
    model: str
    protocol: DecisionProtocol
    headers: dict[str, str]


def _resolve_decision_protocol(
    parsed: httpx.URL | None,
    requested_protocol: Literal["auto", "system_one", "openai_decision"],
) -> DecisionProtocol:
    if requested_protocol != "auto":
        return requested_protocol
    if parsed is None:
        return "system_one"
    host = parsed.host.lower().rstrip(".")
    stripped_path = parsed.path.rstrip("/")
    if host == "api.openai.com":
        return "openai_decision"
    if stripped_path.endswith("/v1/decisions") and not _host_matches(
        host, "openrouter.ai"
    ):
        return "openai_decision"
    return "system_one"


def resolve_decision_endpoint(
    base_url: str,
    *,
    protocol: Literal["auto", "system_one", "openai_decision"] = "auto",
) -> str:
    """Resolve the full POST endpoint URL for System One or OpenAI Decisions."""
    raw = base_url.strip()
    if not raw:
        return ""
    parsed = _parse_url(raw)
    effective_protocol = _resolve_decision_protocol(parsed, protocol)

    if parsed is None or not parsed.scheme or not parsed.host:
        stripped = raw.rstrip("/")
        if effective_protocol == "openai_decision":
            return (
                stripped if stripped.endswith("/decisions") else f"{stripped}/decisions"
            )
        if stripped.endswith("/systemone") or stripped.endswith("/decisions"):
            return stripped
        return f"{stripped}/systemone"

    host = parsed.host
    stripped_path = parsed.path.rstrip("/")

    if _host_matches(host, "openrouter.ai") and effective_protocol == "system_one":
        if stripped_path.endswith("/decisions") or stripped_path.endswith("/systemone"):
            return str(parsed.copy_with(path=stripped_path))
        return str(parsed.copy_with(path="/api/alpha/decisions"))

    if effective_protocol == "system_one":
        if stripped_path.endswith("/systemone") or stripped_path.endswith("/decisions"):
            return str(parsed.copy_with(path=stripped_path))
        return str(parsed.copy_with(path=f"{stripped_path}/systemone"))

    # openai_decision
    if stripped_path.endswith("/decisions"):
        return str(parsed.copy_with(path=stripped_path))
    return str(parsed.copy_with(path=f"{stripped_path}/decisions"))


def _normalize_decision_model(
    model: str,
    *,
    host: str,
    protocol: DecisionProtocol,
) -> str:
    normalized = model.strip()
    if not normalized or not host:
        return normalized

    if _host_matches(host, "openrouter.ai") and protocol == "system_one":
        if normalized.startswith("jev-"):
            normalized = f"typesafe/{normalized}"
        normalized = _OPENROUTER_JEV_ZERO_PATCH_RE.sub(r"\1", normalized)
    elif host.lower().rstrip(".") == "api.typesafe.ai":
        if normalized.startswith("typesafe/"):
            stripped = normalized.removeprefix("typesafe/")
            if stripped:
                normalized = stripped
    return normalized


def resolve_decision_profile(config: JevConfig) -> DecisionEndpointProfile:
    """Resolve a complete :class:`DecisionEndpointProfile` from ``JevConfig``."""
    raw_url = config.base_url.strip()
    parsed = _parse_url(raw_url)
    host = parsed.host if parsed is not None else ""

    protocol = _resolve_decision_protocol(parsed, config.protocol)
    endpoint_url = resolve_decision_endpoint(raw_url, protocol=protocol)
    model = _normalize_decision_model(config.model, host=host, protocol=protocol)

    gateway_headers: dict[str, str] = {}
    if host and _host_matches(host, "openrouter.ai"):
        gateway_headers["HTTP-Referer"] = OPENROUTER_REFERER
        gateway_headers["X-OpenRouter-Title"] = OPENROUTER_TITLE
    api_key = config.api_key.strip()
    if api_key:
        gateway_headers["Authorization"] = f"Bearer {api_key}"

    headers = build_provider_headers(
        config.custom_headers,
        gateway_headers=gateway_headers,
    )
    return DecisionEndpointProfile(
        endpoint_url=endpoint_url,
        model=model,
        protocol=protocol,
        headers=headers,
    )


def _format_predicate_instructions(question: Mapping[str, Any]) -> str:
    instructions = str(question.get("instructions", "")).strip()
    criteria = question.get("criteria")
    if isinstance(criteria, Mapping):
        true_desc = str(criteria.get("true", "")).strip()
        false_desc = str(criteria.get("false", "")).strip()
        parts = [instructions] if instructions else []
        if true_desc:
            parts.append(f"判定为是的情形：{true_desc}")
        if false_desc:
            parts.append(f"判定为否的情形：{false_desc}")
        return "\n".join(parts)
    return instructions


def build_decision_payload(
    profile: DecisionEndpointProfile,
    *,
    state: Any,
    questions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Serialize canonical decision questions & state into wire JSON payload."""
    if profile.protocol == "system_one":
        return {
            "model": profile.model,
            "state": dict(state) if isinstance(state, Mapping) else state,
            "questions": {
                name: copy.deepcopy(dict(spec)) for name, spec in questions.items()
            },
        }

    # OpenAI Decisions protocol (aligned with AstrBot openai_decision_source.py).
    if isinstance(state, str):
        request_input: Any = state
    elif isinstance(state, Mapping):
        request_input = [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_json", "data": dict(state)}],
            }
        ]
    else:
        request_input = json.dumps(state, ensure_ascii=False, allow_nan=False)

    payload_questions: list[dict[str, Any]] = []
    for name, spec in questions.items():
        q_type = spec.get("type")
        if q_type in ("noul", "predicate"):
            payload_questions.append(
                {
                    "type": "predicate",
                    "name": name,
                    "instructions": _format_predicate_instructions(spec),
                }
            )
        elif q_type == "choice":
            criteria = spec.get("criteria")
            choices = (
                [
                    {"value": str(opt_id), "description": str(desc)}
                    for opt_id, desc in criteria.items()
                ]
                if isinstance(criteria, Mapping)
                else []
            )
            payload_questions.append(
                {
                    "type": "choice",
                    "name": name,
                    "instructions": str(spec.get("instructions", "")).strip(),
                    "choices": choices,
                }
            )
        else:
            raise ValueError(f"Unsupported decision question type: {q_type!r}")

    return {
        "model": profile.model,
        "input": request_input,
        "questions": payload_questions,
    }


def normalize_decision_answers(
    data: Any,
    *,
    protocol: DecisionProtocol,
) -> dict[str, Any] | None:
    """Normalize raw decision API response ``answers`` into a name-keyed mapping.

    Returns ``None`` on malformed payloads or OpenAI Decisions ``refusal``
    entries so callers can degrade safely.
    """
    if not isinstance(data, Mapping):
        return None
    raw_answers = data.get("answers")

    if protocol == "system_one":
        if not isinstance(raw_answers, Mapping):
            return None
        return dict(raw_answers)

    # OpenAI Decisions protocol: answers is a list matched by "name".
    if not isinstance(raw_answers, list):
        return None

    normalized: dict[str, Any] = {}
    for entry in raw_answers:
        if not isinstance(entry, Mapping):
            return None
        name = entry.get("name")
        if not isinstance(name, str) or not name or name in normalized:
            return None
        answer_type = entry.get("type")
        if answer_type == "refusal":
            return None

        if answer_type == "predicate":
            probability = entry.get("probability")
            if not _is_unit_number(probability):
                return None
            normalized[name] = {
                "type": "predicate",
                "probability": float(probability),
            }
        elif answer_type == "choice":
            choice = entry.get("choice")
            confidence = entry.get("confidence")
            raw_probs = entry.get("probabilities")
            if not (
                isinstance(choice, str)
                and _is_unit_number(confidence)
                and isinstance(raw_probs, list)
                and raw_probs
            ):
                return None
            probs_map: dict[str, float] = {}
            for item in raw_probs:
                if not isinstance(item, Mapping):
                    return None
                val = item.get("value")
                prob = item.get("probability")
                if not (
                    isinstance(val, str)
                    and val not in probs_map
                    and _is_unit_number(prob)
                ):
                    return None
                probs_map[val] = float(prob)
            if choice not in probs_map or not math.isclose(
                sum(probs_map.values()),
                1.0,
                abs_tol=_OPENAI_DECISION_PROB_SUM_TOL,
            ):
                return None
            normalized[name] = {
                "type": "choice",
                "choice": choice,
                "confidence": float(confidence),
                "probabilities": probs_map,
            }
        else:
            return None

    return normalized
