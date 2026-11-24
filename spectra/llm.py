"""Generic OpenAI-compatible async LLM client and agent tool-loop base.

Retry policy, message sanitization, usage extraction, content normalization,
and Gemini tool schema handling align directly with AstrBot's
``openai_source.py`` and ``request_retry.py``.
"""

from __future__ import annotations

import ast
import copy
import inspect
import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Self

import httpx
import openai
from openai import AsyncOpenAI
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)
from tenacity.wait import wait_base

from spectra.config import LLMConfig
from spectra.providers import (
    OpenAICompatProfile,
    resolve_openai_compat,
    should_omit_empty_tool_parameters,
)

logger = logging.getLogger(__name__)

# Constants aligned with AstrBot's request_retry.py.
REQUEST_RETRY_ATTEMPTS = 5
REQUEST_RETRY_WAIT_MIN_S = 0.2
REQUEST_RETRY_WAIT_MAX_S = 30.0
REQUEST_RETRY_STATUS_CODES: frozenset[int] = frozenset(
    {408, 409, 429, 500, 502, 503, 504, 529}
)


class LLMError(Exception):
    """Base exception for LLM client configuration and runtime errors."""


class EmptyLLMResponseError(LLMError):
    """Raised when an OpenAI-compatible completion returns no choices."""


class MaxToolStepsExceededError(LLMError):
    """Raised when ``run_tool_loop`` exhausts ``max_steps`` without finishing."""


_FALLBACK_EXCEPTIONS = (
    openai.APIError,
    httpx.HTTPError,
    TimeoutError,
    ConnectionError,
    OSError,
    EmptyLLMResponseError,
)


@dataclass(slots=True, frozen=True)
class TokenUsage:
    """Token accounting for one or more LLM completion calls."""

    input_other: int = 0
    input_cached: int = 0
    output: int = 0

    @property
    def total_input(self) -> int:
        """Return total prompt tokens (cached + uncached)."""
        return self.input_other + self.input_cached

    @property
    def total(self) -> int:
        """Return total tokens across input and output."""
        return self.total_input + self.output

    def __add__(self, other: TokenUsage) -> TokenUsage:
        if not isinstance(other, TokenUsage):
            return NotImplemented
        return TokenUsage(
            input_other=self.input_other + other.input_other,
            input_cached=self.input_cached + other.input_cached,
            output=self.output + other.output,
        )


@dataclass(slots=True, frozen=True)
class ToolCall:
    """Single function tool call returned by an assistant completion."""

    id: str
    name: str
    arguments: str


@dataclass(slots=True, frozen=True)
class LLMResponse:
    """Normalized result of a single chat completion call."""

    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    reasoning_content: str | None = None
    model: str = ""
    finish_reason: str | None = None
    usage: TokenUsage = field(default_factory=TokenUsage)

    def to_assistant_message(
        self,
        *,
        reasoning_key: str = "reasoning_content",
    ) -> dict[str, Any]:
        """Build a clean OpenAI-formatted ``assistant`` message dictionary."""
        msg: dict[str, Any] = {
            "role": "assistant",
            "content": (
                None if (self.content == "" and self.tool_calls) else self.content
            ),
        }
        if self.reasoning_content:
            msg[reasoning_key] = self.reasoning_content
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments,
                    },
                }
                for tc in self.tool_calls
            ]
        return msg


@dataclass(slots=True, frozen=True)
class ToolSpec:
    """Declarative specification and handler for a local agent tool."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any]

    def to_openai_tool(self, *, model: str = "") -> dict[str, Any]:
        """Serialize to an OpenAI ``tools`` entry.

        Mirrors AstrBot ``openai_source.py:542``: when targeting Gemini models
        and ``parameters.get("properties")`` is empty, ``"parameters"`` is
        omitted from the function descriptor for API compatibility.
        """
        func: dict[str, Any] = {"name": self.name}
        if self.description:
            func["description"] = self.description
        omit_empty = should_omit_empty_tool_parameters(model)
        has_properties = bool(
            isinstance(self.parameters, Mapping) and self.parameters.get("properties")
        )
        if has_properties or not omit_empty:
            func["parameters"] = copy.deepcopy(self.parameters)
        return {"type": "function", "function": func}


@dataclass(slots=True, frozen=True)
class AgentResult:
    """Final outcome of :meth:`LLMClient.run_tool_loop`."""

    content: str
    steps: int
    usage: TokenUsage
    messages: list[dict[str, Any]]


def _is_empty_content(content: Any) -> bool:
    return content is None or content == "" or content == []


def _clean_assistant_tool_calls(raw_tool_calls: Any) -> list[dict[str, Any]]:
    """Keep only valid mapping tool-call entries with unique non-empty string IDs."""
    if not isinstance(raw_tool_calls, list):
        return []
    deduped: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for tc in raw_tool_calls:
        if not isinstance(tc, Mapping):
            continue
        tc_id = tc.get("id")
        if not isinstance(tc_id, str) or not tc_id.strip() or tc_id in seen_ids:
            continue
        seen_ids.add(tc_id)
        deduped.append(dict(tc))
    return deduped


def _normalize_assistant_reasoning(
    msg: dict[str, Any],
    *,
    reasoning_key: str,
) -> None:
    """Normalize ``reasoning_content``/``reasoning`` into ``reasoning_key``."""
    candidate_keys: list[str] = [reasoning_key]
    for key in ("reasoning_content", "reasoning"):
        if key not in candidate_keys:
            candidate_keys.append(key)

    resolved: str | None = None
    for key in candidate_keys:
        raw = msg.pop(key, None)
        if resolved is None and isinstance(raw, str) and raw:
            resolved = raw
    if resolved is not None:
        msg[reasoning_key] = resolved


def _finalize_assistant_without_tool_calls(
    msg: dict[str, Any],
    *,
    idx: int,
    reasoning_key: str = "reasoning_content",
) -> dict[str, Any] | None:
    """Apply empty-assistant normalization when no valid tool_calls remain."""
    content = msg.get("content")
    reasoning_content = msg.get(reasoning_key)
    if _is_empty_content(content):
        if not reasoning_content:
            logger.debug("Dropping empty assistant message at index %d.", idx)
            return None
        msg["content"] = ""
    return msg


def sanitize_messages(
    messages: Sequence[Mapping[str, Any]],
    *,
    reasoning_key: str = "reasoning_content",
) -> list[dict[str, Any]]:
    """Return a sanitized deep copy of chat messages without mutating inputs.

    Mirrors AstrBot's ``OpenAIProvider._sanitize_assistant_messages`` and
    additionally reconciles ``assistant.tool_calls`` against surviving
    contiguous ``role == "tool"`` responses:

    - Normalizes assistant reasoning fields (accepting either
      ``reasoning_content`` or ``reasoning``) into ``reasoning_key``.
    - Drops empty ``assistant`` messages lacking ``content``, ``tool_calls``,
      and reasoning text.
    - Keeps ``assistant`` messages with reasoning text but no text or
      tool calls by setting ``content`` to ``""``.
    - Normalizes ``assistant`` messages with ``tool_calls`` and empty content
      to ``content=None`` per the OpenAI specification.
    - Pairs each ``assistant(tool_calls)`` with its immediately following
      contiguous ``role == "tool"`` block: filters orphaned/duplicate ``tool``
      messages and prunes any unanswered ``tool_calls`` (removing or
      downgrading the ``assistant`` message if none were answered).
    """
    cleaned: list[dict[str, Any]] = []
    for idx, raw_msg in enumerate(messages):
        msg = copy.deepcopy(dict(raw_msg))
        if msg.get("role") != "assistant":
            cleaned.append(msg)
            continue

        _normalize_assistant_reasoning(msg, reasoning_key=reasoning_key)

        if "tool_calls" in msg:
            valid_tool_calls = _clean_assistant_tool_calls(msg["tool_calls"])
            if valid_tool_calls:
                msg["tool_calls"] = valid_tool_calls
            else:
                msg.pop("tool_calls")

        if msg.get("tool_calls"):
            if _is_empty_content(msg.get("content")):
                msg["content"] = None
            cleaned.append(msg)
        else:
            normalized = _finalize_assistant_without_tool_calls(
                msg,
                idx=idx,
                reasoning_key=reasoning_key,
            )
            if normalized is not None:
                cleaned.append(normalized)

    final: list[dict[str, Any]] = []
    removed_tool_messages = 0
    i = 0
    while i < len(cleaned):
        msg = cleaned[i]
        role = msg.get("role")

        if role == "tool":
            removed_tool_messages += 1
            i += 1
            continue

        if role == "assistant" and msg.get("tool_calls"):
            tool_calls: list[dict[str, Any]] = msg["tool_calls"]
            valid_ids = {tc["id"] for tc in tool_calls}
            matched_ids: set[str] = set()
            matched_tool_msgs: list[dict[str, Any]] = []

            j = i + 1
            while j < len(cleaned) and cleaned[j].get("role") == "tool":
                tmsg = cleaned[j]
                tc_id = tmsg.get("tool_call_id")
                if (
                    isinstance(tc_id, str)
                    and tc_id in valid_ids
                    and tc_id not in matched_ids
                ):
                    matched_ids.add(tc_id)
                    matched_tool_msgs.append(tmsg)
                else:
                    removed_tool_messages += 1
                j += 1

            if matched_ids:
                msg["tool_calls"] = [tc for tc in tool_calls if tc["id"] in matched_ids]
                final.append(msg)
                final.extend(matched_tool_msgs)
            else:
                msg.pop("tool_calls", None)
                downgraded = _finalize_assistant_without_tool_calls(
                    msg,
                    idx=i,
                    reasoning_key=reasoning_key,
                )
                if downgraded is not None:
                    final.append(downgraded)

            i = j
            continue

        final.append(msg)
        i += 1

    if removed_tool_messages:
        logger.debug(
            "Filtered %d orphaned or duplicate tool message(s).",
            removed_tool_messages,
        )
    return final


def _as_non_negative_int(value: Any) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return max(0, value)
    return 0


def _extract_usage(usage: Any) -> TokenUsage:
    """Extract :class:`TokenUsage` from an OpenAI usage object or mapping."""
    if usage is None:
        return TokenUsage()

    if isinstance(usage, Mapping):
        ptd = usage.get("prompt_tokens_details")
        raw_prompt = usage.get("prompt_tokens", 0)
        raw_completion = usage.get("completion_tokens", 0)
    else:
        ptd = getattr(usage, "prompt_tokens_details", None)
        raw_prompt = getattr(usage, "prompt_tokens", 0)
        raw_completion = getattr(usage, "completion_tokens", 0)

    if isinstance(ptd, Mapping):
        raw_cached = ptd.get("cached_tokens", 0)
    elif ptd is not None:
        raw_cached = getattr(ptd, "cached_tokens", 0)
    else:
        raw_cached = 0

    prompt_tokens = _as_non_negative_int(raw_prompt)
    cached_tokens = _as_non_negative_int(raw_cached)
    completion_tokens = _as_non_negative_int(raw_completion)
    return TokenUsage(
        input_other=max(0, prompt_tokens - cached_tokens),
        input_cached=cached_tokens,
        output=completion_tokens,
    )


def _extract_text_parts(parts: list[Any]) -> str | None:
    has_text_part = False
    texts: list[str] = []
    for part in parts:
        if isinstance(part, Mapping):
            part_type = part.get("type")
            text_val = part.get("text", "")
        else:
            part_type = getattr(part, "type", None)
            text_val = getattr(part, "text", "")
        if part_type == "text":
            has_text_part = True
            texts.append(str(text_val) if text_val is not None else "")
    if has_text_part:
        return "".join(texts)
    return None


def _normalize_content(raw_content: Any) -> str:
    """Normalize completion message content across str, list, dict, and None."""
    if raw_content is None:
        return ""

    if isinstance(raw_content, Mapping):
        if "text" in raw_content:
            text_val = raw_content.get("text", "")
            return str(text_val) if text_val is not None else ""
        logger.warning("Unexpected dict format content from LLM: %r", raw_content)
        return ""

    if isinstance(raw_content, list):
        if not raw_content:
            return ""
        extracted = _extract_text_parts(raw_content)
        if extracted is not None:
            return extracted
        return str(raw_content)

    if isinstance(raw_content, str):
        content = raw_content.strip()
        if content.startswith("[") and content.endswith("]") and len(content) < 8192:
            parsed: Any = None
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                try:
                    parsed = ast.literal_eval(content)
                except (
                    ValueError,
                    SyntaxError,
                    TypeError,
                    MemoryError,
                    RecursionError,
                ):
                    parsed = None
            if isinstance(parsed, list):
                extracted = _extract_text_parts(parsed)
                if extracted is not None:
                    return extracted
        return content

    return str(raw_content)


def _parse_tool_calls(raw_tool_calls: Any) -> tuple[ToolCall, ...]:
    if not raw_tool_calls:
        return ()

    parsed: list[ToolCall] = []
    seen_ids: set[str] = set()
    synthetic_counter = 0

    for tc in raw_tool_calls:
        if isinstance(tc, Mapping):
            tc_type = tc.get("type", "function")
            if tc_type != "function":
                continue
            raw_id = tc.get("id")
            func = tc.get("function") or {}
            raw_name = func.get("name") if isinstance(func, Mapping) else None
            raw_args = func.get("arguments") if isinstance(func, Mapping) else None
        else:
            tc_type = getattr(tc, "type", "function")
            if tc_type != "function":
                continue
            raw_id = getattr(tc, "id", None)
            func = getattr(tc, "function", None)
            raw_name = getattr(func, "name", None) if func is not None else None
            raw_args = getattr(func, "arguments", None) if func is not None else None

        tc_id = "" if raw_id is None else str(raw_id).strip()
        if not tc_id or tc_id in seen_ids:
            while True:
                synthetic_counter += 1
                candidate = f"call_{synthetic_counter}"
                if candidate not in seen_ids:
                    tc_id = candidate
                    break
        seen_ids.add(tc_id)

        name = "" if raw_name is None else str(raw_name)

        if raw_args is None:
            arguments = ""
        elif isinstance(raw_args, (dict, list)):
            arguments = json.dumps(raw_args, ensure_ascii=False)
        else:
            arguments = str(raw_args)

        parsed.append(ToolCall(id=tc_id, name=name, arguments=arguments))
    return tuple(parsed)


def _extract_reasoning(
    message: Any,
    *,
    reasoning_key: str = "reasoning_content",
) -> str | None:
    """Extract reasoning text preferring ``reasoning_key`` then known fallbacks."""
    candidate_keys: list[str] = [reasoning_key]
    for key in ("reasoning_content", "reasoning"):
        if key not in candidate_keys:
            candidate_keys.append(key)

    model_extra = getattr(message, "model_extra", None)
    first_empty_str: str | None = None
    for key in candidate_keys:
        if isinstance(message, Mapping):
            value = message.get(key)
        else:
            value = getattr(message, key, None)
            if value is None and isinstance(model_extra, Mapping):
                value = model_extra.get(key)
        if not isinstance(value, str):
            continue
        if value:
            return value
        if first_empty_str is None:
            first_empty_str = value
    return first_empty_str


def _parse_completion(
    completion: Any,
    requested_model: str,
    *,
    reasoning_key: str = "reasoning_content",
) -> LLMResponse:
    choices = getattr(completion, "choices", None)
    if not choices:
        raise EmptyLLMResponseError(
            f"OpenAI completion has no choices (model={requested_model!r})."
        )

    choice = choices[0]
    message = getattr(choice, "message", None)
    if message is None:
        raise EmptyLLMResponseError(
            f"OpenAI completion choice has no message (model={requested_model!r})."
        )
    content = _normalize_content(getattr(message, "content", None))
    reasoning_content = _extract_reasoning(message, reasoning_key=reasoning_key)
    tool_calls = _parse_tool_calls(getattr(message, "tool_calls", None))
    usage = _extract_usage(getattr(completion, "usage", None))
    model = getattr(completion, "model", None) or requested_model
    finish_reason = getattr(choice, "finish_reason", None)

    return LLMResponse(
        content=content,
        tool_calls=tool_calls,
        reasoning_content=reasoning_content,
        model=str(model),
        finish_reason=finish_reason,
        usage=usage,
    )


def _get_status_code(error: BaseException) -> int | None:
    for attr in ("status_code", "status", "code"):
        value = getattr(error, attr, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value

    response = getattr(error, "response", None)
    if response is not None:
        status_code = getattr(response, "status_code", None)
        if isinstance(status_code, int) and not isinstance(status_code, bool):
            return status_code

    return None


def _is_retryable_error(error: BaseException) -> bool:
    """Determine whether an LLM request exception should be retried."""
    if not isinstance(error, Exception):
        return False

    if isinstance(
        error,
        (
            openai.APIConnectionError,
            openai.APITimeoutError,
            httpx.TimeoutException,
            httpx.TransportError,
            TimeoutError,
            ConnectionError,
        ),
    ):
        return True

    status_code = _get_status_code(error)
    if status_code is not None:
        return status_code in REQUEST_RETRY_STATUS_CODES or 500 <= status_code <= 599

    seen: set[int] = {id(error)}
    cause = getattr(error, "__cause__", None)
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        if isinstance(
            cause,
            (
                httpx.TimeoutException,
                httpx.TransportError,
                TimeoutError,
                ConnectionError,
            ),
        ):
            return True
        cause = getattr(cause, "__cause__", None)

    return False


def _serialize_tool_output(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


async def _execute_tool_call(
    tool_call: ToolCall,
    tool_map: Mapping[str, ToolSpec],
) -> str:
    spec = tool_map.get(tool_call.name)
    if spec is None:
        return f"Error: Unknown tool '{tool_call.name}'."

    raw_args = tool_call.arguments.strip()
    if not raw_args:
        parsed_args: dict[str, Any] = {}
    else:
        try:
            decoded = json.loads(raw_args)
        except json.JSONDecodeError as exc:
            return f"Error: Tool arguments must be a JSON object (invalid JSON: {exc})."
        if not isinstance(decoded, dict):
            return (
                "Error: Tool arguments must be a JSON object, "
                f"got {type(decoded).__name__}."
            )
        parsed_args = decoded

    try:
        result = spec.handler(**parsed_args)
        if inspect.isawaitable(result):
            result = await result
    except Exception as exc:
        return f"Error executing tool '{tool_call.name}': {exc}"

    return _serialize_tool_output(result)


class LLMClient:
    """Async OpenAI-compatible chat completion client with retry, fallback, and tools.

    Args:
        config: LLM configuration model.
        openai_client: Optional externally managed :class:`openai.AsyncOpenAI`.
            When provided, :meth:`close` will not close it.
        retry_wait: Optional tenacity wait strategy override (useful in tests).
    """

    def __init__(
        self,
        config: LLMConfig,
        openai_client: AsyncOpenAI | None = None,
        *,
        retry_wait: wait_base | None = None,
    ) -> None:
        self._config = config
        self._compat: OpenAICompatProfile = resolve_openai_compat(
            base_url=config.base_url,
            model=config.model,
            custom_headers=config.custom_headers,
            extra_body=config.extra_body,
        )
        self._openai_client = openai_client
        self._owns_client = openai_client is None
        self._closed = False
        self._retry_wait = retry_wait or wait_exponential(
            multiplier=1,
            min=REQUEST_RETRY_WAIT_MIN_S,
            max=REQUEST_RETRY_WAIT_MAX_S,
        )

    @property
    def compat(self) -> OpenAICompatProfile:
        """Return the resolved gateway compatibility profile."""
        return self._compat

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the underlying OpenAI client if owned by this instance."""
        self._closed = True
        if self._owns_client and self._openai_client is not None:
            client = self._openai_client
            self._openai_client = None
            await client.close()

    def _get_openai_client(self) -> AsyncOpenAI:
        if self._closed:
            raise LLMError("LLMClient has been closed.")
        if self._openai_client is None:
            api_key = self._config.api_key.strip()
            if not api_key:
                raise LLMError(
                    "LLM api_key must be configured when no custom openai_client is "
                    "provided."
                )
            self._openai_client = AsyncOpenAI(
                api_key=api_key,
                base_url=self._compat.base_url or None,
                default_headers=dict(self._compat.default_headers),
                timeout=self._config.timeout_sec,
                # Retries are managed by tenacity; disable SDK retries to avoid
                # multiplying attempts.
                max_retries=0,
            )
        return self._openai_client

    def _resolve_effort(self, effort: Literal["default", "high"]) -> str:
        if effort == "default":
            return self._config.default_effort.strip()
        if effort == "high":
            return self._config.high_effort.strip()
        raise ValueError(f"Unsupported effort level: {effort!r}")

    def _log_retry(self, model: str, retry_state: RetryCallState) -> None:
        error = retry_state.outcome.exception() if retry_state.outcome else None
        logger.warning(
            "LLM request to model %r failed with retryable error; "
            "retrying (%d/%d): %s: %s",
            model,
            retry_state.attempt_number + 1,
            self._config.max_retries,
            type(error).__name__ if error is not None else "UnknownError",
            error,
        )

    async def _complete_with_model(
        self,
        client: AsyncOpenAI,
        model: str,
        sanitized_messages: list[dict[str, Any]],
        *,
        effort_value: str,
        tools: Sequence[ToolSpec],
        session_key: str | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": sanitized_messages,
            "timeout": self._config.timeout_sec,
        }
        if effort_value:
            kwargs["reasoning_effort"] = effort_value
        if tools:
            kwargs["tools"] = [tool.to_openai_tool(model=model) for tool in tools]
            kwargs["tool_choice"] = "auto"
        req_headers = self._compat.build_request_headers(
            session_key,
            default_session="spectra:llm",
        )
        if req_headers:
            kwargs["extra_headers"] = req_headers
        if self._compat.extra_body:
            kwargs["extra_body"] = copy.deepcopy(self._compat.extra_body)

        retrying = AsyncRetrying(
            retry=retry_if_exception(_is_retryable_error),
            stop=stop_after_attempt(self._config.max_retries),
            wait=self._retry_wait,
            before_sleep=lambda state: self._log_retry(model, state),
            reraise=True,
        )

        async for attempt in retrying:
            with attempt:
                completion = await client.chat.completions.create(**kwargs)
                return _parse_completion(
                    completion,
                    model,
                    reasoning_key=self._compat.reasoning_key,
                )

        raise LLMError("LLM retry loop exited unexpectedly.")

    async def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        effort: Literal["default", "high"] = "default",
        tools: Sequence[ToolSpec] = (),
        session_key: str | None = None,
    ) -> LLMResponse:
        """Send a chat completion request with sanitization, retries, and fallback.

        Args:
            messages: Chat message sequence (never mutated).
            effort: Reasoning effort selector (``"default"`` or ``"high"``).
            tools: Optional tool specifications exposed to the model.
            session_key: Optional conversation session key forwarded to gateways
                requiring session affinity headers (e.g. OpenCode Go).

        Returns:
            Parsed :class:`LLMResponse`.

        Raises:
            LLMError: If the client is closed or lacks an API key.
            EmptyLLMResponseError: If the API returns no choices.
            ValueError: If ``effort`` is not ``"default"`` or ``"high"``.
            Exception: Propagates the final API/transport error after retries
                (and fallback, if configured).
        """
        effort_value = self._resolve_effort(effort)
        primary_model = self._config.model.strip()
        if not primary_model:
            raise LLMError("LLM model must be a non-empty string.")
        client = self._get_openai_client()
        sanitized_messages = sanitize_messages(
            messages,
            reasoning_key=self._compat.reasoning_key,
        )

        try:
            return await self._complete_with_model(
                client,
                primary_model,
                sanitized_messages,
                effort_value=effort_value,
                tools=tools,
                session_key=session_key,
            )
        except openai.AuthenticationError:
            raise
        except _FALLBACK_EXCEPTIONS as primary_exc:
            fallback_model = self._config.fallback_model.strip()
            if fallback_model and fallback_model != primary_model:
                logger.warning(
                    "Primary model %r failed (%s: %s); falling back to %r.",
                    primary_model,
                    type(primary_exc).__name__,
                    primary_exc,
                    fallback_model,
                )
                return await self._complete_with_model(
                    client,
                    fallback_model,
                    sanitized_messages,
                    effort_value=effort_value,
                    tools=tools,
                    session_key=session_key,
                )
            raise

    async def run_tool_loop(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[ToolSpec] = (),
        effort: Literal["default", "high"] = "default",
        max_steps: int | None = None,
        session_key: str | None = None,
    ) -> AgentResult:
        """Execute an iterative tool-calling loop until a final response or step limit.

        Args:
            messages: Initial conversation messages (never mutated).
            tools: Available local tools.
            effort: Reasoning effort selector passed to :meth:`complete`.
            max_steps: Optional override for ``config.max_tool_steps`` (must be >= 1).
            session_key: Optional conversation session key forwarded to each
                :meth:`complete` call.

        Returns:
            :class:`AgentResult` containing the final text, step count,
            accumulated :class:`TokenUsage`, and updated message transcript.

        Raises:
            ValueError: If ``max_steps < 1`` or duplicate tool names are passed.
            MaxToolStepsExceededError: If the model keeps requesting tools
                through all ``max_steps`` rounds.
        """
        effective_max_steps = (
            self._config.max_tool_steps if max_steps is None else max_steps
        )
        if effective_max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {effective_max_steps}.")

        tool_map: dict[str, ToolSpec] = {}
        for spec in tools:
            if spec.name in tool_map:
                raise ValueError(f"Duplicate tool name: {spec.name!r}")
            tool_map[spec.name] = spec

        working_messages: list[dict[str, Any]] = [
            copy.deepcopy(dict(msg)) for msg in messages
        ]
        total_usage = TokenUsage()

        for step in range(1, effective_max_steps + 1):
            resp = await self.complete(
                working_messages,
                effort=effort,
                tools=tools,
                session_key=session_key,
            )
            total_usage = total_usage + resp.usage
            working_messages.append(
                resp.to_assistant_message(reasoning_key=self._compat.reasoning_key)
            )

            if not resp.tool_calls:
                return AgentResult(
                    content=resp.content,
                    steps=step,
                    usage=total_usage,
                    messages=working_messages,
                )

            for tc in resp.tool_calls:
                output = await _execute_tool_call(tc, tool_map)
                working_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": output,
                    }
                )

        raise MaxToolStepsExceededError(
            f"Tool loop exceeded max_steps={effective_max_steps} without a final "
            "response."
        )
