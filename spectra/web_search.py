"""AstrBot-aligned web search and page extraction tools.

Implements 1:1 provider parity with AstrBot's ``web_search_tools.py`` across
all nine backends (Tavily, BoCha, Brave, Firecrawl, Baidu AI Search, Exa,
AnySearch, DeepSeek, and SerpApi) using a persistent :class:`httpx.AsyncClient`
and concurrency-safe per-provider API key rotation with failover.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Self
from urllib.parse import quote, quote_plus, urlsplit

import httpx

from spectra.config import WebSearchConfig
from spectra.llm import ToolSpec
from spectra.providers import DEFAULT_USER_AGENT

logger = logging.getLogger(__name__)

WEB_SEARCH_TOOL_NAMES: tuple[str, ...] = (
    "web_search_baidu",
    "web_search_tavily",
    "tavily_extract_web_page",
    "web_search_bocha",
    "web_search_brave",
    "web_search_firecrawl",
    "firecrawl_extract_web_page",
    "web_search_exa",
    "exa_get_contents",
    "web_search_anysearch",
    "web_search_serpapi",
    "news_search_serpapi",
    "image_search_serpapi",
    "finance_search_serpapi",
    "scholar_search_serpapi",
    "web_search_deepseek",
    "deepseek_fetch_web_page",
)

# Exact per-provider retryable HTTP status codes aligned with AstrBot.
_TAVILY_RETRYABLE_STATUSES: frozenset[int] = frozenset({401, 403, 429, 432, 433})
_BOCHA_RETRYABLE_STATUSES: frozenset[int] = frozenset({401, 403, 429})
_BRAVE_RETRYABLE_STATUSES: frozenset[int] = frozenset({401, 403, 429})
_FIRECRAWL_RETRYABLE_STATUSES: frozenset[int] = frozenset({401, 402, 403, 429})
_EXA_RETRYABLE_STATUSES: frozenset[int] = frozenset({401, 402, 429})
_ANYSEARCH_RETRYABLE_STATUSES: frozenset[int] = frozenset({401, 402, 403, 429})
_DEEPSEEK_RETRYABLE_STATUSES: frozenset[int] = frozenset({401, 402, 429})
_SERPAPI_RETRYABLE_STATUSES: frozenset[int] = frozenset({401, 403, 429})

_SERPAPI_FINANCE_WINDOWS: frozenset[str] = frozenset(
    {"1D", "5D", "1M", "6M", "YTD", "1Y", "5Y", "MAX"}
)
_SERPAPI_API_KEY_PARAM_RE = re.compile(
    r"([?&]api_key=)[^&\s'\"<>\\)\]]+",
    re.IGNORECASE,
)
_SERPAPI_ENCODED_API_KEY_PARAM_RE = re.compile(
    r"((?:%3F|%26)api_key(?:%3D|=))(?:(?!(?:%26|&|\s|'|"
    r'"|<|>|\\|\)|\]))\S)+',
    re.IGNORECASE,
)
_HTTP_LOGGERS_TO_REDACT: tuple[str, ...] = (
    "httpx",
    "httpcore",
    "httpcore.connection",
    "httpcore.http11",
    "httpcore.http2",
    "httpcore.proxy",
    "httpcore.socks",
)


def _scrub_api_key_in_text(text: str) -> str:
    """Scrub ``api_key=`` query parameters (raw and percent-encoded) from text."""
    scrubbed = _SERPAPI_API_KEY_PARAM_RE.sub(r"\1[redacted]", text)
    return _SERPAPI_ENCODED_API_KEY_PARAM_RE.sub(r"\1[redacted]", scrubbed)


class _HttpLogRedactionFilter(logging.Filter):
    """Logging filter preventing ``api_key`` query params from leaking in HTTP logs."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            rendered = record.getMessage()
        except Exception:
            rendered = str(record.msg)
        if "api_key" in rendered.lower():
            record.msg = _scrub_api_key_in_text(rendered)
            record.args = None
        if isinstance(record.exc_text, str) and "api_key" in record.exc_text.lower():
            record.exc_text = _scrub_api_key_in_text(record.exc_text)
        return True


_HTTP_LOG_REDACTION_FILTER = _HttpLogRedactionFilter()


def _install_http_log_redaction_filter() -> None:
    """Idempotently attach the ``api_key`` filter to httpx/httpcore loggers."""
    target_names: set[str] = set(_HTTP_LOGGERS_TO_REDACT)
    for name in tuple(logging.root.manager.loggerDict):
        if name.startswith(("httpx.", "httpcore.")):
            target_names.add(name)
    for name in target_names:
        target_logger = logging.getLogger(name)
        if _HTTP_LOG_REDACTION_FILTER not in target_logger.filters:
            target_logger.addFilter(_HTTP_LOG_REDACTION_FILTER)


@dataclass(slots=True, frozen=True)
class SearchResult:
    """Normalized single web search hit."""

    title: str
    url: str
    snippet: str


def _normalize_keys(raw_keys: Sequence[str]) -> tuple[str, ...]:
    """Return stripped non-empty API keys in order without duplicates."""
    seen: set[str] = set()
    normalized: list[str] = []
    for raw in raw_keys:
        if not isinstance(raw, str):
            continue
        key = raw.strip()
        if key and key not in seen:
            seen.add(key)
            normalized.append(key)
    return tuple(normalized)


class _KeyRotator:
    """Concurrency-safe round-robin API key rotator with per-request failover.

    Each call returns the next untried key and advances the shared round-robin
    cursor under an :class:`asyncio.Lock`.
    """

    def __init__(self, provider_name: str, keys: Sequence[str]) -> None:
        self._provider_name = provider_name
        self._keys: tuple[str, ...] = _normalize_keys(keys)
        self._index = 0
        self._lock = asyncio.Lock()

    @property
    def key_count(self) -> int:
        """Return the number of distinct non-empty keys configured."""
        return len(self._keys)

    @property
    def keys(self) -> tuple[str, ...]:
        """Return the normalized tuple of configured keys."""
        return self._keys

    async def get(self, tried_keys: set[str] | None = None) -> str:
        """Return the next untried key and advance the round-robin index.

        Args:
            tried_keys: Request-local set of keys already attempted; updated in
                place with the selected key.

        Returns:
            The selected API key string.

        Raises:
            ValueError: If no keys are configured or all keys have been tried.
        """
        if not self._keys:
            raise ValueError(f"Error: {self._provider_name} API key is not configured.")

        async with self._lock:
            self._index %= len(self._keys)
            for _ in range(len(self._keys)):
                key = self._keys[self._index]
                self._index = (self._index + 1) % len(self._keys)
                if tried_keys is None or key not in tried_keys:
                    if tried_keys is not None:
                        tried_keys.add(key)
                    return key

        raise ValueError(f"Error: No untried {self._provider_name} API keys remain.")


def _as_text(value: Any) -> str:
    """Convert an optional JSON value into a string, mapping ``None`` to ``''``."""
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def _search_result_payload(results: Sequence[SearchResult]) -> str:
    """Serialize normalized results into AstrBot's indexed JSON payload."""
    ref_uuid = uuid.uuid4().hex[:4]
    items = [
        {
            "title": result.title,
            "url": result.url,
            "snippet": result.snippet,
            "index": f"{ref_uuid}.{idx}",
        }
        for idx, result in enumerate(results, 1)
    ]
    return json.dumps({"results": items}, ensure_ascii=False)


def _redact_secret(text: str, secret: str) -> str:
    """Remove all raw, URL-encoded, and query-param forms of ``secret``."""
    redacted = _scrub_api_key_in_text(text)
    if secret:
        for variant in (
            secret,
            quote(secret, safe=""),
            quote_plus(secret),
        ):
            if variant:
                redacted = redacted.replace(variant, "[redacted]")
    return redacted


def _is_brave_invalid_token_422(reason: str) -> bool:
    """Return whether a Brave HTTP 422 body indicates an invalid subscription token."""
    try:
        parsed = json.loads(reason)
    except json.JSONDecodeError:
        return "SUBSCRIPTION_TOKEN_INVALID" in reason
    if isinstance(parsed, dict):
        error = parsed.get("error")
        if isinstance(error, dict):
            return error.get("code") == "SUBSCRIPTION_TOKEN_INVALID"
        if parsed.get("code") == "SUBSCRIPTION_TOKEN_INVALID":
            return True
    return False


def _validate_non_empty_query(
    value: Any,
    *,
    strip: bool = True,
    max_length: int | None = None,
) -> tuple[str | None, str | None]:
    """Validate a required search query argument."""
    if not isinstance(value, str):
        return None, "Error: query must be a non-empty string."
    cleaned = value.strip() if strip else value
    if not cleaned or not value.strip():
        return None, "Error: query must not be empty."
    if max_length is not None and len(cleaned) > max_length:
        return (
            None,
            (
                f"Error: query must be a non-blank string of 1-{max_length} "
                "Unicode characters."
            ),
        )
    return cleaned, None


def _validate_optional_str(
    value: Any,
    *,
    name: str,
    max_length: int | None = None,
    strip: bool = True,
) -> tuple[str | None, str | None]:
    """Validate an optional string argument."""
    if value is None:
        return None, None
    if not isinstance(value, str):
        return None, f"Error: {name} must be a string."
    result = value.strip() if strip else value
    if max_length is not None and not 1 <= len(result) <= max_length:
        return (
            None,
            f"Error: {name} must be a string of 1-{max_length} Unicode characters.",
        )
    return result, None


def _validate_int_arg(
    value: Any,
    *,
    name: str,
    default: int,
    minimum: int | None = None,
    maximum: int | None = None,
    clamp: bool = False,
) -> tuple[int | None, str | None]:
    """Validate an optional integer argument, rejecting booleans and non-ints."""
    if value is None:
        return default, None
    if isinstance(value, bool) or not isinstance(value, int):
        return None, f"Error: {name} must be an integer."
    if clamp:
        if minimum is not None and value < minimum:
            value = minimum
        if maximum is not None and value > maximum:
            value = maximum
        return value, None
    if minimum is not None and value < minimum:
        if maximum is not None:
            return (
                None,
                f"Error: {name} must be an integer between {minimum} and {maximum}.",
            )
        return None, f"Error: {name} must be an integer >= {minimum}."
    if maximum is not None and value > maximum:
        if minimum is not None:
            return (
                None,
                f"Error: {name} must be an integer between {minimum} and {maximum}.",
            )
        return None, f"Error: {name} must be an integer <= {maximum}."
    return value, None


def _validate_bool_arg(
    value: Any,
    *,
    name: str,
    default: bool,
) -> tuple[bool | None, str | None]:
    """Validate an optional boolean argument."""
    if value is None:
        return default, None
    if not isinstance(value, bool):
        return None, f"Error: {name} must be a boolean."
    return value, None


def _validate_http_url(
    value: Any,
    *,
    max_length: int = 8192,
) -> tuple[str | None, str | None]:
    """Validate an HTTP(S) URL with a host, no credentials, and no whitespace."""
    if not isinstance(value, str) or not 1 <= len(value) <= max_length:
        return None, f"Error: url must be a string of 1-{max_length} characters."
    if any(ch.isspace() for ch in value):
        return (
            None,
            "Error: url must be an HTTP or HTTPS URL with a host, "
            "no embedded credentials, and no whitespace.",
        )
    try:
        parsed = urlsplit(value)
        # Port syntax and range are validated only when this property is read.
        _ = parsed.port
    except ValueError:
        return (
            None,
            "Error: url must be an HTTP or HTTPS URL with a host, "
            "no embedded credentials, and no whitespace.",
        )
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        return (
            None,
            "Error: url must be an HTTP or HTTPS URL with a host, "
            "no embedded credentials, and no whitespace.",
        )
    return value, None


def _parse_json_object(response: httpx.Response, *, provider: str) -> dict[str, Any]:
    """Parse a JSON object response body, raising ``RuntimeError`` on bad JSON."""
    try:
        data = response.json()
    except ValueError:
        raise RuntimeError(f"{provider} returned invalid JSON response.") from None
    if not isinstance(data, dict):
        raise RuntimeError(f"{provider} returned unexpected JSON response type.")
    return data


class WebSearchToolkit:
    """Async toolkit exposing AstrBot-aligned web search and page fetch tools.

    Args:
        config: Validated :class:`spectra.config.WebSearchConfig`.
        http_client: Optional externally managed :class:`httpx.AsyncClient`
            (for example with :class:`httpx.MockTransport` in tests). When
            omitted, a persistent client is created lazily and closed by
            :meth:`close`.
    """

    def __init__(
        self,
        config: WebSearchConfig,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._config = config
        self._http_client = http_client
        self._owns_client = http_client is None
        self._closed = False
        _install_http_log_redaction_filter()

        self._tavily_keys = _KeyRotator("Tavily", config.tavily_keys)
        self._bocha_keys = _KeyRotator("BoCha", config.bocha_keys)
        self._brave_keys = _KeyRotator("Brave", config.brave_keys)
        self._firecrawl_keys = _KeyRotator("Firecrawl", config.firecrawl_keys)
        self._exa_keys = _KeyRotator("Exa", config.exa_keys)
        self._anysearch_keys = _KeyRotator("AnySearch", config.anysearch_keys)
        self._deepseek_keys = _KeyRotator("DeepSeek", config.deepseek_keys)
        self._serpapi_keys = _KeyRotator("SerpApi", config.serpapi_keys)

    @property
    def config(self) -> WebSearchConfig:
        """Return the active :class:`WebSearchConfig`."""
        return self._config

    def _get_http_client(self) -> httpx.AsyncClient:
        if self._closed:
            raise RuntimeError("WebSearchToolkit has been closed.")
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                timeout=self._config.timeout_sec,
                headers={"User-Agent": DEFAULT_USER_AGENT},
            )
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

    def build_tools(self) -> tuple[ToolSpec, ...]:
        """Build :class:`ToolSpec` instances for the active web search provider.

        Returns:
            An empty tuple when ``config.enabled`` is ``False``; otherwise only
            the tools belonging to ``config.provider``.
        """
        if not self._config.enabled:
            return ()

        provider = self._config.provider
        if provider == "tavily":
            return (
                self._build_tavily_search_tool(),
                self._build_tavily_extract_tool(),
            )
        if provider == "bocha":
            return (self._build_bocha_search_tool(),)
        if provider == "brave":
            return (self._build_brave_search_tool(),)
        if provider == "firecrawl":
            return (
                self._build_firecrawl_search_tool(),
                self._build_firecrawl_extract_tool(),
            )
        if provider == "baidu_ai_search":
            return (self._build_baidu_search_tool(),)
        if provider == "exa":
            return (
                self._build_exa_search_tool(),
                self._build_exa_get_contents_tool(),
            )
        if provider == "anysearch":
            return (self._build_anysearch_search_tool(),)
        if provider == "deepseek":
            return (
                self._build_deepseek_search_tool(),
                self._build_deepseek_fetch_tool(),
            )
        if provider == "serpapi":
            return (
                self._build_serpapi_web_search_tool(),
                self._build_serpapi_news_search_tool(),
                self._build_serpapi_image_search_tool(),
                self._build_serpapi_finance_search_tool(),
                self._build_serpapi_scholar_search_tool(),
            )
        return ()

    # ------------------------------------------------------------------
    # Shared HTTP failover helper for header-authenticated JSON providers
    # ------------------------------------------------------------------

    async def _request_with_key_failover(
        self,
        *,
        rotator: _KeyRotator,
        provider_label: str,
        method: str,
        url: str,
        build_headers: Callable[[str], dict[str, str]],
        retryable_statuses: frozenset[int],
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        is_retryable_fn: Callable[[int, str], bool] | None = None,
    ) -> dict[str, Any]:
        """Execute an HTTP request rotating API keys on retryable statuses."""
        if rotator.key_count == 0:
            raise ValueError(f"Error: {provider_label} API key is not configured.")

        client = self._get_http_client()
        last_error: RuntimeError | None = None
        tried_keys: set[str] = set()

        for _ in range(rotator.key_count):
            api_key = await rotator.get(tried_keys)
            headers = build_headers(api_key)
            try:
                response = await client.request(
                    method,
                    url,
                    headers=headers,
                    json=json_body,
                    params=params,
                    timeout=self._config.timeout_sec,
                )
            except (httpx.HTTPError, TimeoutError, OSError) as exc:
                safe_detail = _redact_secret(str(exc), api_key)
                logger.warning("%s request failed: %s", provider_label, safe_detail)
                raise RuntimeError(
                    f"{provider_label} request failed or timed out: {safe_detail}"
                ) from None

            if response.status_code == 200:
                return _parse_json_object(response, provider=provider_label)

            reason = response.text
            retryable = response.status_code in retryable_statuses
            if not retryable and is_retryable_fn is not None:
                retryable = is_retryable_fn(response.status_code, reason)
            safe_reason = _redact_secret(reason, api_key)
            err = RuntimeError(
                f"{provider_label} failed: {safe_reason}, "
                f"status: {response.status_code}"
            )
            if retryable:
                last_error = err
                continue
            raise err

        if last_error is not None:
            raise last_error
        raise RuntimeError(f"{provider_label} failed with all configured keys.")

    # ------------------------------------------------------------------
    # Tavily
    # ------------------------------------------------------------------

    def _build_tavily_search_tool(self) -> ToolSpec:
        return ToolSpec(
            name="web_search_tavily",
            description=(
                "A web search tool that uses Tavily to search the web for relevant "
                "content. Ideal for gathering current information, news, and detailed "
                "web content analysis."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Required. Search query.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": (
                            "Optional. The maximum number of results to return. "
                            "Default is 7. Range is 5-20."
                        ),
                    },
                    "search_depth": {
                        "type": "string",
                        "description": (
                            "Optional. The depth of the search, must be one of "
                            '"basic", "advanced". Default is "basic".'
                        ),
                    },
                    "topic": {
                        "type": "string",
                        "description": (
                            "Optional. The topic of the search, must be one of "
                            '"general", "news". Default is "general".'
                        ),
                    },
                    "days": {
                        "type": "integer",
                        "description": (
                            "Optional. The number of days back from the current date "
                            "to include in the search results. This only applies when "
                            'topic is "news".'
                        ),
                    },
                    "time_range": {
                        "type": "string",
                        "description": (
                            "Optional. The time range back from the current date to "
                            'include in the search results. Must be one of "day", '
                            '"week", "month", "year".'
                        ),
                    },
                    "start_date": {
                        "type": "string",
                        "description": (
                            "Optional. The start date for the search results in the "
                            "format YYYY-MM-DD."
                        ),
                    },
                    "end_date": {
                        "type": "string",
                        "description": (
                            "Optional. The end date for the search results in the "
                            "format YYYY-MM-DD."
                        ),
                    },
                },
                "required": ["query"],
            },
            handler=self._web_search_tavily,
        )

    async def _web_search_tavily(self, **kwargs: Any) -> str:
        if self._tavily_keys.key_count == 0:
            return "Error: Tavily API key is not configured."

        query, err = _validate_non_empty_query(kwargs.get("query"))
        if err is not None:
            return err
        assert query is not None

        max_results, err = _validate_int_arg(
            kwargs.get("max_results"),
            name="max_results",
            default=7,
            minimum=1,
            maximum=20,
        )
        if err is not None:
            return err

        search_depth_raw = kwargs.get("search_depth", "basic")
        if search_depth_raw is not None and not isinstance(search_depth_raw, str):
            return "Error: search_depth must be a string."
        search_depth = (
            search_depth_raw if search_depth_raw in {"basic", "advanced"} else "basic"
        )

        topic_raw = kwargs.get("topic", "general")
        if topic_raw is not None and not isinstance(topic_raw, str):
            return "Error: topic must be a string."
        topic = topic_raw if topic_raw in {"general", "news"} else "general"

        payload: dict[str, Any] = {
            "query": query,
            "max_results": max_results,
            "include_favicon": True,
            "search_depth": search_depth,
            "topic": topic,
        }

        if topic == "news":
            days, err = _validate_int_arg(
                kwargs.get("days"),
                name="days",
                default=3,
                minimum=1,
            )
            if err is not None:
                return err
            payload["days"] = days

        start_date, err = _validate_optional_str(
            kwargs.get("start_date"), name="start_date"
        )
        if err is not None:
            return err
        end_date, err = _validate_optional_str(kwargs.get("end_date"), name="end_date")
        if err is not None:
            return err

        if start_date or end_date:
            if start_date:
                payload["start_date"] = start_date
            if end_date:
                payload["end_date"] = end_date
        else:
            time_range, err = _validate_optional_str(
                kwargs.get("time_range"), name="time_range"
            )
            if err is not None:
                return err
            if time_range in {"day", "week", "month", "year"}:
                payload["time_range"] = time_range

        data = await self._request_with_key_failover(
            rotator=self._tavily_keys,
            provider_label="Tavily web search",
            method="POST",
            url="https://api.tavily.com/search",
            build_headers=lambda key: {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            retryable_statuses=_TAVILY_RETRYABLE_STATUSES,
            json_body=payload,
        )
        raw_results = data.get("results")
        if not isinstance(raw_results, list) or not raw_results:
            return "Error: Tavily web searcher does not return any results."
        results = [
            SearchResult(
                title=_as_text(item.get("title")),
                url=_as_text(item.get("url")),
                snippet=_as_text(item.get("content")),
            )
            for item in raw_results
            if isinstance(item, dict)
        ]
        if not results:
            return "Error: Tavily web searcher does not return any results."
        return _search_result_payload(results)

    def _build_tavily_extract_tool(self) -> ToolSpec:
        return ToolSpec(
            name="tavily_extract_web_page",
            description="Extract the content of a web page using Tavily.",
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Required. A URL to extract content from.",
                    },
                    "extract_depth": {
                        "type": "string",
                        "description": (
                            "Optional. The depth of the extraction, must be one of "
                            '"basic", "advanced". Default is "basic".'
                        ),
                    },
                },
                "required": ["url"],
            },
            handler=self._tavily_extract_web_page,
        )

    async def _tavily_extract_web_page(self, **kwargs: Any) -> str:
        if self._tavily_keys.key_count == 0:
            return "Error: Tavily API key is not configured."

        url, err = _validate_http_url(kwargs.get("url"))
        if err is not None:
            return err
        assert url is not None

        depth_raw = kwargs.get("extract_depth", "basic")
        if depth_raw is not None and not isinstance(depth_raw, str):
            return "Error: extract_depth must be a string."
        extract_depth = depth_raw if depth_raw in {"basic", "advanced"} else "basic"

        data = await self._request_with_key_failover(
            rotator=self._tavily_keys,
            provider_label="Tavily web extract",
            method="POST",
            url="https://api.tavily.com/extract",
            build_headers=lambda key: {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            retryable_statuses=_TAVILY_RETRYABLE_STATUSES,
            json_body={"urls": [url], "extract_depth": extract_depth},
        )
        raw_results = data.get("results")
        if not isinstance(raw_results, list) or not raw_results:
            return "Error: Tavily web searcher does not return any results."
        lines: list[str] = []
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            lines.append(f"URL: {item.get('url', 'No URL')}")
            lines.append(f"Content: {item.get('raw_content', 'No content')}")
        return "\n".join(lines) or (
            "Error: Tavily web searcher does not return any results."
        )

    # ------------------------------------------------------------------
    # BoCha
    # ------------------------------------------------------------------

    def _build_bocha_search_tool(self) -> ToolSpec:
        return ToolSpec(
            name="web_search_bocha",
            description=(
                "A web search tool based on Bocha Search API, used to retrieve web "
                "pages related to the user's query."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Required. User's search query.",
                    },
                    "freshness": {
                        "type": "string",
                        "description": (
                            "Optional. Time range of the search. "
                            'Recommended value is "noLimit".'
                        ),
                    },
                    "summary": {
                        "type": "boolean",
                        "description": (
                            "Optional. Whether to include a text summary for each "
                            "search result."
                        ),
                    },
                    "include": {
                        "type": "string",
                        "description": (
                            "Optional. Domains to include in the search, "
                            "separated by | or ,."
                        ),
                    },
                    "exclude": {
                        "type": "string",
                        "description": (
                            "Optional. Domains to exclude from the search, "
                            "separated by | or ,."
                        ),
                    },
                    "count": {
                        "type": "integer",
                        "description": (
                            "Optional. Number of search results to return. Range: 1-50."
                        ),
                    },
                },
                "required": ["query"],
            },
            handler=self._web_search_bocha,
        )

    async def _web_search_bocha(self, **kwargs: Any) -> str:
        if self._bocha_keys.key_count == 0:
            return "Error: BoCha API key is not configured."

        query, err = _validate_non_empty_query(kwargs.get("query"))
        if err is not None:
            return err
        assert query is not None

        count, err = _validate_int_arg(
            kwargs.get("count"), name="count", default=10, minimum=1, maximum=50
        )
        if err is not None:
            return err

        summary, err = _validate_bool_arg(
            kwargs.get("summary"), name="summary", default=False
        )
        if err is not None:
            return err

        payload: dict[str, Any] = {
            "query": query,
            "count": count,
            "summary": summary,
        }
        for opt_name in ("freshness", "include", "exclude"):
            val, err = _validate_optional_str(kwargs.get(opt_name), name=opt_name)
            if err is not None:
                return err
            if val:
                payload[opt_name] = val

        data = await self._request_with_key_failover(
            rotator=self._bocha_keys,
            provider_label="BoCha web search",
            method="POST",
            url="https://api.bochaai.com/v1/web-search",
            build_headers=lambda key: {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            retryable_statuses=_BOCHA_RETRYABLE_STATUSES,
            json_body=payload,
        )
        inner = data.get("data")
        web_pages = inner.get("webPages") if isinstance(inner, dict) else None
        rows = web_pages.get("value") if isinstance(web_pages, dict) else None
        if not isinstance(rows, list) or not rows:
            return "Error: BoCha web searcher does not return any results."
        results = [
            SearchResult(
                title=_as_text(item.get("name")),
                url=_as_text(item.get("url")),
                snippet=_as_text(item.get("snippet")),
            )
            for item in rows
            if isinstance(item, dict)
        ]
        if not results:
            return "Error: BoCha web searcher does not return any results."
        return _search_result_payload(results)

    # ------------------------------------------------------------------
    # Brave
    # ------------------------------------------------------------------

    def _build_brave_search_tool(self) -> ToolSpec:
        return ToolSpec(
            name="web_search_brave",
            description="A web search tool based on Brave Search API.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Required. Search query.",
                    },
                    "count": {
                        "type": "integer",
                        "description": (
                            "Optional. Number of results to return. Range: 1-20."
                        ),
                    },
                    "country": {
                        "type": "string",
                        "description": (
                            "Optional. Country code for region-specific results, "
                            'for example "US" or "CN".'
                        ),
                    },
                    "search_lang": {
                        "type": "string",
                        "description": (
                            'Optional. Brave language code, for example "zh-hans" '
                            'or "en".'
                        ),
                    },
                    "freshness": {
                        "type": "string",
                        "description": (
                            'Optional. One of "day", "week", "month", "year".'
                        ),
                    },
                },
                "required": ["query"],
            },
            handler=self._web_search_brave,
        )

    async def _web_search_brave(self, **kwargs: Any) -> str:
        if self._brave_keys.key_count == 0:
            return "Error: Brave API key is not configured."

        query, err = _validate_non_empty_query(kwargs.get("query"))
        if err is not None:
            return err
        assert query is not None

        count, err = _validate_int_arg(
            kwargs.get("count"),
            name="count",
            default=10,
            minimum=1,
            maximum=20,
            clamp=True,
        )
        if err is not None:
            return err

        country, err = _validate_optional_str(kwargs.get("country"), name="country")
        if err is not None:
            return err
        search_lang, err = _validate_optional_str(
            kwargs.get("search_lang"), name="search_lang"
        )
        if err is not None:
            return err
        freshness, err = _validate_optional_str(
            kwargs.get("freshness"), name="freshness"
        )
        if err is not None:
            return err

        params: dict[str, Any] = {
            "q": query,
            "count": count,
            "country": country or "US",
            "search_lang": search_lang or "zh-hans",
        }
        if freshness in {"day", "week", "month", "year"}:
            params["freshness"] = freshness

        data = await self._request_with_key_failover(
            rotator=self._brave_keys,
            provider_label="Brave web search",
            method="GET",
            url="https://api.search.brave.com/res/v1/web/search",
            build_headers=lambda key: {
                "Accept": "application/json",
                "X-Subscription-Token": key,
            },
            retryable_statuses=_BRAVE_RETRYABLE_STATUSES,
            params=params,
            is_retryable_fn=lambda status, reason: (
                status == 422 and _is_brave_invalid_token_422(reason)
            ),
        )
        web_section = data.get("web")
        rows = web_section.get("results") if isinstance(web_section, dict) else None
        if not isinstance(rows, list) or not rows:
            return "Error: Brave web searcher does not return any results."
        results = [
            SearchResult(
                title=_as_text(item.get("title")),
                url=_as_text(item.get("url")),
                snippet=_as_text(item.get("description")),
            )
            for item in rows
            if isinstance(item, dict)
        ]
        if not results:
            return "Error: Brave web searcher does not return any results."
        return _search_result_payload(results)

    # ------------------------------------------------------------------
    # Firecrawl
    # ------------------------------------------------------------------

    def _build_firecrawl_search_tool(self) -> ToolSpec:
        return ToolSpec(
            name="web_search_firecrawl",
            description=(
                "A web search tool based on Firecrawl Search API, used to retrieve "
                "web pages related to the user's query."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Required. Search query.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": (
                            "Optional. Number of results to return. Range: 1-100. "
                            "Default is 5."
                        ),
                    },
                    "location": {
                        "type": "string",
                        "description": (
                            "Optional. Geographic location for search results."
                        ),
                    },
                    "country": {
                        "type": "string",
                        "description": (
                            "Optional. Country code for search results, for example "
                            '"US" or "CN".'
                        ),
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Optional. Request timeout in milliseconds.",
                    },
                },
                "required": ["query"],
            },
            handler=self._web_search_firecrawl,
        )

    async def _web_search_firecrawl(self, **kwargs: Any) -> str:
        if self._firecrawl_keys.key_count == 0:
            return "Error: Firecrawl API key is not configured."

        query, err = _validate_non_empty_query(kwargs.get("query"))
        if err is not None:
            return err
        assert query is not None

        limit, err = _validate_int_arg(
            kwargs.get("limit"), name="limit", default=5, minimum=1, maximum=100
        )
        if err is not None:
            return err

        payload: dict[str, Any] = {
            "query": query,
            "limit": limit,
            "sources": ["web"],
        }
        for opt_name in ("location", "country"):
            val, err = _validate_optional_str(kwargs.get(opt_name), name=opt_name)
            if err is not None:
                return err
            if val:
                payload[opt_name] = val

        if kwargs.get("timeout") is not None:
            timeout_ms, err = _validate_int_arg(
                kwargs.get("timeout"), name="timeout", default=0, minimum=1
            )
            if err is not None:
                return err
            payload["timeout"] = timeout_ms

        data = await self._request_with_key_failover(
            rotator=self._firecrawl_keys,
            provider_label="Firecrawl web search",
            method="POST",
            url="https://api.firecrawl.dev/v2/search",
            build_headers=lambda key: {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            retryable_statuses=_FIRECRAWL_RETRYABLE_STATUSES,
            json_body=payload,
        )
        rows = data.get("data", [])
        if isinstance(rows, dict):
            rows = rows.get("web", [])
        if not isinstance(rows, list) or not rows:
            return "Error: Firecrawl web searcher does not return any results."
        results = [
            SearchResult(
                title=_as_text(item.get("title")),
                url=_as_text(item.get("url")),
                snippet=_as_text(
                    item.get("description")
                    or item.get("snippet")
                    or item.get("markdown")
                ),
            )
            for item in rows
            if isinstance(item, dict) and item.get("url")
        ]
        if not results:
            return "Error: Firecrawl web searcher does not return any results."
        return _search_result_payload(results)

    def _build_firecrawl_extract_tool(self) -> ToolSpec:
        return ToolSpec(
            name="firecrawl_extract_web_page",
            description="Extract the content of a web page using Firecrawl.",
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Required. A URL to extract content from.",
                    },
                    "format": {
                        "type": "string",
                        "description": (
                            'Optional. Output format, one of "markdown", "html", '
                            '"rawHtml", "summary". Default is "markdown".'
                        ),
                    },
                    "only_main_content": {
                        "type": "boolean",
                        "description": (
                            "Optional. Whether to extract only the main page content. "
                            "Default is true."
                        ),
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Optional. Request timeout in milliseconds.",
                    },
                    "max_age": {
                        "type": "integer",
                        "description": "Optional. Maximum cache age in milliseconds.",
                    },
                },
                "required": ["url"],
            },
            handler=self._firecrawl_extract_web_page,
        )

    async def _firecrawl_extract_web_page(self, **kwargs: Any) -> str:
        if self._firecrawl_keys.key_count == 0:
            return "Error: Firecrawl API key is not configured."

        url, err = _validate_http_url(kwargs.get("url"))
        if err is not None:
            return err
        assert url is not None

        format_raw = kwargs.get("format", "markdown")
        if format_raw is not None and not isinstance(format_raw, str):
            return "Error: format must be a string."
        output_format = (
            format_raw
            if format_raw in {"markdown", "html", "rawHtml", "summary"}
            else "markdown"
        )

        only_main_content, err = _validate_bool_arg(
            kwargs.get("only_main_content"),
            name="only_main_content",
            default=True,
        )
        if err is not None:
            return err

        payload: dict[str, Any] = {
            "url": url,
            "formats": [output_format],
            "onlyMainContent": only_main_content,
        }
        if kwargs.get("timeout") is not None:
            timeout_ms, err = _validate_int_arg(
                kwargs.get("timeout"), name="timeout", default=0, minimum=1
            )
            if err is not None:
                return err
            payload["timeout"] = timeout_ms
        if kwargs.get("max_age") is not None:
            max_age, err = _validate_int_arg(
                kwargs.get("max_age"), name="max_age", default=0, minimum=0
            )
            if err is not None:
                return err
            payload["maxAge"] = max_age

        data = await self._request_with_key_failover(
            rotator=self._firecrawl_keys,
            provider_label="Firecrawl web scraper",
            method="POST",
            url="https://api.firecrawl.dev/v2/scrape",
            build_headers=lambda key: {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            retryable_statuses=_FIRECRAWL_RETRYABLE_STATUSES,
            json_body=payload,
        )
        result = data.get("data")
        if not isinstance(result, dict) or not result:
            return "Error: Firecrawl web scraper does not return any results."
        content = _as_text(result.get(output_format))
        result_url = _as_text(result.get("url")) or url
        if not content:
            return "Error: Firecrawl web scraper does not return any results."
        return f"URL: {result_url}\nContent: {content}"

    # ------------------------------------------------------------------
    # Baidu AI Search
    # ------------------------------------------------------------------

    def _build_baidu_search_tool(self) -> ToolSpec:
        return ToolSpec(
            name="web_search_baidu",
            description=(
                "A web search tool based on Baidu AI Search. Use this for real-time "
                "web retrieval when Baidu AI Search is configured."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Required. Search query.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": (
                            "Optional. Number of web results to return. Maximum 50. "
                            "Default is 10."
                        ),
                    },
                    "search_recency_filter": {
                        "type": "string",
                        "description": (
                            'Optional. One of "week", "month", "semiyear", "year".'
                        ),
                    },
                    "site": {
                        "type": "string",
                        "description": (
                            "Optional. Restrict search to specific sites, "
                            "separated by commas."
                        ),
                    },
                },
                "required": ["query"],
            },
            handler=self._web_search_baidu,
        )

    async def _web_search_baidu(self, **kwargs: Any) -> str:
        api_key = self._config.baidu_app_builder_key.strip()
        if not api_key:
            return "Error: Baidu AI Search API key is not configured."

        query, err = _validate_non_empty_query(kwargs.get("query"))
        if err is not None:
            return err
        assert query is not None

        top_k, err = _validate_int_arg(
            kwargs.get("top_k"),
            name="top_k",
            default=10,
            minimum=1,
            maximum=50,
            clamp=True,
        )
        if err is not None:
            return err

        payload: dict[str, Any] = {
            "messages": [{"role": "user", "content": query[:72]}],
            "search_source": "baidu_search_v2",
            "resource_type_filter": [{"type": "web", "top_k": top_k}],
        }

        recency, err = _validate_optional_str(
            kwargs.get("search_recency_filter"), name="search_recency_filter"
        )
        if err is not None:
            return err
        if recency in {"week", "month", "semiyear", "year"}:
            payload["search_recency_filter"] = recency

        site, err = _validate_optional_str(kwargs.get("site"), name="site")
        if err is not None:
            return err
        if site:
            sites = [s.strip() for s in site.replace("|", ",").split(",") if s.strip()]
            if sites:
                payload["search_filter"] = {"match": {"site": sites[:100]}}

        client = self._get_http_client()
        try:
            response = await client.post(
                "https://qianfan.baidubce.com/v2/ai_search/web_search",
                json=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "X-Appbuilder-Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self._config.timeout_sec,
            )
        except (httpx.HTTPError, TimeoutError, OSError) as exc:
            safe_detail = _redact_secret(str(exc), api_key)
            logger.warning("Baidu AI Search request failed: %s", safe_detail)
            raise RuntimeError(
                f"Baidu AI Search request failed or timed out: {safe_detail}"
            ) from None

        if response.status_code != 200:
            safe_reason = _redact_secret(response.text, api_key)
            raise RuntimeError(
                f"Baidu AI Search failed: {safe_reason}, status: {response.status_code}"
            )
        data = _parse_json_object(response, provider="Baidu AI Search")
        references = data.get("references")
        if not isinstance(references, list) or not references:
            return "Error: Baidu AI Search does not return any results."
        results = [
            SearchResult(
                title=_as_text(item.get("title")),
                url=_as_text(item.get("url")),
                snippet=_as_text(item.get("content")),
            )
            for item in references
            if isinstance(item, dict) and item.get("url")
        ]
        if not results:
            return "Error: Baidu AI Search does not return any results."
        return _search_result_payload(results)

    # ------------------------------------------------------------------
    # Exa
    # ------------------------------------------------------------------

    def _build_exa_search_tool(self) -> ToolSpec:
        return ToolSpec(
            name="web_search_exa",
            description=(
                "A web search tool powered by Exa, an AI-native search engine. "
                "Supports keyword and semantic search with domain, date, and "
                "category filters."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Required. Search query.",
                    },
                    "num_results": {
                        "type": "integer",
                        "description": (
                            "Optional. Number of results to return. Default is 10."
                        ),
                    },
                    "type": {
                        "type": "string",
                        "description": (
                            'Optional. Search type. One of "auto", "keyword", '
                            '"neural". Default is "auto".'
                        ),
                    },
                    "category": {
                        "type": "string",
                        "description": (
                            "Optional. Category filter. One of "
                            '"company", "research paper", "news", "github", '
                            '"tweet", "personal site", "pdf", "linkedin profile".'
                        ),
                    },
                    "include_domains": {
                        "type": "string",
                        "description": (
                            "Optional. Comma-separated domains to restrict results to."
                        ),
                    },
                    "exclude_domains": {
                        "type": "string",
                        "description": (
                            "Optional. Comma-separated domains to exclude from results."
                        ),
                    },
                    "start_published_date": {
                        "type": "string",
                        "description": (
                            "Optional. Start date filter in ISO 8601 format "
                            "(e.g. 2024-01-01T00:00:00.000Z)."
                        ),
                    },
                    "end_published_date": {
                        "type": "string",
                        "description": (
                            "Optional. End date filter in ISO 8601 format."
                        ),
                    },
                },
                "required": ["query"],
            },
            handler=self._web_search_exa,
        )

    async def _web_search_exa(self, **kwargs: Any) -> str:
        if self._exa_keys.key_count == 0:
            return "Error: Exa API key is not configured."

        query, err = _validate_non_empty_query(kwargs.get("query"))
        if err is not None:
            return err
        assert query is not None

        num_results, err = _validate_int_arg(
            kwargs.get("num_results"),
            name="num_results",
            default=10,
            minimum=1,
            clamp=True,
        )
        if err is not None:
            return err

        search_type_raw = kwargs.get("type", "auto")
        if search_type_raw is not None and not isinstance(search_type_raw, str):
            return "Error: type must be a string."
        search_type = (
            search_type_raw
            if search_type_raw in {"auto", "keyword", "neural"}
            else "auto"
        )

        payload: dict[str, Any] = {
            "query": query,
            "numResults": num_results,
            "type": search_type,
            "contents": {"text": {"maxCharacters": 500}},
        }

        category, err = _validate_optional_str(kwargs.get("category"), name="category")
        if err is not None:
            return err
        if category:
            payload["category"] = category

        include_domains, err = _validate_optional_str(
            kwargs.get("include_domains"), name="include_domains"
        )
        if err is not None:
            return err
        if include_domains:
            domains = [d.strip() for d in include_domains.split(",") if d.strip()]
            if domains:
                payload["includeDomains"] = domains

        exclude_domains, err = _validate_optional_str(
            kwargs.get("exclude_domains"), name="exclude_domains"
        )
        if err is not None:
            return err
        if exclude_domains:
            domains = [d.strip() for d in exclude_domains.split(",") if d.strip()]
            if domains:
                payload["excludeDomains"] = domains

        start_date, err = _validate_optional_str(
            kwargs.get("start_published_date"), name="start_published_date"
        )
        if err is not None:
            return err
        if start_date:
            payload["startPublishedDate"] = start_date

        end_date, err = _validate_optional_str(
            kwargs.get("end_published_date"), name="end_published_date"
        )
        if err is not None:
            return err
        if end_date:
            payload["endPublishedDate"] = end_date

        data = await self._request_with_key_failover(
            rotator=self._exa_keys,
            provider_label="Exa web search",
            method="POST",
            url="https://api.exa.ai/search",
            build_headers=lambda key: {
                "x-api-key": key,
                "Content-Type": "application/json",
            },
            retryable_statuses=_EXA_RETRYABLE_STATUSES,
            json_body=payload,
        )
        raw_results = data.get("results")
        if not isinstance(raw_results, list) or not raw_results:
            return "Error: Exa web search does not return any results."
        results: list[SearchResult] = []
        for item in raw_results:
            if not isinstance(item, dict) or not item.get("url"):
                continue
            highlights = item.get("highlights")
            first_highlight = (
                highlights[0] if isinstance(highlights, list) and highlights else ""
            )
            snippet = _as_text(
                item.get("text") or first_highlight or item.get("summary")
            )
            results.append(
                SearchResult(
                    title=_as_text(item.get("title")),
                    url=_as_text(item.get("url")),
                    snippet=snippet,
                )
            )
        if not results:
            return "Error: Exa web search does not return any results."
        return _search_result_payload(results)

    def _build_exa_get_contents_tool(self) -> ToolSpec:
        return ToolSpec(
            name="exa_get_contents",
            description="Extract the content of a web page using Exa.",
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Required. A URL to extract content from.",
                    },
                    "max_characters": {
                        "type": "integer",
                        "description": (
                            "Optional. Maximum number of characters to return. "
                            "Default is 3000."
                        ),
                    },
                },
                "required": ["url"],
            },
            handler=self._exa_get_contents,
        )

    async def _exa_get_contents(self, **kwargs: Any) -> str:
        if self._exa_keys.key_count == 0:
            return "Error: Exa API key is not configured."

        url, err = _validate_http_url(kwargs.get("url"))
        if err is not None:
            return err
        assert url is not None

        max_characters, err = _validate_int_arg(
            kwargs.get("max_characters"),
            name="max_characters",
            default=3000,
            minimum=1,
        )
        if err is not None:
            return err

        data = await self._request_with_key_failover(
            rotator=self._exa_keys,
            provider_label="Exa get contents",
            method="POST",
            url="https://api.exa.ai/contents",
            build_headers=lambda key: {
                "x-api-key": key,
                "Content-Type": "application/json",
            },
            retryable_statuses=_EXA_RETRYABLE_STATUSES,
            json_body={
                "ids": [url],
                "text": {"maxCharacters": max_characters},
            },
        )
        raw_results = data.get("results")
        if not isinstance(raw_results, list) or not raw_results:
            return "Error: Exa get contents does not return any results."
        lines: list[str] = []
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            lines.append(f"URL: {item.get('url', 'No URL')}")
            lines.append(f"Content: {item.get('text', 'No content')}")
        return "\n".join(lines) or (
            "Error: Exa get contents does not return any results."
        )

    # ------------------------------------------------------------------
    # AnySearch (supports anonymous mode when anysearch_keys is empty)
    # ------------------------------------------------------------------

    def _build_anysearch_search_tool(self) -> ToolSpec:
        return ToolSpec(
            name="web_search_anysearch",
            description=(
                "A web search tool powered by AnySearch. Supports general web search "
                "and 16 vertical domains: "
                "academic(search/biomedical/citation/preprint/dataset), "
                "business(company/jobs/people/trade), code(doc/snippet), "
                "energy(production/electricity), environment(aqi), "
                "finance(quote/fundamental/news/calendar/screen/macro), "
                "film(torrent), gaming(esports/store), health(drug/stats/trial), "
                "ip(global), legal(case/statute/legislation), resource(image), "
                "security(vuln/noise/intel/scan), social_media, "
                "travel(flight/flight_status), agriculture(fao), "
                "and general web search."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Required. Search query.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": (
                            "Optional. The maximum number of results to return. "
                            "Default is 10. Range is 1-10."
                        ),
                    },
                    "tag": {
                        "type": "string",
                        "description": (
                            'Optional. Domain capability tag in "{domain}.{subdomain}" '
                            'form, for example "finance.quote" or "academic.search". '
                            "Available domains: general, resource, social_media, "
                            "finance(quote/fundamental/news/calendar/screen/macro), "
                            "academic(search/biomedical/citation/preprint/dataset), "
                            "legal(case/statute/legislation), "
                            "health(drug/stats/trial), "
                            "business(company/jobs/people/trade), "
                            "security(vuln/noise/intel/scan), ip(global), "
                            "code(doc/snippet), energy(production/electricity), "
                            "environment(aqi), agriculture(fao), "
                            "travel(flight/flight_status), film(torrent), "
                            "gaming(esports/store). "
                            "Omit for general web search."
                        ),
                    },
                    "zone": {
                        "type": "string",
                        "description": (
                            'Optional. Result region, must be one of "cn", "intl", '
                            '"global".'
                        ),
                    },
                    "language": {
                        "type": "string",
                        "description": (
                            'Optional. Preferred result language, for example "zh-CN" '
                            'or "en".'
                        ),
                    },
                    "params": {
                        "type": "object",
                        "description": (
                            "Optional. Extra parameters required by specific vertical "
                            "tags. "
                            'Examples: {"symbol": "AAPL", "type": "stock"} for '
                            "finance.quote, "
                            '{"type": "cve", "value": "CVE-2021-44228"} for '
                            "security.vuln, "
                            '{"doi": "10.1038/s41586-021-03819-2"} for '
                            "academic.search, "
                            '{"departure": "SHA", "arrival": "PEK", '
                            '"date": "2026-09-10"} for travel.flight.'
                        ),
                    },
                },
                "required": ["query"],
            },
            handler=self._web_search_anysearch,
        )

    async def _web_search_anysearch(self, **kwargs: Any) -> str:
        query, err = _validate_non_empty_query(kwargs.get("query"))
        if err is not None:
            return err
        assert query is not None

        max_results, err = _validate_int_arg(
            kwargs.get("max_results"),
            name="max_results",
            default=10,
            minimum=1,
            maximum=10,
            clamp=True,
        )
        if err is not None:
            return err

        payload: dict[str, Any] = {
            "query": query,
            "max_results": max_results,
            "format": "json",
        }

        tag, err = _validate_optional_str(kwargs.get("tag"), name="tag")
        if err is not None:
            return err
        if tag:
            payload["tag"] = tag

        zone, err = _validate_optional_str(kwargs.get("zone"), name="zone")
        if err is not None:
            return err
        if zone in {"cn", "intl", "global"}:
            payload["zone"] = zone

        language, err = _validate_optional_str(kwargs.get("language"), name="language")
        if err is not None:
            return err
        if language:
            payload["language"] = language

        extra_params = kwargs.get("params")
        if extra_params is not None:
            if not isinstance(extra_params, dict):
                return "Error: params must be a JSON object."
            payload["params"] = extra_params

        data = await self._anysearch_request(payload)
        code = data.get("code")
        if code not in (None, 0):
            message = str(data.get("message") or code)
            for key in self._anysearch_keys.keys:
                message = _redact_secret(message, key)
            message = _redact_secret(message, "")
            raise RuntimeError(f"AnySearch web search failed: {message}")
        body = data.get("data") if isinstance(data.get("data"), dict) else data
        raw_results = body.get("results")
        if not isinstance(raw_results, list) or not raw_results:
            return "Error: AnySearch web search does not return any results."

        results: list[SearchResult] = []
        for item in raw_results:
            if not isinstance(item, dict) or not item.get("url"):
                continue
            snippet = _as_text(item.get("snippet") or item.get("content"))
            extras: list[str] = []
            for key, value in item.items():
                if (
                    key in {"title", "url", "snippet", "content", "favicon"}
                    or value is None
                ):
                    continue
                if isinstance(value, (dict, list)):
                    formatted = json.dumps(value, ensure_ascii=False)
                elif isinstance(value, (str, int, float, bool)):
                    formatted = str(value)
                else:
                    continue
                extras.append(f"{key}: {formatted}")
            if extras:
                snippet = "\n".join(filter(None, [snippet, *extras])).strip()
            results.append(
                SearchResult(
                    title=_as_text(item.get("title")),
                    url=_as_text(item.get("url")),
                    snippet=snippet,
                )
            )
        if not results:
            return "Error: AnySearch web search does not return any results."
        return _search_result_payload(results)

    async def _anysearch_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        client = self._get_http_client()
        key_count = self._anysearch_keys.key_count
        if key_count == 0:
            # Anonymous mode: single request without Authorization header.
            try:
                response = await client.post(
                    "https://api.anysearch.com/v1/search",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=self._config.timeout_sec,
                )
            except (httpx.HTTPError, TimeoutError, OSError) as exc:
                safe_detail = _redact_secret(str(exc), "")
                logger.warning("AnySearch web search request failed: %s", safe_detail)
                raise RuntimeError(
                    f"AnySearch web search request failed or timed out: {safe_detail}"
                ) from None
            if response.status_code == 200:
                return _parse_json_object(response, provider="AnySearch web search")
            safe_reason = _redact_secret(response.text, "")
            raise RuntimeError(
                f"AnySearch web search failed: {safe_reason}, "
                f"status: {response.status_code}"
            )

        return await self._request_with_key_failover(
            rotator=self._anysearch_keys,
            provider_label="AnySearch web search",
            method="POST",
            url="https://api.anysearch.com/v1/search",
            build_headers=lambda key: {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            retryable_statuses=_ANYSEARCH_RETRYABLE_STATUSES,
            json_body=payload,
        )

    # ------------------------------------------------------------------
    # DeepSeek
    # ------------------------------------------------------------------

    def _build_deepseek_search_tool(self) -> ToolSpec:
        return ToolSpec(
            name="web_search_deepseek",
            description=(
                "Search the web using DeepSeek and return summaries with source "
                "references. Use deepseek_fetch_web_page for full page content "
                "when needed."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 128,
                        "description": (
                            "Search query, 1-128 Unicode characters including filters, "
                            "not all whitespace. Filters can be combined. Examples:\n"
                            "Find DeepSeek API references on GitHub or Hugging Face "
                            '-> DeepSeek "API reference" site:github.com OR '
                            "site:huggingface.co\n"
                            "Find Chinese or English DeepSeek news published after "
                            "2026-01-01 and before 2027-01-01 -> "
                            "DeepSeek news lang:zh,en after:2026-01-01 "
                            "before:2027-01-01\n"
                            "Find DeepSeek information excluding ads, example.com and "
                            'the phrase "release notes" -> DeepSeek -Ads '
                            '-site:example.com -"release notes"\n'
                            "Date filters accept YYYY or YYYY-MM-DD."
                        ),
                    },
                    "count": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 50,
                        "default": 10,
                        "description": (
                            "Optional. Number of results, 1-50. Default is 10."
                        ),
                    },
                    "ori_question": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 128,
                        "description": (
                            "Optional. Original user question, "
                            "1-128 Unicode characters."
                        ),
                    },
                    "need_content": {
                        "type": "boolean",
                        "default": False,
                        "description": (
                            "Optional. Include full page content in results. "
                            "Default is false."
                        ),
                    },
                },
                "required": ["query"],
            },
            handler=self._web_search_deepseek,
        )

    async def _web_search_deepseek(self, **kwargs: Any) -> str:
        if self._deepseek_keys.key_count == 0:
            return "Error: DeepSeek API key is not configured."

        query, err = _validate_non_empty_query(
            kwargs.get("query"), strip=False, max_length=128
        )
        if err is not None:
            return (
                "Error: query must be a non-blank string of 1-128 Unicode characters."
            )
        assert query is not None

        count, err = _validate_int_arg(
            kwargs.get("count"), name="count", default=10, minimum=1, maximum=50
        )
        if err is not None:
            return "Error: count must be an integer between 1 and 50."

        ori_question, err = _validate_optional_str(
            kwargs.get("ori_question"),
            name="ori_question",
            max_length=128,
            strip=False,
        )
        if err is not None:
            return "Error: ori_question must be a string of 1-128 Unicode characters."

        need_content, err = _validate_bool_arg(
            kwargs.get("need_content"), name="need_content", default=False
        )
        if err is not None:
            return err

        payload: dict[str, Any] = {
            "query": query,
            "count": count,
            "need_content": need_content,
        }
        if ori_question is not None:
            payload["ori_question"] = ori_question

        data = await self._request_with_key_failover(
            rotator=self._deepseek_keys,
            provider_label="DeepSeek web search",
            method="POST",
            url="https://api.deepseek.com/v1/tools/search",
            build_headers=lambda key: {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            retryable_statuses=_DEEPSEEK_RETRYABLE_STATUSES,
            json_body=payload,
        )
        raw_results = data.get("results")
        if not isinstance(raw_results, list) or not raw_results:
            return "Error: DeepSeek web search does not return any results."
        results: list[SearchResult] = []
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            snippet = _as_text(item.get("summary") or item.get("snippet"))
            if need_content and item.get("content"):
                snippet = "\n\n".join(
                    filter(None, [snippet, _as_text(item.get("content"))])
                )
            results.append(
                SearchResult(
                    title=_as_text(item.get("title")),
                    url=_as_text(item.get("url")),
                    snippet=snippet,
                )
            )
        if not results:
            return "Error: DeepSeek web search does not return any results."
        return _search_result_payload(results)

    def _build_deepseek_fetch_tool(self) -> ToolSpec:
        return ToolSpec(
            name="deepseek_fetch_web_page",
            description=(
                "Fetch a single web page using DeepSeek and return its title and "
                "Markdown content. Use this to read a known URL, including a page "
                "found through web search."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 8192,
                        "description": (
                            "Required. HTTP or HTTPS URL with a host, no embedded "
                            "credentials, and at most 8192 characters."
                        ),
                    },
                },
                "required": ["url"],
            },
            handler=self._deepseek_fetch_web_page,
        )

    async def _deepseek_fetch_web_page(self, **kwargs: Any) -> str:
        if self._deepseek_keys.key_count == 0:
            return "Error: DeepSeek API key is not configured."

        url, err = _validate_http_url(kwargs.get("url"), max_length=8192)
        if err is not None:
            return err
        assert url is not None

        data = await self._request_with_key_failover(
            rotator=self._deepseek_keys,
            provider_label="DeepSeek fetch",
            method="POST",
            url="https://api.deepseek.com/v1/tools/fetch",
            build_headers=lambda key: {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            retryable_statuses=_DEEPSEEK_RETRYABLE_STATUSES,
            json_body={"url": url},
        )
        content = data.get("content")
        if not content:
            return "Error: DeepSeek fetch does not return any content."
        title = _as_text(data.get("title"))
        return f"URL: {url}\nTitle: {title}\nContent: {_as_text(content)}"

    # ------------------------------------------------------------------
    # SerpApi (strict api_key redaction on all responses and exceptions)
    # ------------------------------------------------------------------

    async def _serpapi_request(
        self,
        payload: dict[str, Any],
        *,
        engine: str,
    ) -> dict[str, Any]:
        if self._serpapi_keys.key_count == 0:
            raise ValueError("Error: SerpApi API key is not configured.")

        _install_http_log_redaction_filter()
        client = self._get_http_client()
        last_error: RuntimeError | None = None
        tried_keys: set[str] = set()

        for _ in range(self._serpapi_keys.key_count):
            api_key = await self._serpapi_keys.get(tried_keys)
            try:
                response = await client.get(
                    "https://serpapi.com/search.json",
                    params={**payload, "engine": engine, "api_key": api_key},
                    timeout=self._config.timeout_sec,
                )
            except (httpx.HTTPError, TimeoutError, OSError) as exc:
                safe_detail = _redact_secret(str(exc), api_key)
                logger.warning(
                    "SerpApi %s request failed: %s",
                    engine,
                    safe_detail,
                )
                raise RuntimeError(
                    f"SerpApi search request failed or timed out: {safe_detail}"
                ) from None

            if response.status_code == 200:
                try:
                    data = response.json()
                except ValueError:
                    raise RuntimeError(
                        "SerpApi returned invalid JSON response."
                    ) from None
                if not isinstance(data, dict):
                    raise RuntimeError(
                        "SerpApi returned unexpected JSON response type."
                    )
                meta = data.get("search_metadata")
                status = meta.get("status") if isinstance(meta, dict) else None
                if status == "Error" or (data.get("error") and status != "Success"):
                    raw_reason = str(data.get("error") or "Unknown API error.")
                    safe_reason = _redact_secret(raw_reason, api_key)
                    raise RuntimeError(f"SerpApi search failed: {safe_reason}")
                return data

            safe_reason = _redact_secret(response.text, api_key)
            err = RuntimeError(
                f"SerpApi search failed: {safe_reason}, status: {response.status_code}"
            )
            if response.status_code in _SERPAPI_RETRYABLE_STATUSES:
                last_error = err
                continue
            raise err

        if last_error is not None:
            raise last_error
        raise RuntimeError("SerpApi search failed with all configured keys.")

    def _build_serpapi_web_search_tool(self) -> ToolSpec:
        return ToolSpec(
            name="web_search_serpapi",
            description=(
                "Search the web using SerpApi Google Search for current information. "
                "Returns page titles, URLs, and snippets."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Required. Search query.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": (
                            "Maximum results to return from the first page, 1-10. "
                            "Default is 5; fewer may be available."
                        ),
                    },
                    "country": {
                        "type": "string",
                        "description": (
                            "Optional. Google country code, such as us, uk, or cn."
                        ),
                    },
                    "search_lang": {
                        "type": "string",
                        "description": (
                            "Optional. Google interface language code, "
                            "such as en or zh-cn."
                        ),
                    },
                    "location": {
                        "type": "string",
                        "description": (
                            "Optional. Search origin, such as Austin, Texas, "
                            "United States."
                        ),
                    },
                },
                "required": ["query"],
            },
            handler=self._web_search_serpapi,
        )

    async def _web_search_serpapi(self, **kwargs: Any) -> str:
        if self._serpapi_keys.key_count == 0:
            return "Error: SerpApi API key is not configured."

        query, err = _validate_non_empty_query(kwargs.get("query"))
        if err is not None:
            return err
        assert query is not None

        max_results, err = _validate_int_arg(
            kwargs.get("max_results"),
            name="max_results",
            default=5,
            minimum=1,
            maximum=10,
            clamp=True,
        )
        if err is not None:
            return err
        assert max_results is not None

        payload: dict[str, Any] = {"q": query}
        for arg_name, param_name in (
            ("country", "gl"),
            ("search_lang", "hl"),
            ("location", "location"),
        ):
            val, err = _validate_optional_str(kwargs.get(arg_name), name=arg_name)
            if err is not None:
                return err
            if val:
                payload[param_name] = val

        data = await self._serpapi_request(payload, engine="google")
        organic = data.get("organic_results")
        rows = organic if isinstance(organic, list) else []
        results = [
            SearchResult(
                title=_as_text(item.get("title")),
                url=_as_text(item.get("link")),
                snippet=_as_text(item.get("snippet")),
            )
            for item in rows
            if isinstance(item, dict) and item.get("link")
        ]
        return _search_result_payload(results[:max_results])

    def _build_serpapi_news_search_tool(self) -> ToolSpec:
        return ToolSpec(
            name="news_search_serpapi",
            description=(
                "Search current news using the SerpApi Google News API. Returns "
                "article titles, source URLs, publishers, authors, snippets, and "
                "publication times."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Required. News search query. Google News operators "
                            "such as site: and when: are supported."
                        ),
                    },
                    "max_results": {
                        "type": "integer",
                        "description": (
                            "Maximum articles to return, 1-10. Default is 5; "
                            "fewer may be available."
                        ),
                    },
                    "country": {
                        "type": "string",
                        "description": (
                            "Optional. Google News country code, such as us, uk, or cn."
                        ),
                    },
                    "search_lang": {
                        "type": "string",
                        "description": (
                            "Optional. Google News language code, such as en or zh-cn."
                        ),
                    },
                },
                "required": ["query"],
            },
            handler=self._news_search_serpapi,
        )

    async def _news_search_serpapi(self, **kwargs: Any) -> str:
        if self._serpapi_keys.key_count == 0:
            return "Error: SerpApi API key is not configured."

        query, err = _validate_non_empty_query(kwargs.get("query"))
        if err is not None:
            return err
        assert query is not None

        max_results, err = _validate_int_arg(
            kwargs.get("max_results"),
            name="max_results",
            default=5,
            minimum=1,
            maximum=10,
            clamp=True,
        )
        if err is not None:
            return err
        assert max_results is not None

        payload: dict[str, Any] = {"q": query}
        for arg_name, param_name in (("country", "gl"), ("search_lang", "hl")):
            val, err = _validate_optional_str(kwargs.get(arg_name), name=arg_name)
            if err is not None:
                return err
            if val:
                payload[param_name] = val

        data = await self._serpapi_request(payload, engine="google_news")
        raw_groups = data.get("news_results")
        groups = raw_groups if isinstance(raw_groups, list) else []

        ref_uuid = uuid.uuid4().hex[:4]
        results: list[dict[str, Any]] = []
        seen_urls: set[str] = set()

        for group in groups:
            if not isinstance(group, dict):
                continue
            articles: list[dict[str, Any]] = []
            if group.get("link"):
                articles.append(group)
            if isinstance(group.get("highlight"), dict):
                articles.append(group["highlight"])
            raw_stories = group.get("stories")
            if isinstance(raw_stories, list):
                articles.extend(s for s in raw_stories if isinstance(s, dict))

            for article in articles:
                source_url = article.get("link")
                if not isinstance(source_url, str) or not source_url:
                    continue
                if source_url in seen_urls:
                    continue
                seen_urls.add(source_url)
                source = article.get("source")
                source_dict = source if isinstance(source, dict) else {}
                authors = source_dict.get("authors")
                results.append(
                    {
                        "title": _as_text(article.get("title")),
                        "url": source_url,
                        "snippet": _as_text(article.get("snippet")),
                        "source": _as_text(
                            source_dict.get("name") or source_dict.get("title")
                        ),
                        "authors": authors if isinstance(authors, list) else [],
                        "published_at": _as_text(
                            article.get("iso_date") or article.get("date")
                        ),
                        "thumbnail_url": _as_text(
                            article.get("thumbnail") or article.get("thumbnail_small")
                        ),
                        "index": f"{ref_uuid}.{len(results) + 1}",
                    }
                )
                if len(results) >= max_results:
                    break
            if len(results) >= max_results:
                break

        return json.dumps({"results": results}, ensure_ascii=False)

    def _build_serpapi_image_search_tool(self) -> ToolSpec:
        return ToolSpec(
            name="image_search_serpapi",
            description=(
                "Find images using SerpApi Google Images Light. Returns direct image "
                "URLs, thumbnails, dimensions, and source page URLs. Use image_url to "
                "display or send an image, and url to cite its source page."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Required. Image search query.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": (
                            "Maximum images to return, 1-10. Default is 5; "
                            "fewer may be available."
                        ),
                    },
                    "country": {
                        "type": "string",
                        "description": (
                            "Optional. Google country code, such as us, uk, or cn."
                        ),
                    },
                    "search_lang": {
                        "type": "string",
                        "description": (
                            "Optional. Google interface language code, "
                            "such as en or zh-cn."
                        ),
                    },
                },
                "required": ["query"],
            },
            handler=self._image_search_serpapi,
        )

    async def _image_search_serpapi(self, **kwargs: Any) -> str:
        if self._serpapi_keys.key_count == 0:
            return "Error: SerpApi API key is not configured."

        query, err = _validate_non_empty_query(kwargs.get("query"))
        if err is not None:
            return err
        assert query is not None

        max_results, err = _validate_int_arg(
            kwargs.get("max_results"),
            name="max_results",
            default=5,
            minimum=1,
            maximum=10,
            clamp=True,
        )
        if err is not None:
            return err
        assert max_results is not None

        payload: dict[str, Any] = {"q": query}
        for arg_name, param_name in (("country", "gl"), ("search_lang", "hl")):
            val, err = _validate_optional_str(kwargs.get(arg_name), name=arg_name)
            if err is not None:
                return err
            if val:
                payload[param_name] = val

        data = await self._serpapi_request(payload, engine="google_images_light")
        raw_images = data.get("images_results")
        images = raw_images if isinstance(raw_images, list) else []

        ref_uuid = uuid.uuid4().hex[:4]
        results: list[dict[str, Any]] = []
        for item in images:
            if not isinstance(item, dict):
                continue
            original = _as_text(item.get("original"))
            thumbnail_url = _as_text(
                item.get("thumbnail") or item.get("serpapi_thumbnail")
            )
            image_url = original
            if not image_url.startswith(("https://", "http://")):
                image_url = thumbnail_url
            if not image_url.startswith(("https://", "http://")):
                continue
            source_url = _as_text(item.get("link")) or image_url
            is_original = image_url == original
            results.append(
                {
                    "title": _as_text(item.get("title")),
                    "url": source_url,
                    "snippet": _as_text(item.get("source")),
                    "image_url": image_url,
                    "thumbnail_url": thumbnail_url,
                    "width": item.get("original_width") if is_original else None,
                    "height": item.get("original_height") if is_original else None,
                    "index": f"{ref_uuid}.{len(results) + 1}",
                }
            )
            if len(results) >= max_results:
                break

        return json.dumps({"results": results}, ensure_ascii=False)

    def _build_serpapi_finance_search_tool(self) -> ToolSpec:
        return ToolSpec(
            name="finance_search_serpapi",
            description=(
                "Look up stocks, indices, funds, currencies, and futures using Google "
                "Finance. Use ticker symbols such as AAPL:NASDAQ or currency pairs "
                "such as EUR-USD. Returns quote prices, currency, timestamps, key "
                "statistics, financials, and news when available. Set window only "
                "when price history is needed."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Required. Google Finance symbol, such as AAPL:NASDAQ, "
                            ".INX:INDEXSP, BTC-USD, or EUR-USD."
                        ),
                    },
                    "search_lang": {
                        "type": "string",
                        "description": (
                            "Optional. Google interface language code, "
                            "such as en or zh-cn."
                        ),
                    },
                    "window": {
                        "type": "string",
                        "enum": ["1D", "5D", "1M", "6M", "YTD", "1Y", "5Y", "MAX"],
                        "description": (
                            "Optional. Price history range. When set, Google Finance "
                            "returns a quote summary and graph instead of full "
                            "financials and news."
                        ),
                    },
                },
                "required": ["query"],
            },
            handler=self._finance_search_serpapi,
        )

    async def _finance_search_serpapi(self, **kwargs: Any) -> str:
        if self._serpapi_keys.key_count == 0:
            return "Error: SerpApi API key is not configured."

        query, err = _validate_non_empty_query(kwargs.get("query"))
        if err is not None:
            return err
        assert query is not None

        window, err = _validate_optional_str(kwargs.get("window"), name="window")
        if err is not None:
            return err
        if window and window not in _SERPAPI_FINANCE_WINDOWS:
            return "Error: window must be one of 1D, 5D, 1M, 6M, YTD, 1Y, 5Y, MAX."

        search_lang, err = _validate_optional_str(
            kwargs.get("search_lang"), name="search_lang"
        )
        if err is not None:
            return err

        payload: dict[str, Any] = {"q": query}
        if window:
            payload["window"] = window
        if search_lang:
            payload["hl"] = search_lang

        data = await self._serpapi_request(payload, engine="google_finance")
        summary = data.get("summary")
        summary_dict = summary if isinstance(summary, dict) else {}
        result: dict[str, Any] = {
            "results": [],
            "summary": summary_dict,
            "knowledge_graph": data.get("knowledge_graph") or {},
            "financials": data.get("financials") or [],
            "news_results": data.get("news_results") or [],
            "suggestions": data.get("suggestions") or [],
        }
        if summary_dict:
            meta = data.get("search_metadata")
            source_url = (
                meta.get("google_finance_url") if isinstance(meta, dict) else None
            )
            if not source_url:
                source_url = (
                    f"https://www.google.com/finance/quote/{quote(query, safe=':-.')}"
                )
            result["results"].append(
                {
                    "title": _as_text(summary_dict.get("title")) or query,
                    "url": _as_text(source_url),
                    "snippet": json.dumps(summary_dict, ensure_ascii=False),
                    "index": f"{uuid.uuid4().hex[:4]}.1",
                }
            )
        if window:
            result["graph"] = data.get("graph") or []
            result["key_events"] = data.get("key_events") or []
        return json.dumps(result, ensure_ascii=False)

    def _build_serpapi_scholar_search_tool(self) -> ToolSpec:
        return ToolSpec(
            name="scholar_search_serpapi",
            description=(
                "Search academic publications using Google Scholar. Returns titles, "
                "source URLs, snippets, publication details, citation counts, and "
                "available PDF or full-text links. Supports publication year filters."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Required. Academic search query.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": (
                            "Maximum publications to return, 1-20. Default is 5."
                        ),
                    },
                    "search_lang": {
                        "type": "string",
                        "description": (
                            "Optional. Google Scholar interface language, "
                            "such as en or zh-cn."
                        ),
                    },
                    "year_from": {
                        "type": "integer",
                        "description": (
                            "Optional. Earliest publication year, inclusive."
                        ),
                    },
                    "year_to": {
                        "type": "integer",
                        "description": "Optional. Latest publication year, inclusive.",
                    },
                },
                "required": ["query"],
            },
            handler=self._scholar_search_serpapi,
        )

    async def _scholar_search_serpapi(self, **kwargs: Any) -> str:
        if self._serpapi_keys.key_count == 0:
            return "Error: SerpApi API key is not configured."

        query, err = _validate_non_empty_query(kwargs.get("query"))
        if err is not None:
            return err
        assert query is not None

        max_results, err = _validate_int_arg(
            kwargs.get("max_results"),
            name="max_results",
            default=5,
            minimum=1,
            maximum=20,
            clamp=True,
        )
        if err is not None:
            return err
        assert max_results is not None

        payload: dict[str, Any] = {"q": query, "num": max_results}
        for arg_name, param_name in (("year_from", "as_ylo"), ("year_to", "as_yhi")):
            if kwargs.get(arg_name) is not None:
                year, err = _validate_int_arg(
                    kwargs.get(arg_name),
                    name=arg_name,
                    default=0,
                    minimum=1,
                    maximum=9999,
                )
                if err is not None:
                    return "Error: publication years must be between 1 and 9999."
                payload[param_name] = year

        if payload.get("as_ylo", 1) > payload.get("as_yhi", 9999):
            return "Error: year_from must not be greater than year_to."

        search_lang, err = _validate_optional_str(
            kwargs.get("search_lang"), name="search_lang"
        )
        if err is not None:
            return err
        if search_lang:
            payload["hl"] = search_lang

        data = await self._serpapi_request(payload, engine="google_scholar")
        raw_organic = data.get("organic_results")
        organic = raw_organic if isinstance(raw_organic, list) else []

        ref_uuid = uuid.uuid4().hex[:4]
        results: list[dict[str, Any]] = []
        for item in organic:
            if not isinstance(item, dict):
                continue
            raw_resources = item.get("resources")
            resources = raw_resources if isinstance(raw_resources, list) else []
            source_url = item.get("link") or next(
                (r["link"] for r in resources if isinstance(r, dict) and r.get("link")),
                None,
            )
            if not source_url:
                fallback_q = quote(_as_text(item.get("title")) or query, safe="")
                source_url = f"https://scholar.google.com/scholar?q={fallback_q}"
            inline_links = item.get("inline_links")
            cited_by = (
                inline_links.get("cited_by")
                if isinstance(inline_links, dict)
                and isinstance(inline_links.get("cited_by"), dict)
                else {}
            )
            pub_info = item.get("publication_info")
            results.append(
                {
                    "title": _as_text(item.get("title")),
                    "url": _as_text(source_url),
                    "snippet": _as_text(item.get("snippet")),
                    "publication_info": pub_info if isinstance(pub_info, dict) else {},
                    "cited_by_count": cited_by.get("total"),
                    "cited_by_url": cited_by.get("link"),
                    "resources": resources,
                    "index": f"{ref_uuid}.{len(results) + 1}",
                }
            )
            if len(results) >= max_results:
                break

        return json.dumps({"results": results}, ensure_ascii=False)
