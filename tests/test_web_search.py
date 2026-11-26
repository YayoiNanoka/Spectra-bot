"""Unit tests for spectra.web_search (all 9 AstrBot-aligned providers)."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any
from urllib.parse import parse_qs, quote, urlsplit

import httpx
import pytest

from spectra.config import WebSearchConfig
from spectra.web_search import WEB_SEARCH_TOOL_NAMES, WebSearchToolkit

_INDEX_RE = re.compile(r"^[0-9a-f]{4}\.\d+$")


def _make_toolkit(
    cfg: WebSearchConfig,
    handler: Any,
) -> tuple[WebSearchToolkit, httpx.AsyncClient]:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return WebSearchToolkit(cfg, http_client=client), client


def _tool_map(toolkit: WebSearchToolkit) -> dict[str, Any]:
    return {tool.name: tool for tool in toolkit.build_tools()}


# ---------------------------------------------------------------------------
# Provider tool registration & lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("provider", "expected_names"),
    [
        ("tavily", ("web_search_tavily", "tavily_extract_web_page")),
        ("bocha", ("web_search_bocha",)),
        ("brave", ("web_search_brave",)),
        ("firecrawl", ("web_search_firecrawl", "firecrawl_extract_web_page")),
        ("baidu_ai_search", ("web_search_baidu",)),
        ("exa", ("web_search_exa", "exa_get_contents")),
        ("anysearch", ("web_search_anysearch",)),
        ("deepseek", ("web_search_deepseek", "deepseek_fetch_web_page")),
        (
            "serpapi",
            (
                "web_search_serpapi",
                "news_search_serpapi",
                "image_search_serpapi",
                "finance_search_serpapi",
                "scholar_search_serpapi",
            ),
        ),
    ],
)
def test_build_tools_returns_only_active_provider(
    provider: Any,
    expected_names: tuple[str, ...],
) -> None:
    disabled = WebSearchToolkit(WebSearchConfig(enabled=False, provider=provider))
    assert disabled.build_tools() == ()

    enabled = WebSearchToolkit(WebSearchConfig(enabled=True, provider=provider))
    names = tuple(t.name for t in enabled.build_tools())
    assert names == expected_names
    assert set(names).issubset(set(WEB_SEARCH_TOOL_NAMES))


async def test_toolkit_client_ownership_and_idempotent_close() -> None:
    # Owned client: created lazily, closed idempotently, rejects post-close calls.
    owned = WebSearchToolkit(
        WebSearchConfig(enabled=True, provider="tavily", tavily_keys=["k1"])
    )
    client = owned._get_http_client()
    assert client.is_closed is False
    async with owned:
        assert owned._get_http_client() is client
    assert client.is_closed is True
    await owned.close()  # idempotent
    with pytest.raises(RuntimeError, match="closed"):
        owned._get_http_client()

    # Injected client: toolkit.close() MUST NOT close external client.
    external = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda req: httpx.Response(200, json={}))
    )
    try:
        injected = WebSearchToolkit(
            WebSearchConfig(enabled=True, provider="tavily", tavily_keys=["k1"]),
            http_client=external,
        )
        await injected.close()
        assert external.is_closed is False
    finally:
        await external.aclose()


# ---------------------------------------------------------------------------
# Tavily
# ---------------------------------------------------------------------------


async def test_tavily_search_and_extract_with_failover() -> None:
    seen_auth: list[str] = []
    seen_bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_auth.append(request.headers["Authorization"])
        body = json.loads(request.content)
        seen_bodies.append(body)
        if request.url.path == "/search":
            if len(seen_auth) == 1:
                return httpx.Response(432, text="quota exceeded")
            if len(seen_auth) == 2:
                return httpx.Response(433, text="plan limit")
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "title": "Tavily Hit",
                            "url": "https://example.com/tavily",
                            "content": "Snippet text",
                        }
                    ]
                },
            )
        assert request.url.path == "/extract"
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "url": "https://example.com/article",
                        "raw_content": "Full article text",
                    }
                ]
            },
        )

    cfg = WebSearchConfig(
        enabled=True,
        provider="tavily",
        tavily_keys=["  k1 ", "", "k2", "k3"],
    )
    toolkit, http_client = _make_toolkit(cfg, handler)
    try:
        tools = _tool_map(toolkit)
        out = await tools["web_search_tavily"].handler(
            query="python 3.13",
            topic="news",
            days=5,
            start_date="2026-01-01",
            time_range="week",  # ignored because start_date is set
            extra_ignored="ok",
        )
        parsed = json.loads(out)
        assert len(parsed["results"]) == 1
        item = parsed["results"][0]
        assert item["title"] == "Tavily Hit"
        assert item["url"] == "https://example.com/tavily"
        assert item["snippet"] == "Snippet text"
        assert _INDEX_RE.match(item["index"])
        assert seen_auth == ["Bearer k1", "Bearer k2", "Bearer k3"]
        assert seen_bodies[0]["topic"] == "news"
        assert seen_bodies[0]["days"] == 5
        assert seen_bodies[0]["start_date"] == "2026-01-01"
        assert "time_range" not in seen_bodies[0]

        extracted = await tools["tavily_extract_web_page"].handler(
            url="https://example.com/article",
            extract_depth="advanced",
        )
        assert "URL: https://example.com/article" in extracted
        assert "Content: Full article text" in extracted
        assert seen_bodies[-1] == {
            "urls": ["https://example.com/article"],
            "extract_depth": "advanced",
        }
    finally:
        await http_client.aclose()


async def test_fail_fast_on_500_without_rotating_keys() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500, text="internal server error")

    cfg = WebSearchConfig(
        enabled=True,
        provider="tavily",
        tavily_keys=["k1", "k2", "k3"],
    )
    toolkit, http_client = _make_toolkit(cfg, handler)
    try:
        tools = _tool_map(toolkit)
        with pytest.raises(RuntimeError, match="status: 500"):
            await tools["web_search_tavily"].handler(query="boom")
        assert calls == 1
    finally:
        await http_client.aclose()


async def test_concurrent_round_robin_key_rotation() -> None:
    seen_keys: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_keys.append(request.headers["Authorization"].removeprefix("Bearer "))
        return httpx.Response(
            200,
            json={
                "results": [
                    {"title": "T", "url": "https://example.com", "content": "C"}
                ]
            },
        )

    cfg = WebSearchConfig(
        enabled=True,
        provider="tavily",
        tavily_keys=["k1", "k2", "k3"],
    )
    toolkit, http_client = _make_toolkit(cfg, handler)
    try:
        search = _tool_map(toolkit)["web_search_tavily"].handler
        await asyncio.gather(*(search(query=f"q{i}") for i in range(6)))
        assert sorted(seen_keys) == ["k1", "k1", "k2", "k2", "k3", "k3"]
    finally:
        await http_client.aclose()


# ---------------------------------------------------------------------------
# BoCha
# ---------------------------------------------------------------------------


async def test_bocha_search_and_failover() -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.headers["Authorization"], json.loads(request.content)))
        if len(calls) == 1:
            return httpx.Response(429, text="rate limited")
        return httpx.Response(
            200,
            json={
                "data": {
                    "webPages": {
                        "value": [
                            {
                                "name": "BoCha Result",
                                "url": "https://bocha.example/1",
                                "snippet": "BoCha snippet",
                            }
                        ]
                    }
                }
            },
        )

    cfg = WebSearchConfig(
        enabled=True,
        provider="bocha",
        bocha_keys=["b1", "b2"],
    )
    toolkit, http_client = _make_toolkit(cfg, handler)
    try:
        out = await _tool_map(toolkit)["web_search_bocha"].handler(
            query="人工智能",
            count=5,
            summary=True,
            freshness="noLimit",
            include="example.com",
            exclude="ads.example",
        )
        parsed = json.loads(out)
        assert parsed["results"][0]["title"] == "BoCha Result"
        assert parsed["results"][0]["url"] == "https://bocha.example/1"
        assert parsed["results"][0]["snippet"] == "BoCha snippet"
        assert [c[0] for c in calls] == ["Bearer b1", "Bearer b2"]
        assert calls[0][1] == {
            "query": "人工智能",
            "count": 5,
            "summary": True,
            "freshness": "noLimit",
            "include": "example.com",
            "exclude": "ads.example",
        }
    finally:
        await http_client.aclose()


# ---------------------------------------------------------------------------
# Brave
# ---------------------------------------------------------------------------


async def test_brave_422_token_invalid_failover_and_validation_failfast() -> None:
    tokens_seen: list[str] = []
    params_seen: list[dict[str, list[str]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        tokens_seen.append(request.headers["X-Subscription-Token"])
        params_seen.append(parse_qs(urlsplit(str(request.url)).query))
        if len(tokens_seen) == 1:
            return httpx.Response(
                422,
                json={"error": {"code": "SUBSCRIPTION_TOKEN_INVALID"}},
            )
        if len(tokens_seen) == 2:
            return httpx.Response(
                200,
                json={
                    "web": {
                        "results": [
                            {
                                "title": "Brave Hit",
                                "url": "https://brave.example/1",
                                "description": "Brave desc",
                            }
                        ]
                    }
                },
            )
        # Non-token 422 must NOT retry.
        return httpx.Response(
            422,
            json={"error": {"code": "VALIDATION", "detail": "bad query"}},
        )

    cfg = WebSearchConfig(
        enabled=True,
        provider="brave",
        brave_keys=["br1", "br2"],
    )
    toolkit, http_client = _make_toolkit(cfg, handler)
    try:
        search = _tool_map(toolkit)["web_search_brave"].handler
        out = await search(
            query="rust async",
            count=99,  # clamped to 20
            country="CN",
            search_lang="en",
            freshness="week",
        )
        parsed = json.loads(out)
        assert parsed["results"][0]["title"] == "Brave Hit"
        assert parsed["results"][0]["snippet"] == "Brave desc"
        assert tokens_seen == ["br1", "br2"]
        assert params_seen[0]["count"] == ["20"]
        assert params_seen[0]["country"] == ["CN"]
        assert params_seen[0]["search_lang"] == ["en"]
        assert params_seen[0]["freshness"] == ["week"]

        with pytest.raises(RuntimeError, match="status: 422"):
            await search(query="bad")
        # Only 1 additional call (no failover on non-token 422).
        assert len(tokens_seen) == 3
    finally:
        await http_client.aclose()


# ---------------------------------------------------------------------------
# Firecrawl
# ---------------------------------------------------------------------------


async def test_firecrawl_search_and_scrape_with_402_failover() -> None:
    calls: list[tuple[str, str, dict[str, Any]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append((request.url.path, request.headers["Authorization"], body))
        if len(calls) == 1:
            return httpx.Response(402, text="payment required")
        if request.url.path == "/v2/search":
            return httpx.Response(
                200,
                json={
                    "data": {
                        "web": [
                            {
                                "title": "FC Hit",
                                "url": "https://fc.example/1",
                                "markdown": "# Heading",
                            },
                            {"title": "No URL"},
                        ]
                    }
                },
            )
        return httpx.Response(
            200,
            json={
                "data": {
                    "url": "https://fc.example/page",
                    "markdown": "Scraped body",
                }
            },
        )

    cfg = WebSearchConfig(
        enabled=True,
        provider="firecrawl",
        firecrawl_keys=["fc1", "fc2"],
    )
    toolkit, http_client = _make_toolkit(cfg, handler)
    try:
        tools = _tool_map(toolkit)
        out = await tools["web_search_firecrawl"].handler(
            query="firecrawl docs",
            limit=3,
            country="US",
            timeout=5000,
        )
        parsed = json.loads(out)
        assert len(parsed["results"]) == 1
        assert parsed["results"][0]["snippet"] == "# Heading"
        assert calls[0][1] == "Bearer fc1"
        assert calls[1][1] == "Bearer fc2"

        scraped = await tools["firecrawl_extract_web_page"].handler(
            url="https://fc.example/page",
            format="markdown",
            only_main_content=False,
            max_age=0,
        )
        assert scraped == "URL: https://fc.example/page\nContent: Scraped body"
        assert calls[-1][2]["onlyMainContent"] is False
        assert calls[-1][2]["maxAge"] == 0
    finally:
        await http_client.aclose()


# ---------------------------------------------------------------------------
# Baidu AI Search
# ---------------------------------------------------------------------------


async def test_baidu_ai_search() -> None:
    captured: list[tuple[dict[str, str], dict[str, Any]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append((dict(request.headers), json.loads(request.content)))
        return httpx.Response(
            200,
            json={
                "references": [
                    {
                        "title": "百度结果",
                        "url": "https://baidu.example/1",
                        "content": "摘要内容",
                    }
                ]
            },
        )

    cfg = WebSearchConfig(
        enabled=True,
        provider="baidu_ai_search",
        baidu_app_builder_key="  baidu-secret  ",
    )
    toolkit, http_client = _make_toolkit(cfg, handler)
    try:
        long_query = "a" * 100
        out = await _tool_map(toolkit)["web_search_baidu"].handler(
            query=long_query,
            top_k=99,
            search_recency_filter="month",
            site="github.com|python.org, docs.python.org",
        )
        parsed = json.loads(out)
        assert parsed["results"][0]["title"] == "百度结果"
        headers, body = captured[0]
        assert headers["authorization"] == "Bearer baidu-secret"
        assert headers["x-appbuilder-authorization"] == "Bearer baidu-secret"
        assert len(body["messages"][0]["content"]) == 72
        assert body["resource_type_filter"] == [{"type": "web", "top_k": 50}]
        assert body["search_recency_filter"] == "month"
        assert body["search_filter"] == {
            "match": {"site": ["github.com", "python.org", "docs.python.org"]}
        }
    finally:
        await http_client.aclose()


# ---------------------------------------------------------------------------
# Exa
# ---------------------------------------------------------------------------


async def test_exa_search_and_get_contents() -> None:
    seen: list[tuple[str, str, dict[str, Any]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(
            (
                request.url.path,
                request.headers["x-api-key"],
                json.loads(request.content),
            )
        )
        if request.url.path == "/search":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "title": "Exa Paper",
                            "url": "https://arxiv.org/abs/1234",
                            "highlights": ["Highlight snippet"],
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "url": "https://arxiv.org/abs/1234",
                        "text": "Paper full text",
                    }
                ]
            },
        )

    cfg = WebSearchConfig(enabled=True, provider="exa", exa_keys=["exa-k1"])
    toolkit, http_client = _make_toolkit(cfg, handler)
    try:
        tools = _tool_map(toolkit)
        out = await tools["web_search_exa"].handler(
            query="attention mechanism",
            num_results=3,
            type="neural",
            category="research paper",
            include_domains="arxiv.org, openreview.net",
            exclude_domains="spam.com",
            start_published_date="2024-01-01T00:00:00.000Z",
        )
        parsed = json.loads(out)
        assert parsed["results"][0]["snippet"] == "Highlight snippet"
        assert seen[0][2]["includeDomains"] == ["arxiv.org", "openreview.net"]
        assert seen[0][2]["excludeDomains"] == ["spam.com"]
        assert seen[0][2]["numResults"] == 3

        content = await tools["exa_get_contents"].handler(
            url="https://arxiv.org/abs/1234",
            max_characters=1000,
        )
        assert "URL: https://arxiv.org/abs/1234" in content
        assert "Content: Paper full text" in content
        assert seen[1][2] == {
            "ids": ["https://arxiv.org/abs/1234"],
            "text": {"maxCharacters": 1000},
        }
    finally:
        await http_client.aclose()


# ---------------------------------------------------------------------------
# AnySearch (anonymous + authenticated + vertical extras + business error)
# ---------------------------------------------------------------------------


async def test_anysearch_anonymous_mode_and_vertical_extras() -> None:
    seen_headers: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_headers.append(dict(request.headers))
        return httpx.Response(
            200,
            json={
                "code": 0,
                "data": {
                    "results": [
                        {
                            "title": "AAPL Quote",
                            "url": "https://finance.example/aapl",
                            "snippet": "Apple Inc.",
                            "price": 215.5,
                            "meta": {"currency": "USD"},
                        }
                    ]
                },
            },
        )

    cfg = WebSearchConfig(enabled=True, provider="anysearch", anysearch_keys=[])
    toolkit, http_client = _make_toolkit(cfg, handler)
    try:
        out = await _tool_map(toolkit)["web_search_anysearch"].handler(
            query="AAPL",
            tag="finance.quote",
            zone="global",
            language="en",
            params={"symbol": "AAPL", "type": "stock"},
        )
        parsed = json.loads(out)
        snippet = parsed["results"][0]["snippet"]
        assert "Apple Inc." in snippet
        assert "price: 215.5" in snippet
        assert '"currency": "USD"' in snippet
        assert "authorization" not in {k.lower() for k in seen_headers[0]}
    finally:
        await http_client.aclose()


async def test_anysearch_authenticated_failover_and_nonzero_code_error() -> None:
    seen_auth: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_auth.append(request.headers["Authorization"])
        if len(seen_auth) == 1:
            return httpx.Response(402, text="out of credits")
        return httpx.Response(200, json={"code": 4001, "message": "missing param"})

    cfg = WebSearchConfig(
        enabled=True,
        provider="anysearch",
        anysearch_keys=["a1", "a2"],
    )
    toolkit, http_client = _make_toolkit(cfg, handler)
    try:
        with pytest.raises(RuntimeError, match="missing param"):
            await _tool_map(toolkit)["web_search_anysearch"].handler(query="test")
        assert seen_auth == ["Bearer a1", "Bearer a2"]
    finally:
        await http_client.aclose()


# ---------------------------------------------------------------------------
# DeepSeek
# ---------------------------------------------------------------------------


async def test_deepseek_search_and_fetch_with_failover() -> None:
    calls: list[tuple[str, str, dict[str, Any]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(
            (
                request.url.path,
                request.headers["Authorization"],
                json.loads(request.content),
            )
        )
        if len(calls) == 1:
            return httpx.Response(402, text="insufficient balance")
        if request.url.path == "/v1/tools/search":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "title": "DeepSeek Docs",
                            "url": "https://api-docs.deepseek.com/",
                            "summary": "Short summary",
                            "content": "Full body",
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            json={"title": "Docs", "content": "# Markdown Body"},
        )

    cfg = WebSearchConfig(
        enabled=True,
        provider="deepseek",
        deepseek_keys=["ds1", "ds2"],
    )
    toolkit, http_client = _make_toolkit(cfg, handler)
    try:
        tools = _tool_map(toolkit)
        out = await tools["web_search_deepseek"].handler(
            query="DeepSeek API",
            count=5,
            ori_question="How to use DeepSeek?",
            need_content=True,
        )
        parsed = json.loads(out)
        assert parsed["results"][0]["snippet"] == "Short summary\n\nFull body"
        assert calls[0][1] == "Bearer ds1"
        assert calls[1][1] == "Bearer ds2"

        fetched = await tools["deepseek_fetch_web_page"].handler(
            url="https://api-docs.deepseek.com/"
        )
        assert fetched == (
            "URL: https://api-docs.deepseek.com/\nTitle: Docs\nContent: # Markdown Body"
        )
    finally:
        await http_client.aclose()


# ---------------------------------------------------------------------------
# SerpApi (all 5 engines + strict api_key redaction)
# ---------------------------------------------------------------------------


async def test_serpapi_all_five_engines() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        qs = parse_qs(urlsplit(str(request.url)).query)
        engine = qs["engine"][0]
        if engine == "google":
            return httpx.Response(
                200,
                json={
                    "organic_results": [
                        {
                            "title": "Web 1",
                            "link": "https://web.example/1",
                            "snippet": "Web snippet",
                        },
                        {
                            "title": "Web 2",
                            "link": "https://web.example/2",
                            "snippet": "Web snippet 2",
                        },
                    ]
                },
            )
        if engine == "google_news":
            return httpx.Response(
                200,
                json={
                    "news_results": [
                        {
                            "highlight": {
                                "title": "Headline",
                                "link": "https://news.example/1",
                                "snippet": "Lead story",
                                "iso_date": "2026-01-01T00:00:00Z",
                                "source": {"name": "Daily", "authors": ["Alice"]},
                                "thumbnail": "https://news.example/thumb.jpg",
                            },
                            "stories": [
                                {
                                    "title": "Duplicate",
                                    "link": "https://news.example/1",
                                },
                                {
                                    "title": "Follow-up",
                                    "link": "https://news.example/2",
                                    "date": "2 hours ago",
                                    "source": {"title": "Times"},
                                },
                            ],
                        }
                    ]
                },
            )
        if engine == "google_images_light":
            return httpx.Response(
                200,
                json={
                    "images_results": [
                        {
                            "title": "PDF Img",
                            "original": "x-raw-image:///abc",
                            "thumbnail": "https://img.example/thumb1.jpg",
                            "link": "https://site.example/doc",
                            "source": "Site",
                            "original_width": 800,
                            "original_height": 600,
                        },
                        {
                            "title": "Direct Img",
                            "original": "https://img.example/full.png",
                            "thumbnail": "https://img.example/thumb2.png",
                            "link": "https://site.example/photo",
                            "source": "Photos",
                            "original_width": 1024,
                            "original_height": 768,
                        },
                    ]
                },
            )
        if engine == "google_finance":
            return httpx.Response(
                200,
                json={
                    "summary": {"title": "Apple Inc", "price": "$215.00"},
                    "graph": [{"price": 215.0}],
                    "key_events": [],
                },
            )
        assert engine == "google_scholar"
        return httpx.Response(
            200,
            json={
                "organic_results": [
                    {
                        "title": "Attention Is All You Need",
                        "snippet": "Transformer paper",
                        "inline_links": {
                            "cited_by": {
                                "total": 100000,
                                "link": "https://scholar.google.com/cites",
                            }
                        },
                        "publication_info": {"summary": "NeurIPS 2017"},
                    }
                ]
            },
        )

    cfg = WebSearchConfig(
        enabled=True,
        provider="serpapi",
        serpapi_keys=["serp-key-1"],
    )
    toolkit, http_client = _make_toolkit(cfg, handler)
    try:
        tools = _tool_map(toolkit)

        web_out = json.loads(
            await tools["web_search_serpapi"].handler(query="spectra", max_results=1)
        )
        assert len(web_out["results"]) == 1
        assert web_out["results"][0]["url"] == "https://web.example/1"

        news_out = json.loads(
            await tools["news_search_serpapi"].handler(query="ai news", country="us")
        )
        assert len(news_out["results"]) == 2
        assert news_out["results"][0]["source"] == "Daily"
        assert news_out["results"][0]["authors"] == ["Alice"]

        img_out = json.loads(await tools["image_search_serpapi"].handler(query="cat"))
        assert len(img_out["results"]) == 2
        # Fallback from x-raw-image:// clears width/height.
        assert img_out["results"][0]["image_url"] == "https://img.example/thumb1.jpg"
        assert img_out["results"][0]["width"] is None
        assert img_out["results"][1]["image_url"] == "https://img.example/full.png"
        assert img_out["results"][1]["width"] == 1024

        fin_out = json.loads(
            await tools["finance_search_serpapi"].handler(
                query="AAPL:NASDAQ", window="1M"
            )
        )
        assert len(fin_out["results"]) == 1
        assert fin_out["results"][0]["url"].startswith(
            "https://www.google.com/finance/quote/AAPL:NASDAQ"
        )
        assert fin_out["graph"] == [{"price": 215.0}]

        sch_out = json.loads(
            await tools["scholar_search_serpapi"].handler(
                query="transformer", year_from=2017, year_to=2025
            )
        )
        assert len(sch_out["results"]) == 1
        assert sch_out["results"][0]["cited_by_count"] == 100000
        assert sch_out["results"][0]["url"].startswith(
            "https://scholar.google.com/scholar?q="
        )
    finally:
        await http_client.aclose()


async def test_serpapi_redacts_api_key_in_http_api_and_transport_errors(
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "serp+secret/key=12345"
    encoded_secret = quote(secret, safe="")
    mode = "http_error"

    def handler(request: httpx.Request) -> httpx.Response:
        if mode == "http_error":
            return httpx.Response(
                401,
                text=f"Invalid API key {secret} in {request.url}",
            )
        if mode == "api_error":
            return httpx.Response(
                200,
                json={
                    "search_metadata": {"status": "Error"},
                    "error": f"Bad key {secret} ({encoded_secret})",
                },
            )
        raise httpx.ConnectError(
            f"Connection refused for https://serpapi.com/search.json?q=hi&api_key={encoded_secret}",
            request=request,
        )

    cfg = WebSearchConfig(enabled=True, provider="serpapi", serpapi_keys=[secret])
    toolkit, http_client = _make_toolkit(cfg, handler)
    try:
        search = _tool_map(toolkit)["web_search_serpapi"].handler

        with pytest.raises(RuntimeError) as exc_info:
            await search(query="test")
        msg = str(exc_info.value)
        assert secret not in msg
        assert encoded_secret not in msg
        assert "[redacted]" in msg

        mode = "api_error"
        with pytest.raises(RuntimeError) as exc_info2:
            await search(query="test")
        msg2 = str(exc_info2.value)
        assert secret not in msg2
        assert encoded_secret not in msg2
        assert "[redacted]" in msg2

        mode = "transport_error"
        with caplog.at_level(logging.WARNING):
            with pytest.raises(RuntimeError) as exc_info3:
                await search(query="test")
        msg3 = str(exc_info3.value)
        assert secret not in msg3
        assert encoded_secret not in msg3
        assert "[redacted]" in msg3
        assert exc_info3.value.__cause__ is None
        for record in caplog.records:
            assert secret not in record.getMessage()
            assert encoded_secret not in record.getMessage()
    finally:
        await http_client.aclose()


# ---------------------------------------------------------------------------
# URL & malformed argument validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("provider", "tool_name", "key_field"),
    [
        ("tavily", "tavily_extract_web_page", "tavily_keys"),
        ("firecrawl", "firecrawl_extract_web_page", "firecrawl_keys"),
        ("exa", "exa_get_contents", "exa_keys"),
        ("deepseek", "deepseek_fetch_web_page", "deepseek_keys"),
    ],
)
@pytest.mark.parametrize(
    "bad_url",
    [
        "",
        "   ",
        123,
        ["https://example.com"],
        "ftp://example.com/file",
        "javascript:alert(1)",
        "https:///no-host",
        "https://user:pass@example.com/secret",
        "https://example.com/has space",
        "https://example.com/tab\there",
        "https://example.com:99999/bad-port",
        "https://example.com/" + "a" * 8200,
    ],
)
async def test_extract_and_fetch_tools_reject_invalid_urls(
    provider: Any,
    tool_name: str,
    key_field: str,
    bad_url: Any,
) -> None:
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json={})

    cfg = WebSearchConfig(enabled=True, provider=provider, **{key_field: ["k1"]})
    toolkit, http_client = _make_toolkit(cfg, handler)
    try:
        result = await _tool_map(toolkit)[tool_name].handler(url=bad_url)
        assert result.startswith("Error:")
        assert called is False
    finally:
        await http_client.aclose()


async def test_malformed_arguments_return_error_strings_without_raising() -> None:
    def fail_if_called(request: httpx.Request) -> httpx.Response:
        raise AssertionError("HTTP should not be called for invalid arguments")

    # Tavily malformed args
    t_kit, t_http = _make_toolkit(
        WebSearchConfig(enabled=True, provider="tavily", tavily_keys=["k1"]),
        fail_if_called,
    )
    # DeepSeek malformed args
    d_kit, d_http = _make_toolkit(
        WebSearchConfig(enabled=True, provider="deepseek", deepseek_keys=["k1"]),
        fail_if_called,
    )
    # SerpApi malformed args
    s_kit, s_http = _make_toolkit(
        WebSearchConfig(enabled=True, provider="serpapi", serpapi_keys=["k1"]),
        fail_if_called,
    )
    # AnySearch malformed args
    a_kit, a_http = _make_toolkit(
        WebSearchConfig(enabled=True, provider="anysearch"),
        fail_if_called,
    )
    try:
        t_search = _tool_map(t_kit)["web_search_tavily"].handler
        assert (await t_search(query="   ")).startswith("Error:")
        assert (await t_search(query=123)).startswith("Error:")
        assert (await t_search(query="ok", max_results=True)).startswith("Error:")
        assert (await t_search(query="ok", max_results=0)).startswith("Error:")
        assert (await t_search(query="ok", max_results=21)).startswith("Error:")
        assert (await t_search(query="ok", topic="news", days=0)).startswith("Error:")

        d_search = _tool_map(d_kit)["web_search_deepseek"].handler
        assert (await d_search(query="x" * 129)).startswith("Error:")
        assert (await d_search(query="ok", count=0)).startswith("Error:")
        assert (await d_search(query="ok", count=51)).startswith("Error:")
        assert (await d_search(query="ok", count=True)).startswith("Error:")
        assert (await d_search(query="ok", ori_question="")).startswith("Error:")
        assert (await d_search(query="ok", need_content="yes")).startswith("Error:")

        s_tools = _tool_map(s_kit)
        assert (
            await s_tools["finance_search_serpapi"].handler(
                query="AAPL:NASDAQ", window="2Y"
            )
        ).startswith("Error:")
        assert (
            await s_tools["scholar_search_serpapi"].handler(query="ml", year_from=0)
        ).startswith("Error:")
        assert (
            await s_tools["scholar_search_serpapi"].handler(
                query="ml", year_from=2025, year_to=2020
            )
        ).startswith("Error:")
        assert (
            await s_tools["web_search_serpapi"].handler(query="ml", max_results="five")
        ).startswith("Error:")

        a_search = _tool_map(a_kit)["web_search_anysearch"].handler
        assert (await a_search(query="ok", params=["bad"])).startswith("Error:")
    finally:
        await t_http.aclose()
        await d_http.aclose()
        await s_http.aclose()
        await a_http.aclose()


async def test_missing_api_key_returns_clear_error_for_keyed_providers() -> None:
    for provider, tool_name in (
        ("tavily", "web_search_tavily"),
        ("tavily", "tavily_extract_web_page"),
        ("bocha", "web_search_bocha"),
        ("brave", "web_search_brave"),
        ("firecrawl", "web_search_firecrawl"),
        ("firecrawl", "firecrawl_extract_web_page"),
        ("baidu_ai_search", "web_search_baidu"),
        ("exa", "web_search_exa"),
        ("exa", "exa_get_contents"),
        ("deepseek", "web_search_deepseek"),
        ("deepseek", "deepseek_fetch_web_page"),
        ("serpapi", "web_search_serpapi"),
        ("serpapi", "news_search_serpapi"),
        ("serpapi", "image_search_serpapi"),
        ("serpapi", "finance_search_serpapi"),
        ("serpapi", "scholar_search_serpapi"),
    ):
        kit = WebSearchToolkit(WebSearchConfig(enabled=True, provider=provider))
        tool = _tool_map(kit)[tool_name]
        arg = (
            {"url": "https://example.com"}
            if "extract" in tool_name or "fetch" in tool_name or "contents" in tool_name
            else {"query": "hello"}
        )
        out = await tool.handler(**arg)
        assert out.startswith("Error:")
        assert "not configured" in out
