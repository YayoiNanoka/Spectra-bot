"""Unit tests for spectra.providers."""

from __future__ import annotations

import json

import pytest

from spectra.config import EmbeddingConfig, JevConfig
from spectra.providers import (
    DEFAULT_USER_AGENT,
    OPENCODE_SESSION_HEADER,
    OPENROUTER_REFERER,
    OPENROUTER_TITLE,
    build_decision_payload,
    build_provider_headers,
    normalize_decision_answers,
    resolve_decision_endpoint,
    resolve_decision_profile,
    resolve_embedding_endpoint,
    resolve_embedding_profile,
    resolve_openai_compat,
    should_omit_empty_tool_parameters,
    should_send_embedding_dimensions,
)


def test_build_provider_headers_defaults_and_case_insensitive_overrides() -> None:
    assert build_provider_headers() == {"User-Agent": DEFAULT_USER_AGENT}

    # Empty or whitespace user-agent overrides are ignored (AstrBot headers.py).
    assert build_provider_headers({"user-agent": "   ", "X-Test": "1"}) == {
        "User-Agent": DEFAULT_USER_AGENT,
        "X-Test": "1",
    }
    assert build_provider_headers({"User-Agent": ""}) == {
        "User-Agent": DEFAULT_USER_AGENT,
    }

    # Non-empty case-insensitive user-agent replaces User-Agent without duplicates.
    overridden = build_provider_headers({"user-agent": "custom-agent/2.0"})
    assert overridden == {"User-Agent": "custom-agent/2.0"}

    # Custom headers override gateway headers case-insensitively without duplicates.
    merged = build_provider_headers(
        {
            "http-referer": "https://custom.example",
            "x-openrouter-title": "CustomBot",
            "   ": "ignored",
        },
        gateway_headers={
            "HTTP-Referer": OPENROUTER_REFERER,
            "X-OpenRouter-Title": OPENROUTER_TITLE,
        },
    )
    assert merged == {
        "User-Agent": DEFAULT_USER_AGENT,
        "http-referer": "https://custom.example",
        "x-openrouter-title": "CustomBot",
    }
    assert len([k for k in merged if k.lower() == "http-referer"]) == 1


@pytest.mark.parametrize(
    ("raw_url", "expected_url", "expected_session_header"),
    [
        (
            "https://opencode.ai/zen/go/v1",
            "https://opencode.ai/inference/go/openai/v1",
            OPENCODE_SESSION_HEADER,
        ),
        (
            "https://opencode.ai/zen/go/v1/",
            "https://opencode.ai/inference/go/openai/v1",
            OPENCODE_SESSION_HEADER,
        ),
        (
            "https://opencode.ai/zen/go",
            "https://opencode.ai/inference/go/openai/v1",
            OPENCODE_SESSION_HEADER,
        ),
        (
            "https://sub.opencode.ai:8443/zen/go/",
            "https://sub.opencode.ai:8443/inference/go/openai/v1",
            OPENCODE_SESSION_HEADER,
        ),
        (
            "https://opencode.ai/inference/go/openai/v1",
            "https://opencode.ai/inference/go/openai/v1",
            OPENCODE_SESSION_HEADER,
        ),
        (
            "https://opencode.ai/inference/go",
            "https://opencode.ai/inference/go",
            OPENCODE_SESSION_HEADER,
        ),
        (
            "https://opencode.ai/zen/v1",
            "https://opencode.ai/zen/v1",
            None,
        ),
        (
            "https://opencode.ai/good/v1",
            "https://opencode.ai/good/v1",
            None,
        ),
        (
            "https://evil-opencode.ai/zen/go/v1",
            "https://evil-opencode.ai/zen/go/v1",
            None,
        ),
    ],
)
def test_resolve_openai_compat_opencode_go_rules(
    raw_url: str,
    expected_url: str,
    expected_session_header: str | None,
) -> None:
    profile = resolve_openai_compat(base_url=raw_url, model="kimi-k2.5")
    assert profile.base_url == expected_url
    assert profile.session_header_name == expected_session_header
    assert profile.reasoning_key == "reasoning_content"
    assert profile.default_headers == {"User-Agent": DEFAULT_USER_AGENT}


def test_openai_compat_session_header_behavior_and_override() -> None:
    profile = resolve_openai_compat(
        base_url="https://opencode.ai/zen/go/v1",
        model="glm-5",
    )
    assert profile.build_request_headers("group:12345") == {
        "x-opencode-session": "group:12345"
    }
    assert profile.build_request_headers("  ") == {
        "x-opencode-session": "spectra:default"
    }
    assert profile.build_request_headers(None, default_session="spectra:vision") == {
        "x-opencode-session": "spectra:vision"
    }

    # Explicit user override in custom_headers suppresses dynamic session header.
    overridden = resolve_openai_compat(
        base_url="https://opencode.ai/inference/go/openai/v1",
        model="glm-5",
        custom_headers={"X-OpenCode-Session": "pinned-session"},
    )
    assert overridden.default_headers["X-OpenCode-Session"] == "pinned-session"
    assert overridden.build_request_headers("group:999") == {}


def test_resolve_openai_compat_openrouter_and_extra_body_isolation() -> None:
    source_extra = {"provider": {"order": ["anthropic"]}}
    profile = resolve_openai_compat(
        base_url="https://openrouter.ai/api/v1",
        model="deepseek/deepseek-r1",
        extra_body=source_extra,
    )
    assert profile.base_url == "https://openrouter.ai/api/v1"
    assert profile.session_header_name is None
    assert profile.reasoning_key == "reasoning"
    assert profile.default_headers == {
        "User-Agent": DEFAULT_USER_AGENT,
        "HTTP-Referer": OPENROUTER_REFERER,
        "X-OpenRouter-Title": OPENROUTER_TITLE,
    }
    assert profile.build_request_headers("group:1") == {}

    # Mutating caller dict does not mutate profile.extra_body.
    source_extra["provider"]["order"].append("openai")
    assert profile.extra_body == {"provider": {"order": ["anthropic"]}}

    # Subdomain + custom header override.
    sub = resolve_openai_compat(
        base_url="https://api.openrouter.ai/api/v1",
        model="x",
        custom_headers={"HTTP-Referer": "https://my.app"},
    )
    assert sub.reasoning_key == "reasoning"
    assert sub.default_headers["HTTP-Referer"] == "https://my.app"
    assert sub.default_headers["X-OpenRouter-Title"] == OPENROUTER_TITLE


def test_should_omit_empty_tool_parameters() -> None:
    assert should_omit_empty_tool_parameters("gemini-2.5-flash") is True
    assert should_omit_empty_tool_parameters("google/Gemini-2.5-Pro") is True
    assert should_omit_empty_tool_parameters("gpt-4o") is False
    assert should_omit_empty_tool_parameters("deepseek-v4.1-flash") is False


@pytest.mark.parametrize(
    ("base_url", "protocol", "expected_url"),
    [
        (
            "https://openrouter.ai",
            "auto",
            "https://openrouter.ai/api/alpha/decisions",
        ),
        (
            "https://openrouter.ai/api/v1",
            "auto",
            "https://openrouter.ai/api/alpha/decisions",
        ),
        (
            "https://openrouter.ai/api/alpha",
            "auto",
            "https://openrouter.ai/api/alpha/decisions",
        ),
        (
            "https://openrouter.ai/api/alpha/decisions/",
            "auto",
            "https://openrouter.ai/api/alpha/decisions",
        ),
        (
            "https://openrouter.ai/api/alpha/systemone",
            "auto",
            "https://openrouter.ai/api/alpha/systemone",
        ),
        (
            "https://api.typesafe.ai/v1",
            "auto",
            "https://api.typesafe.ai/v1/systemone",
        ),
        (
            "https://api.openai.com/v1",
            "auto",
            "https://api.openai.com/v1/decisions",
        ),
        (
            "https://api.openai.com/v1/decisions/",
            "auto",
            "https://api.openai.com/v1/decisions",
        ),
        (
            "https://proxy.example.com/v1/decisions",
            "auto",
            "https://proxy.example.com/v1/decisions",
        ),
        (
            "https://proxy.example.com/custom",
            "openai_decision",
            "https://proxy.example.com/custom/decisions",
        ),
        ("   ", "auto", ""),
    ],
)
def test_resolve_decision_endpoint(
    base_url: str,
    protocol: str,
    expected_url: str,
) -> None:
    assert (
        resolve_decision_endpoint(base_url, protocol=protocol)  # type: ignore[arg-type]
        == expected_url
    )


@pytest.mark.parametrize(
    ("base_url", "model", "protocol", "expected_protocol", "expected_model"),
    [
        (
            "https://openrouter.ai/api/alpha/decisions",
            "jev-1.14",
            "auto",
            "system_one",
            "typesafe/jev-1.14",
        ),
        (
            "https://openrouter.ai/api/v1",
            "typesafe/jev-1.14.0",
            "auto",
            "system_one",
            "typesafe/jev-1.14",
        ),
        (
            "https://openrouter.ai/api/alpha/decisions",
            "typesafe/jev-1.13.0",
            "auto",
            "system_one",
            "typesafe/jev-1.13",
        ),
        (
            "https://openrouter.ai/api/alpha/decisions",
            "jev-1.14.0",
            "auto",
            "system_one",
            "typesafe/jev-1.14",
        ),
        (
            "https://openrouter.ai/api/alpha/decisions",
            "typesafe/jev-1.14.1",
            "auto",
            "system_one",
            "typesafe/jev-1.14.1",
        ),
        (
            "https://api.typesafe.ai/v1",
            "typesafe/jev-1.14",
            "auto",
            "system_one",
            "jev-1.14",
        ),
        (
            "https://api.openai.com/v1",
            "gpt-4.1-mini",
            "auto",
            "openai_decision",
            "gpt-4.1-mini",
        ),
        (
            "https://gateway.example.com/v1/decisions",
            "gpt-4.1-mini",
            "auto",
            "openai_decision",
            "gpt-4.1-mini",
        ),
        (
            "https://openrouter.ai/v1/decisions",
            "typesafe/jev-1.14",
            "auto",
            "system_one",
            "typesafe/jev-1.14",
        ),
        (
            "https://api.openai.com/v1",
            "jev-1.14",
            "system_one",
            "system_one",
            "jev-1.14",
        ),
    ],
)
def test_resolve_decision_profile_protocol_and_model_normalization(
    base_url: str,
    model: str,
    protocol: str,
    expected_protocol: str,
    expected_model: str,
) -> None:
    cfg = JevConfig(
        base_url=base_url,
        api_key="sk-test",
        model=model,
        protocol=protocol,  # type: ignore[arg-type]
    )
    profile = resolve_decision_profile(cfg)
    assert profile.protocol == expected_protocol
    assert profile.model == expected_model
    assert profile.headers["Authorization"] == "Bearer sk-test"
    assert profile.headers["User-Agent"] == DEFAULT_USER_AGENT
    if "openrouter.ai" in base_url:
        assert profile.headers["HTTP-Referer"] == OPENROUTER_REFERER
        assert profile.headers["X-OpenRouter-Title"] == OPENROUTER_TITLE
    else:
        assert "HTTP-Referer" not in profile.headers


def test_build_decision_payload_openai_decision_input_variants() -> None:
    profile = resolve_decision_profile(
        JevConfig(
            base_url="https://api.openai.com/v1",
            api_key="sk-oa",
            model="gpt-4.1-mini",
        )
    )
    questions = {
        "should_reply": {
            "type": "noul",
            "instructions": "是否回复？",
            "criteria": {"true": "开放话题", "false": "单聊对线"},
        },
        "effort": {
            "type": "choice",
            "instructions": "思考深度？",
            "criteria": {"default": "闲聊", "high": "复杂推理"},
        },
    }

    # Mapping state -> input_json message
    payload_dict = build_decision_payload(
        profile,
        state={"bot_name": "Spectra", "latest_batch": ["hi"]},
        questions=questions,
    )
    assert payload_dict["model"] == "gpt-4.1-mini"
    assert payload_dict["input"] == [
        {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_json",
                    "data": {"bot_name": "Spectra", "latest_batch": ["hi"]},
                }
            ],
        }
    ]
    assert len(payload_dict["questions"]) == 2
    q_sr = payload_dict["questions"][0]
    assert q_sr["type"] == "predicate"
    assert q_sr["name"] == "should_reply"
    assert "是否回复？" in q_sr["instructions"]
    assert "判定为是的情形：开放话题" in q_sr["instructions"]
    assert "判定为否的情形：单聊对线" in q_sr["instructions"]
    assert "criteria" not in q_sr

    q_eff = payload_dict["questions"][1]
    assert q_eff == {
        "type": "choice",
        "name": "effort",
        "instructions": "思考深度？",
        "choices": [
            {"value": "default", "description": "闲聊"},
            {"value": "high", "description": "复杂推理"},
        ],
    }

    # String state -> sent verbatim
    payload_str = build_decision_payload(
        profile, state="plain text", questions=questions
    )
    assert payload_str["input"] == "plain text"

    # List state -> JSON string
    payload_list = build_decision_payload(
        profile,
        state=["line1", "line2"],
        questions=questions,
    )
    assert json.loads(payload_list["input"]) == ["line1", "line2"]


def test_normalize_decision_answers_openai_edge_cases() -> None:
    valid = {
        "answers": [
            {
                "type": "choice",
                "name": "effort",
                "choice": "high",
                "confidence": 0.8,
                "probabilities": [
                    {"value": "default", "probability": 0.2},
                    {"value": "high", "probability": 0.8},
                ],
            },
            {
                "type": "predicate",
                "name": "should_reply",
                "probability": 0.72,
            },
        ]
    }
    parsed = normalize_decision_answers(valid, protocol="openai_decision")
    assert parsed == {
        "effort": {
            "type": "choice",
            "choice": "high",
            "confidence": 0.8,
            "probabilities": {"default": 0.2, "high": 0.8},
        },
        "should_reply": {"type": "predicate", "probability": 0.72},
    }

    # Refusal degrades cleanly
    assert (
        normalize_decision_answers(
            {"answers": [{"type": "refusal", "name": "should_reply"}]},
            protocol="openai_decision",
        )
        is None
    )
    # Duplicate answer names rejected
    assert (
        normalize_decision_answers(
            {
                "answers": [
                    {"type": "predicate", "name": "should_reply", "probability": 0.5},
                    {"type": "predicate", "name": "should_reply", "probability": 0.6},
                ]
            },
            protocol="openai_decision",
        )
        is None
    )
    # Out-of-range or boolean probability rejected
    assert (
        normalize_decision_answers(
            {
                "answers": [
                    {"type": "predicate", "name": "should_reply", "probability": 1.2}
                ]
            },
            protocol="openai_decision",
        )
        is None
    )
    assert (
        normalize_decision_answers(
            {
                "answers": [
                    {"type": "predicate", "name": "should_reply", "probability": True}
                ]
            },
            protocol="openai_decision",
        )
        is None
    )
    # Choice probabilities not summing to 1 rejected
    assert (
        normalize_decision_answers(
            {
                "answers": [
                    {
                        "type": "choice",
                        "name": "effort",
                        "choice": "high",
                        "confidence": 0.9,
                        "probabilities": [
                            {"value": "default", "probability": 0.9},
                            {"value": "high", "probability": 0.9},
                        ],
                    }
                ]
            },
            protocol="openai_decision",
        )
        is None
    )


@pytest.mark.parametrize(
    ("raw_url", "expected_url"),
    [
        ("", ""),
        ("   ", ""),
        (
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings",
        ),
        (
            "https://dashscope.aliyuncs.com/compatible-mode/v1/",
            "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings",
        ),
        (
            "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings",
            "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings",
        ),
        (
            "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings/",
            "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings",
        ),
        (
            "https://openrouter.ai",
            "https://openrouter.ai/api/v1/embeddings",
        ),
        (
            "https://openrouter.ai/",
            "https://openrouter.ai/api/v1/embeddings",
        ),
        (
            "https://openrouter.ai/api/v1",
            "https://openrouter.ai/api/v1/embeddings",
        ),
        (
            "https://openrouter.ai/api/v1/embeddings",
            "https://openrouter.ai/api/v1/embeddings",
        ),
        (
            "https://opencode.ai/zen/go",
            "https://opencode.ai/inference/go/openai/v1/embeddings",
        ),
        (
            "https://opencode.ai/zen/go/v1",
            "https://opencode.ai/inference/go/openai/v1/embeddings",
        ),
        (
            "https://opencode.ai/zen/go/v1/embeddings",
            "https://opencode.ai/inference/go/openai/v1/embeddings",
        ),
        (
            "https://opencode.ai/zen/go/embeddings/",
            "https://opencode.ai/inference/go/openai/v1/embeddings",
        ),
        (
            "https://opencode.ai/inference/go/openai/v1/embeddings",
            "https://opencode.ai/inference/go/openai/v1/embeddings",
        ),
        (
            "http://localhost:11434",
            "http://localhost:11434/v1/embeddings",
        ),
        (
            "http://localhost:11434/",
            "http://localhost:11434/v1/embeddings",
        ),
        (
            "http://localhost:11434/api/embed",
            "http://localhost:11434/v1/embeddings",
        ),
        (
            "http://localhost:11434/api/embed/",
            "http://localhost:11434/v1/embeddings",
        ),
        (
            "http://proxy.local/ollama/api/embed",
            "http://proxy.local/ollama/v1/embeddings",
        ),
        (
            "https://generativelanguage.googleapis.com/v1beta/openai",
            "https://generativelanguage.googleapis.com/v1beta/openai/embeddings",
        ),
        (
            "https://generativelanguage.googleapis.com/v1beta/openai/",
            "https://generativelanguage.googleapis.com/v1beta/openai/embeddings",
        ),
        (
            "https://generativelanguage.googleapis.com/v1beta/openai/embeddings",
            "https://generativelanguage.googleapis.com/v1beta/openai/embeddings",
        ),
        (
            "https://api.example.com/v2/custom",
            "https://api.example.com/v2/custom/embeddings",
        ),
        (
            "https://api.example.com/v1.5/openai",
            "https://api.example.com/v1.5/openai/embeddings",
        ),
        (
            "https://api.example.com/custom-proxy",
            "https://api.example.com/custom-proxy/v1/embeddings",
        ),
    ],
)
def test_resolve_embedding_endpoint_rules_and_idempotency(
    raw_url: str,
    expected_url: str,
) -> None:
    resolved = resolve_embedding_endpoint(raw_url)
    assert resolved == expected_url
    # Idempotency check
    assert resolve_embedding_endpoint(resolved) == expected_url


def test_resolve_embedding_profile_headers_auth_and_extra_body() -> None:
    # 1. Keyless local Ollama: omits Authorization header
    ollama_profile = resolve_embedding_profile(
        EmbeddingConfig(
            base_url="http://localhost:11434",
            api_key="   ",
            model="  nomic-embed-text  ",
        )
    )
    assert ollama_profile.endpoint_url == "http://localhost:11434/v1/embeddings"
    assert ollama_profile.model == "nomic-embed-text"
    assert "Authorization" not in ollama_profile.headers
    assert ollama_profile.headers == {"User-Agent": DEFAULT_USER_AGENT}

    # 2. OpenRouter headers + API key + extra_body deep copy
    source_extra = {"provider": {"order": ["openai"]}}
    or_cfg = EmbeddingConfig(
        base_url="https://openrouter.ai",
        api_key="sk-or-emb",
        model="qwen/qwen3-embedding-8b",
        custom_headers={"HTTP-Referer": "https://custom.example"},
        extra_body=source_extra,
    )
    or_profile = resolve_embedding_profile(or_cfg)
    assert or_profile.endpoint_url == "https://openrouter.ai/api/v1/embeddings"
    assert or_profile.headers == {
        "User-Agent": DEFAULT_USER_AGENT,
        "HTTP-Referer": "https://custom.example",
        "X-OpenRouter-Title": OPENROUTER_TITLE,
        "Authorization": "Bearer sk-or-emb",
    }
    source_extra["provider"]["order"].append("dashscope")
    assert or_profile.extra_body == {"provider": {"order": ["openai"]}}

    # 3. OpenCode Go session header & user override
    go_profile = resolve_embedding_profile(
        EmbeddingConfig(
            base_url="https://opencode.ai/zen/go/v1/embeddings",
            api_key="sk-go",
            model="text-embedding-3-small",
        )
    )
    assert (
        go_profile.endpoint_url
        == "https://opencode.ai/inference/go/openai/v1/embeddings"
    )
    assert go_profile.headers[OPENCODE_SESSION_HEADER] == "spectra:embedding"
    assert go_profile.headers["Authorization"] == "Bearer sk-go"

    go_override = resolve_embedding_profile(
        EmbeddingConfig(
            base_url="https://opencode.ai/zen/go",
            api_key="sk-go",
            model="text-embedding-3-small",
            custom_headers={"X-OpenCode-Session": "custom-emb-session"},
        )
    )
    assert go_override.headers["X-OpenCode-Session"] == "custom-emb-session"
    assert OPENCODE_SESSION_HEADER not in go_override.headers

    # 4. OpenCode non-Go path does not attach session header
    zen_profile = resolve_embedding_profile(
        EmbeddingConfig(
            base_url="https://opencode.ai/zen/v1",
            api_key="sk-zen",
            model="text-embedding-3-small",
        )
    )
    assert OPENCODE_SESSION_HEADER not in zen_profile.headers


@pytest.mark.parametrize(
    ("model", "mode", "expected"),
    [
        ("text-embedding-3-small", "auto", True),
        ("openai/text-embedding-3-large", "auto", True),
        ("text-embedding-v3", "auto", True),
        ("text-embedding-v4", "auto", True),
        ("qwen/qwen3-embedding-8b", "auto", True),
        ("Qwen/Qwen3-Embedding-4B", "auto", True),
        ("jina-embeddings-v3", "auto", True),
        ("cohere/embed-v4.0", "auto", True),
        ("text-embedding-ada-002", "auto", False),
        ("bge-m3", "auto", False),
        ("nomic-embed-text", "auto", False),
        ("", "auto", False),
        ("bge-m3", "always", True),
        ("text-embedding-v4", "never", False),
    ],
)
def test_should_send_embedding_dimensions(
    model: str,
    mode: str,
    expected: bool,
) -> None:
    assert (
        should_send_embedding_dimensions(
            model,
            mode,  # type: ignore[arg-type]
        )
        is expected
    )
