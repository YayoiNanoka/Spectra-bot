"""Unit tests for spectra.config."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from spectra.config import (
    DEFAULT_GROUP_CONSOLIDATION_PROMPT,
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
    DEFAULT_MONTAGE_PROMPT_NOTE,
    DEFAULT_PRIVATE_CONSOLIDATION_PROMPT,
    DEFAULT_VISION_PROMPT,
    AccessConfig,
    ActiveReplyConfig,
    AppConfig,
    EmbeddingConfig,
    GateConfig,
    JevConfig,
    JevPromptsConfig,
    LLMConfig,
    MemoryConfig,
    MemoryPromptsConfig,
    SegmentedReplyConfig,
    SessionConfig,
    TraceConfig,
    VisionConfig,
    WebSearchConfig,
    WebUIConfig,
    load_config,
    save_config,
)


def test_default_config_values() -> None:
    cfg = AppConfig()
    assert cfg.db.path == Path("data/bot.db")
    assert cfg.db.message_retention_days == 7
    assert cfg.access.group_mode == "blacklist"
    assert cfg.access.private_mode == "blacklist"
    assert cfg.access.ignored_users == ["2854196310"]
    assert cfg.napcat.host == "0.0.0.0"
    assert cfg.napcat.port == 6199
    assert cfg.napcat.bot_name == "Spectra"
    assert cfg.napcat.segmented_reply.enable is True
    assert cfg.napcat.segmented_reply.interval_method == "random"
    assert cfg.napcat.segmented_reply.interval == "1.5,3.5"
    assert cfg.napcat.segmented_reply.log_base == 2.6
    assert cfg.napcat.segmented_reply.words_count_threshold == 150
    assert cfg.napcat.segmented_reply.split_mode == "regex"
    assert cfg.napcat.segmented_reply.regex == r".*?[。？！~…]+|.+$"
    assert cfg.napcat.segmented_reply.split_words == ["。", "？", "！", "~", "…"]
    assert cfg.napcat.segmented_reply.content_cleanup_rule == ""
    assert cfg.napcat.forward_threshold == 1500
    assert cfg.vision.enabled is True
    assert cfg.vision.prompt == DEFAULT_VISION_PROMPT
    assert cfg.vision.montage_prompt_note == DEFAULT_MONTAGE_PROMPT_NOTE
    assert cfg.vision.max_size == 1280
    assert cfg.vision.reasoning_effort == ""
    assert cfg.vision.custom_headers == {}
    assert cfg.vision.extra_body == {}
    assert not hasattr(cfg.vision, "max_input_bytes")
    assert cfg.llm.base_url == "https://api.openai.com/v1"
    assert cfg.llm.api_key == ""
    assert cfg.llm.model == "gpt-4o"
    assert cfg.llm.fallback_model == ""
    assert cfg.llm.default_effort == ""
    assert cfg.llm.high_effort == "high"
    assert cfg.llm.timeout_sec == 60.0
    assert cfg.llm.max_retries == 5
    assert cfg.llm.max_tool_steps == 10
    assert cfg.llm.custom_headers == {}
    assert cfg.llm.extra_body == {}
    assert cfg.embedding.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert cfg.embedding.api_key == ""
    assert cfg.embedding.model == ""
    assert cfg.embedding.dimensions is None
    assert cfg.embedding.dimensions_mode == "auto"
    assert cfg.embedding.timeout_sec == 15.0
    assert cfg.embedding.max_retries == 2
    assert cfg.embedding.batch_size == 10
    assert cfg.embedding.custom_headers == {}
    assert cfg.embedding.extra_body == {}
    assert cfg.embedding.is_configured is False
    assert cfg.jev.enabled is True
    assert cfg.jev.base_url == "https://openrouter.ai/api/alpha/decisions"
    assert cfg.jev.api_key == ""
    assert cfg.jev.model == "typesafe/jev-1.14"
    assert cfg.jev.protocol == "auto"
    assert cfg.jev.custom_headers == {}
    assert cfg.jev.timeout_sec == 1.5
    assert cfg.jev.reply_threshold == 0.65
    assert cfg.jev.high_effort_threshold == 0.60
    assert cfg.jev.is_configured is False
    assert cfg.jev.prompts.reply_instructions == DEFAULT_JEV_REPLY_INSTRUCTIONS
    assert cfg.jev.prompts.reply_true_criteria == DEFAULT_JEV_REPLY_TRUE_CRITERIA
    assert cfg.jev.prompts.reply_false_criteria == DEFAULT_JEV_REPLY_FALSE_CRITERIA
    assert cfg.jev.prompts.effort_instructions == DEFAULT_JEV_EFFORT_INSTRUCTIONS
    assert (
        cfg.jev.prompts.effort_default_criteria == DEFAULT_JEV_EFFORT_DEFAULT_CRITERIA
    )
    assert cfg.jev.prompts.effort_high_criteria == DEFAULT_JEV_EFFORT_HIGH_CRITERIA
    assert (
        cfg.jev.prompts.continuity_instructions == DEFAULT_JEV_CONTINUITY_INSTRUCTIONS
    )
    assert (
        cfg.jev.prompts.continuity_true_criteria == DEFAULT_JEV_CONTINUITY_TRUE_CRITERIA
    )
    assert (
        cfg.jev.prompts.continuity_false_criteria
        == DEFAULT_JEV_CONTINUITY_FALSE_CRITERIA
    )
    assert cfg.jev.prompts.rerank_instructions == DEFAULT_JEV_RERANK_INSTRUCTIONS
    assert cfg.jev.prompts.rerank_true_criteria == DEFAULT_JEV_RERANK_TRUE_CRITERIA
    assert cfg.jev.prompts.rerank_false_criteria == DEFAULT_JEV_RERANK_FALSE_CRITERIA
    assert cfg.gate.proactive_mode == "on"
    assert cfg.gate.private_debounce_sec == 1.5
    assert cfg.gate.group_debounce_sec == 2.5
    assert cfg.gate.max_batch_wait_sec == 6.0
    assert cfg.gate.proactive_cooldown_sec == 60.0
    assert cfg.gate.proactive_max_per_hour == 15
    assert cfg.gate.hot_window_sec == 30.0
    assert cfg.gate.state_message_limit == 10
    assert cfg.gate.active_reply.method == "possibility_reply"
    assert cfg.gate.active_reply.possibility_reply == 0.1
    assert cfg.gate.active_reply.whitelist == []
    assert cfg.session.prompts_dir == Path("prompts")
    assert cfg.session.default_persona_id == "default"
    assert cfg.session.window_max_messages == 60
    assert cfg.session.window_keep_messages == 30
    assert cfg.session.timezone == "Asia/Shanghai"
    assert cfg.session.datetime_system_prompt is True
    assert cfg.session.group_idle_timeout_minutes == 30
    assert cfg.session.private_idle_timeout_minutes == 180
    assert cfg.session.group_hard_timeout_hours == 6
    assert cfg.session.private_hard_timeout_hours == 24
    assert cfg.session.continuity_threshold == 0.55
    assert cfg.session.continuity_tail_messages == 6
    assert cfg.session.lead_in_gap_sec == 300
    assert cfg.session.lead_in_max_messages == 12
    assert cfg.memory.enabled is True
    assert cfg.memory.vector_top_k == 12
    assert cfg.memory.min_vector_similarity == 0.25
    assert cfg.memory.rerank_threshold == 0.50
    assert cfg.memory.max_recall_items == 5
    assert cfg.memory.jargon_min_confidence == 0.60
    assert cfg.memory.jargon_max_matches == 8
    assert cfg.memory.consolidation_enabled is True
    assert cfg.memory.consolidation_max_messages == 120
    assert cfg.memory.cleanup_interval_sec == 3600.0
    assert (
        cfg.memory.prompts.group_consolidation_prompt
        == DEFAULT_GROUP_CONSOLIDATION_PROMPT
    )
    assert (
        cfg.memory.prompts.private_consolidation_prompt
        == DEFAULT_PRIVATE_CONSOLIDATION_PROMPT
    )
    assert cfg.trace.enabled is True
    assert cfg.trace.path == Path("data/traces.jsonl")
    assert cfg.web_search.enabled is False
    assert cfg.web_search.provider == "tavily"
    assert cfg.web_search.timeout_sec == 30.0
    assert cfg.web_search.tavily_keys == []
    assert cfg.web_search.bocha_keys == []
    assert cfg.web_search.brave_keys == []
    assert cfg.web_search.firecrawl_keys == []
    assert cfg.web_search.baidu_app_builder_key == ""
    assert cfg.web_search.exa_keys == []
    assert cfg.web_search.anysearch_keys == []
    assert cfg.web_search.deepseek_keys == []
    assert cfg.web_search.serpapi_keys == []
    assert cfg.webui.enabled is True
    assert cfg.webui.host == "127.0.0.1"
    assert cfg.webui.port == 6200
    assert cfg.webui.auth_token == ""


def test_load_missing_file_returns_defaults(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "does_not_exist.json")
    assert cfg == AppConfig()


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    config_path = tmp_path / "nested" / "config.json"
    original = AppConfig(
        db={"path": Path("custom/spectra.db"), "message_retention_days": 14},
        access={
            "admin_users": ["10001"],
            "ignored_users": ["2854196310", "99999"],
            "group_mode": "whitelist",
            "group_whitelist": ["20001"],
            "private_mode": "whitelist",
            "private_whitelist": ["30001"],
        },
        napcat={
            "host": "127.0.0.1",
            "port": 7000,
            "access_token": "secret",
            "bot_name": "SpectraTest",
            "segmented_reply": {
                "enable": False,
                "interval_method": "log",
                "interval": "0.5,1.0",
                "log_base": 3.0,
                "words_count_threshold": 200,
                "split_mode": "words",
                "split_words": ["。", "！"],
                "content_cleanup_rule": r"\s+$",
            },
        },
        vision={
            "enabled": True,
            "base_url": "https://example.invalid/v1",
            "api_key": "sk-test",
            "model": "vision-x",
            "prompt": "自定义识图提示词",
            "montage_prompt_note": "自定义动图拼图提示词",
            "max_size": 1024,
            "reasoning_effort": "low",
            "custom_headers": {"X-Vision": "1"},
            "extra_body": {"thinking": {"type": "enabled"}},
        },
        llm={
            "base_url": "https://llm.example.invalid/v1",
            "api_key": "sk-llm",
            "model": "gpt-5",
            "fallback_model": "gpt-4o-mini",
            "default_effort": "low",
            "high_effort": "high",
            "timeout_sec": 45.0,
            "max_retries": 3,
            "max_tool_steps": 6,
            "custom_headers": {"X-LLM": "yes"},
            "extra_body": {"provider": {"sort": "latency"}},
        },
        embedding={
            "base_url": "https://embed.example.invalid/v1",
            "api_key": "sk-emb",
            "model": "text-embedding-v4",
            "dimensions": 1024,
            "dimensions_mode": "always",
            "timeout_sec": 20.0,
            "max_retries": 0,
            "batch_size": 8,
            "custom_headers": {"X-Emb": "1"},
            "extra_body": {"encoding_format": "float"},
        },
        jev={
            "enabled": True,
            "base_url": "https://openrouter.ai/api/alpha/decisions",
            "api_key": "sk-jev",
            "model": "typesafe/jev-1.14",
            "protocol": "system_one",
            "custom_headers": {"X-Jev": "ok"},
            "timeout_sec": 1.2,
            "reply_threshold": 0.7,
            "high_effort_threshold": 0.55,
            "prompts": {
                "reply_instructions": "自定义插话指令",
                "rerank_instructions": "候选 {index} 与 {literal} 有关吗？",
            },
        },
        gate={
            "proactive_mode": "shadow",
            "private_debounce_sec": 1.0,
            "group_debounce_sec": 2.0,
            "max_batch_wait_sec": 5.0,
            "proactive_cooldown_sec": 45.0,
            "proactive_max_per_hour": 10,
            "hot_window_sec": 20.0,
            "state_message_limit": 8,
            "active_reply": {
                "method": "possibility_reply",
                "possibility_reply": 0.25,
                "whitelist": ["20001"],
            },
        },
        session={
            "prompts_dir": Path("custom_prompts"),
            "default_persona_id": "catgirl",
            "window_max_messages": 80,
            "window_keep_messages": 40,
            "timezone": "UTC",
            "datetime_system_prompt": False,
            "group_idle_timeout_minutes": 45,
            "private_idle_timeout_minutes": 120,
            "group_hard_timeout_hours": 8,
            "private_hard_timeout_hours": 36,
            "continuity_threshold": 0.65,
            "continuity_tail_messages": 8,
            "lead_in_gap_sec": 180,
            "lead_in_max_messages": 10,
        },
        memory={
            "enabled": False,
            "vector_top_k": 20,
            "min_vector_similarity": 0.15,
            "rerank_threshold": 0.65,
            "max_recall_items": 8,
            "jargon_min_confidence": 0.75,
            "jargon_max_matches": 4,
            "consolidation_enabled": False,
            "consolidation_max_messages": 80,
            "cleanup_interval_sec": 900.0,
            "prompts": {
                "group_consolidation_prompt": "自定义群聊记忆整理提示词",
                "private_consolidation_prompt": "自定义私聊记忆整理提示词",
            },
        },
        trace={
            "enabled": False,
            "path": Path("logs/custom_traces.jsonl"),
        },
        web_search={
            "enabled": True,
            "provider": "serpapi",
            "timeout_sec": 15.0,
            "tavily_keys": ["tvly-1"],
            "bocha_keys": ["bocha-1"],
            "brave_keys": ["brave-1"],
            "firecrawl_keys": ["fc-1"],
            "baidu_app_builder_key": "baidu-1",
            "exa_keys": ["exa-1"],
            "anysearch_keys": ["any-1"],
            "deepseek_keys": ["ds-1"],
            "serpapi_keys": ["serp-1", "serp-2"],
        },
        webui={
            "enabled": False,
            "host": "0.0.0.0",
            "port": 6201,
            "auth_token": "tok-webui-secret",
        },
    )

    save_config(original, config_path)
    assert config_path.is_file()
    loaded = load_config(config_path)
    assert loaded == original


def test_extra_fields_and_invalid_log_base_rejected(tmp_path: Path) -> None:
    bad_path = tmp_path / "bad.json"
    bad_path.write_text('{"unknown_section": {}}', encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(bad_path)

    for invalid_base in (1.0, 0.5, -2.0, float("inf"), float("nan")):
        with pytest.raises(ValidationError):
            SegmentedReplyConfig(log_base=invalid_base)

    with pytest.raises(ValidationError):
        LLMConfig(unknown_field="x")  # type: ignore[call-arg]
    for invalid_timeout in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValidationError):
            VisionConfig(timeout_sec=invalid_timeout)
        with pytest.raises(ValidationError):
            LLMConfig(timeout_sec=invalid_timeout)
    with pytest.raises(ValidationError):
        LLMConfig(max_retries=0)
    with pytest.raises(ValidationError):
        LLMConfig(max_tool_steps=0)

    with pytest.raises(ValidationError):
        EmbeddingConfig(unknown_field="x")  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        EmbeddingConfig(dimensions=0)
    with pytest.raises(ValidationError):
        EmbeddingConfig(dimensions_mode="invalid")  # type: ignore[arg-type]
    for invalid_timeout in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValidationError):
            EmbeddingConfig(timeout_sec=invalid_timeout)
    with pytest.raises(ValidationError):
        EmbeddingConfig(max_retries=-1)
    with pytest.raises(ValidationError):
        EmbeddingConfig(batch_size=0)
    assert EmbeddingConfig(max_retries=0).max_retries == 0

    with pytest.raises(ValidationError):
        JevConfig(unknown_field="x")  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        JevPromptsConfig(unknown_field="x")  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        JevConfig(protocol="invalid")  # type: ignore[arg-type]
    for invalid_timeout in (0.0, -0.5, float("inf"), float("nan")):
        with pytest.raises(ValidationError):
            JevConfig(timeout_sec=invalid_timeout)
    for invalid_prob in (-0.01, 1.01, float("inf"), float("nan")):
        with pytest.raises(ValidationError):
            JevConfig(reply_threshold=invalid_prob)
        with pytest.raises(ValidationError):
            JevConfig(high_effort_threshold=invalid_prob)
        with pytest.raises(ValidationError):
            ActiveReplyConfig(possibility_reply=invalid_prob)

    with pytest.raises(ValidationError):
        ActiveReplyConfig(unknown_field="x")  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        GateConfig(unknown_field="x")  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        GateConfig(proactive_mode="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        GateConfig(proactive_max_per_hour=-1)
    with pytest.raises(ValidationError):
        GateConfig(state_message_limit=0)
    for invalid_sec in (-0.1, float("inf"), float("nan")):
        with pytest.raises(ValidationError):
            GateConfig(private_debounce_sec=invalid_sec)
        with pytest.raises(ValidationError):
            GateConfig(group_debounce_sec=invalid_sec)
        with pytest.raises(ValidationError):
            GateConfig(max_batch_wait_sec=invalid_sec)
        with pytest.raises(ValidationError):
            GateConfig(proactive_cooldown_sec=invalid_sec)
        with pytest.raises(ValidationError):
            GateConfig(hot_window_sec=invalid_sec)

    with pytest.raises(ValidationError):
        SessionConfig(unknown_field="x")  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        SessionConfig(window_max_messages=1)
    with pytest.raises(ValidationError):
        SessionConfig(window_keep_messages=0)
    with pytest.raises(ValidationError):
        SessionConfig(window_max_messages=30, window_keep_messages=30)
    with pytest.raises(ValidationError):
        SessionConfig(window_max_messages=20, window_keep_messages=25)
    for bad_tz in ("", "Not/A_Timezone", "Asia/Shanghai ", "/etc/passwd"):
        with pytest.raises(ValidationError):
            SessionConfig(timezone=bad_tz)
    assert SessionConfig(timezone="UTC").timezone == "UTC"
    assert SessionConfig(timezone="America/New_York").timezone == "America/New_York"
    with pytest.raises(ValidationError):
        SessionConfig(group_idle_timeout_minutes=0)
    with pytest.raises(ValidationError):
        SessionConfig(private_idle_timeout_minutes=0)
    with pytest.raises(ValidationError):
        SessionConfig(group_hard_timeout_hours=0)
    with pytest.raises(ValidationError):
        SessionConfig(private_hard_timeout_hours=0)
    with pytest.raises(ValidationError):
        SessionConfig(continuity_tail_messages=0)
    with pytest.raises(ValidationError):
        SessionConfig(lead_in_gap_sec=-1)
    with pytest.raises(ValidationError):
        SessionConfig(lead_in_max_messages=0)
    for invalid_prob in (-0.01, 1.01, float("inf"), float("nan")):
        with pytest.raises(ValidationError):
            SessionConfig(continuity_threshold=invalid_prob)
    with pytest.raises(ValidationError):
        SessionConfig(group_idle_timeout_minutes=360, group_hard_timeout_hours=6)
    with pytest.raises(ValidationError):
        SessionConfig(group_idle_timeout_minutes=400, group_hard_timeout_hours=6)
    with pytest.raises(ValidationError):
        SessionConfig(private_idle_timeout_minutes=1440, private_hard_timeout_hours=24)
    with pytest.raises(ValidationError):
        SessionConfig(private_idle_timeout_minutes=1500, private_hard_timeout_hours=24)
    assert SessionConfig(lead_in_gap_sec=0).lead_in_gap_sec == 0

    with pytest.raises(ValidationError):
        MemoryConfig(unknown_field="x")  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        MemoryPromptsConfig(unknown_field="x")  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        MemoryConfig(vector_top_k=0)
    with pytest.raises(ValidationError):
        MemoryConfig(max_recall_items=0)
    with pytest.raises(ValidationError):
        MemoryConfig(jargon_max_matches=0)
    with pytest.raises(ValidationError):
        MemoryConfig(consolidation_max_messages=9)
    for invalid_interval in (0.0, -10.0, float("inf"), float("nan")):
        with pytest.raises(ValidationError):
            MemoryConfig(cleanup_interval_sec=invalid_interval)
    for invalid_sim in (-1.01, 1.01, float("inf"), float("nan")):
        with pytest.raises(ValidationError):
            MemoryConfig(min_vector_similarity=invalid_sim)
    for invalid_prob in (-0.01, 1.01, float("inf"), float("nan")):
        with pytest.raises(ValidationError):
            MemoryConfig(rerank_threshold=invalid_prob)
        with pytest.raises(ValidationError):
            MemoryConfig(jargon_min_confidence=invalid_prob)

    with pytest.raises(ValidationError):
        TraceConfig(unknown_field="x")  # type: ignore[call-arg]

    with pytest.raises(ValidationError):
        WebSearchConfig(unknown_field="x")  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        WebSearchConfig(provider="unknown")  # type: ignore[arg-type]
    for invalid_timeout in (0.0, -5.0, float("inf"), float("nan")):
        with pytest.raises(ValidationError):
            WebSearchConfig(timeout_sec=invalid_timeout)
    with pytest.raises(ValidationError):
        WebUIConfig(unknown_field="x")  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        WebUIConfig(port=0)
    with pytest.raises(ValidationError):
        WebUIConfig(port=65536)


def test_jev_is_configured_and_no_cross_model_inheritance() -> None:
    cfg = AppConfig(
        llm={"api_key": "sk-llm-only", "base_url": "https://llm.example/v1"},
        vision={"api_key": "sk-vision-only"},
    )
    # Jev MUST NOT inherit api_key or base_url from llm or vision.
    assert cfg.jev.api_key == ""
    assert cfg.jev.is_configured is False

    assert JevConfig(api_key="sk-jev").is_configured is True
    assert JevConfig(enabled=False, api_key="sk-jev").is_configured is False
    assert JevConfig(api_key="   ").is_configured is False
    assert JevConfig(api_key="sk-jev", base_url="   ").is_configured is False
    assert JevConfig(api_key="sk-jev", model="   ").is_configured is False

    # Embedding MUST NOT inherit api_key/base_url/model from llm or vision,
    # and keyless local endpoints are valid once base_url and model are set.
    assert cfg.embedding.api_key == ""
    assert cfg.embedding.model == ""
    assert cfg.embedding.is_configured is False
    assert EmbeddingConfig(model="nomic-embed-text").is_configured is True
    assert (
        EmbeddingConfig(base_url="http://localhost:11434", model="bge-m3").is_configured
        is True
    )
    assert EmbeddingConfig(base_url="   ", model="bge-m3").is_configured is False
    assert EmbeddingConfig(model="   ").is_configured is False


def test_model_json_schema_has_webui_metadata() -> None:
    cjk_re = re.compile(r"[\u4e00-\u9fff]")
    schema = AppConfig.model_json_schema()
    assert "properties" in schema
    assert cjk_re.search(schema.get("title", "")), "root title must be localized"
    assert cjk_re.search(schema.get("description", "")), (
        "root description must be localized"
    )
    for section in (
        "db",
        "access",
        "napcat",
        "vision",
        "llm",
        "embedding",
        "jev",
        "gate",
        "session",
        "memory",
        "trace",
        "web_search",
        "webui",
    ):
        assert section in schema["properties"]
        sec_meta = schema["properties"][section]
        assert sec_meta.get("title"), f"{section} missing title"
        assert sec_meta.get("description"), f"{section} missing description"
        assert cjk_re.search(sec_meta["title"]), f"{section} title must be Chinese"
        assert cjk_re.search(sec_meta["description"]), (
            f"{section} description must be Chinese"
        )

    defs = schema["$defs"]
    for model_name in (
        "DatabaseConfig",
        "AccessConfig",
        "SegmentedReplyConfig",
        "NapCatConfig",
        "VisionConfig",
        "LLMConfig",
        "EmbeddingConfig",
        "JevConfig",
        "JevPromptsConfig",
        "ActiveReplyConfig",
        "GateConfig",
        "SessionConfig",
        "MemoryConfig",
        "MemoryPromptsConfig",
        "TraceConfig",
        "WebSearchConfig",
        "WebUIConfig",
    ):
        model_def = defs[model_name]
        assert cjk_re.search(model_def.get("title", "")), (
            f"{model_name} title must be Chinese"
        )
        assert cjk_re.search(model_def.get("description", "")), (
            f"{model_name} description must be Chinese"
        )
        props = model_def["properties"]
        assert props, f"{model_name} must define properties"
        for field_name, meta in props.items():
            assert meta.get("title"), f"{model_name}.{field_name} missing title"
            assert meta.get("description"), (
                f"{model_name}.{field_name} missing description"
            )
            assert cjk_re.search(meta["title"]), (
                f"{model_name}.{field_name} title must be Chinese"
            )
            assert cjk_re.search(meta["description"]), (
                f"{model_name}.{field_name} description must be Chinese"
            )

    # Verify prompt widget and collapsible schema metadata.
    vision_props = defs["VisionConfig"]["properties"]
    assert vision_props["prompt"]["x_widget"] == "textarea"
    assert vision_props["prompt"]["x_rows"] == 3
    assert vision_props["montage_prompt_note"]["x_widget"] == "textarea"
    assert vision_props["montage_prompt_note"]["x_rows"] == 3
    assert vision_props["montage_prompt_note"]["x_collapsed"] is True

    assert defs["JevConfig"]["properties"]["prompts"]["x_collapsed"] is True
    for field_name, meta in defs["JevPromptsConfig"]["properties"].items():
        assert meta["x_widget"] == "textarea", f"JevPromptsConfig.{field_name}"
        assert meta["x_rows"] == 3, f"JevPromptsConfig.{field_name}"

    assert defs["MemoryConfig"]["properties"]["prompts"]["x_collapsed"] is True
    for field_name, meta in defs["MemoryPromptsConfig"]["properties"].items():
        assert meta["x_widget"] == "textarea", f"MemoryPromptsConfig.{field_name}"
        assert meta["x_rows"] == 10, f"MemoryPromptsConfig.{field_name}"


def test_example_config_matches_defaults(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    example_path = repo_root / "config.example.json"
    assert example_path.is_file()
    assert load_config(example_path) == AppConfig()

    # Verify byte-for-byte match with save_config(AppConfig()).
    regenerated = tmp_path / "config.example.json"
    save_config(AppConfig(), regenerated)
    assert example_path.read_bytes() == regenerated.read_bytes()


def test_access_matrix_self_and_ignored_unconditional() -> None:
    access = AccessConfig(
        admin_users=["10000", "2854196310"],
        ignored_users=["2854196310"],
        group_mode="whitelist",
        group_whitelist=["500"],
        private_mode="whitelist",
        private_whitelist=["10000", "2854196310"],
    )
    # Self messages are always blocked, even for admins.
    assert access.is_allowed(group_id="500", user_id="10000", self_id="10000") is False
    assert access.is_allowed(group_id=None, user_id="10000", self_id="10000") is False
    # Ignored users are always blocked, even if also listed as admins or whitelisted.
    assert (
        access.is_allowed(group_id="500", user_id="2854196310", self_id="99999")
        is False
    )
    assert (
        access.is_allowed(group_id=None, user_id="2854196310", self_id="99999") is False
    )


def test_access_matrix_group_blacklist_and_whitelist() -> None:
    bl = AccessConfig(
        admin_users=["1"],
        group_mode="blacklist",
        group_blacklist=["900"],
        private_blacklist=["1", "2"],
    )
    # Blacklisted group is rejected even for admin.
    assert bl.is_allowed(group_id="900", user_id="1", self_id="999") is False
    assert bl.is_allowed(group_id="900", user_id="3", self_id="999") is False
    # Allowed group: normal non-blacklisted user passes.
    assert bl.is_allowed(group_id="800", user_id="3", self_id="999") is True
    # Allowed group: private-blacklisted non-admin is blocked.
    assert bl.is_allowed(group_id="800", user_id="2", self_id="999") is False
    # Allowed group: admin is exempt from private_blacklist.
    assert bl.is_allowed(group_id="800", user_id="1", self_id="999") is True

    wl = AccessConfig(
        admin_users=["1"],
        group_mode="whitelist",
        group_whitelist=["800"],
        private_blacklist=["1", "2"],
    )
    assert wl.is_allowed(group_id="800", user_id="3", self_id="999") is True
    # Private-blacklisted non-admin is still blocked inside a whitelisted group.
    assert wl.is_allowed(group_id="800", user_id="2", self_id="999") is False
    # Admin is exempt from private_blacklist inside a whitelisted group.
    assert wl.is_allowed(group_id="800", user_id="1", self_id="999") is True
    # Non-whitelisted group is rejected even for admin.
    assert wl.is_allowed(group_id="900", user_id="1", self_id="999") is False
    assert wl.is_allowed(group_id="900", user_id="3", self_id="999") is False


def test_access_matrix_private_blacklist_and_whitelist() -> None:
    bl = AccessConfig(
        admin_users=["1"],
        private_mode="blacklist",
        private_blacklist=["1", "2"],
    )
    assert bl.is_allowed(group_id=None, user_id="1", self_id="999") is True
    assert bl.is_allowed(group_id=None, user_id="2", self_id="999") is False
    assert bl.is_allowed(group_id=None, user_id="3", self_id="999") is True

    wl = AccessConfig(
        admin_users=["1"],
        private_mode="whitelist",
        private_whitelist=["3"],
    )
    assert wl.is_allowed(group_id=None, user_id="1", self_id="999") is True
    assert wl.is_allowed(group_id=None, user_id="3", self_id="999") is True
    assert wl.is_allowed(group_id=None, user_id="4", self_id="999") is False
