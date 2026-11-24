"""Unit tests for spectra.config."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from spectra.config import (
    AccessConfig,
    ActiveReplyConfig,
    AppConfig,
    GateConfig,
    JevConfig,
    LLMConfig,
    SegmentedReplyConfig,
    SessionConfig,
    TraceConfig,
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
    assert cfg.vision.max_size == 1280
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
    assert cfg.jev.enabled is True
    assert cfg.jev.base_url == "https://opencode.ai/inference/go/systemone/v1/systemone"
    assert cfg.jev.api_key == ""
    assert cfg.jev.model == "jev-1.14"
    assert cfg.jev.timeout_sec == 1.5
    assert cfg.jev.reply_threshold == 0.65
    assert cfg.jev.high_effort_threshold == 0.60
    assert cfg.jev.is_configured is False
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
    assert cfg.session.window_max_messages == 60
    assert cfg.session.window_keep_messages == 30
    assert cfg.session.timezone == "Asia/Shanghai"
    assert cfg.trace.enabled is True
    assert cfg.trace.path == Path("data/traces.jsonl")


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
            "max_size": 1024,
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
        },
        jev={
            "enabled": True,
            "base_url": "https://openrouter.ai/api/alpha/decisions",
            "api_key": "sk-jev",
            "model": "jev-1.14",
            "timeout_sec": 1.2,
            "reply_threshold": 0.7,
            "high_effort_threshold": 0.55,
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
            "window_max_messages": 80,
            "window_keep_messages": 40,
            "timezone": "UTC",
        },
        trace={
            "enabled": False,
            "path": Path("logs/custom_traces.jsonl"),
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
            LLMConfig(timeout_sec=invalid_timeout)
    with pytest.raises(ValidationError):
        LLMConfig(max_retries=0)
    with pytest.raises(ValidationError):
        LLMConfig(max_tool_steps=0)

    with pytest.raises(ValidationError):
        JevConfig(unknown_field="x")  # type: ignore[call-arg]
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
        TraceConfig(unknown_field="x")  # type: ignore[call-arg]


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


def test_model_json_schema_has_webui_metadata() -> None:
    schema = AppConfig.model_json_schema()
    assert "properties" in schema
    for section in (
        "db",
        "access",
        "napcat",
        "vision",
        "llm",
        "jev",
        "gate",
        "session",
        "trace",
    ):
        assert section in schema["properties"]
        assert "title" in schema["properties"][section]
        assert "description" in schema["properties"][section]

    defs = schema["$defs"]
    for model_name in (
        "DatabaseConfig",
        "AccessConfig",
        "SegmentedReplyConfig",
        "NapCatConfig",
        "VisionConfig",
        "LLMConfig",
        "JevConfig",
        "ActiveReplyConfig",
        "GateConfig",
        "SessionConfig",
        "TraceConfig",
    ):
        props = defs[model_name]["properties"]
        assert props, f"{model_name} must define properties"
        for field_name, meta in props.items():
            assert meta.get("title"), f"{model_name}.{field_name} missing title"
            assert meta.get("description"), (
                f"{model_name}.{field_name} missing description"
            )


def test_example_config_matches_defaults() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    example_path = repo_root / "config.example.json"
    assert example_path.is_file()
    assert load_config(example_path) == AppConfig()


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
