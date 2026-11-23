"""Unit tests for spectra.config."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from spectra.config import AccessConfig, AppConfig, load_config, save_config


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
    assert cfg.napcat.enable_segmented_reply is True
    assert cfg.napcat.segmented_words_threshold == 150
    assert cfg.napcat.forward_threshold == 1500
    assert cfg.vision.enabled is True
    assert cfg.vision.max_input_bytes == 20 * 1024 * 1024
    assert cfg.vision.max_image_edge == 1280


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
            "enable_segmented_reply": False,
        },
        vision={
            "enabled": True,
            "base_url": "https://example.invalid/v1",
            "api_key": "sk-test",
            "model": "vision-x",
        },
    )

    save_config(original, config_path)
    assert config_path.is_file()
    loaded = load_config(config_path)
    assert loaded == original


def test_extra_fields_rejected(tmp_path: Path) -> None:
    bad_path = tmp_path / "bad.json"
    bad_path.write_text('{"unknown_section": {}}', encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(bad_path)


def test_model_json_schema_has_webui_metadata() -> None:
    schema = AppConfig.model_json_schema()
    assert "properties" in schema
    for section in ("db", "access", "napcat", "vision"):
        assert section in schema["properties"]
        assert "title" in schema["properties"][section]
        assert "description" in schema["properties"][section]

    defs = schema["$defs"]
    for model_name in (
        "DatabaseConfig",
        "AccessConfig",
        "NapCatConfig",
        "VisionConfig",
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
