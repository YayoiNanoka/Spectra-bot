"""Unit tests for spectra.prompts."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from spectra.prompts import _DEFAULT_SYSTEM_PROMPT, PromptLoader


def test_lookup_priority_and_fallbacks(tmp_path: Path) -> None:
    prompts_dir = tmp_path / "prompts"
    loader = PromptLoader(prompts_dir)

    # Missing directory -> built-in fallback.
    assert loader.load(group_id="100", user_id="200") == _DEFAULT_SYSTEM_PROMPT
    assert loader.load(group_id=None, user_id="200") == _DEFAULT_SYSTEM_PROMPT

    prompts_dir.mkdir()
    (prompts_dir / "default.md").write_text("  默认人设  \n", encoding="utf-8")
    assert loader.load(group_id="100", user_id="200") == "默认人设"
    assert loader.load(group_id=None, user_id="200") == "默认人设"

    (prompts_dir / "group_100.md").write_text("群100专属人设", encoding="utf-8")
    (prompts_dir / "private_200.md").write_text("私聊200专属人设", encoding="utf-8")
    assert loader.load(group_id="100", user_id="200") == "群100专属人设"
    assert loader.load(group_id="999", user_id="200") == "默认人设"
    assert loader.load(group_id=None, user_id="200") == "私聊200专属人设"
    assert loader.load(group_id=None, user_id="300") == "默认人设"


def test_empty_files_fall_through_to_next_tier(tmp_path: Path) -> None:
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "group_100.md").write_text("   \n\t  ", encoding="utf-8")
    (prompts_dir / "default.md").write_text("有效默认人设", encoding="utf-8")

    loader = PromptLoader(prompts_dir)
    assert loader.load(group_id="100", user_id="200") == "有效默认人设"

    (prompts_dir / "default.md").write_text("   \n", encoding="utf-8")
    assert loader.load(group_id="100", user_id="200") == _DEFAULT_SYSTEM_PROMPT


def test_mtime_cache_hits_and_hot_reload_on_create_edit_delete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    default_file = prompts_dir / "default.md"
    default_file.write_text("第一版默认", encoding="utf-8")

    read_calls: list[Path] = []
    real_read_text = Path.read_text

    def counting_read_text(self: Path, *args: object, **kwargs: object) -> str:
        read_calls.append(self)
        return real_read_text(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "read_text", counting_read_text)

    loader = PromptLoader(prompts_dir)
    assert loader.load(group_id="100", user_id="200") == "第一版默认"
    assert len(read_calls) == 1

    # Repeated loads with unchanged stat must hit cache without disk reads.
    assert loader.load(group_id="100", user_id="200") == "第一版默认"
    assert loader.load(group_id="100", user_id="200") == "第一版默认"
    assert len(read_calls) == 1

    # Creating a per-group override takes effect immediately.
    group_file = prompts_dir / "group_100.md"
    group_file.write_text("群聊V1", encoding="utf-8")
    assert loader.load(group_id="100", user_id="200") == "群聊V1"
    assert len(read_calls) == 2

    # Same-size edit with bumped mtime_ns invalidates cache immediately.
    old_stat = group_file.stat()
    group_file.write_text("群聊V2", encoding="utf-8")
    os.utime(
        group_file,
        ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns + 1_000_000),
    )
    assert loader.load(group_id="100", user_id="200") == "群聊V2"
    assert len(read_calls) == 3

    # Deleting the override immediately falls back to default.md (still cached).
    group_file.unlink()
    assert loader.load(group_id="100", user_id="200") == "第一版默认"
    assert len(read_calls) == 3


def test_path_traversal_and_symlink_escape_rejected(tmp_path: Path) -> None:
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "default.md").write_text("安全默认", encoding="utf-8")

    secret = tmp_path / "secret.md"
    secret.write_text("绝密内容", encoding="utf-8")

    loader = PromptLoader(prompts_dir)
    for malicious_id in (
        "../secret",
        "..\\secret",
        "foo/bar",
        "foo\\bar",
        "123\x00evil",
        "..",
    ):
        assert loader.load(group_id=malicious_id, user_id="100") == "安全默认"
        assert loader.load(group_id=None, user_id=malicious_id) == "安全默认"

    symlink_override = prompts_dir / "group_777.md"
    symlink_override.symlink_to(secret)
    assert loader.load(group_id="777", user_id="100") == "安全默认"


def test_repo_default_prompt_file_loads() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    loader = PromptLoader(repo_root / "prompts")
    loaded = loader.load(group_id="10000", user_id="20000")
    assert "Spectra" in loaded
    assert "[Spectra]:" in loaded
    assert "[昵称(QQ号) HH:MM]:" in loaded
