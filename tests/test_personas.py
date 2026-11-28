"""Unit tests for spectra.personas (PersonaStore, bindings, and resolution)."""

from __future__ import annotations

from pathlib import Path

import pytest

from spectra.db import Database
from spectra.personas import (
    SYSTEM_DEFAULT_PERSONA_ID,
    PersonaExistsError,
    PersonaNotFoundError,
    PersonaStore,
    validate_begin_dialogs,
    validate_binding_target,
    validate_persona_id,
)
from spectra.prompts import _DEFAULT_SYSTEM_PROMPT


async def test_init_seeds_default_from_markdown_and_is_idempotent(
    tmp_path: Path,
) -> None:
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "default.md").write_text(
        "  来自 default.md 的默认人设  \n", encoding="utf-8"
    )

    async with Database(":memory:") as db:
        store = PersonaStore(db, prompts_dir=prompts_dir)
        await store.init()

        default_rec = await store.get_persona(SYSTEM_DEFAULT_PERSONA_ID)
        assert default_rec is not None
        assert default_rec.persona_id == "default"
        assert default_rec.system_prompt == "来自 default.md 的默认人设"
        assert default_rec.begin_dialogs == ()
        assert default_rec.custom_error_message == ""

        # Editing default then re-running init() must preserve the DB edit.
        await store.update_persona(
            "default",
            system_prompt="数据库里编辑后的默认人设",
            begin_dialogs=["你好", "你好呀！"],
            custom_error_message="稍后再试哦",
            now=200,
        )
        await store.init()
        after_reinit = await store.get_persona("default")
        assert after_reinit is not None
        assert after_reinit.system_prompt == "数据库里编辑后的默认人设"
        assert after_reinit.begin_dialogs == ("你好", "你好呀！")
        assert after_reinit.custom_error_message == "稍后再试哦"


async def test_init_falls_back_to_builtin_prompt_when_default_md_missing_or_blank(
    tmp_path: Path,
) -> None:
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "default.md").write_text("   \n\t  ", encoding="utf-8")

    async with Database(":memory:") as db:
        store = PersonaStore(db, prompts_dir=prompts_dir)
        await store.init()
        default_rec = await store.get_persona("default")
        assert default_rec is not None
        assert default_rec.system_prompt == _DEFAULT_SYSTEM_PROMPT


async def test_persona_crud_and_undeletable_default(tmp_path: Path) -> None:
    async with Database(":memory:") as db:
        store = PersonaStore(db, prompts_dir=tmp_path / "prompts")
        await store.init()

        created = await store.create_persona(
            persona_id="  coder  ",
            system_prompt="  你是严谨的代码助手。  ",
            begin_dialogs=["写个 hello", "print('hello')"],
            custom_error_message="  代码服务开小差了  ",
            now=1000,
        )
        assert created.persona_id == "coder"
        assert created.system_prompt == "你是严谨的代码助手。"
        assert created.begin_dialogs == ("写个 hello", "print('hello')")
        assert created.custom_error_message == "代码服务开小差了"
        assert created.created_at == 1000
        assert created.updated_at == 1000

        # Duplicate creation raises PersonaExistsError.
        with pytest.raises(PersonaExistsError):
            await store.create_persona(
                persona_id="coder",
                system_prompt="重复创建",
            )

        await store.create_persona(
            persona_id="alpha",
            system_prompt="Alpha 人设",
            now=1001,
        )

        # Listing always puts "default" first, then sorted alphabetically.
        listed = await store.list_personas()
        assert [p.persona_id for p in listed] == ["default", "alpha", "coder"]

        # Update existing persona.
        updated = await store.update_persona(
            "coder",
            system_prompt="更新后的代码助手",
            begin_dialogs=["问题1", "回答1", "问题2", "回答2"],
            custom_error_message="",
            now=2000,
        )
        assert updated.system_prompt == "更新后的代码助手"
        assert updated.begin_dialogs == ("问题1", "回答1", "问题2", "回答2")
        assert updated.custom_error_message == ""
        assert updated.created_at == 1000
        assert updated.updated_at == 2000

        # Update missing persona raises PersonaNotFoundError.
        with pytest.raises(PersonaNotFoundError):
            await store.update_persona("missing", system_prompt="不存在")

        # Default persona CANNOT be deleted.
        with pytest.raises(ValueError, match="default"):
            await store.delete_persona("default")
        with pytest.raises(ValueError, match="default"):
            await store.delete_persona("  default  ")

        # Deleting non-default returns True once, then False.
        assert await store.delete_persona("alpha") is True
        assert await store.delete_persona("alpha") is False
        assert await store.get_persona("alpha") is None


def test_validation_rejects_unsafe_ids_and_invalid_begin_dialogs() -> None:
    # persona_id validation
    assert validate_persona_id("  valid_id-01  ") == "valid_id-01"
    for bad_id in (
        "",
        "   ",
        "a" * 65,
        "a/b",
        "a\\b",
        "..",
        ".",
        "foo..bar",
        "bad\x00id",
        "bad\nid",
        "bad\tid",
        "bad\x7fid",
        123,
    ):
        with pytest.raises(ValueError):
            validate_persona_id(bad_id)

    # begin_dialogs validation
    assert validate_begin_dialogs([]) == ()
    assert validate_begin_dialogs(None) == ()
    assert validate_begin_dialogs([" u1 ", " a1 "]) == ("u1", "a1")
    for bad_dialogs in (
        ["only_user"],
        ["u1", "a1", "u2"],
        ["u1", "   "],
        ["", "a1"],
        "not-a-list",
        [1, "a1"],
    ):
        with pytest.raises(ValueError):
            validate_begin_dialogs(bad_dialogs)  # type: ignore[arg-type]

    # binding target_id validation
    assert validate_binding_target("10001") == "10001"
    for bad_target in (
        "",
        "   ",
        " 10001",
        "10001 ",
        "100 01",
        "100/01",
        "100\\01",
        "..",
        "100..01",
        "100\x0001",
        "100\n01",
        12345,
    ):
        with pytest.raises(ValueError):
            validate_binding_target(bad_target)


async def test_bindings_crud_and_cascade_delete(tmp_path: Path) -> None:
    async with Database(":memory:") as db:
        store = PersonaStore(db, prompts_dir=tmp_path / "prompts")
        await store.init()
        await store.create_persona(persona_id="catgirl", system_prompt="喵喵人设")
        await store.create_persona(persona_id="coder", system_prompt="程序员人设")

        # Non-existent persona raises PersonaNotFoundError.
        with pytest.raises(PersonaNotFoundError):
            await store.set_binding("group", "50001", "nonexistent")

        # Invalid scope or target raises ValueError.
        with pytest.raises(ValueError):
            await store.set_binding("channel", "50001", "catgirl")  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            await store.set_binding("group", "bad/id", "catgirl")

        b1 = await store.set_binding("group", "50001", "catgirl", now=100)
        assert b1.session_key == "group:50001"
        assert b1.scope == "group"
        assert b1.target_id == "50001"
        assert b1.persona_id == "catgirl"

        b2 = await store.set_binding("private", "20002", "catgirl", now=101)
        assert b2.session_key == "private:20002"

        # Upsert existing binding to another persona.
        b1_updated = await store.set_binding("group", "50001", "coder", now=102)
        assert b1_updated.persona_id == "coder"
        assert b1_updated.updated_at == 102

        bindings = await store.list_bindings()
        assert [(b.session_key, b.persona_id) for b in bindings] == [
            ("group:50001", "coder"),
            ("private:20002", "catgirl"),
        ]

        # Deleting "catgirl" cascades and removes private:20002, keeping group:50001.
        assert await store.delete_persona("catgirl") is True
        remaining = await store.list_bindings()
        assert [(b.session_key, b.persona_id) for b in remaining] == [
            ("group:50001", "coder")
        ]

        # Explicit delete_binding returns True then False.
        assert await store.delete_binding("group", "50001") is True
        assert await store.delete_binding("group", "50001") is False


async def test_resolve_effective_persona_four_level_priority(tmp_path: Path) -> None:
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "default.md").write_text("默认MD人设", encoding="utf-8")
    (prompts_dir / "group_50001.md").write_text("群50001文件覆写", encoding="utf-8")
    (prompts_dir / "private_20002.md").write_text("私聊20002文件覆写", encoding="utf-8")

    async with Database(":memory:") as db:
        store = PersonaStore(db, prompts_dir=prompts_dir)
        await store.init()

        await store.update_persona(
            "default",
            system_prompt="DB默认人设",
            begin_dialogs=["默认问", "默认答"],
            custom_error_message="默认报错兜底",
        )
        await store.create_persona(
            persona_id="global_alt",
            system_prompt="全局配置默认人设",
            begin_dialogs=["全局问", "全局答"],
            custom_error_message="全局报错兜底",
        )
        await store.create_persona(
            persona_id="vip_group",
            system_prompt="DB群绑定人设",
            begin_dialogs=["群问", "群答"],
            custom_error_message="群报错兜底",
        )
        await store.create_persona(
            persona_id="vip_dm",
            system_prompt="DB私聊绑定人设",
            begin_dialogs=["私问", "私答"],
            custom_error_message="私聊报错兜底",
        )

        # Level 4: Unbound session without override file and missing default_persona_id
        # falls back to DB "default" row.
        p4 = await store.resolve_effective_persona(
            group_id="99999",
            user_id="10001",
            default_persona_id="does_not_exist",
        )
        assert p4.persona_id == "default"
        assert p4.system_prompt == "DB默认人设"
        assert p4.begin_dialogs == ("默认问", "默认答")
        assert p4.custom_error_message == "默认报错兜底"

        # Level 3: Configured default_persona_id wins over "default" row.
        p3 = await store.resolve_effective_persona(
            group_id="99999",
            user_id="10001",
            default_persona_id="global_alt",
        )
        assert p3.persona_id == "global_alt"
        assert p3.system_prompt == "全局配置默认人设"
        assert p3.begin_dialogs == ("全局问", "全局答")
        assert p3.custom_error_message == "全局报错兜底"

        # Level 2: Local file override wins over configured default_persona_id.
        p2_group = await store.resolve_effective_persona(
            group_id="50001",
            user_id="10001",
            default_persona_id="global_alt",
        )
        assert p2_group.persona_id == "file:group_50001"
        assert p2_group.system_prompt == "群50001文件覆写"
        assert p2_group.begin_dialogs == ()
        assert p2_group.custom_error_message == ""

        p2_private = await store.resolve_effective_persona(
            group_id=None,
            user_id="20002",
            default_persona_id="global_alt",
        )
        assert p2_private.persona_id == "file:private_20002"
        assert p2_private.system_prompt == "私聊20002文件覆写"
        assert p2_private.begin_dialogs == ()
        assert p2_private.custom_error_message == ""

        # Level 1: Explicit DB binding wins over local file override!
        await store.set_binding("group", "50001", "vip_group")
        await store.set_binding("private", "20002", "vip_dm")

        p1_group = await store.resolve_effective_persona(
            group_id="50001",
            user_id="10001",
            default_persona_id="global_alt",
        )
        assert p1_group.persona_id == "vip_group"
        assert p1_group.system_prompt == "DB群绑定人设"
        assert p1_group.begin_dialogs == ("群问", "群答")
        assert p1_group.custom_error_message == "群报错兜底"

        p1_private = await store.resolve_effective_persona(
            group_id=None,
            user_id="20002",
            default_persona_id="global_alt",
        )
        assert p1_private.persona_id == "vip_dm"
        assert p1_private.system_prompt == "DB私聊绑定人设"
        assert p1_private.begin_dialogs == ("私问", "私答")
        assert p1_private.custom_error_message == "私聊报错兜底"

        # Deleting DB binding immediately falls back to local file override.
        await store.delete_binding("group", "50001")
        p_fallback_file = await store.resolve_effective_persona(
            group_id="50001",
            user_id="10001",
            default_persona_id="global_alt",
        )
        assert p_fallback_file.persona_id == "file:group_50001"
        assert p_fallback_file.system_prompt == "群50001文件覆写"
