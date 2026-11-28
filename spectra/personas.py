"""SQLite-backed AstrBot-aligned persona presets and per-session bindings.

Owns the ``personas`` SQLite migration module with two tables:

* ``personas``: reusable persona definitions storing ``system_prompt``,
  optional few-shot ``begin_dialogs`` pairs, and optional
  ``custom_error_message`` fallback text.
* ``persona_bindings``: per-session (``group:{id}`` or ``private:{id}``)
  overrides pointing to a ``personas.persona_id`` with ``ON DELETE CASCADE``.

Effective persona resolution keeps full backward compatibility with local
``prompts/group_{gid}.md`` and ``prompts/private_{uid}.md`` overrides while
giving explicit WebUI/DB bindings top priority.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import aiosqlite

from spectra.db import Database
from spectra.prompts import _DEFAULT_SYSTEM_PROMPT, PromptLoader

SYSTEM_DEFAULT_PERSONA_ID = "default"
MAX_PERSONA_ID_LENGTH = 64

BindingScope = Literal["group", "private"]

_MIGRATIONS: tuple[str, ...] = (
    """
    CREATE TABLE personas (
        persona_id TEXT PRIMARY KEY CHECK (length(trim(persona_id)) > 0),
        system_prompt TEXT NOT NULL CHECK (length(trim(system_prompt)) > 0),
        begin_dialogs TEXT NOT NULL DEFAULT '[]'
            CHECK (json_valid(begin_dialogs) AND json_type(begin_dialogs) = 'array'),
        custom_error_message TEXT NOT NULL DEFAULT '',
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL
    );
    CREATE TABLE persona_bindings (
        session_key TEXT PRIMARY KEY
            CHECK (session_key GLOB 'group:*' OR session_key GLOB 'private:*'),
        persona_id TEXT NOT NULL REFERENCES personas(persona_id) ON DELETE CASCADE,
        updated_at INTEGER NOT NULL
    );
    CREATE INDEX idx_persona_bindings_persona_id ON persona_bindings (persona_id);
    """,
)

_PERSONA_COLUMNS = (
    "persona_id, system_prompt, begin_dialogs, custom_error_message, "
    "created_at, updated_at"
)


class PersonaExistsError(ValueError):
    """Raised when creating a persona whose ``persona_id`` already exists."""


class PersonaNotFoundError(LookupError):
    """Raised when an operation references a non-existent ``persona_id``."""


@dataclass(slots=True, frozen=True)
class PersonaRecord:
    """Persisted row from the ``personas`` table."""

    persona_id: str
    system_prompt: str
    begin_dialogs: tuple[str, ...]
    custom_error_message: str
    created_at: int
    updated_at: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize into a JSON-compatible dictionary."""
        return {
            "persona_id": self.persona_id,
            "system_prompt": self.system_prompt,
            "begin_dialogs": list(self.begin_dialogs),
            "custom_error_message": self.custom_error_message,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(slots=True, frozen=True)
class PersonaBinding:
    """Persisted row from the ``persona_bindings`` table."""

    session_key: str
    scope: BindingScope
    target_id: str
    persona_id: str
    updated_at: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize into a JSON-compatible dictionary."""
        return {
            "session_key": self.session_key,
            "scope": self.scope,
            "target_id": self.target_id,
            "persona_id": self.persona_id,
            "updated_at": self.updated_at,
        }


@dataclass(slots=True, frozen=True)
class EffectivePersona:
    """Resolved runtime persona applied to a single conversation turn."""

    persona_id: str
    system_prompt: str
    begin_dialogs: tuple[str, ...] = ()
    custom_error_message: str = ""


def _has_control_chars(text: str) -> bool:
    return any(ord(ch) < 32 or ord(ch) == 127 for ch in text)


def _has_unsafe_path_tokens(text: str) -> bool:
    if text in (".", "..") or ".." in text or "\x00" in text:
        return True
    separators = {"/", "\\"}
    if os.sep:
        separators.add(os.sep)
    if os.altsep:
        separators.add(os.altsep)
    return any(sep in text for sep in separators)


def validate_persona_id(raw_id: Any) -> str:
    """Validate and normalize a ``persona_id``.

    Args:
        raw_id: Candidate persona identifier string.

    Returns:
        Trimmed, validated persona identifier.

    Raises:
        ValueError: If ``raw_id`` is not a string, empty, longer than 64 chars,
            or contains path separators, ``..``, NUL, or control characters.
    """
    if not isinstance(raw_id, str):
        raise ValueError("人格 ID 必须是字符串。")
    pid = raw_id.strip()
    if not pid:
        raise ValueError("人格 ID 不能为空。")
    if len(pid) > MAX_PERSONA_ID_LENGTH:
        raise ValueError(f"人格 ID 长度不能超过 {MAX_PERSONA_ID_LENGTH} 个字符。")
    if _has_control_chars(pid) or _has_unsafe_path_tokens(pid):
        raise ValueError("人格 ID 不能包含路径分隔符（/、\\）、.. 或控制字符。")
    return pid


def validate_system_prompt(raw_prompt: Any) -> str:
    """Validate and normalize a persona ``system_prompt``."""
    if not isinstance(raw_prompt, str):
        raise ValueError("系统提示词必须是字符串。")
    prompt = raw_prompt.strip()
    if not prompt:
        raise ValueError("系统提示词不能为空。")
    return prompt


def validate_begin_dialogs(raw_dialogs: Sequence[Any] | None) -> tuple[str, ...]:
    """Validate and normalize ``begin_dialogs`` into paired non-empty strings.

    Args:
        raw_dialogs: Optional sequence of strings ordered as
            ``[user_0, assistant_0, user_1, assistant_1, ...]``.

    Returns:
        Immutable tuple of stripped, non-empty strings of even length.

    Raises:
        ValueError: If ``raw_dialogs`` is not a sequence of strings, has odd
            length, or contains any blank/whitespace-only message.
    """
    if raw_dialogs is None:
        return ()
    if isinstance(raw_dialogs, (str, bytes, bytearray)) or not isinstance(
        raw_dialogs, Sequence
    ):
        raise ValueError("预设示范对话（begin_dialogs）必须是字符串列表。")
    items: list[str] = []
    for idx, entry in enumerate(raw_dialogs):
        if not isinstance(entry, str):
            raise ValueError(f"预设示范对话第 {idx + 1} 条必须是字符串。")
        cleaned = entry.strip()
        if not cleaned:
            raise ValueError("预设示范对话中的每条用户与助手消息都不能为空。")
        items.append(cleaned)
    if len(items) % 2 != 0:
        raise ValueError(
            "预设示范对话必须成对出现（长度必须为偶数，按用户/助手交替）。"
        )
    return tuple(items)


def validate_custom_error_message(raw_message: Any) -> str:
    """Validate and normalize optional ``custom_error_message`` text."""
    if raw_message is None:
        return ""
    if not isinstance(raw_message, str):
        raise ValueError("报错兜底回复必须是字符串。")
    return raw_message.strip()


def validate_binding_scope(scope: Any) -> BindingScope:
    """Validate a session binding scope (``"group"`` or ``"private"``)."""
    if scope not in ("group", "private"):
        raise ValueError("会话类型（scope）必须为 group 或 private。")
    return scope


def validate_binding_target(target_id: Any) -> str:
    """Validate a group or user target identifier for session bindings."""
    if not isinstance(target_id, str):
        raise ValueError("目标会话 ID 必须是字符串。")
    if not target_id or any(ch.isspace() for ch in target_id):
        raise ValueError("目标会话 ID 不能为空且不能包含空白字符。")
    if _has_control_chars(target_id) or _has_unsafe_path_tokens(target_id):
        raise ValueError("目标会话 ID 不能包含路径分隔符（/、\\）、.. 或控制字符。")
    return target_id


def make_binding_session_key(scope: BindingScope, target_id: str) -> str:
    """Build canonical ``group:{id}`` or ``private:{id}`` session key."""
    valid_scope = validate_binding_scope(scope)
    valid_target = validate_binding_target(target_id)
    return f"{valid_scope}:{valid_target}"


def _parse_begin_dialogs_json(raw: Any) -> tuple[str, ...]:
    if not isinstance(raw, str):
        return ()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return ()
    if not isinstance(parsed, list):
        return ()
    try:
        return validate_begin_dialogs(parsed)
    except ValueError:
        return ()


def _row_to_persona(row: aiosqlite.Row) -> PersonaRecord:
    return PersonaRecord(
        persona_id=str(row["persona_id"]),
        system_prompt=str(row["system_prompt"]),
        begin_dialogs=_parse_begin_dialogs_json(row["begin_dialogs"]),
        custom_error_message=str(row["custom_error_message"]),
        created_at=int(row["created_at"]),
        updated_at=int(row["updated_at"]),
    )


def _row_to_binding(row: aiosqlite.Row) -> PersonaBinding:
    session_key = str(row["session_key"])
    scope_raw, _, target_id = session_key.partition(":")
    scope: BindingScope = "group" if scope_raw == "group" else "private"
    return PersonaBinding(
        session_key=session_key,
        scope=scope,
        target_id=target_id,
        persona_id=str(row["persona_id"]),
        updated_at=int(row["updated_at"]),
    )


class PersonaStore:
    """Async SQLite store for persona presets, bindings, and runtime resolution.

    Args:
        db: Connected :class:`spectra.db.Database` instance.
        prompts_dir: Directory holding optional ``default.md``,
            ``group_{group_id}.md``, and ``private_{user_id}.md`` overrides.
        prompt_loader: Optional pre-built :class:`spectra.prompts.PromptLoader`
            to share stat-cache state.
    """

    def __init__(
        self,
        db: Database,
        *,
        prompts_dir: Path | str = Path("prompts"),
        prompt_loader: PromptLoader | None = None,
    ) -> None:
        self._db = db
        self._prompts_dir = Path(prompts_dir)
        self._prompt_loader = (
            prompt_loader
            if prompt_loader is not None
            else PromptLoader(self._prompts_dir)
        )

    @property
    def prompts_dir(self) -> Path:
        """Return the configured local prompts directory."""
        return self._prompts_dir

    @property
    def prompt_loader(self) -> PromptLoader:
        """Return the underlying :class:`PromptLoader`."""
        return self._prompt_loader

    async def init(self) -> None:
        """Apply ``personas`` schema migrations and seed the ``default`` row."""
        await self._db.migrate("personas", _MIGRATIONS)
        existing = await self.get_persona(SYSTEM_DEFAULT_PERSONA_ID)
        if existing is not None:
            return
        seed_prompt = self._prompt_loader.load_default_file() or _DEFAULT_SYSTEM_PROMPT
        now_ts = int(time.time())
        await self._db.execute(
            """
            INSERT OR IGNORE INTO personas (
                persona_id, system_prompt, begin_dialogs, custom_error_message,
                created_at, updated_at
            )
            VALUES (?, ?, '[]', '', ?, ?)
            """,
            (SYSTEM_DEFAULT_PERSONA_ID, seed_prompt, now_ts, now_ts),
        )

    async def get_persona(self, persona_id: str) -> PersonaRecord | None:
        """Fetch a single persona by ID, or ``None`` if absent."""
        pid = persona_id.strip()
        if not pid:
            return None
        row = await self._db.fetchone(
            f"SELECT {_PERSONA_COLUMNS} FROM personas WHERE persona_id = ?",
            (pid,),
        )
        return _row_to_persona(row) if row is not None else None

    async def list_personas(self) -> list[PersonaRecord]:
        """List all personas with ``default`` first, then sorted by ID."""
        rows = await self._db.fetchall(
            f"""
            SELECT {_PERSONA_COLUMNS}
            FROM personas
            ORDER BY
                CASE WHEN persona_id = ? THEN 0 ELSE 1 END,
                persona_id ASC
            """,
            (SYSTEM_DEFAULT_PERSONA_ID,),
        )
        return [_row_to_persona(row) for row in rows]

    async def create_persona(
        self,
        *,
        persona_id: str,
        system_prompt: str,
        begin_dialogs: Sequence[str] = (),
        custom_error_message: str = "",
        now: int | None = None,
    ) -> PersonaRecord:
        """Create and persist a new persona preset.

        Raises:
            ValueError: If any field fails validation.
            PersonaExistsError: If ``persona_id`` already exists.
        """
        pid = validate_persona_id(persona_id)
        prompt = validate_system_prompt(system_prompt)
        dialogs = validate_begin_dialogs(begin_dialogs)
        error_msg = validate_custom_error_message(custom_error_message)
        ts = int(time.time()) if now is None else int(now)
        dialogs_json = json.dumps(list(dialogs), ensure_ascii=False)

        try:
            await self._db.execute(
                """
                INSERT INTO personas (
                    persona_id, system_prompt, begin_dialogs, custom_error_message,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (pid, prompt, dialogs_json, error_msg, ts, ts),
            )
        except sqlite3.IntegrityError as exc:
            raise PersonaExistsError(f"人格预设「{pid}」已存在。") from exc

        return PersonaRecord(
            persona_id=pid,
            system_prompt=prompt,
            begin_dialogs=dialogs,
            custom_error_message=error_msg,
            created_at=ts,
            updated_at=ts,
        )

    async def update_persona(
        self,
        persona_id: str,
        *,
        system_prompt: str,
        begin_dialogs: Sequence[str] = (),
        custom_error_message: str = "",
        now: int | None = None,
    ) -> PersonaRecord:
        """Update an existing persona preset (including ``default``).

        Raises:
            ValueError: If any field fails validation.
            PersonaNotFoundError: If ``persona_id`` does not exist.
        """
        pid = validate_persona_id(persona_id)
        prompt = validate_system_prompt(system_prompt)
        dialogs = validate_begin_dialogs(begin_dialogs)
        error_msg = validate_custom_error_message(custom_error_message)
        ts = int(time.time()) if now is None else int(now)
        dialogs_json = json.dumps(list(dialogs), ensure_ascii=False)

        res = await self._db.execute(
            """
            UPDATE personas
            SET system_prompt = ?,
                begin_dialogs = ?,
                custom_error_message = ?,
                updated_at = ?
            WHERE persona_id = ?
            """,
            (prompt, dialogs_json, error_msg, ts, pid),
        )
        if res.rowcount == 0:
            raise PersonaNotFoundError(f"人格预设「{pid}」不存在。")

        updated = await self.get_persona(pid)
        assert updated is not None
        return updated

    async def delete_persona(self, persona_id: str) -> bool:
        """Delete a non-default persona preset (cascading its session bindings).

        Returns:
            ``True`` if a row was deleted, ``False`` if it did not exist.

        Raises:
            ValueError: If ``persona_id`` is invalid or is ``"default"``.
        """
        pid = validate_persona_id(persona_id)
        if pid == SYSTEM_DEFAULT_PERSONA_ID:
            raise ValueError("系统内置默认人格（default）不可删除。")
        res = await self._db.execute(
            "DELETE FROM personas WHERE persona_id = ?",
            (pid,),
        )
        return res.rowcount > 0

    async def set_binding(
        self,
        scope: BindingScope,
        target_id: str,
        persona_id: str,
        *,
        now: int | None = None,
    ) -> PersonaBinding:
        """Bind a group or private session to an existing persona preset.

        Raises:
            ValueError: If ``scope``, ``target_id``, or ``persona_id`` is invalid.
            PersonaNotFoundError: If ``persona_id`` does not exist in ``personas``.
        """
        valid_scope = validate_binding_scope(scope)
        valid_target = validate_binding_target(target_id)
        pid = validate_persona_id(persona_id)
        session_key = f"{valid_scope}:{valid_target}"
        ts = int(time.time()) if now is None else int(now)

        async with self._db.transaction() as tx:
            row = await tx.fetchone(
                "SELECT 1 FROM personas WHERE persona_id = ?",
                (pid,),
            )
            if row is None:
                raise PersonaNotFoundError(f"人格预设「{pid}」不存在。")
            await tx.execute(
                """
                INSERT INTO persona_bindings (session_key, persona_id, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(session_key) DO UPDATE SET
                    persona_id = excluded.persona_id,
                    updated_at = excluded.updated_at
                """,
                (session_key, pid, ts),
            )

        return PersonaBinding(
            session_key=session_key,
            scope=valid_scope,
            target_id=valid_target,
            persona_id=pid,
            updated_at=ts,
        )

    async def get_binding(
        self,
        scope: BindingScope,
        target_id: str,
    ) -> PersonaBinding | None:
        """Return the binding for ``(scope, target_id)``, or ``None``."""
        session_key = make_binding_session_key(scope, target_id)
        row = await self._db.fetchone(
            """
            SELECT session_key, persona_id, updated_at
            FROM persona_bindings
            WHERE session_key = ?
            """,
            (session_key,),
        )
        return _row_to_binding(row) if row is not None else None

    async def delete_binding(
        self,
        scope: BindingScope,
        target_id: str,
    ) -> bool:
        """Remove a session binding, returning whether a row was deleted."""
        session_key = make_binding_session_key(scope, target_id)
        res = await self._db.execute(
            "DELETE FROM persona_bindings WHERE session_key = ?",
            (session_key,),
        )
        return res.rowcount > 0

    async def list_bindings(self) -> list[PersonaBinding]:
        """List all active session-to-persona bindings ordered by session key."""
        rows = await self._db.fetchall(
            """
            SELECT session_key, persona_id, updated_at
            FROM persona_bindings
            ORDER BY session_key ASC
            """
        )
        return [_row_to_binding(row) for row in rows]

    async def resolve_effective_persona(
        self,
        *,
        group_id: str | None,
        user_id: str,
        default_persona_id: str = SYSTEM_DEFAULT_PERSONA_ID,
    ) -> EffectivePersona:
        """Resolve the effective persona for a group or private conversation.

        Priority order:
        1. Explicit DB binding in ``persona_bindings`` for ``group:{group_id}``
           (when ``group_id is not None``) or ``private:{user_id}``.
        2. Local file override via :class:`PromptLoader`
           (``group_{group_id}.md`` or ``private_{user_id}.md``).
        3. Configured ``default_persona_id`` looked up in ``personas``.
        4. ``"default"`` row in ``personas``, falling back to built-in
           ``_DEFAULT_SYSTEM_PROMPT``.

        Args:
            group_id: Group QQ ID for group chat, or ``None`` for private chat.
            user_id: Peer user QQ ID.
            default_persona_id: Configured fallback persona preset identifier.

        Returns:
            Resolved :class:`EffectivePersona`.
        """
        session_key = (
            f"group:{group_id}" if group_id is not None else f"private:{user_id}"
        )

        # 1. Explicit DB binding (highest priority).
        bound_row = await self._db.fetchone(
            """
            SELECT p.persona_id, p.system_prompt, p.begin_dialogs,
                   p.custom_error_message, p.created_at, p.updated_at
            FROM persona_bindings b
            JOIN personas p ON p.persona_id = b.persona_id
            WHERE b.session_key = ?
            """,
            (session_key,),
        )
        if bound_row is not None:
            rec = _row_to_persona(bound_row)
            return EffectivePersona(
                persona_id=rec.persona_id,
                system_prompt=rec.system_prompt,
                begin_dialogs=rec.begin_dialogs,
                custom_error_message=rec.custom_error_message,
            )

        # 2. Local Markdown override file (group_{gid}.md / private_{uid}.md).
        file_override = self._prompt_loader.load_override(
            group_id=group_id,
            user_id=user_id,
        )
        if file_override is not None:
            file_persona_id, override_text = file_override
            return EffectivePersona(
                persona_id=file_persona_id,
                system_prompt=override_text,
                begin_dialogs=(),
                custom_error_message="",
            )

        # 3. Configured default_persona_id in personas table.
        configured_id = default_persona_id.strip() if default_persona_id else ""
        if configured_id and configured_id != SYSTEM_DEFAULT_PERSONA_ID:
            configured = await self.get_persona(configured_id)
            if configured is not None:
                return EffectivePersona(
                    persona_id=configured.persona_id,
                    system_prompt=configured.system_prompt,
                    begin_dialogs=configured.begin_dialogs,
                    custom_error_message=configured.custom_error_message,
                )

        # 4. System "default" row in personas table (or built-in prompt fallback).
        default_rec = await self.get_persona(SYSTEM_DEFAULT_PERSONA_ID)
        if default_rec is not None:
            return EffectivePersona(
                persona_id=default_rec.persona_id,
                system_prompt=default_rec.system_prompt,
                begin_dialogs=default_rec.begin_dialogs,
                custom_error_message=default_rec.custom_error_message,
            )

        return EffectivePersona(
            persona_id=SYSTEM_DEFAULT_PERSONA_ID,
            system_prompt=_DEFAULT_SYSTEM_PROMPT,
            begin_dialogs=(),
            custom_error_message="",
        )
