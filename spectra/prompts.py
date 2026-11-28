"""Mtime-cached local Markdown persona prompt loader.

Resolves per-group (``group_{group_id}.md``) or per-private-user
(``private_{user_id}.md``) overrides under a configured prompts directory,
falling back to ``default.md`` and finally an embedded base prompt.
"""

from __future__ import annotations

import logging
import os
import stat
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM_PROMPT = (
    "你是 Spectra，一个常驻 QQ 群聊与私聊里的普通群友型聊天伙伴。\n"
    "语气自然、真诚、简洁，有活人感，不使用“作为AI助手”等机械套话。\n"
    "日常闲聊简短自然回应，认真提问时给出清晰准确的解答。\n"
    "直接输出回复正文，严禁在开头添加 [昵称(QQ号) HH:MM]: 或 [Spectra]: 等任何前缀。"
)


def _is_safe_id(raw_id: str) -> bool:
    """Return whether ``raw_id`` is safe to interpolate into a filename."""
    if not raw_id or raw_id in (".", ".."):
        return False
    if "\x00" in raw_id or ".." in raw_id:
        return False
    separators = {"/", "\\"}
    if os.sep:
        separators.add(os.sep)
    if os.altsep:
        separators.add(os.altsep)
    return not any(sep in raw_id for sep in separators)


class PromptLoader:
    """Hot-reloading local Markdown system-prompt loader with stat caching.

    Caches read file contents keyed by resolved :class:`~pathlib.Path` and
    ``(st_mtime_ns, st_size)`` so unchanged files avoid disk reads while edits,
    creations, and deletions take effect immediately without restarting.

    Args:
        prompts_dir: Directory holding ``default.md`` and optional
            ``group_{group_id}.md`` / ``private_{user_id}.md`` files.
    """

    def __init__(self, prompts_dir: Path | str = Path("prompts")) -> None:
        self._prompts_dir = Path(prompts_dir)
        self._cache: dict[Path, tuple[tuple[int, int], str]] = {}

    @property
    def prompts_dir(self) -> Path:
        """Return the configured prompts directory path."""
        return self._prompts_dir

    def _read_prompt_file(self, filename: str) -> str | None:
        try:
            root = self._prompts_dir.resolve()
            candidate = (self._prompts_dir / filename).resolve()
        except OSError as exc:
            logger.warning("Failed to resolve prompt path %r: %s", filename, exc)
            return None

        if candidate.parent != root:
            logger.warning(
                "Rejected prompt path outside prompts_dir: %r -> %s",
                filename,
                candidate,
            )
            return None

        try:
            st = candidate.stat()
        except OSError:
            self._cache.pop(candidate, None)
            return None

        if not stat.S_ISREG(st.st_mode):
            self._cache.pop(candidate, None)
            return None

        signature = (st.st_mtime_ns, st.st_size)
        cached = self._cache.get(candidate)
        if cached is not None and cached[0] == signature:
            return cached[1] or None

        try:
            text = candidate.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError) as exc:
            logger.warning("Failed to read prompt file %s: %s", candidate, exc)
            self._cache.pop(candidate, None)
            return None

        self._cache[candidate] = (signature, text)
        return text or None

    def load_default_file(self) -> str | None:
        """Load non-empty ``default.md`` from ``prompts_dir``, or ``None``."""
        return self._read_prompt_file("default.md")

    def load_override(
        self,
        *,
        group_id: str | None,
        user_id: str,
    ) -> tuple[str, str] | None:
        """Load a per-group or per-private Markdown override file if present.

        Args:
            group_id: Group QQ ID for group chat, or ``None`` for private chat.
            user_id: Peer user QQ ID.

        Returns:
            ``(source_id, prompt_text)`` where ``source_id`` is
            ``"file:group_{gid}"`` or ``"file:private_{uid}"``, or ``None``
            when no valid non-empty override file exists.
        """
        if group_id is not None:
            gid = str(group_id)
            if _is_safe_id(gid):
                override = self._read_prompt_file(f"group_{gid}.md")
                if override is not None:
                    return f"file:group_{gid}", override
            else:
                logger.warning("Ignoring unsafe group_id for prompt lookup: %r", gid)
        else:
            uid = str(user_id)
            if _is_safe_id(uid):
                override = self._read_prompt_file(f"private_{uid}.md")
                if override is not None:
                    return f"file:private_{uid}", override
            else:
                logger.warning("Ignoring unsafe user_id for prompt lookup: %r", uid)
        return None

    def load(self, *, group_id: str | None, user_id: str) -> str:
        """Load the effective system prompt for a group or private conversation.

        Lookup priority:
        1. ``group_{group_id}.md`` (group chat) or ``private_{user_id}.md``
           (private chat), when the ID passes path-traversal validation and the
           file exists with non-whitespace content.
        2. ``default.md`` under ``prompts_dir`` when non-empty.
        3. Built-in fallback prompt constant.

        Args:
            group_id: Group QQ ID for group chat, or ``None`` for private chat.
            user_id: Peer user QQ ID.

        Returns:
            Non-empty system prompt string.
        """
        override = self.load_override(group_id=group_id, user_id=user_id)
        if override is not None:
            return override[1]

        default_prompt = self.load_default_file()
        if default_prompt is not None:
            return default_prompt

        return _DEFAULT_SYSTEM_PROMPT
