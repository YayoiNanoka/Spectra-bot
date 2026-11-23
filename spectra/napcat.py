"""NapCat (OneBot v11 reverse WebSocket) inbound/outbound adapter."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import math
import random
import re
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from aiocqhttp import CQHttp

from spectra.config import AppConfig
from spectra.messages import MessageStore
from spectra.vision import ImageCaptioner

logger = logging.getLogger(__name__)

_DEFAULT_SEGMENTED_REGEX = r".*?[。？！~…]+|.+$"
_MEMBER_CACHE_MAX_SIZE = 4096
_REPLY_SUMMARY_MAX_LEN = 200
_FORWARD_SUMMARY_MAX_LEN = 300
_CARD_FIELD_MAX_LEN = 200


@dataclass(slots=True, frozen=True)
class InboundMessage:
    """Normalized inbound OneBot v11 message persisted in :class:`MessageStore`.

    Attributes:
        db_id: Auto-incremented ``messages.id`` assigned upon persistence.
        message_id: OneBot v11 platform message ID.
        self_id: Bot's own QQ ID.
        group_id: Group QQ ID for group messages, or ``None`` for private chat.
        user_id: Sender QQ ID.
        sender_name: Sender's group card or nickname at receive time.
        content: Flattened text content with resolved mentions, replies, and media.
        is_mentioned_bot: ``True`` if the message explicitly ``@Bot`` or replies
            to a message sent by the bot.
        starts_with_at_other: ``True`` if the first substantive (non-empty,
            non-reply) segment ``@``-mentions another user (neither bot nor all).
        timestamp: Unix timestamp in seconds.
    """

    db_id: int
    message_id: str
    self_id: str
    group_id: str | None
    user_id: str
    sender_name: str
    content: str
    is_mentioned_bot: bool
    starts_with_at_other: bool
    timestamp: int


def _maybe_int(value: str | int) -> int | str:
    if isinstance(value, int):
        return value
    s = str(value).strip()
    if s and (s.isdigit() or (s[0] == "-" and s[1:].isdigit())):
        return int(s)
    return s


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def _join_tokens(tokens: list[tuple[bool, str]]) -> str:
    """Join ``(is_text, chunk)`` tokens without gluing non-text tags together."""
    out: list[str] = []
    prev_is_text: bool | None = None
    for is_text, chunk in tokens:
        if not chunk:
            continue
        if out and not (prev_is_text and is_text):
            if not out[-1][-1].isspace() and not chunk[0].isspace():
                out.append(" ")
        out.append(chunk)
        prev_is_text = is_text
    return "".join(out).strip()


def _format_json_card(raw_data: Any) -> str:
    """Extract a readable summary from an OneBot ``json`` share-card payload."""
    card_data: Any = raw_data
    if isinstance(card_data, str):
        try:
            card_data = json.loads(card_data)
        except json.JSONDecodeError:
            return "[分享卡片]"

    if isinstance(card_data, dict) and isinstance(card_data.get("data"), str):
        try:
            nested = json.loads(card_data["data"])
            if isinstance(nested, dict):
                card_data = nested
        except json.JSONDecodeError:
            pass
    elif isinstance(card_data, dict) and isinstance(card_data.get("data"), dict):
        card_data = card_data["data"]

    detail: dict[str, Any] = {}
    if isinstance(card_data, dict):
        meta = card_data.get("meta")
        if isinstance(meta, dict):
            candidate = meta.get("detail_1") or meta.get("news")
            if isinstance(candidate, dict):
                detail = candidate

    fields: list[str] = []
    for label, value in (
        ("标题", detail.get("title")),
        ("简介", detail.get("desc")),
        ("链接", detail.get("qqdocurl") or detail.get("jumpUrl")),
    ):
        if isinstance(value, str) and value.strip():
            normalized = " ".join(value.split())
            fields.append(f"{label}: {_truncate(normalized, _CARD_FIELD_MAX_LEN)}")

    if not fields:
        return "[分享卡片]"
    return f"[分享卡片: {'; '.join(fields)}]"


def _summarize_segments(segments: Any, *, max_len: int) -> str:
    """Non-recursive summary of OneBot message segments for quotes and forwards."""
    if isinstance(segments, str):
        return _truncate(" ".join(segments.split()), max_len)
    if not isinstance(segments, list):
        return ""

    tokens: list[tuple[bool, str]] = []
    for seg in segments:
        if not isinstance(seg, Mapping):
            continue
        seg_type = str(seg.get("type") or "")
        data = seg.get("data")
        if not isinstance(data, Mapping):
            data = {}

        if seg_type == "text":
            tokens.append((True, str(data.get("text") or "")))
        elif seg_type == "markdown":
            tokens.append(
                (True, str(data.get("content") or data.get("markdown") or ""))
            )
        elif seg_type == "image":
            tokens.append((False, "[图片]"))
        elif seg_type == "at":
            qq = str(data.get("qq") or "").strip()
            if qq.lower() == "all":
                tokens.append((False, "@全体成员"))
            elif qq:
                tokens.append((False, f"@{qq}"))
        elif seg_type == "reply":
            tokens.append((False, "[回复]"))
        elif seg_type in ("face", "mface"):
            tokens.append((False, "[表情]"))
        elif seg_type == "record":
            tokens.append((False, "[语音]"))
        elif seg_type == "video":
            tokens.append((False, "[视频]"))
        elif seg_type == "file":
            name = str(
                data.get("name") or data.get("file_name") or data.get("file") or ""
            ).strip()
            tokens.append((False, f"[文件: {name}]" if name else "[文件]"))
        elif seg_type == "json":
            tokens.append((False, _format_json_card(data)))
        elif seg_type == "forward":
            tokens.append((False, "[合并转发]"))

    joined = " ".join(_join_tokens(tokens).split())
    return _truncate(joined, max_len)


def _count_words(text: str) -> int:
    if all(ord(c) < 128 for c in text):
        return len(text.split())
    return sum(1 for c in text if c.isalnum())


class NapCatAdapter:
    """OneBot v11 reverse WebSocket adapter backed by :class:`aiocqhttp.CQHttp`.

    Args:
        config: Root :class:`spectra.config.AppConfig`.
        message_store: Initialized :class:`spectra.messages.MessageStore`.
        captioner: Optional :class:`spectra.vision.ImageCaptioner` for images.
        on_message: Optional async callback invoked after persisting each
            allowed inbound message.
        sleep_fn: Optional async sleep function override for segmented reply
            delays (defaults to :func:`asyncio.sleep`).
    """

    def __init__(
        self,
        config: AppConfig,
        message_store: MessageStore,
        captioner: ImageCaptioner | None = None,
        on_message: Callable[[InboundMessage], Awaitable[None]] | None = None,
        *,
        sleep_fn: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self._config = config
        self._message_store = message_store
        self._captioner = captioner
        self._on_message = on_message
        self._sleep = sleep_fn or asyncio.sleep

        self._bot = CQHttp(
            use_ws_reverse=True,
            access_token=config.napcat.access_token or None,
            api_timeout_sec=60,
        )
        self._member_cache: OrderedDict[tuple[str, str], str] = OrderedDict()
        self._shutdown_event = asyncio.Event()
        self._last_self_id: str | None = None

        @self._bot.on_message("group")
        async def _on_group_message(event: Mapping[str, Any]) -> None:
            await self.handle_event(event)

        @self._bot.on_message("private")
        async def _on_private_message(event: Mapping[str, Any]) -> None:
            await self.handle_event(event)

    @property
    def bot(self) -> CQHttp:
        """Return the underlying :class:`aiocqhttp.CQHttp` client."""
        return self._bot

    async def _call_action(self, action: str, **params: Any) -> Any:
        return await self._bot.call_action(action, **params)

    def _cache_member_name(self, group_id: str, user_id: str, name: str) -> None:
        key = (group_id, user_id)
        self._member_cache[key] = name
        self._member_cache.move_to_end(key)
        while len(self._member_cache) > _MEMBER_CACHE_MAX_SIZE:
            self._member_cache.popitem(last=False)

    async def _resolve_member_name(
        self,
        group_id: str | None,
        qq: str,
        routing: dict[str, Any],
    ) -> str | None:
        if group_id is None:
            return None
        key = (group_id, qq)
        cached = self._member_cache.get(key)
        if cached is not None:
            self._member_cache.move_to_end(key)
            return cached

        try:
            info = await self._call_action(
                "get_group_member_info",
                group_id=_maybe_int(group_id),
                user_id=_maybe_int(qq),
                no_cache=False,
                **routing,
            )
        except Exception as exc:
            logger.warning(
                "Failed to resolve member name for (%s, %s): %s", group_id, qq, exc
            )
            return None

        if not isinstance(info, Mapping):
            return None
        name = str(info.get("card") or info.get("nickname") or "").strip()
        if not name:
            return None
        self._cache_member_name(group_id, qq, name)
        return name

    async def _resolve_reply_segment(
        self,
        data: Mapping[str, Any],
        self_id: str,
        routing: dict[str, Any],
    ) -> tuple[str, bool]:
        reply_id = data.get("id")
        if reply_id is None or str(reply_id).strip() == "":
            return "[回复]", False

        try:
            resp = await self._call_action(
                "get_msg",
                message_id=_maybe_int(str(reply_id)),
                **routing,
            )
        except Exception as exc:
            logger.warning("Failed to fetch quoted message %s: %s", reply_id, exc)
            return "[回复]", False

        if not isinstance(resp, Mapping):
            return "[回复]", False

        sender = resp.get("sender")
        sender_map = sender if isinstance(sender, Mapping) else {}
        reply_sender_id = str(
            sender_map.get("user_id") or resp.get("user_id") or ""
        ).strip()
        is_reply_to_bot = bool(reply_sender_id and reply_sender_id == self_id)
        reply_name = str(
            sender_map.get("card")
            or sender_map.get("nickname")
            or reply_sender_id
            or "未知用户"
        ).strip()
        summary = _summarize_segments(
            resp.get("message"),
            max_len=_REPLY_SUMMARY_MAX_LEN,
        )
        if summary:
            return f"[回复 {reply_name}: {summary}]", is_reply_to_bot
        return f"[回复 {reply_name}]", is_reply_to_bot

    async def _resolve_forward_segment(
        self,
        data: Mapping[str, Any],
        routing: dict[str, Any],
    ) -> str:
        forward_id = data.get("id")
        if forward_id is None or str(forward_id).strip() == "":
            return "[合并转发]"

        try:
            resp = await self._call_action(
                "get_forward_msg",
                id=str(forward_id),
                **routing,
            )
        except Exception as exc:
            logger.warning("Failed to fetch forward message %s: %s", forward_id, exc)
            return "[合并转发]"

        if not isinstance(resp, Mapping):
            return "[合并转发]"

        nodes = resp.get("messages")
        if not isinstance(nodes, list):
            nodes = resp.get("message")
        if not isinstance(nodes, list):
            return "[合并转发]"

        entries: list[str] = []
        for node in nodes:
            if not isinstance(node, Mapping):
                continue
            node_data = node.get("data")
            payload = node_data if isinstance(node_data, Mapping) else node
            sender = payload.get("sender")
            sender_map = sender if isinstance(sender, Mapping) else {}
            name = str(
                sender_map.get("card")
                or sender_map.get("nickname")
                or payload.get("nickname")
                or payload.get("name")
                or sender_map.get("user_id")
                or payload.get("user_id")
                or payload.get("uin")
                or "未知用户"
            ).strip()
            body = payload.get("content")
            if body is None:
                body = payload.get("message")
            text = _summarize_segments(body, max_len=_FORWARD_SUMMARY_MAX_LEN)
            if text:
                entries.append(f"{name}: {text}")

        if not entries:
            return "[合并转发]"
        summary = _truncate(" | ".join(entries), _FORWARD_SUMMARY_MAX_LEN)
        return f"[合并转发: {summary}]"

    async def handle_event(
        self,
        event: Mapping[str, Any],
        *,
        resolve_reply: bool = True,
    ) -> InboundMessage | None:
        """Process a raw OneBot v11 message event, persist it, and dispatch callback.

        Args:
            event: OneBot v11 message event mapping.
            resolve_reply: Whether ``reply`` segments should be resolved via
                ``get_msg`` (1 level only).

        Returns:
            Persisted :class:`InboundMessage`, or ``None`` if ignored or empty.
        """
        raw_self_id = event.get("self_id")
        raw_user_id = event.get("user_id")
        if raw_self_id is None or raw_user_id is None:
            return None

        self_id = str(raw_self_id).strip()
        user_id = str(raw_user_id).strip()
        if not self_id or not user_id:
            return None

        message_type = str(event.get("message_type") or "").strip().lower()
        raw_group_id = event.get("group_id")
        if message_type == "private" or raw_group_id is None:
            group_id: str | None = None
        else:
            gid_str = str(raw_group_id).strip()
            group_id = None if gid_str in ("", "0") else gid_str

        if not self._config.access.is_allowed(
            group_id=group_id,
            user_id=user_id,
            self_id=self_id,
        ):
            return None

        raw_segments = event.get("message")
        if not isinstance(raw_segments, list):
            return None

        self._last_self_id = self_id
        routing: dict[str, Any] = {"self_id": _maybe_int(self_id)} if self_id else {}

        sender = event.get("sender")
        sender_map = sender if isinstance(sender, Mapping) else {}
        sender_name = (
            str(sender_map.get("card") or sender_map.get("nickname") or user_id).strip()
            or user_id
        )
        message_id = str(event.get("message_id") or "")
        raw_time = event.get("time")
        try:
            timestamp = int(raw_time) if raw_time is not None else int(time.time())
        except (TypeError, ValueError):
            timestamp = int(time.time())

        is_mentioned_bot = False
        starts_with_at_other = False
        saw_substantive_segment = False
        tokens: list[tuple[bool, str]] = []

        for seg in raw_segments:
            if not isinstance(seg, Mapping):
                continue
            seg_type = str(seg.get("type") or "")
            data = seg.get("data")
            if not isinstance(data, Mapping):
                data = {}

            if not saw_substantive_segment and seg_type != "reply":
                if seg_type == "text":
                    if str(data.get("text") or "").strip():
                        saw_substantive_segment = True
                elif seg_type == "markdown":
                    md = str(data.get("content") or data.get("markdown") or "")
                    if md.strip():
                        saw_substantive_segment = True
                elif seg_type == "at":
                    qq_val = str(data.get("qq") or "").strip()
                    if qq_val:
                        saw_substantive_segment = True
                        if qq_val.lower() != "all" and qq_val != self_id:
                            starts_with_at_other = True
                elif seg_type in {
                    "image",
                    "json",
                    "forward",
                    "face",
                    "mface",
                    "record",
                    "video",
                    "file",
                }:
                    saw_substantive_segment = True

            if seg_type == "reply":
                if resolve_reply:
                    reply_text, replied_to_bot = await self._resolve_reply_segment(
                        data,
                        self_id,
                        routing,
                    )
                    if replied_to_bot:
                        is_mentioned_bot = True
                    tokens.append((False, reply_text))
                else:
                    tokens.append((False, "[回复]"))
            elif seg_type == "at":
                qq = str(data.get("qq") or "").strip()
                if qq.lower() == "all":
                    tokens.append((False, "@全体成员"))
                elif qq == self_id:
                    is_mentioned_bot = True
                    tokens.append((False, "@Bot"))
                elif qq:
                    member_name = await self._resolve_member_name(
                        group_id,
                        qq,
                        routing,
                    )
                    if member_name and member_name != qq:
                        tokens.append((False, f"@{member_name}({qq})"))
                    else:
                        tokens.append((False, f"@{qq}({qq})"))
            elif seg_type == "image":
                source = str(data.get("url") or data.get("file") or "").strip()
                caption: str | None = None
                if source and self._captioner is not None:
                    caption = await self._captioner.caption(source)
                if caption:
                    tokens.append((False, f"[图片: {caption}]"))
                else:
                    tokens.append((False, "[图片]"))
            elif seg_type == "json":
                tokens.append((False, _format_json_card(data)))
            elif seg_type == "forward":
                tokens.append(
                    (False, await self._resolve_forward_segment(data, routing))
                )
            elif seg_type == "text":
                tokens.append((True, str(data.get("text") or "")))
            elif seg_type == "markdown":
                tokens.append(
                    (True, str(data.get("content") or data.get("markdown") or ""))
                )
            elif seg_type in ("face", "mface"):
                tokens.append((False, "[表情]"))
            elif seg_type == "record":
                tokens.append((False, "[语音]"))
            elif seg_type == "video":
                tokens.append((False, "[视频]"))
            elif seg_type == "file":
                name = str(
                    data.get("name") or data.get("file_name") or data.get("file") or ""
                ).strip()
                tokens.append((False, f"[文件: {name}]" if name else "[文件]"))

        content = _join_tokens(tokens)
        if not content.strip():
            return None

        db_id = await self._message_store.append_message(
            group_id=group_id,
            user_id=user_id,
            sender_name=sender_name,
            role="user",
            content=content,
            created_at=timestamp,
        )
        inbound = InboundMessage(
            db_id=db_id,
            message_id=message_id,
            self_id=self_id,
            group_id=group_id,
            user_id=user_id,
            sender_name=sender_name,
            content=content,
            is_mentioned_bot=is_mentioned_bot,
            starts_with_at_other=starts_with_at_other,
            timestamp=timestamp,
        )

        if self._on_message is not None:
            try:
                await self._on_message(inbound)
            except Exception:
                logger.exception(
                    "Unhandled exception in inbound message callback (db_id=%d)",
                    db_id,
                )

        return inbound

    async def _send_segments(
        self,
        *,
        group_id: str | None,
        user_id: str,
        segments: list[dict[str, Any]],
        routing: dict[str, Any],
    ) -> None:
        if group_id is not None:
            await self._call_action(
                "send_group_msg",
                group_id=_maybe_int(group_id),
                message=segments,
                **routing,
            )
        else:
            await self._call_action(
                "send_private_msg",
                user_id=_maybe_int(user_id),
                message=segments,
                **routing,
            )

    @staticmethod
    def _build_text_segments(
        text: str,
        *,
        reply_to_msg_id: str | None,
    ) -> list[dict[str, Any]]:
        segments: list[dict[str, Any]] = []
        if reply_to_msg_id:
            segments.append({"type": "reply", "data": {"id": str(reply_to_msg_id)}})
        segments.append({"type": "text", "data": {"text": text}})
        return segments

    def _split_reply_text(self, text: str) -> list[str]:
        pattern = self._config.napcat.segmented_regex
        try:
            matches = list(re.finditer(pattern, text, re.DOTALL | re.MULTILINE))
        except re.error as exc:
            logger.warning(
                "Invalid segmented_regex %r (%s); falling back to default.",
                pattern,
                exc,
            )
            matches = list(
                re.finditer(
                    _DEFAULT_SEGMENTED_REGEX,
                    text,
                    re.DOTALL | re.MULTILINE,
                )
            )
        pieces = [m.group(0).strip() for m in matches if m.group(0).strip()]
        return pieces or [text]

    async def send_reply(
        self,
        text: str,
        *,
        group_id: str | None,
        user_id: str,
        reply_to_msg_id: str | None = None,
        self_id: str | None = None,
        segmented: bool | None = None,
    ) -> int | None:
        """Send an outbound reply via NapCat and persist one merged assistant row.

        Args:
            text: Full reply text to send.
            group_id: Target group QQ ID, or ``None`` for private chat.
            user_id: Target peer QQ ID (always required; persisted as
                ``messages.user_id`` for private chats).
            reply_to_msg_id: Optional OneBot message ID to quote on the first
                normal message segment.
            self_id: Optional bot QQ ID for multi-account routing and group
                persistence.
            segmented: Optional override for ``config.napcat.enable_segmented_reply``.

        Returns:
            Inserted ``messages.id`` for the persisted assistant row, or
            ``None`` if ``text`` is empty/whitespace.
        """
        cleaned = text.strip()
        if not cleaned:
            return None

        resolved_self_id = self_id or self._last_self_id
        routing: dict[str, Any] = (
            {"self_id": _maybe_int(resolved_self_id)} if resolved_self_id else {}
        )
        napcat_cfg = self._config.napcat
        use_segmented = (
            napcat_cfg.enable_segmented_reply if segmented is None else segmented
        )

        if len(cleaned) > napcat_cfg.forward_threshold:
            uin = (
                resolved_self_id
                if resolved_self_id and resolved_self_id.isdigit()
                else "10000"
            )
            nodes = [
                {
                    "type": "node",
                    "data": {
                        "name": napcat_cfg.bot_name,
                        "uin": uin,
                        "content": [{"type": "text", "data": {"text": cleaned}}],
                    },
                }
            ]
            try:
                if group_id is not None:
                    await self._call_action(
                        "send_group_forward_msg",
                        group_id=_maybe_int(group_id),
                        messages=nodes,
                        **routing,
                    )
                else:
                    await self._call_action(
                        "send_private_forward_msg",
                        user_id=_maybe_int(user_id),
                        messages=nodes,
                        **routing,
                    )
            except Exception as exc:
                logger.warning(
                    "Forward message failed (%s); falling back to single message.",
                    exc,
                )
                await self._send_segments(
                    group_id=group_id,
                    user_id=user_id,
                    segments=self._build_text_segments(
                        cleaned,
                        reply_to_msg_id=reply_to_msg_id,
                    ),
                    routing=routing,
                )
        elif (
            use_segmented
            and len(cleaned) <= napcat_cfg.segmented_words_threshold
            and "```" not in cleaned
        ):
            pieces = self._split_reply_text(cleaned)
            for idx, piece in enumerate(pieces):
                if idx > 0:
                    word_count = _count_words(piece)
                    delay = math.log(
                        word_count + 1,
                        napcat_cfg.segmented_log_base,
                    ) + random.uniform(0.0, 0.5)
                    await self._sleep(delay)
                await self._send_segments(
                    group_id=group_id,
                    user_id=user_id,
                    segments=self._build_text_segments(
                        piece,
                        reply_to_msg_id=reply_to_msg_id if idx == 0 else None,
                    ),
                    routing=routing,
                )
        else:
            await self._send_segments(
                group_id=group_id,
                user_id=user_id,
                segments=self._build_text_segments(
                    cleaned,
                    reply_to_msg_id=reply_to_msg_id,
                ),
                routing=routing,
            )

        persisted_user_id = (
            (resolved_self_id or "bot") if group_id is not None else str(user_id)
        )
        return await self._message_store.append_message(
            group_id=group_id,
            user_id=persisted_user_id,
            sender_name=napcat_cfg.bot_name,
            role="assistant",
            content=cleaned,
        )

    async def run(self) -> None:
        """Start the reverse WebSocket server until :meth:`shutdown` is called."""
        try:
            await self._bot.run_task(
                host=self._config.napcat.host,
                port=self._config.napcat.port,
                shutdown_trigger=self._shutdown_event.wait,
            )
        finally:
            self._shutdown_event.clear()

    async def shutdown(self) -> None:
        """Signal server shutdown and close all active reverse WebSocket clients."""
        self._shutdown_event.set()
        await self._close_reverse_ws_connections()

    async def _close_reverse_ws_connections(self) -> None:
        api_clients = getattr(self._bot, "_wsr_api_clients", None)
        event_clients = getattr(self._bot, "_wsr_event_clients", None)

        ws_clients: set[Any] = set()
        if isinstance(api_clients, dict):
            ws_clients.update(api_clients.values())
        if isinstance(event_clients, set):
            ws_clients.update(event_clients)

        close_tasks: list[Awaitable[Any]] = []
        for ws in ws_clients:
            close_func = getattr(ws, "close", None)
            if not callable(close_func):
                continue
            try:
                result = close_func(code=1000, reason="Adapter shutdown")
            except TypeError:
                result = close_func()
            except Exception:
                continue
            if inspect.isawaitable(result):
                close_tasks.append(result)

        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)

        if isinstance(api_clients, dict):
            api_clients.clear()
        if isinstance(event_clients, set):
            event_clients.clear()
