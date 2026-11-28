"""Comprehensive deterministic full-stack E2E test for the complete SpectraBot runtime.

Exercises every subsystem wired inside :class:`spectra.bot.SpectraBot` across a
unified 4-act timeline (AccessGate, NapCatAdapter, ImageCaptioner 3x3 GIF cache,
EntryGate read-the-air, Jev Q1/Q2/continuity/rerank, WebSearchToolkit Tavily,
LLMClient tool loop, SessionStore short episodes & intra-episode rollover,
MemoryStore profiles/memories/jargon, MemoryConsolidator privacy guardrails &
retention cleanup, and TraceLogger JSONL audit).

Only the two outermost edges are replaced in-memory:
1. OneBot v11 WebSocket RPC edge (``bot.adapter.bot.call_action``).
2. HTTP wire transport edge (``httpx.MockTransport`` on ``LLMClient``,
   ``ImageCaptioner``, ``EmbeddingClient``, ``JevClient``, and ``WebSearchToolkit``),
   plus a zero-delay ``sleep_fn``.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import httpx
from openai import AsyncOpenAI
from PIL import Image
from tenacity import wait_none

from spectra.bot import SpectraBot
from spectra.config import (
    AccessConfig,
    AppConfig,
    DatabaseConfig,
    EmbeddingConfig,
    GateConfig,
    JevConfig,
    LLMConfig,
    MemoryConfig,
    NapCatConfig,
    SegmentedReplyConfig,
    SessionConfig,
    TraceConfig,
    VisionConfig,
    WebSearchConfig,
)
from spectra.db import Database
from spectra.embedding import EmbeddingClient
from spectra.jev import JevClient
from spectra.llm import LLMClient
from spectra.vision import MODEL_IMAGE_MAX_BYTES, ImageCaptioner
from spectra.web_search import WebSearchToolkit

BOT_SELF_ID = 99999
GROUP_ID = 80001
BLOCKED_GROUP_ID = 88888
BLOCKED_USER_ID = 44444
AZHE_UID = 10001
XIAOXIA_UID = 10002
PASSERBY_UID = 10003


def _make_animated_gif(
    frame_colors: list[tuple[int, int, int]],
    size: tuple[int, int] = (72, 54),
) -> bytes:
    """Build a real multi-frame animated GIF in memory via Pillow."""
    frames = [Image.new("RGB", size, color) for color in frame_colors]
    try:
        buf = io.BytesIO()
        frames[0].save(
            buf,
            format="GIF",
            save_all=True,
            append_images=frames[1:],
            duration=100,
            loop=0,
        )
        return buf.getvalue()
    finally:
        for frame in frames:
            frame.close()


def _chat_completion_json(
    *,
    content: Any,
    model: str = "gpt-4o",
    finish_reason: str = "stop",
    tool_calls: list[dict[str, Any]] | None = None,
    prompt_tokens: int = 120,
    cached_tokens: int = 40,
    completion_tokens: int = 25,
) -> dict[str, Any]:
    """Construct an OpenAI-compatible ``/v1/chat/completions`` response body."""
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-fullstack",
        "object": "chat.completion",
        "created": 1700000000,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_tokens_details": {"cached_tokens": cached_tokens},
        },
    }


def _make_sdk_client(
    *,
    api_key: str,
    base_url: str,
    handler: Callable[[httpx.Request], httpx.Response],
) -> tuple[AsyncOpenAI, httpx.AsyncClient]:
    """Wire a real ``AsyncOpenAI`` SDK client onto an ``httpx.MockTransport``."""
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    sdk_client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=http_client,
        max_retries=0,
    )
    return sdk_client, http_client


async def _zero_sleep(_: float) -> None:
    await asyncio.sleep(0)


class _TraceWaiter:
    """Event-driven observer wrapping ``TraceLogger.write`` for deterministic waits."""

    def __init__(self, bot: SpectraBot) -> None:
        assert bot.trace_logger is not None
        self._bot = bot
        self._real_write = bot.trace_logger.write
        self._count = 0
        self._changed = asyncio.Event()
        bot.trace_logger.write = self._wrapped_write  # type: ignore[method-assign]

    async def _wrapped_write(self, entry: Any) -> None:
        await self._real_write(entry)
        self._count += 1
        self._changed.set()

    async def wait_count(self, expected: int, *, max_wait_sec: float = 5.0) -> None:
        async with asyncio.timeout(max_wait_sec):
            while self._count < expected:
                self._changed.clear()
                await self._changed.wait()
            # Ensure every EntryGate task has exited its finally block cleanly.
            assert self._bot.gate is not None
            while pending := [t for t in self._bot.gate._tasks if not t.done()]:
                await asyncio.gather(*pending)


class _OneBotRpcStub:
    """In-memory OneBot v11 ``call_action`` boundary stub recording all RPCs."""

    def __init__(
        self,
        *,
        members_by_key: dict[tuple[int, int], dict[str, Any]] | None = None,
    ) -> None:
        self._members_by_key = members_by_key or {}
        self._messages_by_id: dict[int, dict[str, Any]] = {}
        self._next_msg_id = 90000
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def __call__(self, action: str, **params: Any) -> Any:
        self.calls.append((action, params))
        if action == "get_group_member_info":
            key = (int(params["group_id"]), int(params["user_id"]))
            return self._members_by_key.get(key)
        if action == "get_msg":
            msg_id = int(params["message_id"])
            return self._messages_by_id.get(msg_id)
        if action in {"send_group_msg", "send_private_msg"}:
            self._next_msg_id += 1
            msg_id = self._next_msg_id
            # Store outbound bot bubbles so subsequent reply quotes resolve accurately.
            self._messages_by_id[msg_id] = {
                "message_id": msg_id,
                "user_id": BOT_SELF_ID,
                "sender": {
                    "user_id": BOT_SELF_ID,
                    "card": "Spectra",
                    "nickname": "Spectra",
                },
                "message": params.get("message", []),
            }
            return {"message_id": msg_id}
        raise AssertionError(f"Unexpected OneBot action: {action!r} ({params!r})")


def _group_event(
    *,
    msg_id: int,
    user_id: int,
    sender_name: str,
    timestamp: int,
    segments: list[dict[str, Any]],
    group_id: int = GROUP_ID,
) -> dict[str, Any]:
    return {
        "post_type": "message",
        "message_type": "group",
        "self_id": BOT_SELF_ID,
        "group_id": group_id,
        "user_id": user_id,
        "message_id": msg_id,
        "time": timestamp,
        "sender": {"card": sender_name, "nickname": sender_name},
        "message": segments,
    }


def _private_event(
    *,
    msg_id: int,
    user_id: int,
    sender_name: str,
    timestamp: int,
    text: str,
) -> dict[str, Any]:
    return {
        "post_type": "message",
        "message_type": "private",
        "self_id": BOT_SELF_ID,
        "user_id": user_id,
        "message_id": msg_id,
        "time": timestamp,
        "sender": {"nickname": sender_name},
        "message": [{"type": "text", "data": {"text": text}}],
    }


async def test_fullstack_e2e_four_act_lifecycle(tmp_path: Path) -> None:
    """Unified 4-act full-stack E2E test covering the entire SpectraBot runtime."""
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "default.md").write_text(
        "你是 Spectra，自然随和的朋友型聊天助手。",
        encoding="utf-8",
    )
    (prompts_dir / f"group_{GROUP_ID}.md").write_text(
        "你是群 80001 里的熟人朋友 Spectra，说话简短自然。",
        encoding="utf-8",
    )
    (prompts_dir / f"private_{AZHE_UID}.md").write_text(
        "你是阿哲的私聊好友 Spectra，替他严守私聊秘密。",
        encoding="utf-8",
    )

    cfg = AppConfig(
        db=DatabaseConfig(
            path=tmp_path / "spectra_fullstack.db",
            message_retention_days=7,
        ),
        access=AccessConfig(
            group_mode="blacklist",
            group_blacklist=[str(BLOCKED_GROUP_ID)],
            private_mode="blacklist",
            private_blacklist=[str(BLOCKED_USER_ID)],
        ),
        napcat=NapCatConfig(
            bot_name="Spectra",
            forward_threshold=1500,
            segmented_reply=SegmentedReplyConfig(
                enable=True,
                interval_method="random",
                interval="0.1,0.2",
                words_count_threshold=150,
                split_mode="regex",
            ),
        ),
        vision=VisionConfig(
            enabled=True,
            base_url="https://vision.example.test/v1",
            api_key="sk-vision-e2e",
            model="gpt-4o-mini-vision",
            max_size=1280,
        ),
        llm=LLMConfig(
            base_url="https://llm.example.test/v1",
            api_key="sk-llm-e2e",
            model="gpt-4o",
            default_effort="",
            high_effort="high",
            max_retries=1,
            max_tool_steps=6,
        ),
        embedding=EmbeddingConfig(
            base_url="https://embed.example.test/v1",
            api_key="sk-embed-e2e",
            model="text-embedding-v4",
            max_retries=0,
        ),
        jev=JevConfig(
            enabled=True,
            base_url="https://jev.example.test/v1/systemone",
            api_key="sk-jev-e2e",
            model="jev-1.14",
            reply_threshold=0.65,
            high_effort_threshold=0.60,
        ),
        gate=GateConfig(
            proactive_mode="on",
            private_debounce_sec=0.0,
            group_debounce_sec=0.0,
            max_batch_wait_sec=0.0,
            proactive_cooldown_sec=0.0,
            hot_window_sec=30.0,
        ),
        session=SessionConfig(
            prompts_dir=prompts_dir,
            window_max_messages=8,
            window_keep_messages=4,
            timezone="Asia/Shanghai",
            group_idle_timeout_minutes=30,
            private_idle_timeout_minutes=60,
            group_hard_timeout_hours=6,
            private_hard_timeout_hours=24,
            continuity_threshold=0.55,
            continuity_tail_messages=6,
            lead_in_gap_sec=300,
            lead_in_max_messages=12,
        ),
        memory=MemoryConfig(
            enabled=True,
            consolidation_enabled=True,
            vector_top_k=12,
            min_vector_similarity=0.25,
            rerank_threshold=0.50,
            max_recall_items=5,
            jargon_min_confidence=0.60,
            jargon_max_matches=8,
            cleanup_interval_sec=3600.0,
        ),
        trace=TraceConfig(
            enabled=True,
            path=tmp_path / "data" / "traces.jsonl",
        ),
        web_search=WebSearchConfig(
            enabled=True,
            provider="tavily",
            tavily_keys=["tvly-e2e-test-key"],
        ),
    )

    # ------------------------------------------------------------------
    # Simulated clocks & real 6-frame GIF bytes
    # ------------------------------------------------------------------
    tz = ZoneInfo("Asia/Shanghai")
    t0 = int(datetime(2026, 11, 27, 12, 0, 0, tzinfo=tz).timestamp())
    mono_clock = [1000.0]
    wall_clock = [t0]

    frame_w, frame_h = 72, 54
    gif_bytes = _make_animated_gif(
        [
            (220, 40, 40),
            (40, 200, 40),
            (40, 60, 220),
            (220, 200, 40),
            (200, 40, 200),
            (40, 200, 220),
        ],
        size=(frame_w, frame_h),
    )
    gif_sha256 = hashlib.sha256(gif_bytes).hexdigest()
    gif_url_primary = "https://img.example.test/rank_hype.gif"
    gif_url_mirror = "https://cdn2.example.test/rank_hype_mirror.gif"

    # ------------------------------------------------------------------
    # Transport edge recorders & deterministic HTTP MockTransports
    # ------------------------------------------------------------------
    image_download_urls: list[str] = []
    vision_requests: list[dict[str, Any]] = []
    tavily_requests: list[dict[str, Any]] = []
    embedding_requests: list[list[str]] = []
    jev_requests: list[dict[str, Any]] = []
    jev_rerank_requests: list[dict[str, Any]] = []
    llm_chat_requests: list[dict[str, Any]] = []
    llm_consolidation_requests: list[dict[str, Any]] = []

    # Dynamic IDs filled during the timeline so consolidation mock can emit
    # exact reinforce / supersede target_ids discovered from existing_memories.
    def media_handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        image_download_urls.append(url)
        if url in {gif_url_primary, gif_url_mirror}:
            return httpx.Response(
                200,
                content=gif_bytes,
                headers={
                    "content-type": "image/gif",
                    "content-length": str(len(gif_bytes)),
                },
            )
        return httpx.Response(404, text="not found")

    def vision_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.content)
        vision_requests.append(body)
        return httpx.Response(
            200,
            json=_chat_completion_json(
                content="像素猫举着双爪欢呼冲分",
                model=body["model"],
            ),
        )

    def tavily_handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://api.tavily.com/search"
        assert request.headers["authorization"] == "Bearer tvly-e2e-test-key"
        body = json.loads(request.content)
        tavily_requests.append(body)
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "2026京都红叶最佳观赏期预报",
                        "url": "https://travel.example.test/kyoto-autumn-2026",
                        "content": (
                            "2026年京都清水寺与岚山红叶预计"
                            "11月下旬至12月上旬进入最佳观赏期。"
                        ),
                    }
                ]
            },
        )

    def embedding_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        raw_inputs = body["input"]
        inputs = [raw_inputs] if isinstance(raw_inputs, str) else list(raw_inputs)
        embedding_requests.append(inputs)
        data = []
        for idx, text in enumerate(inputs):
            if "上海" in text or "跳槽" in text:
                # Deliberately near-parallel to the Act 4 recall query [1, 0, 0]
                # so private isolation is proven at the SQL boundary, not by chance.
                vec = [0.99, 0.14, 0.0]
            elif "北海道" in text or "滑雪" in text:
                vec = [0.98, 0.20, 0.0]
            elif "京都" in text or "红叶" in text:
                vec = [0.95, 0.31, 0.0]
            elif "旅游" in text or "去哪儿玩" in text:
                vec = [1.0, 0.0, 0.0]
            else:
                vec = [0.0, 0.0, 1.0]
            data.append({"index": idx, "embedding": vec})
        return httpx.Response(200, json={"data": data})

    def jev_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        jev_requests.append(body)
        questions = body["questions"]
        state = body["state"]

        # 1. Memory rerank
        if "m0" in questions:
            jev_rerank_requests.append(body)
            candidates: list[str] = state["candidates"]
            answers: dict[str, Any] = {}
            for idx, cand in enumerate(candidates):
                score = 0.94 if ("北海道" in cand or "滑雪" in cand) else 0.10
                answers[f"m{idx}"] = {"noul": score}
            return httpx.Response(
                200,
                json={
                    "model": "jev-1.14",
                    "usage": {"input_tokens": 45},
                    "answers": answers,
                },
            )

        # 2. Short-episode continuity check
        if "should_continue" in questions:
            new_text = "\n".join(state.get("new_messages", []))
            if "清水寺夜枫" in new_text:
                prob = 0.93
            else:
                prob = 0.05
            return httpx.Response(
                200,
                json={
                    "model": "jev-1.14",
                    "usage": {"input_tokens": 40},
                    "answers": {"should_continue": {"noul": prob}},
                },
            )

        # 3. EntryGate Q1 + Q2 or Q2-only
        latest_text = "\n".join(state.get("latest_batch", []))
        answers = {}
        if "should_reply" in questions:
            # Reject all unmentioned ambient chatter (passive lead-ins & mirror GIF).
            answers["should_reply"] = {"noul": 0.12}

        if "联网查下" in latest_text or "最后决定去哪儿玩" in latest_text:
            answers["effort"] = {
                "choice": "high",
                "probabilities": {"default": 0.10, "high": 0.90},
            }
        else:
            answers["effort"] = {
                "choice": "default",
                "probabilities": {"default": 0.88, "high": 0.12},
            }
        return httpx.Response(
            200,
            json={
                "model": "jev-1.14",
                "usage": {"input_tokens": 55},
                "answers": answers,
            },
        )

    def _handle_consolidation_llm(body: dict[str, Any]) -> dict[str, Any]:
        llm_consolidation_requests.append(body)
        sys_prompt = body["messages"][0]["content"]
        user_prompt = body["messages"][1]["content"]

        # Parse the embedded existing context JSON to obtain real memory IDs.
        json_start = user_prompt.index("{")
        json_end = user_prompt.index("\n\n## 本次已结束对话片段原文")
        ctx = json.loads(user_prompt[json_start:json_end])

        # Private consolidation
        if "私聊模式" in sys_prompt:
            if "上海新公司 offer" in user_prompt:
                # Deliberately attempt every possible cross-scope leak!
                return _chat_completion_json(
                    content=json.dumps(
                        {
                            "group_profile_patch": {
                                "interaction_style": "试图从私聊污染群画像",
                            },
                            "user_profile_patches": [
                                {
                                    "user_id": str(AZHE_UID),
                                    "add_aliases": ["沪漂阿哲"],
                                    "interaction_style": "私聊很紧张",
                                    "topic_preferences": "上海跳槽",
                                    "relationship": "密友",
                                    "recent_focus": "准备跳槽上海",
                                    "private_notes": (
                                        "私下透露拿到上海 offer 准备跳槽，需保密"
                                    ),
                                }
                            ],
                            "memory_ops": [
                                {
                                    "action": "add",
                                    "kind": "fact",
                                    "content": "阿哲私下拿到上海新公司 offer 准备跳槽",
                                    "owner_user_id": str(AZHE_UID),
                                    "visibility": "public",
                                    "confidence": 0.95,
                                    "event_time": None,
                                }
                            ],
                            "jargon_ops": [
                                {
                                    "term": "沪漂秘密",
                                    "meaning": "绝不能写入群黑话",
                                    "confidence": 0.95,
                                }
                            ],
                        },
                        ensure_ascii=False,
                    )
                )
            return _chat_completion_json(
                content=json.dumps(
                    {
                        "group_profile_patch": None,
                        "user_profile_patches": [],
                        "memory_ops": [],
                        "jargon_ops": [],
                    }
                )
            )

        # Group consolidation: Episode 1 (Kyoto + jargon + profiles)
        if "12月6日" in user_prompt and "拉满" in user_prompt:
            return _chat_completion_json(
                content=json.dumps(
                    {
                        "group_profile_patch": {
                            "interaction_style": "轻松熟人闲聊、爱分享旅行和游戏",
                            "frequent_topics": "旅行规划、排位开黑、美食",
                            "taboos_and_boundaries": None,
                            "member_notes": "阿哲喜欢旅行，小夏爱组织排位",
                        },
                        "user_profile_patches": [
                            {
                                "user_id": str(AZHE_UID),
                                "add_aliases": ["哲哥"],
                                "interaction_style": "随性爱分享",
                                "topic_preferences": "旅行、美食",
                                "relationship": "群里熟人",
                                "recent_focus": "准备12月去京都看红叶",
                                "private_notes": "试图从群聊篡改私聊备注_BLOCKED",
                            },
                            {
                                "user_id": str(XIAOXIA_UID),
                                "add_aliases": ["夏夏"],
                                "interaction_style": "热情爱玩梗",
                                "topic_preferences": "游戏排位",
                                "relationship": "群里熟人",
                                "recent_focus": "通宵排位上分",
                                "private_notes": None,
                            },
                        ],
                        "memory_ops": [
                            {
                                "action": "add",
                                "kind": "event",
                                "content": "阿哲计划 2026-12-06 去京都看红叶",
                                "owner_user_id": str(AZHE_UID),
                                "visibility": "public",
                                "confidence": 0.85,
                                "event_time": "2026-12-06",
                            },
                            {
                                "action": "add",
                                "kind": "fact",
                                "content": "群友经常一起讨论游戏排位上分",
                                "owner_user_id": None,
                                "visibility": "group_only",
                                "confidence": 0.75,
                                "event_time": None,
                            },
                        ],
                        "jargon_ops": [
                            {
                                "term": "拉满",
                                "meaning": "通宵排位上分",
                                "confidence": 0.85,
                            }
                        ],
                    },
                    ensure_ascii=False,
                )
            )

        # Group consolidation: Episode 3 (Hokkaido supersedes Kyoto + reinforce gaming)
        if "北海道滑雪" in user_prompt:
            existing_mems: list[dict[str, Any]] = ctx.get("existing_memories", [])
            kyoto_id = next(m["id"] for m in existing_mems if "京都" in m["content"])
            gaming_id = next(
                m["id"] for m in existing_mems if "排位上分" in m["content"]
            )
            return _chat_completion_json(
                content=json.dumps(
                    {
                        "group_profile_patch": None,
                        "user_profile_patches": [],
                        "memory_ops": [
                            {"action": "reinforce", "target_id": gaming_id},
                            {
                                "action": "supersede",
                                "target_id": kyoto_id,
                                "kind": "event",
                                "content": (
                                    "阿哲取消京都行程，改在 2026-12-12 去北海道滑雪"
                                ),
                                "owner_user_id": str(AZHE_UID),
                                "visibility": "public",
                                "confidence": 0.92,
                                "event_time": "2026-12-12",
                            },
                        ],
                        "jargon_ops": [],
                    },
                    ensure_ascii=False,
                )
            )

        # Other transitional episodes (GPU question, morning check-in) -> no-op.
        return _chat_completion_json(
            content=json.dumps(
                {
                    "group_profile_patch": None,
                    "user_profile_patches": [],
                    "memory_ops": [],
                    "jargon_ops": [],
                }
            )
        )

    def llm_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.content)
        sys_prompt = body["messages"][0]["content"]
        if "后台长期记忆整理器" in sys_prompt:
            return httpx.Response(200, json=_handle_consolidation_llm(body))

        llm_chat_requests.append(body)
        messages = body["messages"]
        last_msg = messages[-1]

        # Tool result follow-up steps
        if last_msg["role"] == "tool":
            prev_assistant = messages[-2]
            tool_name = prev_assistant["tool_calls"][0]["function"]["name"]
            if tool_name == "web_search_tavily":
                # 3 sentences -> 3 segmented bubbles (90001, 90002, 90003).
                return httpx.Response(
                    200,
                    json=_chat_completion_json(
                        content=(
                            "查到啦，京都红叶在11月下旬到12月上旬最美！"
                            "阿哲下周五去正好赶上最佳观赏期。"
                            "顺便记住咱们群的「拉满」就是通宵排位上分啦！"
                        ),
                        prompt_tokens=220,
                        cached_tokens=140,
                        completion_tokens=40,
                    ),
                )
            if tool_name == "recall_memory":
                return httpx.Response(
                    200,
                    json=_chat_completion_json(
                        content=(
                            "今晚继续冲排位！阿哲最后定好12月12日去北海道滑雪了，"
                            "至于工作跳槽的事群里没听他提过呢。"
                        ),
                        prompt_tokens=260,
                        cached_tokens=160,
                        completion_tokens=36,
                    ),
                )
            raise AssertionError(f"Unexpected tool follow-up: {tool_name!r}")

        user_text = str(last_msg["content"])

        # Act 1: Tavily web search tool call
        if "帮我联网查下2026京都红叶最佳观赏期" in user_text:
            return httpx.Response(
                200,
                json=_chat_completion_json(
                    content=None,
                    finish_reason="tool_calls",
                    tool_calls=[
                        {
                            "id": "call_tavily_kyoto",
                            "type": "function",
                            "function": {
                                "name": "web_search_tavily",
                                "arguments": json.dumps(
                                    {"query": "2026 京都红叶 最佳观赏期"},
                                    ensure_ascii=False,
                                ),
                            },
                        }
                    ],
                ),
            )

        # Act 2: Private DM turns
        if "上海新公司 offer" in user_text:
            return httpx.Response(
                200,
                json=_chat_completion_json(
                    content="恭喜拿到上海新 offer，放心我一定替你严守秘密！"
                ),
            )
        if "深夜写代码听的纯音乐" in user_text:
            return httpx.Response(
                200,
                json=_chat_completion_json(
                    content="推荐坂本龙一的钢琴曲，深夜写代码特别静心。"
                ),
            )
        if "今晚加班有点累" in user_text:
            return httpx.Response(
                200,
                json=_chat_completion_json(content="辛苦啦，早点歇着别把自己累坏了。"),
            )

        # Act 3: Group Episode 1 continuation, rollover, and Episode 2 cut
        if "改到下周日 12月6日" in user_text:
            return httpx.Response(
                200,
                json=_chat_completion_json(content="收到，12月6日周日出发也很棒！"),
            )
        if "清水寺夜枫要提前预约吗" in user_text:
            return httpx.Response(
                200,
                json=_chat_completion_json(
                    content="清水寺夜枫通常现场购票即可，建议早点去排队。"
                ),
            )
        if "好吃的汤豆腐" in user_text:
            return httpx.Response(
                200,
                json=_chat_completion_json(
                    content="奥丹汤豆腐和顺正都很经典，冬天吃超暖和。"
                ),
            )
        if "显卡价格会降吗" in user_text:
            return httpx.Response(
                200,
                json=_chat_completion_json(
                    content="中端卡大促时可能会小降，刚需可以先蹲蹲好价。"
                ),
            )

        # Act 4: Hokkaido switch, transitional episode, and final recall_memory turn
        if "改成 12月12日 去北海道滑雪" in user_text:
            return httpx.Response(
                200,
                json=_chat_completion_json(
                    content="北海道滑雪也太爽了，12月中旬雪质超好！"
                ),
            )
        if "早安打卡" in user_text:
            return httpx.Response(
                200,
                json=_chat_completion_json(content="早安呀，新的一天元气满满！"),
            )
        if "阿哲最后决定去哪儿玩" in user_text:
            return httpx.Response(
                200,
                json=_chat_completion_json(
                    content=None,
                    finish_reason="tool_calls",
                    tool_calls=[
                        {
                            "id": "call_recall_trip",
                            "type": "function",
                            "function": {
                                "name": "recall_memory",
                                "arguments": json.dumps(
                                    {"query": "阿哲 旅游 去哪儿玩 工作 跳槽"},
                                    ensure_ascii=False,
                                ),
                            },
                        }
                    ],
                ),
            )

        raise AssertionError(f"Unexpected live chat prompt: {user_text!r}")

    # ------------------------------------------------------------------
    # Instantiate real production clients over MockTransports
    # ------------------------------------------------------------------
    db = Database(cfg.db.path)
    media_http = httpx.AsyncClient(transport=httpx.MockTransport(media_handler))
    vision_sdk, vision_http = _make_sdk_client(
        api_key=cfg.vision.api_key,
        base_url=cfg.vision.base_url,
        handler=vision_handler,
    )
    llm_sdk, llm_http = _make_sdk_client(
        api_key=cfg.llm.api_key,
        base_url=cfg.llm.base_url,
        handler=llm_handler,
    )
    embed_http = httpx.AsyncClient(transport=httpx.MockTransport(embedding_handler))
    jev_http = httpx.AsyncClient(transport=httpx.MockTransport(jev_handler))
    tavily_http = httpx.AsyncClient(transport=httpx.MockTransport(tavily_handler))

    captioner = ImageCaptioner(
        db,
        cfg.vision,
        http_client=media_http,
        openai_client=vision_sdk,
        retry_wait=wait_none(),
    )
    llm_client = LLMClient(
        cfg.llm,
        openai_client=llm_sdk,
        retry_wait=wait_none(),
    )
    embedding_client = EmbeddingClient(cfg.embedding, http_client=embed_http)
    jev_client = JevClient(cfg.jev, http_client=jev_http)
    web_search = WebSearchToolkit(cfg.web_search, http_client=tavily_http)

    rpc_stub = _OneBotRpcStub(
        members_by_key={
            (GROUP_ID, PASSERBY_UID): {
                "group_id": GROUP_ID,
                "user_id": PASSERBY_UID,
                "card": "路人丙",
                "nickname": "路人丙",
            }
        }
    )

    bot = SpectraBot(
        cfg,
        db=db,
        captioner=captioner,
        jev_client=jev_client,
        llm_client=llm_client,
        embedding_client=embedding_client,
        web_search=web_search,
        sleep_fn=_zero_sleep,
        clock_fn=lambda: mono_clock[0],
    )

    try:
        async with bot:
            assert bot.adapter is not None
            assert bot.message_store is not None
            assert bot.session_store is not None
            assert bot.memory_store is not None
            assert bot.consolidator is not None
            assert bot.trace_logger is not None

            bot.adapter.bot.call_action = rpc_stub  # type: ignore[method-assign]
            bot.consolidator._time_fn = lambda: float(wall_clock[0])
            bot.consolidator._clock_fn = lambda: mono_clock[0]
            waiter = _TraceWaiter(bot)
            trace_count = 0

            async def dispatch_event(
                event: dict[str, Any],
                *,
                expect_trace: bool = True,
            ) -> Any:
                nonlocal trace_count
                mono_clock[0] += 1.0
                inbound = await bot.adapter.handle_event(event)  # type: ignore[union-attr]
                if expect_trace:
                    trace_count += 1
                    await waiter.wait_count(trace_count)
                return inbound

            async def count_rows(sql: str, params: tuple[Any, ...] = ()) -> int:
                row = await db.fetchone(sql, params)
                assert row is not None
                return int(row[0])

            # ==============================================================
            # ACT 1: Access Gate + EntryGate Read-the-Air + Vision 3x3 GIF
            #        Cache + Tavily Web Search + Segmented Multi-Bubble Reply
            #        + Passive Lead-In
            # ==============================================================
            dropped_group = await dispatch_event(
                _group_event(
                    msg_id=1001,
                    group_id=BLOCKED_GROUP_ID,
                    user_id=AZHE_UID,
                    sender_name="阿哲",
                    timestamp=t0 - 700,
                    segments=[
                        {"type": "at", "data": {"qq": str(BOT_SELF_ID)}},
                        {"type": "text", "data": {"text": " 黑名单群测试"}},
                    ],
                ),
                expect_trace=False,
            )
            dropped_private = await dispatch_event(
                _private_event(
                    msg_id=1002,
                    user_id=BLOCKED_USER_ID,
                    sender_name="黑名单用户",
                    timestamp=t0 - 690,
                    text="黑名单私聊测试",
                ),
                expect_trace=False,
            )
            assert dropped_group is None
            assert dropped_private is None
            assert await count_rows("SELECT COUNT(*) FROM messages") == 0
            assert rpc_stub.calls == []
            assert vision_requests == []
            assert jev_requests == []
            assert llm_chat_requests == []

            # @Other message (> lead_in_gap_sec before Episode 1) -> filtered locally.
            at_other_in = await dispatch_event(
                _group_event(
                    msg_id=1003,
                    user_id=XIAOXIA_UID,
                    sender_name="小夏",
                    timestamp=t0 - 600,
                    segments=[
                        {"type": "at", "data": {"qq": str(PASSERBY_UID)}},
                        {"type": "text", "data": {"text": " 午饭吃啥"}},
                    ],
                )
            )
            assert at_other_in is not None
            assert at_other_in.starts_with_at_other is True
            assert len(jev_requests) == 0  # Zero-cost local pre-filter!
            assert len(llm_chat_requests) == 0

            # Unmentioned ambient message from 阿哲 -> rejected by Jev Q1 (lead-in).
            azhe_leadin = await dispatch_event(
                _group_event(
                    msg_id=1004,
                    user_id=AZHE_UID,
                    sender_name="阿哲",
                    timestamp=t0,
                    segments=[
                        {
                            "type": "text",
                            "data": {"text": "下周五去京都看红叶机票订好啦"},
                        }
                    ],
                )
            )
            assert azhe_leadin is not None
            assert len(jev_requests) == 1
            assert "should_reply" in jev_requests[0]["questions"]
            assert len(llm_chat_requests) == 0

            # Within lead_in_gap_sec: 小夏 @Bot + 6-frame GIF + jargon + Tavily query.
            xiaoxia_ep1 = await dispatch_event(
                _group_event(
                    msg_id=1005,
                    user_id=XIAOXIA_UID,
                    sender_name="小夏",
                    timestamp=t0 + 80,
                    segments=[
                        {"type": "at", "data": {"qq": str(BOT_SELF_ID)}},
                        {"type": "image", "data": {"url": gif_url_primary}},
                        {
                            "type": "text",
                            "data": {
                                "text": (
                                    " 顺便科普下咱们群黑话「拉满」= 通宵排位上分！"
                                    "另外帮我联网查下2026京都红叶最佳观赏期"
                                )
                            },
                        },
                    ],
                )
            )
            assert xiaoxia_ep1 is not None
            assert "[图片: 像素猫举着双爪欢呼冲分]" in xiaoxia_ep1.content

            # Verify 3x3 JPEG montage sent to vision endpoint & cached by SHA-256.
            assert image_download_urls == [gif_url_primary]
            assert len(vision_requests) == 1
            v_content = vision_requests[0]["messages"][0]["content"]
            v_text = next(p["text"] for p in v_content if p["type"] == "text")
            v_img_url = next(
                p["image_url"]["url"] for p in v_content if p["type"] == "image_url"
            )
            assert "3×3" in v_text
            assert v_img_url.startswith("data:image/jpeg;base64,")
            montage_bytes = base64.b64decode(
                v_img_url.removeprefix("data:image/jpeg;base64,")
            )
            assert len(montage_bytes) < MODEL_IMAGE_MAX_BYTES
            with Image.open(io.BytesIO(montage_bytes)) as montage_img:
                assert montage_img.format == "JPEG"
                assert montage_img.size == (3 * frame_w, 3 * frame_h)

            cache_row = await db.fetchone(
                "SELECT sha256, caption FROM image_cache WHERE sha256 = ?",
                (gif_sha256,),
            )
            assert cache_row is not None
            assert str(cache_row["caption"]) == "像素猫举着双爪欢呼冲分"

            # Verify passive lead-in + high-effort routing + Tavily tool loop.
            sess_g_ep1 = await bot.session_store.get_or_create(f"group:{GROUP_ID}")
            assert sess_g_ep1.episode_id == 1
            assert sess_g_ep1.episode_start_id == azhe_leadin.db_id
            assert sess_g_ep1.window_start_id == azhe_leadin.db_id

            assert len(llm_chat_requests) == 2
            act1_step1_req = llm_chat_requests[0]
            assert act1_step1_req["reasoning_effort"] == "high"
            act1_user_turn = act1_step1_req["messages"][1]["content"]
            assert "下周五去京都看红叶机票订好啦" in act1_user_turn
            assert "午饭吃啥" not in act1_user_turn
            assert len(tavily_requests) == 1
            assert "京都红叶" in tavily_requests[0]["query"]

            # Verify segmented reply (90001, 90002, 90003) & platform ID map.
            send_group_calls_act1 = [
                params
                for action, params in rpc_stub.calls
                if action == "send_group_msg"
            ]
            assert len(send_group_calls_act1) == 3
            assert send_group_calls_act1[0]["message"][0] == {
                "type": "reply",
                "data": {"id": "1005"},
            }
            for follow in send_group_calls_act1[1:]:
                assert all(seg["type"] != "reply" for seg in follow["message"])

            g_rows_act1 = await bot.message_store.get_group_messages(str(GROUP_ID))
            bot_ep1_row = g_rows_act1[-1]
            assert bot_ep1_row.role == "assistant"
            assert bot_ep1_row.platform_message_id == "90001"
            for bubble_pid in ("90001", "90002", "90003"):
                resolved = await bot.message_store.resolve_platform_message_id(
                    platform_message_id=bubble_pid,
                    group_id=str(GROUP_ID),
                    user_id=str(XIAOXIA_UID),
                )
                assert resolved == bot_ep1_row.id

            # Same GIF from mirror URL hits image_cache with 0 extra vision calls.
            mirror_in = await dispatch_event(
                _group_event(
                    msg_id=1006,
                    user_id=XIAOXIA_UID,
                    sender_name="小夏",
                    timestamp=t0 + 100,
                    segments=[
                        {"type": "image", "data": {"url": gif_url_mirror}},
                        {"type": "text", "data": {"text": " 再发一遍冲分猫"}},
                    ],
                )
            )
            assert mirror_in is not None
            assert "[图片: 像素猫举着双爪欢呼冲分]" in mirror_in.content
            assert image_download_urls == [gif_url_primary, gif_url_mirror]
            assert len(vision_requests) == 1

            # ==============================================================
            # ACT 2: Interleaved Private DM Secret Confession & Strict
            #        One-Way Privacy Wall
            # ==============================================================
            t_dm_1 = t0 + 300
            wall_clock[0] = t_dm_1
            await dispatch_event(
                _private_event(
                    msg_id=2001,
                    user_id=AZHE_UID,
                    sender_name="阿哲私聊昵称",
                    timestamp=t_dm_1,
                    text="刚拿到上海新公司 offer 准备跳槽，千万别在群里说",
                )
            )

            # After > private_idle_timeout_minutes (60m), unrelated DM triggers
            # Jev continuity -> jev_new_episode, closing DM Episode 1.
            t_dm_2 = t_dm_1 + 70 * 60
            wall_clock[0] = t_dm_2
            await dispatch_event(
                _private_event(
                    msg_id=2002,
                    user_id=AZHE_UID,
                    sender_name="阿哲私聊昵称",
                    timestamp=t_dm_2,
                    text="对了推荐几首适合深夜写代码听的纯音乐吧",
                )
            )
            await bot.consolidator.flush()

            # Verify strict one-way privacy wall: zero DM leak into PublicUserProfile,
            # private memory forced visibility='private' & source_group_id=None,
            # zero group_profiles / group_jargon pollution.
            pub_azhe = await bot.memory_store.get_public_user_profile(str(AZHE_UID))
            assert pub_azhe is not None
            assert pub_azhe.aliases == ()
            assert pub_azhe.interaction_style == ""
            assert pub_azhe.topic_preferences == ""
            assert pub_azhe.relationship == ""
            assert pub_azhe.recent_focus == ""
            pub_hint = await bot.memory_store.render_public_user_hint(str(AZHE_UID))
            assert pub_hint == ""
            assert "上海" not in pub_hint
            assert "沪漂阿哲" not in pub_hint

            full_azhe = await bot.memory_store.get_user_profile(str(AZHE_UID))
            assert full_azhe is not None
            assert full_azhe.private_notes == "私下透露拿到上海 offer 准备跳槽，需保密"

            dm_mems = await bot.memory_store.list_active_memories(
                group_id=None,
                user_id=str(AZHE_UID),
            )
            assert len(dm_mems) == 1
            private_shanghai_mem = dm_mems[0]
            assert private_shanghai_mem.visibility == "private"
            assert private_shanghai_mem.owner_user_id == str(AZHE_UID)
            assert private_shanghai_mem.source_group_id is None
            assert private_shanghai_mem.embedding is not None
            assert await count_rows("SELECT COUNT(*) FROM group_profiles") == 0
            assert await count_rows("SELECT COUNT(*) FROM group_jargon") == 0

            # DM Episode 3 (> 60m idle) freezes private profile into system prompt.
            t_dm_3 = t_dm_2 + 70 * 60
            wall_clock[0] = t_dm_3
            await dispatch_event(
                _private_event(
                    msg_id=2003,
                    user_id=AZHE_UID,
                    sender_name="阿哲私聊昵称",
                    timestamp=t_dm_3,
                    text="今晚加班有点累，想找人随便聊聊",
                )
            )
            dm3_sys_prompt = llm_chat_requests[-1]["messages"][0]["content"]
            assert "## 当前私聊对象画像" in dm3_sys_prompt
            assert (
                "- 私聊备注：私下透露拿到上海 offer 准备跳槽，需保密" in dm3_sys_prompt
            )
            await bot.consolidator.flush()

            # ==============================================================
            # ACT 3: Group Short-Episode Mechanics (Multi-Bubble Quote
            #        Shortcut + Jev Continuity + Intra-Episode Rollover
            #        Preserving episode_start_id + Cut & Consolidation)
            # ==============================================================
            # 45m after Act 1 -> quote secondary bubble 90003 -> in_window_reply!
            t_act3_1 = t0 + 80 + 45 * 60
            wall_clock[0] = t_act3_1
            jev_continuity_before_quote = sum(
                1 for r in jev_requests if "should_continue" in r["questions"]
            )
            await dispatch_event(
                _group_event(
                    msg_id=3001,
                    user_id=AZHE_UID,
                    sender_name="阿哲",
                    timestamp=t_act3_1,
                    segments=[
                        {"type": "reply", "data": {"id": "90003"}},
                        {
                            "type": "text",
                            "data": {
                                "text": "改主意啦，京都行程改到下周日 12月6日 出发"
                            },
                        },
                    ],
                )
            )
            jev_continuity_after_quote = sum(
                1 for r in jev_requests if "should_continue" in r["questions"]
            )
            assert jev_continuity_after_quote == jev_continuity_before_quote
            sess_after_quote = await bot.session_store.get_or_create(
                f"group:{GROUP_ID}"
            )
            assert sess_after_quote.episode_id == 1

            # 40m later (no quote): same-topic -> Jev continuity 0.93 -> jev_continue!
            t_act3_2 = t_act3_1 + 40 * 60
            wall_clock[0] = t_act3_2
            await dispatch_event(
                _group_event(
                    msg_id=3002,
                    user_id=XIAOXIA_UID,
                    sender_name="小夏",
                    timestamp=t_act3_2,
                    segments=[
                        {"type": "at", "data": {"qq": str(BOT_SELF_ID)}},
                        {"type": "text", "data": {"text": " 清水寺夜枫要提前预约吗？"}},
                    ],
                )
            )
            sess_after_cont = await bot.session_store.get_or_create(f"group:{GROUP_ID}")
            assert sess_after_cont.episode_id == 1

            # Turn #9 in Episode 1 exceeds window_max_messages=8 -> intra-episode
            # rollover advances window_start_id while keeping episode_start_id anchored!
            t_act3_3 = t_act3_2 + 5 * 60
            wall_clock[0] = t_act3_3
            await dispatch_event(
                _group_event(
                    msg_id=3003,
                    user_id=AZHE_UID,
                    sender_name="阿哲",
                    timestamp=t_act3_3,
                    segments=[
                        {"type": "at", "data": {"qq": str(BOT_SELF_ID)}},
                        {
                            "type": "text",
                            "data": {"text": " 清水寺附近有什么好吃的汤豆腐？"},
                        },
                    ],
                )
            )
            sess_after_rollover = await bot.session_store.get_or_create(
                f"group:{GROUP_ID}"
            )
            assert sess_after_rollover.episode_id == 1
            assert sess_after_rollover.episode_start_id == azhe_leadin.db_id
            assert (
                sess_after_rollover.window_start_id
                > sess_after_rollover.episode_start_id
            )

            # 45m later, topic switch -> Jev continuity = 0.05 -> jev_new_episode!
            # Closes FULL Group Episode 1 [episode_start_id .. end_msg_id].
            t_act3_4 = t_act3_3 + 45 * 60
            wall_clock[0] = t_act3_4
            consolidation_count_before_g1 = len(llm_consolidation_requests)
            await dispatch_event(
                _group_event(
                    msg_id=3004,
                    user_id=XIAOXIA_UID,
                    sender_name="小夏",
                    timestamp=t_act3_4,
                    segments=[
                        {"type": "at", "data": {"qq": str(BOT_SELF_ID)}},
                        {
                            "type": "text",
                            "data": {"text": " 换个话题，今年显卡价格会降吗？"},
                        },
                    ],
                )
            )
            await bot.consolidator.flush()

            sess_g_ep2 = await bot.session_store.get_or_create(f"group:{GROUP_ID}")
            assert sess_g_ep2.episode_id == 2
            assert len(llm_consolidation_requests) == consolidation_count_before_g1 + 1

            # Inspect Group Episode 1 consolidation prompt: MUST include pre-rollover
            # messages AND zero DM secrets!
            g1_cons_prompt = llm_consolidation_requests[-1]["messages"][1]["content"]
            assert "下周五去京都看红叶机票订好啦" in g1_cons_prompt
            assert "拉满" in g1_cons_prompt
            assert "像素猫举着双爪欢呼冲分" in g1_cons_prompt
            assert "12月6日" in g1_cons_prompt
            assert "上海" not in g1_cons_prompt
            assert "沪漂阿哲" not in g1_cons_prompt
            assert "私下透露" not in g1_cons_prompt
            assert f'"id": {private_shanghai_mem.id}' not in g1_cons_prompt

            # Verify GroupProfile, PublicUserProfiles, Kyoto memory, and group_jargon.
            gp = await bot.memory_store.get_group_profile(str(GROUP_ID))
            assert gp is not None
            assert "轻松熟人闲聊" in gp.interaction_style

            pub_azhe_after_g1 = await bot.memory_store.get_public_user_profile(
                str(AZHE_UID)
            )
            assert pub_azhe_after_g1 is not None
            assert {"阿哲", "哲哥"}.issubset(set(pub_azhe_after_g1.aliases))
            assert "上海" not in pub_azhe_after_g1.recent_focus
            assert "沪漂阿哲" not in pub_azhe_after_g1.aliases

            # Group consolidation must NEVER overwrite existing private_notes!
            full_azhe_after_g1 = await bot.memory_store.get_user_profile(str(AZHE_UID))
            assert full_azhe_after_g1 is not None
            assert (
                full_azhe_after_g1.private_notes
                == "私下透露拿到上海 offer 准备跳槽，需保密"
            )

            pub_xiaoxia = await bot.memory_store.get_public_user_profile(
                str(XIAOXIA_UID)
            )
            assert pub_xiaoxia is not None
            assert {"小夏", "夏夏"}.issubset(set(pub_xiaoxia.aliases))

            g_mems_ep1 = await bot.memory_store.list_active_memories(
                group_id=str(GROUP_ID),
                active_speaker_ids=[str(AZHE_UID), str(XIAOXIA_UID)],
            )
            kyoto_mem = next(m for m in g_mems_ep1 if "京都" in m.content)
            gaming_mem = next(m for m in g_mems_ep1 if "排位上分" in m.content)
            assert kyoto_mem.event_time == "2026-12-06"
            assert kyoto_mem.visibility == "public"
            assert kyoto_mem.embedding is not None
            assert gaming_mem.evidence_count == 1

            jargon_list = await bot.memory_store.list_group_jargon(str(GROUP_ID))
            assert len(jargon_list) == 1
            assert jargon_list[0].term == "拉满"
            assert jargon_list[0].meaning == "通宵排位上分"

            # ==============================================================
            # ACT 4: Memory Evolution (reinforce + supersede), Online Jargon
            #        <system_reminder>, Autonomous recall_memory + Anti-Leak,
            #        Retention Cleanup & traces.jsonl Audit
            # ==============================================================
            # Episode 3 (hard timeout >= 6h): 阿哲 cancels Kyoto for Hokkaido skiing.
            t_act4_1 = t_act3_4 + 8 * 3600
            wall_clock[0] = t_act4_1
            await dispatch_event(
                _group_event(
                    msg_id=4001,
                    user_id=AZHE_UID,
                    sender_name="阿哲",
                    timestamp=t_act4_1,
                    segments=[
                        {"type": "at", "data": {"qq": str(BOT_SELF_ID)}},
                        {
                            "type": "text",
                            "data": {
                                "text": (
                                    " 临时改行程了！京都取消不去了，"
                                    "改成 12月12日 去北海道滑雪！今晚继续排位上分"
                                )
                            },
                        },
                    ],
                )
            )
            await bot.consolidator.flush()

            # Close Episode 3 via hard timeout (Episode 4) & flush consolidation so
            # reinforce + supersede land in SQLite before the final recall episode.
            t_act4_2 = t_act4_1 + 8 * 3600
            wall_clock[0] = t_act4_2
            await dispatch_event(
                _group_event(
                    msg_id=4002,
                    user_id=AZHE_UID,
                    sender_name="阿哲",
                    timestamp=t_act4_2,
                    segments=[
                        {"type": "at", "data": {"qq": str(BOT_SELF_ID)}},
                        {"type": "text", "data": {"text": " 早安打卡"}},
                    ],
                )
            )
            await bot.consolidator.flush()

            # Verify reinforce incremented evidence_count and supersede atomically
            # replaced Kyoto with active Hokkaido memory.
            gaming_after = await bot.memory_store.get_memory(gaming_mem.id)
            assert gaming_after is not None
            assert gaming_after.evidence_count == 2

            kyoto_after = await bot.memory_store.get_memory(kyoto_mem.id)
            assert kyoto_after is not None
            assert kyoto_after.status == "superseded"
            assert kyoto_after.superseded_by is not None

            hokkaido_mem = await bot.memory_store.get_memory(kyoto_after.superseded_by)
            assert hokkaido_mem is not None
            assert hokkaido_mem.status == "active"
            assert "北海道滑雪" in hokkaido_mem.content
            assert hokkaido_mem.event_time == "2026-12-12"
            assert hokkaido_mem.embedding is not None
            assert hokkaido_mem.last_accessed_at == 0

            # Final Group Episode (Episode 5, 9 days after T0 -> hard timeout cut).
            # Seed an ambient message from 阿哲 60s before 小夏 so BOTH 阿哲(10001)
            # and 小夏(10002) are in active_speaker_ids during recall_memory,
            # proving 阿哲's private Shanghai memory STILL cannot leak!
            t_act4_final = t0 + 9 * 86400
            wall_clock[0] = t_act4_final

            ep5_azhe_leadin = await dispatch_event(
                _group_event(
                    msg_id=4003,
                    user_id=AZHE_UID,
                    sender_name="阿哲",
                    timestamp=t_act4_final - 60,
                    segments=[{"type": "text", "data": {"text": "晚上好呀大家"}}],
                )
            )
            assert ep5_azhe_leadin is not None

            chat_req_idx_before_final = len(llm_chat_requests)
            rerank_count_before_final = len(jev_rerank_requests)
            await dispatch_event(
                _group_event(
                    msg_id=4004,
                    user_id=XIAOXIA_UID,
                    sender_name="小夏",
                    timestamp=t_act4_final,
                    segments=[
                        {"type": "at", "data": {"qq": str(BOT_SELF_ID)}},
                        {
                            "type": "text",
                            "data": {
                                "text": (
                                    " 今晚继续拉满！阿哲最后决定去哪儿玩？"
                                    "他最近工作上有跳槽打算吗？"
                                )
                            },
                        },
                    ],
                )
            )
            await bot.consolidator.flush()

            # Verify system prompt has frozen ## 当前群聊画像 (NO <system_reminder>),
            # while ephemeral last user turn carries <system_reminder> for "拉满".
            final_step1_req = llm_chat_requests[chat_req_idx_before_final]
            final_sys_prompt = final_step1_req["messages"][0]["content"]
            assert "## 当前群聊画像" in final_sys_prompt
            assert "轻松熟人闲聊" in final_sys_prompt
            assert "<system_reminder>" not in final_sys_prompt

            final_user_turn = final_step1_req["messages"][-1]["content"]
            assert "<system_reminder>" in final_user_turn
            assert "拉满：通宵排位上分" in final_user_turn

            # SQLite messages table must NEVER persist <system_reminder>!
            leaked_reminder_row = await db.fetchone(
                "SELECT COUNT(*) AS cnt FROM messages "
                "WHERE content LIKE '%<system_reminder>%'"
            )
            assert leaked_reminder_row is not None
            assert int(leaked_reminder_row["cnt"]) == 0

            # Verify recall_memory executed vector + Jev rerank, returned ONLY active
            # Hokkaido memory (no superseded Kyoto, no private Shanghai), and updated
            # last_accessed_at.
            assert len(jev_rerank_requests) == rerank_count_before_final + 1
            rerank_candidates = jev_rerank_requests[-1]["state"]["candidates"]
            assert any("北海道滑雪" in c for c in rerank_candidates)
            assert all("京都看红叶" not in c for c in rerank_candidates)
            assert all("上海" not in c for c in rerank_candidates)

            final_step2_req = llm_chat_requests[chat_req_idx_before_final + 1]
            recall_tool_msg = final_step2_req["messages"][-1]
            assert recall_tool_msg["role"] == "tool"
            recalled_payload = json.loads(recall_tool_msg["content"])
            recalled_contents = [m["content"] for m in recalled_payload["memories"]]
            assert len(recalled_contents) == 1
            assert "北海道滑雪" in recalled_contents[0]
            assert all("京都看红叶" not in c for c in recalled_contents)
            assert all("上海" not in c for c in recalled_contents)

            hokkaido_accessed = await bot.memory_store.get_memory(hokkaido_mem.id)
            assert hokkaido_accessed is not None
            assert hokkaido_accessed.last_accessed_at > 0

            # Retention cleanup: prune expired already-extracted messages (> 7 days old
            # & behind min_protected_message_id) while keeping active Episode 5 intact.
            deleted_count = await bot.consolidator.maybe_cleanup_expired_messages(
                force=True
            )
            assert deleted_count > 0

            # Act 1 segmented reply row and its cascaded message_platform_ids are gone.
            assert (
                await db.fetchone(
                    "SELECT id FROM messages WHERE id = ?", (bot_ep1_row.id,)
                )
                is None
            )
            assert (
                await count_rows(
                    "SELECT COUNT(*) FROM message_platform_ids WHERE message_id = ?",
                    (bot_ep1_row.id,),
                )
                == 0
            )
            for old_pid in ("90001", "90002", "90003"):
                assert (
                    await bot.message_store.resolve_platform_message_id(
                        platform_message_id=old_pid,
                        group_id=str(GROUP_ID),
                        user_id=str(XIAOXIA_UID),
                    )
                    is None
                )

            # Active Episode 5 window messages remain completely intact.
            remaining_group = await bot.message_store.get_group_messages(
                str(GROUP_ID),
                limit=50,
            )
            remaining_ids = {r.id for r in remaining_group}
            assert ep5_azhe_leadin.db_id in remaining_ids
            assert any("阿哲最后决定去哪儿玩" in r.content for r in remaining_group)
            assert any("北海道滑雪" in r.content for r in remaining_group)

            # --------------------------------------------------------------
            # Audit traces.jsonl end-to-end
            # --------------------------------------------------------------
            trace_lines = (
                cfg.trace.path.read_text(encoding="utf-8").strip().splitlines()
            )
            traces = [json.loads(line) for line in trace_lines]
            assert len(traces) == trace_count
            assert all(t["error"] is None for t in traces)

            expected_signatures = [
                # Act 1
                (f"group:{GROUP_ID}", False, "filtered_at_other", None, None, None, []),
                (
                    f"group:{GROUP_ID}",
                    False,
                    "filtered_jev_rejected",
                    None,
                    None,
                    None,
                    [],
                ),
                (
                    f"group:{GROUP_ID}",
                    True,
                    "mentioned",
                    1,
                    False,
                    "initial",
                    ["web_search_tavily"],
                ),
                (
                    f"group:{GROUP_ID}",
                    False,
                    "filtered_jev_rejected",
                    None,
                    None,
                    None,
                    [],
                ),
                # Act 2 (private DMs)
                (f"private:{AZHE_UID}", True, "private", 1, False, "initial", []),
                (
                    f"private:{AZHE_UID}",
                    True,
                    "private",
                    2,
                    True,
                    "jev_new_episode",
                    [],
                ),
                (
                    f"private:{AZHE_UID}",
                    True,
                    "private",
                    3,
                    True,
                    "jev_new_episode",
                    [],
                ),
                # Act 3 (quote shortcut, Jev continue, rollover within_idle, cut)
                (
                    f"group:{GROUP_ID}",
                    True,
                    "mentioned",
                    1,
                    False,
                    "in_window_reply",
                    [],
                ),
                (
                    f"group:{GROUP_ID}",
                    True,
                    "mentioned",
                    1,
                    False,
                    "jev_continue",
                    [],
                ),
                (
                    f"group:{GROUP_ID}",
                    True,
                    "mentioned",
                    1,
                    False,
                    "within_idle",
                    [],
                ),
                (
                    f"group:{GROUP_ID}",
                    True,
                    "mentioned",
                    2,
                    True,
                    "jev_new_episode",
                    [],
                ),
                # Act 4 (Hokkaido cut, morning cut, ambient lead-in, final recall)
                (
                    f"group:{GROUP_ID}",
                    True,
                    "mentioned",
                    3,
                    True,
                    "hard_timeout",
                    [],
                ),
                (
                    f"group:{GROUP_ID}",
                    True,
                    "mentioned",
                    4,
                    True,
                    "hard_timeout",
                    [],
                ),
                (
                    f"group:{GROUP_ID}",
                    False,
                    "filtered_jev_rejected",
                    None,
                    None,
                    None,
                    [],
                ),
                (
                    f"group:{GROUP_ID}",
                    True,
                    "mentioned",
                    5,
                    True,
                    "hard_timeout",
                    ["recall_memory"],
                ),
            ]
            actual_signatures = [
                (
                    t["session_key"],
                    t["should_reply"],
                    t["reason"],
                    t["episode_id"],
                    t["new_episode"],
                    t["continuity_reason"],
                    t["tool_calls"],
                )
                for t in traces
            ]
            assert actual_signatures == expected_signatures
            assert all(t["reply_sent"] is True for t in traces if t["should_reply"])
    finally:
        await web_search.close()
        await tavily_http.aclose()
        await embedding_client.close()
        await embed_http.aclose()
        await jev_client.close()
        await jev_http.aclose()
        await llm_client.close()
        await llm_sdk.close()
        await llm_http.aclose()
        await captioner.close()
        await vision_sdk.close()
        await vision_http.aclose()
        await media_http.aclose()
        await db.close()
