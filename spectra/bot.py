"""MVP orchestrator wiring NapCat, Gate, Prompts, Sessions, LLM, and Trace."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, Self

from spectra.config import AppConfig
from spectra.consolidation import MemoryConsolidator
from spectra.db import Database
from spectra.embedding import EmbeddingClient
from spectra.gate import EntryGate, GateDecision
from spectra.jev import JevClient
from spectra.llm import LLMClient, ToolSpec
from spectra.memory import (
    MemoryStore,
    build_recall_memory_tool,
    format_jargon_reminder,
)
from spectra.messages import MessageStore
from spectra.napcat import NapCatAdapter
from spectra.prompts import PromptLoader
from spectra.sessions import ClosedEpisode, SessionStore
from spectra.trace import TraceEntry, TraceLogger
from spectra.vision import ImageCaptioner
from spectra.web_search import WebSearchToolkit

logger = logging.getLogger(__name__)


def _merge_tools(
    primary: Sequence[ToolSpec],
    secondary: Sequence[ToolSpec],
) -> tuple[ToolSpec, ...]:
    """Merge tool sequences, preserving order and primary precedence on collisions."""
    merged: list[ToolSpec] = []
    seen: set[str] = set()
    for spec in (*primary, *secondary):
        if spec.name in seen:
            continue
        seen.add(spec.name)
        merged.append(spec)
    return tuple(merged)


def _extract_tool_call_names(
    messages: Sequence[Mapping[str, Any]],
    *,
    start_index: int,
) -> list[str]:
    """Extract ordered tool names invoked during ``LLMClient.run_tool_loop``."""
    names: list[str] = []
    for msg in messages[start_index:]:
        if msg.get("role") != "assistant":
            continue
        raw_calls = msg.get("tool_calls")
        if not isinstance(raw_calls, list):
            continue
        for call in raw_calls:
            if not isinstance(call, Mapping):
                continue
            func = call.get("function")
            if isinstance(func, Mapping):
                name = func.get("name")
                if isinstance(name, str) and name:
                    names.append(name)
    return names


class SpectraBot:
    """Main async runtime orchestrating all Spectra-bot MVP subsystems.

    Accepts optional pre-built dependencies so both the production CLI and
    offline end-to-end tests exercise the exact same orchestration code path.

    Args:
        config: Validated root :class:`spectra.config.AppConfig`.
        db: Optional externally managed :class:`spectra.db.Database`.
        captioner: Optional externally managed :class:`spectra.vision.ImageCaptioner`.
        jev_client: Optional externally managed :class:`spectra.jev.JevClient`.
        llm_client: Optional externally managed :class:`spectra.llm.LLMClient`.
        adapter: Optional externally managed :class:`spectra.napcat.NapCatAdapter`.
        web_search: Optional externally managed
            :class:`spectra.web_search.WebSearchToolkit`. When omitted and
            ``config.web_search.enabled`` is ``True``, an owned toolkit is
            created in :meth:`start` and closed in :meth:`close`.
        embedding_client: Optional externally managed
            :class:`spectra.embedding.EmbeddingClient`.
        memory_store: Optional externally managed
            :class:`spectra.memory.MemoryStore`.
        consolidator: Optional externally managed
            :class:`spectra.consolidation.MemoryConsolidator`. When omitted and
            ``on_episode_closed`` is ``None`` while both ``config.memory.enabled``
            and ``config.memory.consolidation_enabled`` are ``True``, an owned
            consolidator is created in :meth:`start` and closed in :meth:`close`.
        on_episode_closed: Optional async callback invoked whenever short-episode
            segmentation closes a non-empty previous episode. Errors raised by
            this callback are logged and never disrupt live replies.
        tools: Optional sequence of :class:`spectra.llm.ToolSpec` exposed to the
            agent tool loop. Caller-provided tools take precedence over web
            search and built-in recall tools on name collisions.
        sleep_fn: Injectable async sleep function for gate debounce and
            segmented replies.
        clock_fn: Injectable monotonic clock function in seconds.
        rand_fn: Injectable uniform ``[0.0, 1.0)`` RNG for AstrBot probability
            fallback.
    """

    def __init__(
        self,
        config: AppConfig,
        *,
        db: Database | None = None,
        captioner: ImageCaptioner | None = None,
        jev_client: JevClient | None = None,
        llm_client: LLMClient | None = None,
        adapter: NapCatAdapter | None = None,
        web_search: WebSearchToolkit | None = None,
        embedding_client: EmbeddingClient | None = None,
        memory_store: MemoryStore | None = None,
        consolidator: MemoryConsolidator | None = None,
        on_episode_closed: Callable[[ClosedEpisode], Awaitable[None]] | None = None,
        tools: Sequence[ToolSpec] = (),
        sleep_fn: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock_fn: Callable[[], float] = time.monotonic,
        rand_fn: Callable[[], float] = random.random,
    ) -> None:
        self._config = config
        self._db = db
        self._owns_db = db is None
        self._captioner = captioner
        self._owns_captioner = captioner is None
        self._jev_client = jev_client
        self._owns_jev_client = jev_client is None
        self._llm_client = llm_client
        self._owns_llm_client = llm_client is None
        self._adapter = adapter
        self._web_search = web_search
        self._owns_web_search = web_search is None
        self._embedding_client = embedding_client
        self._owns_embedding_client = embedding_client is None
        self._memory_store = memory_store
        self._consolidator = consolidator
        self._owns_consolidator = consolidator is None
        self._on_episode_closed = on_episode_closed
        self._base_tools: tuple[ToolSpec, ...] = _merge_tools(tools, ())
        self._tools: tuple[ToolSpec, ...] = self._base_tools
        self._sleep_fn = sleep_fn
        self._clock_fn = clock_fn
        self._rand_fn = rand_fn

        self._message_store: MessageStore | None = None
        self._session_store: SessionStore | None = None
        self._prompt_loader: PromptLoader | None = None
        self._trace_logger: TraceLogger | None = None
        self._gate: EntryGate | None = None

        self._started = False
        self._closed = False
        self._shutdown_requested = False
        self._shutdown_event = asyncio.Event()

    @property
    def config(self) -> AppConfig:
        """Return the active :class:`AppConfig`."""
        return self._config

    @property
    def db(self) -> Database | None:
        """Return the active :class:`Database`, if initialized."""
        return self._db

    @property
    def message_store(self) -> MessageStore | None:
        """Return the active :class:`MessageStore`, if initialized."""
        return self._message_store

    @property
    def session_store(self) -> SessionStore | None:
        """Return the active :class:`SessionStore`, if initialized."""
        return self._session_store

    @property
    def prompt_loader(self) -> PromptLoader | None:
        """Return the active :class:`PromptLoader`, if initialized."""
        return self._prompt_loader

    @property
    def trace_logger(self) -> TraceLogger | None:
        """Return the active :class:`TraceLogger`, if initialized."""
        return self._trace_logger

    @property
    def captioner(self) -> ImageCaptioner | None:
        """Return the active :class:`ImageCaptioner`, if enabled or injected."""
        return self._captioner

    @property
    def jev_client(self) -> JevClient | None:
        """Return the active :class:`JevClient`, if initialized."""
        return self._jev_client

    @property
    def llm_client(self) -> LLMClient | None:
        """Return the active :class:`LLMClient`, if initialized."""
        return self._llm_client

    @property
    def gate(self) -> EntryGate | None:
        """Return the active :class:`EntryGate`, if initialized."""
        return self._gate

    @property
    def adapter(self) -> NapCatAdapter | None:
        """Return the active :class:`NapCatAdapter`, if initialized."""
        return self._adapter

    @property
    def web_search(self) -> WebSearchToolkit | None:
        """Return the active :class:`WebSearchToolkit`, if enabled or injected."""
        return self._web_search

    @property
    def embedding_client(self) -> EmbeddingClient | None:
        """Return the active :class:`EmbeddingClient`, if initialized."""
        return self._embedding_client

    @property
    def memory_store(self) -> MemoryStore | None:
        """Return the active :class:`MemoryStore`, if initialized."""
        return self._memory_store

    @property
    def consolidator(self) -> MemoryConsolidator | None:
        """Return the active :class:`MemoryConsolidator`, if enabled or injected."""
        return self._consolidator

    @property
    def tools(self) -> tuple[ToolSpec, ...]:
        """Return the deduplicated tuple of tools registered for the LLM loop."""
        return self._tools

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    async def start(self) -> None:
        """Initialize all subsystems and wire callbacks cleanly.

        If initialization fails partway through, all already-opened resources
        are closed before re-raising.
        """
        if self._started:
            return
        if self._closed:
            raise RuntimeError("SpectraBot has already been closed.")

        try:
            if self._db is None:
                self._db = Database(self._config.db.path)
            await self._db.connect()

            self._message_store = MessageStore(self._db)
            await self._message_store.init()

            self._session_store = SessionStore(
                self._db,
                self._message_store,
                self._config.session,
            )
            await self._session_store.init()

            if self._captioner is None and self._config.vision.enabled:
                self._captioner = ImageCaptioner(self._db, self._config.vision)
            if self._captioner is not None:
                await self._captioner.init()

            self._prompt_loader = PromptLoader(self._config.session.prompts_dir)
            self._trace_logger = TraceLogger(self._config.trace)

            if self._jev_client is None:
                self._jev_client = JevClient(self._config.jev)
            if self._llm_client is None:
                self._llm_client = LLMClient(self._config.llm)
            if self._embedding_client is None:
                self._embedding_client = EmbeddingClient(self._config.embedding)
            if self._memory_store is None:
                self._memory_store = MemoryStore(
                    self._db,
                    self._config.memory,
                    embedding_client=self._embedding_client,
                    jev_client=self._jev_client,
                )
            await self._memory_store.init()
            if (
                self._consolidator is None
                and self._on_episode_closed is None
                and self._config.memory.enabled
                and self._config.memory.consolidation_enabled
            ):
                self._consolidator = MemoryConsolidator(
                    config=self._config,
                    message_store=self._message_store,
                    session_store=self._session_store,
                    memory_store=self._memory_store,
                    llm_client=self._llm_client,
                )
            if self._web_search is None and self._config.web_search.enabled:
                self._web_search = WebSearchToolkit(self._config.web_search)
            self._tools = _merge_tools(
                self._base_tools,
                self._web_search.build_tools() if self._web_search is not None else (),
            )

            self._gate = EntryGate(
                self._config,
                self._message_store,
                jev_client=self._jev_client,
                on_trigger=self._handle_trigger,
                on_decision=self._handle_decision,
                clock_fn=self._clock_fn,
                sleep_fn=self._sleep_fn,
                rand_fn=self._rand_fn,
            )

            if self._adapter is None:
                self._adapter = NapCatAdapter(
                    self._config,
                    self._message_store,
                    captioner=self._captioner,
                    on_message=self._gate.submit,
                    sleep_fn=self._sleep_fn,
                )
            elif (
                isinstance(self._adapter, NapCatAdapter)
                and self._adapter._on_message is None
            ):
                self._adapter._on_message = self._gate.submit

            self._started = True
        except BaseException:
            await self.close()
            raise

    async def _handle_decision(self, decision: GateDecision) -> None:
        """Record filtered and shadow decisions immediately to the trace log."""
        if decision.should_reply:
            return
        if self._trace_logger is None:
            return
        entry = TraceEntry.from_decision(
            decision,
            total_latency_ms=decision.decision_latency_ms,
        )
        await self._trace_logger.write(entry)

    async def _handle_trigger(self, decision: GateDecision) -> bool:
        """Execute context building, LLM tool loop, outbound send, and tracing."""
        assert self._prompt_loader is not None
        assert self._session_store is not None
        assert self._llm_client is not None
        assert self._adapter is not None
        assert self._trace_logger is not None

        trigger_started = self._clock_fn()
        llm_model: str | None = None
        steps: int | None = None
        tool_calls: list[str] = []
        input_other_tokens: int | None = None
        input_cached_tokens: int | None = None
        output_tokens: int | None = None
        reply_chars: int | None = None
        reply_sent = False
        llm_latency_ms: float | None = None
        error: str | None = None
        episode_id: int | None = None
        new_episode: bool | None = None
        continuity_reason: str | None = None
        continuity_probability: float | None = None
        cancelled = False

        try:

            async def _load_profile() -> str:
                if not self._config.memory.enabled or self._memory_store is None:
                    return ""
                if decision.group_id is not None:
                    return await self._memory_store.render_group_profile(
                        decision.group_id
                    )
                return await self._memory_store.render_private_profile(decision.user_id)

            resolution = await self._session_store.resolve_episode(
                session_key=decision.session_key,
                group_id=decision.group_id,
                user_id=decision.user_id,
                batch_messages=decision.messages,
                jev_client=self._jev_client,
                profile_loader=_load_profile,
                advance_extracted_on_close=(
                    self._on_episode_closed is None and self._consolidator is None
                ),
            )
            episode_id = resolution.session.episode_id
            new_episode = resolution.new_episode
            continuity_reason = resolution.reason
            continuity_probability = resolution.continuity_probability

            if resolution.closed_episode is not None:
                if self._on_episode_closed is not None:
                    try:
                        await self._on_episode_closed(resolution.closed_episode)
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        logger.exception(
                            "Unhandled exception in on_episode_closed callback "
                            "(session=%s, episode=%d)",
                            resolution.closed_episode.session_key,
                            resolution.closed_episode.episode_id,
                        )
                elif self._consolidator is not None:
                    self._consolidator.enqueue(resolution.closed_episode)

            system_prompt = self._prompt_loader.load(
                group_id=decision.group_id,
                user_id=decision.user_id,
            )
            min_keep_id = min(m.db_id for m in decision.messages)
            up_to_id = max(m.db_id for m in decision.messages)
            batch_ts = max(int(m.timestamp) for m in decision.messages)
            context_messages = await self._session_store.build_context(
                session_key=decision.session_key,
                group_id=decision.group_id,
                user_id=decision.user_id,
                system_prompt=system_prompt,
                up_to_id=up_to_id,
                min_keep_id=min_keep_id,
                now=batch_ts,
            )
            if not any(msg.get("role") == "user" for msg in context_messages):
                fallback_content = self._session_store.format_inbound_batch(
                    decision.messages,
                    is_group=decision.group_id is not None,
                )
                if fallback_content:
                    context_messages.append(
                        {"role": "user", "content": fallback_content}
                    )

            turn_tools = self._tools
            if self._config.memory.enabled and self._memory_store is not None:
                active_speaker_ids = await self._collect_active_speaker_ids(
                    decision=decision,
                    window_start_id=resolution.session.window_start_id,
                    up_to_id=up_to_id,
                )
                recall_tool = build_recall_memory_tool(
                    self._memory_store,
                    group_id=decision.group_id,
                    user_id=decision.user_id,
                    active_speaker_ids=active_speaker_ids,
                )
                turn_tools = _merge_tools(self._tools, (recall_tool,))

                if decision.group_id is not None:
                    batch_text = "\n".join(m.content for m in decision.messages)
                    matched_jargon = await self._memory_store.match_jargon_in_text(
                        decision.group_id,
                        batch_text,
                    )
                    reminder = format_jargon_reminder(matched_jargon)
                    if reminder:
                        for idx in range(len(context_messages) - 1, -1, -1):
                            if context_messages[idx].get("role") == "user":
                                base_text = str(
                                    context_messages[idx].get("content") or ""
                                )
                                context_messages[idx] = {
                                    **context_messages[idx],
                                    "content": f"{base_text}\n\n{reminder}",
                                }
                                break

            llm_model = self._config.llm.model
            llm_started = self._clock_fn()
            try:
                result = await self._llm_client.run_tool_loop(
                    context_messages,
                    tools=turn_tools,
                    effort=decision.effort,
                    session_key=decision.session_key,
                )
            finally:
                llm_latency_ms = max(0.0, (self._clock_fn() - llm_started) * 1000.0)
            steps = result.steps
            tool_calls = _extract_tool_call_names(
                result.messages,
                start_index=len(context_messages),
            )
            input_other_tokens = result.usage.input_other
            input_cached_tokens = result.usage.input_cached
            output_tokens = result.usage.output

            reply_text = result.content.strip()
            reply_chars = len(reply_text)
            if not reply_text:
                return False

            reply_to = (
                decision.reply_to_msg_id if decision.group_id is not None else None
            )
            await self._adapter.send_reply(
                reply_text,
                group_id=decision.group_id,
                user_id=decision.user_id,
                reply_to_msg_id=reply_to,
                self_id=decision.self_id,
            )
            reply_sent = True

            await self._session_store.record_turn(
                decision.session_key,
                message_at=batch_ts,
            )
            return True
        except asyncio.CancelledError:
            cancelled = True
            raise
        except Exception as exc:
            logger.exception(
                "Failed to process triggered decision (session=%s, reason=%s)",
                decision.session_key,
                decision.reason,
            )
            error = str(exc) or type(exc).__name__
            return reply_sent
        finally:
            if not cancelled:
                exec_latency_ms = max(
                    0.0,
                    (self._clock_fn() - trigger_started) * 1000.0,
                )
                entry = TraceEntry.from_decision(
                    decision,
                    llm_model=llm_model,
                    steps=steps,
                    tool_calls=tool_calls,
                    input_other_tokens=input_other_tokens,
                    input_cached_tokens=input_cached_tokens,
                    output_tokens=output_tokens,
                    reply_chars=reply_chars,
                    reply_sent=reply_sent,
                    llm_latency_ms=llm_latency_ms,
                    total_latency_ms=decision.decision_latency_ms + exec_latency_ms,
                    error=error,
                    episode_id=episode_id,
                    new_episode=new_episode,
                    continuity_reason=continuity_reason,
                    continuity_probability=continuity_probability,
                )
                await self._trace_logger.write(entry)

    async def _collect_active_speaker_ids(
        self,
        *,
        decision: GateDecision,
        window_start_id: int,
        up_to_id: int,
    ) -> tuple[str, ...]:
        """Collect deduplicated human speaker IDs in the current turn and window."""
        self_id = str(decision.self_id).strip()
        speakers: list[str] = []
        seen: set[str] = set()

        def _add(raw_uid: str) -> None:
            uid = str(raw_uid).strip()
            if uid and uid != self_id and uid not in seen:
                seen.add(uid)
                speakers.append(uid)

        for msg in decision.messages:
            _add(msg.user_id)

        if decision.group_id is None:
            _add(decision.user_id)
            return tuple(speakers)

        if self._message_store is not None and window_start_id <= up_to_id:
            window_rows = await self._message_store.get_messages_in_range(
                group_id=decision.group_id,
                user_id=decision.user_id,
                start_id=window_start_id,
                end_id=up_to_id,
            )
            for rec in window_rows:
                if rec.role == "user" and rec.content.strip():
                    _add(rec.user_id)
        return tuple(speakers)

    def request_shutdown(self) -> None:
        """Signal the bot to stop gracefully (safe to call from signal handlers)."""
        self._shutdown_requested = True
        self._shutdown_event.set()

    async def run(self) -> None:
        """Start subsystems, serve the NapCat reverse WS server, and shut down."""
        await self.start()
        assert self._adapter is not None
        if self._shutdown_requested:
            await self.close()
            return

        adapter_task = asyncio.create_task(
            self._adapter.run(),
            name="spectra-napcat-adapter",
        )
        shutdown_task = asyncio.create_task(
            self._shutdown_event.wait(),
            name="spectra-shutdown-waiter",
        )
        try:
            done, _ = await asyncio.wait(
                {adapter_task, shutdown_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if shutdown_task in done and not adapter_task.done():
                await self._adapter.shutdown()
                await adapter_task
            elif adapter_task in done:
                shutdown_task.cancel()
                await asyncio.gather(shutdown_task, return_exceptions=True)
                adapter_task.result()
        except BaseException:
            for task in (adapter_task, shutdown_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(
                adapter_task,
                shutdown_task,
                return_exceptions=True,
            )
            raise
        finally:
            await self.close()

    async def close(self) -> None:
        """Idempotently shut down all subsystems in strict reverse startup order."""
        if self._closed:
            return
        self._closed = True
        self._started = False
        self._shutdown_requested = True
        self._shutdown_event.set()

        steps: list[tuple[str, Callable[[], Awaitable[Any]]]] = []
        if self._adapter is not None:
            steps.append(("adapter", self._adapter.shutdown))
        if self._gate is not None:
            steps.append(("gate", self._gate.close))
        if self._owns_consolidator and self._consolidator is not None:
            steps.append(("consolidator", self._consolidator.close))
        if self._owns_web_search and self._web_search is not None:
            steps.append(("web_search", self._web_search.close))
        if self._owns_embedding_client and self._embedding_client is not None:
            steps.append(("embedding_client", self._embedding_client.close))
        if self._owns_captioner and self._captioner is not None:
            steps.append(("captioner", self._captioner.close))
        if self._owns_jev_client and self._jev_client is not None:
            steps.append(("jev_client", self._jev_client.close))
        if self._owns_llm_client and self._llm_client is not None:
            steps.append(("llm_client", self._llm_client.close))
        if self._owns_db and self._db is not None:
            steps.append(("db", self._db.close))

        for name, closer in steps:
            try:
                await closer()
            except Exception:
                logger.exception("Error while closing %s during shutdown.", name)
