"""MVP orchestrator wiring NapCat, Gate, Prompts, Sessions, LLM, and Trace."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, Self

from spectra.config import AppConfig
from spectra.db import Database
from spectra.gate import EntryGate, GateDecision
from spectra.jev import JevClient
from spectra.llm import LLMClient, ToolSpec
from spectra.messages import MessageStore
from spectra.napcat import NapCatAdapter
from spectra.prompts import PromptLoader
from spectra.sessions import SessionStore
from spectra.trace import TraceEntry, TraceLogger
from spectra.vision import ImageCaptioner

logger = logging.getLogger(__name__)


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
        tools: Optional sequence of :class:`spectra.llm.ToolSpec` exposed to the
            agent tool loop.
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
        self._tools: tuple[ToolSpec, ...] = tuple(tools)
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
        cancelled = False

        try:
            system_prompt = self._prompt_loader.load(
                group_id=decision.group_id,
                user_id=decision.user_id,
            )
            up_to_id = max(m.db_id for m in decision.messages)
            context_messages = await self._session_store.build_context(
                session_key=decision.session_key,
                group_id=decision.group_id,
                user_id=decision.user_id,
                system_prompt=system_prompt,
                up_to_id=up_to_id,
            )

            llm_model = self._config.llm.model
            llm_started = self._clock_fn()
            try:
                result = await self._llm_client.run_tool_loop(
                    context_messages,
                    tools=self._tools,
                    effort=decision.effort,
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

            await self._session_store.touch(decision.session_key)
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
                )
                await self._trace_logger.write(entry)

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
