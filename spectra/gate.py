"""Session debounce, local pre-filters, and Jev/AstrBot entry gate.

Coordinates per-session message batching, busy-session protection, local
zero-cost group pre-filters, System One / Jev proactive + effort routing,
and AstrBot-aligned ``active_reply`` probabilistic fallback.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import random
import time
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Self

from spectra.config import AppConfig
from spectra.jev import EffortLevel, JevClient, JevDecision
from spectra.messages import MessageRecord, MessageStore
from spectra.napcat import InboundMessage

logger = logging.getLogger(__name__)

_HOUR_SECONDS = 3600.0


def _make_session_key(group_id: str | None, user_id: str) -> str:
    if group_id is not None:
        return f"group:{group_id}"
    return f"private:{user_id}"


@dataclass(slots=True, frozen=True)
class GateDecision:
    """Immutable outcome produced by :class:`EntryGate` for a message batch."""

    session_key: str
    group_id: str | None
    user_id: str
    self_id: str
    reply_to_msg_id: str
    messages: tuple[InboundMessage, ...]
    batch_mentioned: bool
    should_reply: bool
    shadow: bool
    effort: EffortLevel
    reason: str
    jev: JevDecision | None = None
    decision_latency_ms: float = 0.0


@dataclass
class _SessionState:
    pending: list[InboundMessage] = field(default_factory=list)
    first_arrival: float | None = None
    deadline: float | None = None
    max_deadline: float | None = None
    debounce_task: asyncio.Task[None] | None = None
    evaluating: bool = False
    generating: bool = False
    state_changed: asyncio.Event = field(default_factory=asyncio.Event)

    def notify_state_changed(self) -> None:
        old = self.state_changed
        self.state_changed = asyncio.Event()
        old.set()


class EntryGate:
    """Async entry gate sitting between :class:`NapCatAdapter` and reply generation.

    Args:
        config: Root :class:`spectra.config.AppConfig`.
        message_store: Initialized :class:`spectra.messages.MessageStore`.
        jev_client: Optional externally managed :class:`JevClient`. When
            omitted, ``EntryGate`` creates and owns one from ``config.jev``.
        on_trigger: Async callback invoked under the per-session generation
            lock whenever ``decision.should_reply`` is ``True``. Returning
            ``False`` indicates no reply was sent and skips bot-reply /
            proactive quota updates.
        on_decision: Optional sync or async observer invoked on every completed
            decision (triggered, shadow, or filtered).
        clock_fn: Monotonic clock function in seconds (injectable for tests).
        sleep_fn: Async sleep function (injectable for tests).
        rand_fn: Uniform ``[0.0, 1.0)`` random generator (injectable for tests).
    """

    def __init__(
        self,
        config: AppConfig,
        message_store: MessageStore,
        jev_client: JevClient | None = None,
        on_trigger: Callable[[GateDecision], Awaitable[bool | None]] | None = None,
        *,
        on_decision: Callable[[GateDecision], Awaitable[None] | None] | None = None,
        clock_fn: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], Awaitable[None]] = asyncio.sleep,
        rand_fn: Callable[[], float] = random.random,
    ) -> None:
        self._config = config
        self._message_store = message_store
        self._jev_client = (
            jev_client if jev_client is not None else JevClient(config.jev)
        )
        self._owns_jev_client = jev_client is None
        self._on_trigger = on_trigger
        self._on_decision = on_decision
        self._clock = clock_fn
        self._sleep = sleep_fn
        self._rand = rand_fn

        self._sessions: dict[str, _SessionState] = {}
        self._tasks: set[asyncio.Task[None]] = set()
        self._last_bot_reply_at: dict[str, float] = {}
        self._last_proactive_at: dict[str, float] = {}
        self._proactive_timestamps: dict[str, deque[float]] = defaultdict(deque)
        self._closed = False

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    async def close(self) -> None:
        """Cancel all pending debounce/dispatch tasks and close owned resources."""
        self._closed = True
        current = asyncio.current_task()
        tasks = [task for task in self._tasks if task is not current]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.difference_update(tasks)
        self._sessions.clear()
        if self._owns_jev_client:
            await self._jev_client.close()

    def _track_task(self, task: asyncio.Task[None]) -> None:
        self._tasks.add(task)
        task.add_done_callback(self._on_task_done)

    def _on_task_done(self, task: asyncio.Task[None]) -> None:
        self._tasks.discard(task)
        if not task.cancelled():
            exc = task.exception()
            if exc is not None:
                logger.error(
                    "Unhandled exception in EntryGate task %s",
                    task.get_name(),
                    exc_info=exc,
                )

    def _maybe_cleanup_session(self, session_key: str) -> None:
        state = self._sessions.get(session_key)
        if state is None:
            return
        if (
            not state.pending
            and not state.evaluating
            and not state.generating
            and (state.debounce_task is None or state.debounce_task.done())
        ):
            self._sessions.pop(session_key, None)

    async def _emit_decision(self, decision: GateDecision) -> None:
        if self._on_decision is None:
            return
        try:
            res = self._on_decision(decision)
            if inspect.isawaitable(res):
                await res
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "Unhandled exception in gate on_decision hook (session=%s, reason=%s)",
                decision.session_key,
                decision.reason,
            )

    async def submit(self, msg: InboundMessage) -> None:
        """Submit a persisted inbound message to the per-session entry gate."""
        if self._closed:
            return

        session_key = _make_session_key(msg.group_id, msg.user_id)
        state = self._sessions.setdefault(session_key, _SessionState())

        has_mentioned_pending = any(m.is_mentioned_bot for m in state.pending)
        if (
            state.generating
            and msg.group_id is not None
            and not msg.is_mentioned_bot
            and not has_mentioned_pending
        ):
            decision = GateDecision(
                session_key=session_key,
                group_id=msg.group_id,
                user_id=msg.user_id,
                self_id=msg.self_id,
                reply_to_msg_id=msg.message_id,
                messages=(msg,),
                batch_mentioned=False,
                should_reply=False,
                shadow=False,
                effort="default",
                reason="filtered_busy",
                jev=None,
                decision_latency_ms=0.0,
            )
            await self._emit_decision(decision)
            return

        now = self._clock()
        debounce_sec = (
            self._config.gate.private_debounce_sec
            if msg.group_id is None
            else self._config.gate.group_debounce_sec
        )
        max_wait = self._config.gate.max_batch_wait_sec

        if not state.pending:
            state.first_arrival = now
            state.max_deadline = now + max_wait
            state.deadline = min(state.max_deadline, now + debounce_sec)
        else:
            assert state.max_deadline is not None
            state.deadline = min(state.max_deadline, now + debounce_sec)
        state.pending.append(msg)

        if state.debounce_task is None or state.debounce_task.done():
            task = asyncio.create_task(
                self._run_debounce(session_key),
                name=f"gate-debounce:{session_key}",
            )
            state.debounce_task = task
            self._track_task(task)

    async def _run_debounce(self, session_key: str) -> None:
        state = self._sessions.get(session_key)
        if state is None:
            return
        current_task = asyncio.current_task()
        owns_evaluation = False
        try:
            if not state.pending or state.first_arrival is None:
                return

            slept_until = state.first_arrival
            while True:
                if self._closed or not state.pending:
                    return
                now = self._clock()
                effective_now = max(now, slept_until)
                target = state.deadline
                if target is None:
                    return
                remaining = target - effective_now
                if remaining > 0.0:
                    await self._sleep(remaining)
                    slept_until = max(slept_until, target)
                    continue

                is_direct_pending = state.pending[0].group_id is None or any(
                    m.is_mentioned_bot for m in state.pending
                )
                if state.evaluating or (is_direct_pending and state.generating):
                    wait_event = state.state_changed
                    await wait_event.wait()
                    continue

                break

            if self._closed or not state.pending:
                return

            batch = tuple(state.pending)
            state.pending.clear()
            state.first_arrival = None
            state.deadline = None
            state.max_deadline = None
            if state.debounce_task is current_task:
                state.debounce_task = None

            state.evaluating = True
            owns_evaluation = True
            await self._evaluate_and_dispatch(session_key, state, batch)
        finally:
            if owns_evaluation and state.evaluating:
                state.evaluating = False
                state.notify_state_changed()
            if state.debounce_task is current_task:
                state.debounce_task = None
            self._maybe_cleanup_session(session_key)

    async def _build_jev_state(
        self,
        *,
        group_id: str | None,
        user_id: str,
        batch: tuple[InboundMessage, ...],
    ) -> dict[str, Any]:
        bot_name = self._config.napcat.bot_name
        limit = self._config.gate.state_message_limit
        min_batch_db_id = min(m.db_id for m in batch)
        max_fetch_limit = max(256, limit * 16)
        fetch_limit = limit + len(batch)
        prior_rows: list[MessageRecord] = []

        while True:
            if group_id is not None:
                rows = await self._message_store.get_group_messages(
                    group_id,
                    limit=fetch_limit,
                )
            else:
                rows = await self._message_store.get_private_messages(
                    user_id,
                    limit=fetch_limit,
                )

            prior_rows = [rec for rec in rows if rec.id < min_batch_db_id]
            if (
                len(prior_rows) >= limit
                or len(rows) < fetch_limit
                or fetch_limit >= max_fetch_limit
            ):
                break
            fetch_limit = min(max_fetch_limit, fetch_limit * 2)

        recent_rows = prior_rows[-limit:]
        recent_messages = [
            (
                f"[{bot_name}(Bot)]: {rec.content}"
                if rec.role == "assistant"
                else f"[{rec.sender_name}]: {rec.content}"
            )
            for rec in recent_rows
        ]
        latest_batch = [f"[{m.sender_name}]: {m.content}" for m in batch]
        return {
            "chat_type": "group" if group_id is not None else "private",
            "bot_name": bot_name,
            "recent_messages": recent_messages,
            "latest_batch": latest_batch,
        }

    async def _evaluate_and_dispatch(
        self,
        session_key: str,
        state: _SessionState,
        batch: tuple[InboundMessage, ...],
    ) -> None:
        started = self._clock()
        group_id = batch[0].group_id
        batch_mentioned = any(m.is_mentioned_bot for m in batch)
        if group_id is not None and batch_mentioned:
            target = next(m for m in reversed(batch) if m.is_mentioned_bot)
        else:
            target = batch[-1]

        def make_decision(
            *,
            should_reply: bool,
            shadow: bool,
            effort: EffortLevel,
            reason: str,
            jev: JevDecision | None = None,
        ) -> GateDecision:
            latency_ms = max(0.0, (self._clock() - started) * 1000.0)
            return GateDecision(
                session_key=session_key,
                group_id=group_id,
                user_id=target.user_id,
                self_id=target.self_id,
                reply_to_msg_id=target.message_id,
                messages=batch,
                batch_mentioned=batch_mentioned,
                should_reply=should_reply,
                shadow=shadow,
                effort=effort,
                reason=reason,
                jev=jev,
                decision_latency_ms=latency_ms,
            )

        # 1. Direct wake: private chat or group batch containing @Bot / reply-to-bot.
        if group_id is None or batch_mentioned:
            jev_res: JevDecision | None = None
            effort: EffortLevel = "default"
            if self._config.jev.is_configured:
                jev_state = await self._build_jev_state(
                    group_id=group_id,
                    user_id=target.user_id,
                    batch=batch,
                )
                jev_res = await self._jev_client.evaluate(
                    jev_state,
                    include_should_reply=False,
                )
                effort = jev_res.effort

            while state.generating:
                if self._closed:
                    return
                wait_event = state.state_changed
                await wait_event.wait()
            if self._closed:
                return

            reason = "private" if group_id is None else "mentioned"
            decision = make_decision(
                should_reply=True,
                shadow=False,
                effort=effort,
                reason=reason,
                jev=jev_res,
            )
            await self._run_trigger_locked(state, decision, is_proactive=False)
            return

        # 2. Unmentioned group chatter: local zero-cost pre-filters in strict order.
        gate_cfg = self._config.gate
        if gate_cfg.proactive_mode == "off":
            await self._emit_decision(
                make_decision(
                    should_reply=False,
                    shadow=False,
                    effort="default",
                    reason="filtered_proactive_off",
                )
            )
            return

        whitelist = gate_cfg.active_reply.whitelist
        if whitelist and group_id not in whitelist:
            await self._emit_decision(
                make_decision(
                    should_reply=False,
                    shadow=False,
                    effort="default",
                    reason="filtered_whitelist",
                )
            )
            return

        if state.generating:
            await self._emit_decision(
                make_decision(
                    should_reply=False,
                    shadow=False,
                    effort="default",
                    reason="filtered_busy",
                )
            )
            return

        if batch[-1].starts_with_at_other or (
            batch[0].starts_with_at_other
            and all(m.user_id == batch[0].user_id for m in batch)
        ):
            await self._emit_decision(
                make_decision(
                    should_reply=False,
                    shadow=False,
                    effort="default",
                    reason="filtered_at_other",
                )
            )
            return

        now = self._clock()
        timestamps = self._proactive_timestamps[group_id]
        while timestamps and (now - timestamps[0]) >= _HOUR_SECONDS:
            timestamps.popleft()
        if len(timestamps) >= gate_cfg.proactive_max_per_hour:
            await self._emit_decision(
                make_decision(
                    should_reply=False,
                    shadow=False,
                    effort="default",
                    reason="filtered_hourly_cap",
                )
            )
            return

        last_bot = self._last_bot_reply_at.get(group_id)
        in_hot_window = (
            gate_cfg.hot_window_sec > 0.0
            and last_bot is not None
            and (now - last_bot) <= gate_cfg.hot_window_sec
        )
        last_proactive = self._last_proactive_at.get(group_id)
        if (
            not in_hot_window
            and last_proactive is not None
            and (now - last_proactive) < gate_cfg.proactive_cooldown_sec
        ):
            await self._emit_decision(
                make_decision(
                    should_reply=False,
                    shadow=False,
                    effort="default",
                    reason="filtered_cooldown",
                )
            )
            return

        # 3. Model / probability gate.
        if self._config.jev.is_configured:
            jev_state = await self._build_jev_state(
                group_id=group_id,
                user_id=target.user_id,
                batch=batch,
            )
            jev_res = await self._jev_client.evaluate(
                jev_state,
                include_should_reply=True,
            )
            if self._closed:
                return
            if jev_res.degraded:
                await self._emit_decision(
                    make_decision(
                        should_reply=False,
                        shadow=False,
                        effort="default",
                        reason="filtered_jev_degraded",
                        jev=jev_res,
                    )
                )
                return
            if not jev_res.should_reply:
                await self._emit_decision(
                    make_decision(
                        should_reply=False,
                        shadow=False,
                        effort=jev_res.effort,
                        reason="filtered_jev_rejected",
                        jev=jev_res,
                    )
                )
                return
            if gate_cfg.proactive_mode == "shadow":
                await self._emit_decision(
                    make_decision(
                        should_reply=False,
                        shadow=True,
                        effort=jev_res.effort,
                        reason="shadow_jev",
                        jev=jev_res,
                    )
                )
                return
            if state.generating:
                await self._emit_decision(
                    make_decision(
                        should_reply=False,
                        shadow=False,
                        effort=jev_res.effort,
                        reason="filtered_busy",
                        jev=jev_res,
                    )
                )
                return
            decision = make_decision(
                should_reply=True,
                shadow=False,
                effort=jev_res.effort,
                reason="proactive_jev",
                jev=jev_res,
            )
            await self._run_trigger_locked(state, decision, is_proactive=True)
            return

        # Fallback: AstrBot active_reply possibility_reply.
        prob = gate_cfg.active_reply.possibility_reply
        passed = False if prob <= 0.0 else (self._rand() < prob)
        if not passed:
            await self._emit_decision(
                make_decision(
                    should_reply=False,
                    shadow=False,
                    effort="default",
                    reason="filtered_prob_rejected",
                )
            )
            return
        if gate_cfg.proactive_mode == "shadow":
            await self._emit_decision(
                make_decision(
                    should_reply=False,
                    shadow=True,
                    effort="default",
                    reason="shadow_astrbot_prob",
                )
            )
            return
        decision = make_decision(
            should_reply=True,
            shadow=False,
            effort="default",
            reason="proactive_astrbot_prob",
        )
        await self._run_trigger_locked(state, decision, is_proactive=True)

    async def _run_trigger_locked(
        self,
        state: _SessionState,
        decision: GateDecision,
        *,
        is_proactive: bool,
    ) -> None:
        state.generating = True
        if state.evaluating:
            state.evaluating = False
            state.notify_state_changed()
        try:
            await self._emit_decision(decision)
            if self._on_trigger is None:
                succeeded = True
            else:
                try:
                    result = await self._on_trigger(decision)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception(
                        "Unhandled exception in gate on_trigger callback "
                        "(session=%s, reason=%s)",
                        decision.session_key,
                        decision.reason,
                    )
                    succeeded = False
                else:
                    succeeded = result is not False

            if succeeded and decision.group_id is not None:
                done_now = self._clock()
                self._last_bot_reply_at[decision.group_id] = done_now
                if is_proactive:
                    self._last_proactive_at[decision.group_id] = done_now
                    self._proactive_timestamps[decision.group_id].append(done_now)
        finally:
            state.generating = False
            state.notify_state_changed()
