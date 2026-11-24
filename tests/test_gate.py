"""Unit and integration tests for spectra.gate."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from spectra.config import AppConfig
from spectra.db import Database
from spectra.gate import EntryGate, GateDecision
from spectra.jev import JevClient
from spectra.messages import MessageStore
from spectra.napcat import InboundMessage


class FakeClock:
    """Deterministic monotonic clock + async sleep driver for debounce tests."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now: float = start
        self._waiters: list[tuple[float, asyncio.Future[None]]] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, delay: float) -> None:
        if delay <= 0.0:
            await asyncio.sleep(0)
            return
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[None] = loop.create_future()
        self._waiters.append((self.now + delay, fut))
        await fut

    async def advance(self, dt: float) -> None:
        target = self.now + dt
        while True:
            await asyncio.sleep(0)
            ready = [
                (wake, fut)
                for wake, fut in self._waiters
                if wake <= target and not fut.done()
            ]
            if not ready:
                break
            next_wake = min(wake for wake, _ in ready)
            self.now = max(self.now, next_wake)
            still_waiting: list[tuple[float, asyncio.Future[None]]] = []
            for wake, fut in self._waiters:
                if fut.done():
                    continue
                if wake <= self.now:
                    fut.set_result(None)
                else:
                    still_waiting.append((wake, fut))
            self._waiters = still_waiting
            await asyncio.sleep(0)
        self.now = target
        await asyncio.sleep(0)

    async def settle(self, turns: int = 8) -> None:
        for _ in range(turns):
            await asyncio.sleep(0)


class DecisionRecorder:
    """Thread/task-safe decision recorder with deterministic async waiting."""

    def __init__(self) -> None:
        self.items: list[GateDecision] = []
        self._event = asyncio.Event()

    def __call__(self, decision: GateDecision) -> None:
        self.items.append(decision)
        self._event.set()

    async def wait_count(self, count: int) -> GateDecision:
        async with asyncio.timeout(2.0):
            while len(self.items) < count:
                self._event.clear()
                if len(self.items) >= count:
                    break
                await self._event.wait()
            return self.items[count - 1]


@pytest.fixture
async def message_store(tmp_path: Path) -> MessageStore:
    db = Database(tmp_path / "gate.db")
    await db.connect()
    store = MessageStore(db)
    await store.init()
    try:
        yield store
    finally:
        await db.close()


async def _persist_inbound(
    store: MessageStore,
    *,
    group_id: str | None,
    user_id: str,
    sender_name: str,
    content: str,
    message_id: str,
    self_id: str = "999999",
    is_mentioned_bot: bool = False,
    starts_with_at_other: bool = False,
    timestamp: int = 1700000000,
) -> InboundMessage:
    db_id = await store.append_message(
        group_id=group_id,
        user_id=user_id,
        sender_name=sender_name,
        role="user",
        content=content,
        created_at=timestamp,
    )
    return InboundMessage(
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


def _make_jev_client(
    cfg: AppConfig,
    handler: Any,
) -> tuple[httpx.AsyncClient, JevClient]:
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return http, JevClient(cfg.jev, http_client=http)


async def test_private_debounce_and_jev_state_building(
    message_store: MessageStore,
) -> None:
    clock = FakeClock()
    captured_payloads: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_payloads.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "model": "jev-1.14",
                "usage": {"input_tokens": 99},
                "answers": {
                    "effort": {
                        "probabilities": {"default": 0.2, "high": 0.8},
                    }
                },
            },
        )

    cfg = AppConfig(
        napcat={"bot_name": "Spectra"},
        jev={"api_key": "sk-jev", "model": "jev-1.14"},
        gate={"private_debounce_sec": 1.5, "state_message_limit": 2},
    )
    # Seed 3 prior messages (2 user + 1 assistant); limit=2 should keep last 2 only.
    await message_store.append_message(
        group_id=None,
        user_id="1001",
        sender_name="Alice",
        role="user",
        content="很早以前的消息",
    )
    await message_store.append_message(
        group_id=None,
        user_id="1001",
        sender_name="Alice",
        role="user",
        content="上一句",
    )
    await message_store.append_message(
        group_id=None,
        user_id="1001",
        sender_name="Spectra",
        role="assistant",
        content="我收到了",
    )

    decisions = DecisionRecorder()
    triggered: list[GateDecision] = []
    trigger_done = asyncio.Event()

    async def on_trigger(dec: GateDecision) -> bool:
        triggered.append(dec)
        trigger_done.set()
        return True

    http, jev_client = _make_jev_client(cfg, handler)
    async with (
        http,
        EntryGate(
            cfg,
            message_store,
            jev_client=jev_client,
            on_trigger=on_trigger,
            on_decision=decisions,
            clock_fn=clock,
            sleep_fn=clock.sleep,
        ) as gate,
    ):
        m1 = await _persist_inbound(
            message_store,
            group_id=None,
            user_id="1001",
            sender_name="Alice",
            content="帮我看个报错",
            message_id="pm1",
        )
        await gate.submit(m1)
        await clock.advance(0.8)
        assert not triggered

        m2 = await _persist_inbound(
            message_store,
            group_id=None,
            user_id="1001",
            sender_name="Alice",
            content="TypeError: NoneType is not iterable",
            message_id="pm2",
        )
        await gate.submit(m2)
        await clock.advance(1.4)
        assert not triggered

        await clock.advance(0.1)
        await asyncio.wait_for(trigger_done.wait(), timeout=2.0)
        assert len(triggered) == 1
        assert len(decisions.items) == 1

    dec = triggered[0]
    assert dec.session_key == "private:1001"
    assert dec.group_id is None
    assert dec.user_id == "1001"
    assert dec.reply_to_msg_id == "pm2"
    assert [m.message_id for m in dec.messages] == ["pm1", "pm2"]
    assert dec.should_reply is True
    assert dec.shadow is False
    assert dec.effort == "high"
    assert dec.reason == "private"
    assert dec.jev is not None and dec.jev.degraded is False

    assert len(captured_payloads) == 1
    assert list(captured_payloads[0]["questions"].keys()) == ["effort"]
    state = captured_payloads[0]["state"]
    assert state["bot_name"] == "Spectra"
    assert state["recent_messages"] == [
        "[Alice]: 上一句",
        "[Spectra(Bot)]: 我收到了",
    ]
    assert state["latest_batch"] == [
        "[Alice]: 帮我看个报错",
        "[Alice]: TypeError: NoneType is not iterable",
    ]


async def test_group_debounce_max_batch_wait_anti_starvation(
    message_store: MessageStore,
) -> None:
    clock = FakeClock()
    cfg = AppConfig(
        gate={
            "group_debounce_sec": 2.5,
            "max_batch_wait_sec": 6.0,
            "active_reply": {"possibility_reply": 1.0},
        }
    )
    triggered: list[GateDecision] = []

    async def on_trigger(dec: GateDecision) -> None:
        triggered.append(dec)

    async with EntryGate(
        cfg,
        message_store,
        on_trigger=on_trigger,
        clock_fn=clock,
        sleep_fn=clock.sleep,
        rand_fn=lambda: 0.0,
    ) as gate:
        # Send 5 messages 1.5s apart; max_batch_wait_sec prevents starvation.
        for idx in range(5):
            msg = await _persist_inbound(
                message_store,
                group_id="g1",
                user_id=f"u{idx}",
                sender_name=f"User{idx}",
                content=f"msg-{idx}",
                message_id=f"gm{idx}",
            )
            await gate.submit(msg)
            if idx < 4:
                await clock.advance(1.5)
                if idx < 3:
                    assert not triggered

        # At t0 + 6.0s the batch MUST flush despite message 4 arriving at t0 + 6.0s.
        await clock.settle()
        assert len(triggered) == 1
        assert [m.message_id for m in triggered[0].messages] == [
            "gm0",
            "gm1",
            "gm2",
            "gm3",
        ]
        # Message 4 arrived right at t0 + 6.0s so it starts a fresh batch.
        await clock.advance(2.5)
        assert len(triggered) == 2
        assert [m.message_id for m in triggered[1].messages] == ["gm4"]


async def test_batch_mentioned_picks_latest_mentioning_message(
    message_store: MessageStore,
) -> None:
    clock = FakeClock()
    cfg = AppConfig(gate={"group_debounce_sec": 2.5})
    triggered: list[GateDecision] = []

    async def on_trigger(dec: GateDecision) -> None:
        triggered.append(dec)

    async with EntryGate(
        cfg,
        message_store,
        on_trigger=on_trigger,
        clock_fn=clock,
        sleep_fn=clock.sleep,
    ) as gate:
        for mid, uid, mentioned in (
            ("m1", "u1", False),
            ("m2", "u2", True),
            ("m3", "u3", True),
            ("m4", "u4", False),
        ):
            msg = await _persist_inbound(
                message_store,
                group_id="g1",
                user_id=uid,
                sender_name=uid,
                content=f"text-{mid}",
                message_id=mid,
                is_mentioned_bot=mentioned,
            )
            await gate.submit(msg)

        await clock.advance(2.5)
        assert len(triggered) == 1
        dec = triggered[0]
        assert dec.batch_mentioned is True
        assert dec.reason == "mentioned"
        assert dec.reply_to_msg_id == "m3"
        assert dec.user_id == "u3"
        assert dec.effort == "default"
        assert [m.message_id for m in dec.messages] == ["m1", "m2", "m3", "m4"]


async def test_busy_session_drops_group_chatter_but_queues_mentions_and_dms(
    message_store: MessageStore,
) -> None:
    clock = FakeClock()
    cfg = AppConfig(
        gate={
            "private_debounce_sec": 1.0,
            "group_debounce_sec": 1.0,
            "active_reply": {"possibility_reply": 1.0},
        }
    )

    first_started = asyncio.Event()
    release_first = asyncio.Event()
    group_triggers: list[GateDecision] = []
    group_decisions: list[GateDecision] = []

    async def on_group_trigger(dec: GateDecision) -> None:
        group_triggers.append(dec)
        if len(group_triggers) == 1:
            first_started.set()
            await release_first.wait()

    async with EntryGate(
        cfg,
        message_store,
        on_trigger=on_group_trigger,
        on_decision=group_decisions.append,
        clock_fn=clock,
        sleep_fn=clock.sleep,
        rand_fn=lambda: 0.0,
    ) as gate:
        m1 = await _persist_inbound(
            message_store,
            group_id="g1",
            user_id="u1",
            sender_name="A",
            content="@Bot start",
            message_id="m1",
            is_mentioned_bot=True,
        )
        await gate.submit(m1)
        await clock.advance(1.0)
        await first_started.wait()

        # Unmentioned chatter while busy -> immediately dropped as filtered_busy.
        m_chatter = await _persist_inbound(
            message_store,
            group_id="g1",
            user_id="u2",
            sender_name="B",
            content="路过水一句",
            message_id="m_chatter",
            is_mentioned_bot=False,
        )
        await gate.submit(m_chatter)
        assert any(
            d.reason == "filtered_busy" and d.messages == (m_chatter,)
            for d in group_decisions
        )

        # @Bot message while busy -> buffered; follow-up unmentioned appends to it!
        m_at = await _persist_inbound(
            message_store,
            group_id="g1",
            user_id="u3",
            sender_name="C",
            content="@Bot 第二个问题",
            message_id="m_at",
            is_mentioned_bot=True,
        )
        await gate.submit(m_at)
        await clock.advance(1.0)  # debounce expires while first trigger still running

        # Arrives while waiting for first on_trigger to finish -> coalesces!
        m_followup = await _persist_inbound(
            message_store,
            group_id="g1",
            user_id="u3",
            sender_name="C",
            content="补充一下细节",
            message_id="m_followup",
            is_mentioned_bot=False,
        )
        await gate.submit(m_followup)
        assert len(group_triggers) == 1

        release_first.set()
        await clock.settle()
        assert len(group_triggers) == 2
        assert [m.message_id for m in group_triggers[1].messages] == [
            "m_at",
            "m_followup",
        ]
        assert group_triggers[1].reply_to_msg_id == "m_at"

    # Private DM queuing while busy.
    dm_first_started = asyncio.Event()
    dm_release = asyncio.Event()
    dm_triggers: list[GateDecision] = []

    async def on_dm_trigger(dec: GateDecision) -> None:
        dm_triggers.append(dec)
        if len(dm_triggers) == 1:
            dm_first_started.set()
            await dm_release.wait()

    async with EntryGate(
        cfg,
        message_store,
        on_trigger=on_dm_trigger,
        clock_fn=clock,
        sleep_fn=clock.sleep,
    ) as gate:
        d1 = await _persist_inbound(
            message_store,
            group_id=None,
            user_id="u9",
            sender_name="DM",
            content="first",
            message_id="d1",
        )
        await gate.submit(d1)
        await clock.advance(1.0)
        await dm_first_started.wait()

        d2 = await _persist_inbound(
            message_store,
            group_id=None,
            user_id="u9",
            sender_name="DM",
            content="second",
            message_id="d2",
        )
        d3 = await _persist_inbound(
            message_store,
            group_id=None,
            user_id="u9",
            sender_name="DM",
            content="third",
            message_id="d3",
        )
        await gate.submit(d2)
        await clock.advance(1.0)
        await gate.submit(d3)
        assert len(dm_triggers) == 1

        dm_release.set()
        await clock.settle()
        assert len(dm_triggers) == 2
        assert [m.message_id for m in dm_triggers[1].messages] == ["d2", "d3"]


async def test_local_prefilters_off_whitelist_and_at_other(
    message_store: MessageStore,
) -> None:
    clock = FakeClock()
    jev_calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal jev_calls
        jev_calls += 1
        return httpx.Response(
            200,
            json={
                "answers": {
                    "should_reply": {"noul": 0.99},
                    "effort": {"probabilities": {"default": 0.9, "high": 0.1}},
                }
            },
        )

    # 1. proactive_mode = "off"
    cfg_off = AppConfig(
        jev={"api_key": "sk-jev"},
        gate={"proactive_mode": "off", "group_debounce_sec": 1.0},
    )
    decisions: list[GateDecision] = []
    http, jev = _make_jev_client(cfg_off, handler)
    async with (
        http,
        EntryGate(
            cfg_off,
            message_store,
            jev_client=jev,
            on_decision=decisions.append,
            clock_fn=clock,
            sleep_fn=clock.sleep,
        ) as gate,
    ):
        m = await _persist_inbound(
            message_store,
            group_id="g1",
            user_id="u1",
            sender_name="A",
            content="hi",
            message_id="m1",
        )
        await gate.submit(m)
        await clock.advance(1.0)
    assert decisions[-1].reason == "filtered_proactive_off"
    assert jev_calls == 0

    # 2. active_reply.whitelist & @other rules
    cfg_wl = AppConfig(
        jev={"api_key": "sk-jev"},
        gate={
            "proactive_mode": "on",
            "group_debounce_sec": 1.0,
            "proactive_cooldown_sec": 0.0,
            "active_reply": {"whitelist": ["g-ok"]},
        },
    )
    wl_decisions = DecisionRecorder()
    http2, jev2 = _make_jev_client(cfg_wl, handler)
    async with (
        http2,
        EntryGate(
            cfg_wl,
            message_store,
            jev_client=jev2,
            on_decision=wl_decisions,
            clock_fn=clock,
            sleep_fn=clock.sleep,
        ) as gate,
    ):
        # Not in whitelist -> filtered_whitelist
        m_blocked = await _persist_inbound(
            message_store,
            group_id="g-no",
            user_id="u1",
            sender_name="A",
            content="hi",
            message_id="mb",
        )
        await gate.submit(m_blocked)
        await clock.advance(1.0)
        assert (await wl_decisions.wait_count(1)).reason == "filtered_whitelist"
        assert jev_calls == 0

        # Same speaker split messages starting with @other -> filtered_at_other
        m_a1 = await _persist_inbound(
            message_store,
            group_id="g-ok",
            user_id="u1",
            sender_name="A",
            content="@Bob",
            message_id="ma1",
            starts_with_at_other=True,
        )
        m_a2 = await _persist_inbound(
            message_store,
            group_id="g-ok",
            user_id="u1",
            sender_name="A",
            content="你明天来吗",
            message_id="ma2",
            starts_with_at_other=False,
        )
        await gate.submit(m_a1)
        await gate.submit(m_a2)
        await clock.advance(1.0)
        assert (await wl_decisions.wait_count(2)).reason == "filtered_at_other"
        assert jev_calls == 0

        # Different speaker joins after @other -> NOT filtered_at_other!
        m_b1 = await _persist_inbound(
            message_store,
            group_id="g-ok",
            user_id="u1",
            sender_name="A",
            content="@Bob 吃饭吗",
            message_id="mb1",
            starts_with_at_other=True,
        )
        m_b2 = await _persist_inbound(
            message_store,
            group_id="g-ok",
            user_id="u2",
            sender_name="C",
            content="大家晚上吃啥",
            message_id="mb2",
            starts_with_at_other=False,
        )
        await gate.submit(m_b1)
        await gate.submit(m_b2)
        await clock.advance(1.0)
        assert (await wl_decisions.wait_count(3)).reason == "proactive_jev"
        assert jev_calls == 1

        # Last message starts with @other -> filtered_at_other
        m_c1 = await _persist_inbound(
            message_store,
            group_id="g-ok",
            user_id="u2",
            sender_name="C",
            content="大家晚上吃啥",
            message_id="mc1",
            starts_with_at_other=False,
        )
        m_c2 = await _persist_inbound(
            message_store,
            group_id="g-ok",
            user_id="u1",
            sender_name="A",
            content="@Bob 单独问你",
            message_id="mc2",
            starts_with_at_other=True,
        )
        await gate.submit(m_c1)
        await gate.submit(m_c2)
        await clock.advance(1.0)
        assert (await wl_decisions.wait_count(4)).reason == "filtered_at_other"
        assert jev_calls == 1


async def test_cooldown_hot_window_and_hourly_cap(
    message_store: MessageStore,
) -> None:
    clock = FakeClock(start=1000.0)
    cfg = AppConfig(
        gate={
            "proactive_mode": "on",
            "group_debounce_sec": 0.0,
            "proactive_cooldown_sec": 60.0,
            "hot_window_sec": 30.0,
            "proactive_max_per_hour": 3,
            "active_reply": {"possibility_reply": 1.0},
        }
    )
    decisions: list[GateDecision] = []

    async with EntryGate(
        cfg,
        message_store,
        on_decision=decisions.append,
        clock_fn=clock,
        sleep_fn=clock.sleep,
        rand_fn=lambda: 0.0,
    ) as gate:

        async def send_group(
            mid: str,
            *,
            mentioned: bool = False,
        ) -> GateDecision:
            msg = await _persist_inbound(
                message_store,
                group_id="g1",
                user_id="u1",
                sender_name="A",
                content=f"hello-{mid}",
                message_id=mid,
                is_mentioned_bot=mentioned,
            )
            await gate.submit(msg)
            await clock.settle()
            return decisions[-1]

        # t=1000: 1st proactive reply succeeds (count=1, last_bot=1000, last_pro=1000).
        d1 = await send_group("m1")
        assert d1.reason == "proactive_astrbot_prob"

        # t=1020: inside 30s hot window -> bypasses 60s cooldown! (count=2, last=1020).
        await clock.advance(20.0)
        d2 = await send_group("m2")
        assert d2.reason == "proactive_astrbot_prob"

        # t=1055: 35s after t=1020 -> outside hot window, inside 60s cooldown.
        await clock.advance(35.0)
        d3 = await send_group("m3")
        assert d3.reason == "filtered_cooldown"

        # t=1056: @Bot reply opens new 30s hot window without consuming hourly quota!
        await clock.advance(1.0)
        d_at = await send_group("m_at", mentioned=True)
        assert d_at.reason == "mentioned"

        # t=1070: inside @Bot hot window (14s <= 30s), bypasses cooldown (count=3).
        await clock.advance(14.0)
        d4 = await send_group("m4")
        assert d4.reason == "proactive_astrbot_prob"

        # t=1075: inside hot window, but hourly cap=3 reached -> filtered_hourly_cap.
        await clock.advance(5.0)
        d5 = await send_group("m5")
        assert d5.reason == "filtered_hourly_cap"

        # t=4600 (3600s since m1 at 1000): 1 slot frees up and cooldown expires.
        await clock.advance(3525.0)
        d6 = await send_group("m6")
        assert d6.reason == "proactive_astrbot_prob"


async def test_jev_shadow_rejected_and_degraded_modes(
    message_store: MessageStore,
) -> None:
    clock = FakeClock()
    outcomes = iter(
        [
            # shadow approval
            httpx.Response(
                200,
                json={
                    "answers": {
                        "should_reply": {"noul": 0.90},
                        "effort": {"probabilities": {"default": 0.1, "high": 0.9}},
                    }
                },
            ),
            # rejected
            httpx.Response(
                200,
                json={
                    "answers": {
                        "should_reply": {"noul": 0.40},
                        "effort": {"probabilities": {"default": 0.8, "high": 0.2}},
                    }
                },
            ),
            # degraded (HTTP 500)
            httpx.Response(500, text="server error"),
        ]
    )

    cfg = AppConfig(
        jev={
            "api_key": "sk-jev",
            "reply_threshold": 0.65,
            "high_effort_threshold": 0.60,
        },
        gate={
            "proactive_mode": "shadow",
            "group_debounce_sec": 0.0,
            "proactive_cooldown_sec": 60.0,
            "proactive_max_per_hour": 1,
        },
    )
    decisions = DecisionRecorder()
    triggers: list[GateDecision] = []

    async def on_trigger(dec: GateDecision) -> None:
        triggers.append(dec)

    http, jev = _make_jev_client(cfg, lambda _: next(outcomes))
    async with (
        http,
        EntryGate(
            cfg,
            message_store,
            jev_client=jev,
            on_trigger=on_trigger,
            on_decision=decisions,
            clock_fn=clock,
            sleep_fn=clock.sleep,
        ) as gate,
    ):
        for idx, mid in enumerate(("s1", "s2", "s3"), start=1):
            msg = await _persist_inbound(
                message_store,
                group_id="g1",
                user_id="u1",
                sender_name="A",
                content=f"msg-{mid}",
                message_id=mid,
            )
            await gate.submit(msg)
            await decisions.wait_count(idx)

    assert not triggers
    assert [d.reason for d in decisions.items] == [
        "shadow_jev",
        "filtered_jev_rejected",
        "filtered_jev_degraded",
    ]
    assert decisions.items[0].shadow is True
    assert decisions.items[0].should_reply is False
    assert decisions.items[0].effort == "high"
    assert decisions.items[0].jev is not None
    assert decisions.items[0].jev.reply_probability == pytest.approx(0.90)
    # Shadow did NOT consume hourly cap (1) or cooldown (60s); s2/s3 still hit Jev.
    assert (
        decisions.items[2].jev is not None and decisions.items[2].jev.degraded is True
    )


async def test_astrbot_active_reply_fallback_on_and_shadow(
    message_store: MessageStore,
) -> None:
    clock = FakeClock()
    rolls = iter([0.05, 0.25, 0.01])

    # 1. shadow mode
    cfg_shadow = AppConfig(
        gate={
            "proactive_mode": "shadow",
            "group_debounce_sec": 0.0,
            "active_reply": {"possibility_reply": 0.2},
        }
    )
    decisions: list[GateDecision] = []
    triggers: list[GateDecision] = []

    async def on_trigger(dec: GateDecision) -> None:
        triggers.append(dec)

    async with EntryGate(
        cfg_shadow,
        message_store,
        on_trigger=on_trigger,
        on_decision=decisions.append,
        clock_fn=clock,
        sleep_fn=clock.sleep,
        rand_fn=lambda: next(rolls),
    ) as gate:
        for mid in ("p1", "p2"):
            m = await _persist_inbound(
                message_store,
                group_id="g1",
                user_id="u1",
                sender_name="A",
                content=mid,
                message_id=mid,
            )
            await gate.submit(m)
            await clock.settle()

    assert not triggers
    assert [d.reason for d in decisions] == [
        "shadow_astrbot_prob",
        "filtered_prob_rejected",
    ]
    assert decisions[0].shadow is True
    assert decisions[0].should_reply is False
    assert decisions[0].effort == "default"
    assert decisions[0].jev is None

    # 2. possibility_reply = 0.0 never triggers and never needs randomness
    def exploding_rand() -> float:
        raise AssertionError("rand_fn must not be called when possibility_reply == 0.0")

    cfg_zero = AppConfig(
        gate={
            "proactive_mode": "on",
            "group_debounce_sec": 0.0,
            "active_reply": {"possibility_reply": 0.0},
        }
    )
    decisions.clear()
    async with EntryGate(
        cfg_zero,
        message_store,
        on_decision=decisions.append,
        clock_fn=clock,
        sleep_fn=clock.sleep,
        rand_fn=exploding_rand,
    ) as gate:
        m = await _persist_inbound(
            message_store,
            group_id="g1",
            user_id="u1",
            sender_name="A",
            content="zero",
            message_id="pz",
        )
        await gate.submit(m)
        await clock.settle()
    assert decisions[-1].reason == "filtered_prob_rejected"


async def test_on_trigger_false_or_exception_does_not_corrupt_lock_or_quota(
    message_store: MessageStore,
) -> None:
    clock = FakeClock()
    cfg = AppConfig(
        gate={
            "proactive_mode": "on",
            "group_debounce_sec": 0.0,
            "proactive_cooldown_sec": 60.0,
            "hot_window_sec": 0.0,
            "proactive_max_per_hour": 1,
            "active_reply": {"possibility_reply": 1.0},
        }
    )
    behaviors = iter(["false", "raise", "ok"])
    decisions: list[GateDecision] = []
    calls = 0

    async def on_trigger(_: GateDecision) -> bool:
        nonlocal calls
        calls += 1
        mode = next(behaviors)
        if mode == "false":
            return False
        if mode == "raise":
            raise RuntimeError("boom in trigger")
        return True

    async with EntryGate(
        cfg,
        message_store,
        on_trigger=on_trigger,
        on_decision=decisions.append,
        clock_fn=clock,
        sleep_fn=clock.sleep,
        rand_fn=lambda: 0.0,
    ) as gate:
        for mid in ("t1", "t2", "t3", "t4"):
            m = await _persist_inbound(
                message_store,
                group_id="g1",
                user_id="u1",
                sender_name="A",
                content=mid,
                message_id=mid,
            )
            await gate.submit(m)
            await clock.settle()

    # t1 returned False (no quota used), t2 raised (no quota used),
    # t3 returned True (quota=1/1 consumed), t4 hit hourly cap!
    assert calls == 3
    assert [d.reason for d in decisions] == [
        "proactive_astrbot_prob",
        "proactive_astrbot_prob",
        "proactive_astrbot_prob",
        "filtered_hourly_cap",
    ]


async def test_close_cancels_pending_and_inflight_tasks_cleanly(
    message_store: MessageStore,
) -> None:
    clock = FakeClock()
    cfg = AppConfig(
        gate={
            "private_debounce_sec": 10.0,
            "group_debounce_sec": 10.0,
        }
    )
    in_trigger = asyncio.Event()
    cancelled = False

    async def on_trigger(_: GateDecision) -> None:
        nonlocal cancelled
        in_trigger.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled = True
            raise

    gate = EntryGate(
        cfg,
        message_store,
        on_trigger=on_trigger,
        clock_fn=clock,
        sleep_fn=clock.sleep,
    )
    m_pending = await _persist_inbound(
        message_store,
        group_id="g-pending",
        user_id="u1",
        sender_name="A",
        content="pending",
        message_id="p1",
    )
    await gate.submit(m_pending)

    m_running = await _persist_inbound(
        message_store,
        group_id=None,
        user_id="u2",
        sender_name="B",
        content="running",
        message_id="r1",
    )
    await gate.submit(m_running)
    await clock.advance(10.0)
    await in_trigger.wait()

    await gate.close()
    await gate.close()  # idempotent
    assert cancelled is True
    assert not gate._tasks
    assert not gate._sessions

    # Post-close submissions are silently ignored without creating tasks.
    await gate.submit(m_pending)
    assert not gate._tasks
