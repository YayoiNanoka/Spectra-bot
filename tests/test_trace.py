"""Unit tests for spectra.trace."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import pytest

from spectra.config import TraceConfig
from spectra.gate import GateDecision
from spectra.jev import JevDecision
from spectra.napcat import InboundMessage
from spectra.trace import TraceEntry, TraceLogger


def _make_decision(
    *,
    should_reply: bool = True,
    shadow: bool = False,
    reason: str = "mentioned",
    jev: JevDecision | None = None,
) -> GateDecision:
    msg = InboundMessage(
        db_id=42,
        message_id="m-42",
        self_id="99999",
        group_id="10001",
        user_id="20002",
        sender_name="小明",
        content="@Bot 你好",
        is_mentioned_bot=True,
        starts_with_at_other=False,
        timestamp=1700000000,
    )
    return GateDecision(
        session_key="group:10001",
        group_id="10001",
        user_id="20002",
        self_id="99999",
        reply_to_msg_id="m-42",
        messages=(msg,),
        batch_mentioned=True,
        should_reply=should_reply,
        shadow=shadow,
        effort="high" if jev and jev.effort == "high" else "default",
        reason=reason,
        jev=jev,
        decision_latency_ms=12.3456,
    )


async def test_write_utf8_compact_jsonl_and_create_parent_dirs(tmp_path: Path) -> None:
    trace_path = tmp_path / "nested" / "logs" / "traces.jsonl"
    writer = TraceLogger(TraceConfig(enabled=True, path=trace_path))

    decision = _make_decision(
        jev=JevDecision(
            should_reply=True,
            effort="high",
            reply_probability=0.88,
            high_effort_probability=0.77,
            model="jev-1.14",
            input_tokens=120,
            degraded=False,
        )
    )
    entry = TraceEntry.from_decision(
        decision,
        llm_model="gpt-4o",
        steps=2,
        tool_calls=["web_search", "fetch_url"],
        input_other_tokens=300,
        input_cached_tokens=150,
        output_tokens=45,
        reply_chars=18,
        reply_sent=True,
        llm_latency_ms=456.7891,
        total_latency_ms=470.1234,
        error=None,
    )
    await writer.write(entry)

    raw_text = trace_path.read_text(encoding="utf-8")
    lines = raw_text.splitlines()
    assert len(lines) == 1
    assert raw_text.endswith("\n")
    # Compact UTF-8 without ASCII-escaping non-ASCII if present.
    parsed = json.loads(lines[0])
    assert parsed["session_key"] == "group:10001"
    assert parsed["batch_db_ids"] == [42]
    assert parsed["jev_model"] == "jev-1.14"
    assert parsed["jev_reply_probability"] == 0.88
    assert parsed["jev_high_effort_probability"] == 0.77
    assert parsed["jev_input_tokens"] == 120
    assert parsed["jev_degraded"] is False
    assert parsed["llm_model"] == "gpt-4o"
    assert parsed["steps"] == 2
    assert parsed["tool_calls"] == ["web_search", "fetch_url"]
    assert parsed["input_other_tokens"] == 300
    assert parsed["input_cached_tokens"] == 150
    assert parsed["output_tokens"] == 45
    assert parsed["reply_chars"] == 18
    assert parsed["reply_sent"] is True
    assert parsed["decision_latency_ms"] == 12.346
    assert parsed["llm_latency_ms"] == 456.789
    assert parsed["total_latency_ms"] == 470.123
    assert parsed["error"] is None

    # Verify ensure_ascii=False preserves Chinese text directly on disk.
    err_entry = TraceEntry.from_decision(
        _make_decision(should_reply=False, reason="filtered_busy"),
        error="中文错误信息",
    )
    await writer.write(err_entry)
    assert "中文错误信息" in trace_path.read_text(encoding="utf-8")


async def test_disabled_trace_logger_writes_nothing(tmp_path: Path) -> None:
    trace_path = tmp_path / "never_created" / "traces.jsonl"
    writer = TraceLogger(TraceConfig(enabled=False, path=trace_path))
    await writer.write(TraceEntry.from_decision(_make_decision()))
    assert not trace_path.exists()
    assert not trace_path.parent.exists()


async def test_concurrent_writes_produce_valid_lines(tmp_path: Path) -> None:
    trace_path = tmp_path / "traces.jsonl"
    writer = TraceLogger(TraceConfig(enabled=True, path=trace_path))

    entries = [
        TraceEntry.from_decision(
            _make_decision(reason=f"batch-{i}"),
            reply_chars=i,
        )
        for i in range(25)
    ]
    await asyncio.gather(*(writer.write(entry) for entry in entries))

    lines = trace_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 25
    reasons = {json.loads(line)["reason"] for line in lines}
    assert reasons == {f"batch-{i}" for i in range(25)}


async def test_oserror_is_swallowed_with_warning(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Point trace path to a directory so open(..., "a") raises IsADirectoryError.
    dir_path = tmp_path / "dir_as_file"
    dir_path.mkdir()
    writer = TraceLogger(TraceConfig(enabled=True, path=dir_path))

    with caplog.at_level(logging.WARNING):
        await writer.write(TraceEntry.from_decision(_make_decision()))

    assert any("Failed to write trace entry" in rec.message for rec in caplog.records)


async def test_line_separator_unicode_chars_are_escaped_in_jsonl(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "traces.jsonl"
    writer = TraceLogger(TraceConfig(enabled=True, path=trace_path))

    entry = TraceEntry.from_decision(
        _make_decision(should_reply=False, reason="filtered_busy"),
        error="line1\u2028line2\u2029line3",
    )
    await writer.write(entry)

    raw = trace_path.read_text(encoding="utf-8")
    lines = raw.splitlines()
    assert len(lines) == 1
    assert "\u2028" not in lines[0]
    assert "\u2029" not in lines[0]
    parsed = json.loads(lines[0])
    assert parsed["error"] == "line1\u2028line2\u2029line3"
