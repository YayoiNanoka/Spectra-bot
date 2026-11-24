"""Async structured JSONL trace logger for Spectra-bot.

Writes one self-contained compact JSON line per evaluated message batch so
gate filtering, Jev probabilities, and LLM execution telemetry can be analyzed
offline without affecting online chat reliability.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from spectra.config import TraceConfig
from spectra.gate import GateDecision

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


@dataclass(slots=True, frozen=True)
class TraceEntry:
    """Single self-contained trace record for one evaluated message batch."""

    timestamp: str
    session_key: str
    group_id: str | None
    user_id: str
    batch_size: int
    batch_db_ids: list[int]
    batch_mentioned: bool
    should_reply: bool
    shadow: bool
    reason: str
    effort: str
    decision_latency_ms: float
    jev_model: str | None = None
    jev_reply_probability: float | None = None
    jev_high_effort_probability: float | None = None
    jev_input_tokens: int | None = None
    jev_degraded: bool | None = None
    llm_model: str | None = None
    steps: int | None = None
    tool_calls: list[str] = field(default_factory=list)
    input_other_tokens: int | None = None
    input_cached_tokens: int | None = None
    output_tokens: int | None = None
    reply_chars: int | None = None
    reply_sent: bool = False
    llm_latency_ms: float | None = None
    total_latency_ms: float | None = None
    error: str | None = None

    @classmethod
    def from_decision(
        cls,
        decision: GateDecision,
        *,
        timestamp: str | None = None,
        llm_model: str | None = None,
        steps: int | None = None,
        tool_calls: Sequence[str] = (),
        input_other_tokens: int | None = None,
        input_cached_tokens: int | None = None,
        output_tokens: int | None = None,
        reply_chars: int | None = None,
        reply_sent: bool = False,
        llm_latency_ms: float | None = None,
        total_latency_ms: float | None = None,
        error: str | None = None,
    ) -> TraceEntry:
        """Build a :class:`TraceEntry` from a :class:`GateDecision` and LLM stats."""
        jev = decision.jev
        resolved_total = (
            decision.decision_latency_ms
            if total_latency_ms is None
            else total_latency_ms
        )
        return cls(
            timestamp=timestamp or _utc_now_iso(),
            session_key=decision.session_key,
            group_id=decision.group_id,
            user_id=decision.user_id,
            batch_size=len(decision.messages),
            batch_db_ids=[m.db_id for m in decision.messages],
            batch_mentioned=decision.batch_mentioned,
            should_reply=decision.should_reply,
            shadow=decision.shadow,
            reason=decision.reason,
            effort=decision.effort,
            decision_latency_ms=round(decision.decision_latency_ms, 3),
            jev_model=jev.model if jev is not None else None,
            jev_reply_probability=jev.reply_probability if jev is not None else None,
            jev_high_effort_probability=(
                jev.high_effort_probability if jev is not None else None
            ),
            jev_input_tokens=jev.input_tokens if jev is not None else None,
            jev_degraded=jev.degraded if jev is not None else None,
            llm_model=llm_model,
            steps=steps,
            tool_calls=list(tool_calls),
            input_other_tokens=input_other_tokens,
            input_cached_tokens=input_cached_tokens,
            output_tokens=output_tokens,
            reply_chars=reply_chars,
            reply_sent=reply_sent,
            llm_latency_ms=(
                None if llm_latency_ms is None else round(llm_latency_ms, 3)
            ),
            total_latency_ms=round(resolved_total, 3),
            error=error,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize this trace entry into a JSON-compatible dictionary."""
        return asdict(self)


def _append_line_sync(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line)
        fh.flush()


class TraceLogger:
    """Async append-only JSONL trace writer serialized by an :class:`asyncio.Lock`.

    Args:
        config: :class:`spectra.config.TraceConfig` specifying enabled flag and
            destination file path.
    """

    def __init__(self, config: TraceConfig) -> None:
        self._config = config
        self._lock = asyncio.Lock()

    @property
    def enabled(self) -> bool:
        """Return whether trace logging is enabled."""
        return self._config.enabled

    @property
    def path(self) -> Path:
        """Return the configured JSONL trace file path."""
        return self._config.path

    async def write(self, entry: TraceEntry) -> None:
        """Serialize ``entry`` and append it as one UTF-8 JSON line.

        Swallows :class:`OSError` with a warning log so filesystem errors never
        disrupt live chat processing.
        """
        if not self._config.enabled:
            return
        payload = json.dumps(
            entry.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        line = payload.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029") + "\n"
        async with self._lock:
            try:
                await asyncio.to_thread(_append_line_sync, self._config.path, line)
            except OSError as exc:
                logger.warning(
                    "Failed to write trace entry to %s: %s",
                    self._config.path,
                    exc,
                )
