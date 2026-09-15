"""Strategy framework (REQ-STR-*). Rule-based strategies land in Phase 1.

A strategy turns a window of bars (or features) into a :class:`Signal`. Strategies
are config-driven (enabled/disabled/tuned without code redeploys, REQ-STR-10) and
versioned (REQ-STR-04).
"""

from __future__ import annotations

from collections.abc import Sequence

from ai_trading_bot.domain import Bar, Instrument, Signal, SignalDirection


class Strategy:
    """Base class for all strategies (rule-based and ML)."""

    id: str = "base"
    version: str = "1.0.0"

    def required_timeframes(self) -> Sequence[str]:
        return ["5m"]

    def generate(self, instrument: Instrument, bars: Sequence[Bar],
                 model_inputs: dict[str, object] | None = None) -> Signal:
        raise NotImplementedError

    def on_tick(self, instrument: Instrument, signal: Signal) -> Signal:
        """Optional intra-strategy pass; default returns the signal unchanged."""
        return signal


def make_hold(instrument: Instrument, strategy_id: str, reason: str = "NO_SIGNAL") -> Signal:
    return Signal(
        instrument_id=instrument.id,
        direction=SignalDirection.HOLD,
        confidence=0.0,
        strategy_id=strategy_id,
        reason_code=reason,
    )