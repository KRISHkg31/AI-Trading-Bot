"""Signal synthesis filters (REQ-SIG-03/05).

- Minimum-confidence threshold: weak signals are dropped.
- Anti-whipsaw cooldown: no re-entry on a symbol shortly after an opposite signal.
Rejections are returned as auditable HOLD signals with a machine-readable reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ai_trading_bot.domain import Signal, SignalDirection, utc_now_ns


@dataclass(slots=True)
class _LastAction:
    direction: SignalDirection
    ts_ns: int


@dataclass(slots=True)
class SignalFilter:
    min_confidence: float = 0.5
    cooldown_seconds: int = 300
    _last: dict[str, _LastAction] = field(default_factory=dict)

    def apply(self, signal: Signal) -> Signal:
        if signal.confidence < self.min_confidence:
            return Signal(
                instrument_id=signal.instrument_id,
                direction=SignalDirection.HOLD,
                confidence=0.0,
                strategy_id=signal.strategy_id,
                reason_code="LOW_CONFIDENCE",
                suggested_size=None,
            )
        if signal.direction in (SignalDirection.BUY, SignalDirection.SELL):
            prev = self._last.get(signal.instrument_id)
            if prev is not None and prev.direction is not signal.direction:
                elapsed_s = (utc_now_ns() - prev.ts_ns) / 1e9
                if elapsed_s < self.cooldown_seconds:
                    return Signal(
                        instrument_id=signal.instrument_id,
                        direction=SignalDirection.HOLD,
                        confidence=0.0,
                        strategy_id=signal.strategy_id,
                        reason_code="WHIPSAW_COOLDOWN",
                    )
            self._last[signal.instrument_id] = _LastAction(
                signal.direction, utc_now_ns()
            )
        return signal