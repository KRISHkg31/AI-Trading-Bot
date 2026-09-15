import time

from ai_trading_bot.domain import Signal, SignalDirection
from ai_trading_bot.signal import SignalFilter


def _sig(direction: SignalDirection, confidence: float, instrument_id: str = "binance:BTC/USDT") -> Signal:
    return Signal(
        instrument_id=instrument_id, direction=direction, confidence=confidence,
        strategy_id="test", reason_code="TEST",
    )


def test_low_confidence_rejected() -> None:
    filt = SignalFilter(min_confidence=0.5, cooldown_seconds=0)
    out = filt.apply(_sig(SignalDirection.BUY, 0.3))
    assert out.direction == SignalDirection.HOLD
    assert out.reason_code == "LOW_CONFIDENCE"


def test_cooldown_blocks_whipsaw() -> None:
    filt = SignalFilter(min_confidence=0.5, cooldown_seconds=600)
    first = filt.apply(_sig(SignalDirection.BUY, 0.7))
    assert first.direction == SignalDirection.BUY

    # immediate opposite signal is blocked by the cooldown
    second = filt.apply(_sig(SignalDirection.SELL, 0.8))
    assert second.direction == SignalDirection.HOLD
    assert second.reason_code == "WHIPSAW_COOLDOWN"


def test_cooldown_expires() -> None:
    filt = SignalFilter(min_confidence=0.5, cooldown_seconds=1)
    filt.apply(_sig(SignalDirection.BUY, 0.7))
    time.sleep(1.1)
    out = filt.apply(_sig(SignalDirection.SELL, 0.8))
    assert out.direction == SignalDirection.SELL


def test_same_direction_not_blocked() -> None:
    filt = SignalFilter(min_confidence=0.5, cooldown_seconds=600)
    filt.apply(_sig(SignalDirection.BUY, 0.6))
    out = filt.apply(_sig(SignalDirection.BUY, 0.6))
    assert out.direction == SignalDirection.BUY