"""Expected-edge filter (REQ-SIG-04) tests."""

from __future__ import annotations

from ai_trading_bot.backtest.costs import CostModel
from ai_trading_bot.domain import Bar, Signal, SignalDirection, Timeframe
from ai_trading_bot.signal import ExpectedEdgeFilter

from .helpers import bar_series


def _signal(direction: SignalDirection, confidence: float = 0.8) -> Signal:
    return Signal("binance:BTC/USDT", direction, confidence, "test", reason_code="X")


def _tight_bars() -> list[Bar]:
    """Range is ~0.0002 of price -> ATR/px ~ 2e-6, no edge vs default costs."""
    step_ns = 60_000_000_000
    t0 = 1_000_000_000_000_000_000
    return [
        Bar("binance:BTC/USDT", Timeframe.M5, t0 + i * step_ns,
            open=100.0, high=100.0001, low=99.9999, close=100.0, volume=10.0, source="test")
        for i in range(40)
    ]


def _wide_bars() -> list[Bar]:
    """bar_series has ±0.5% intrabar range -> ATR/px ~1%, well past default costs."""
    return bar_series([100.0] * 40)


def test_hold_passes_through() -> None:
    sig = _signal(SignalDirection.HOLD)
    assert ExpectedEdgeFilter().apply(sig, _tight_bars()) is sig


def test_tiny_range_rejected() -> None:
    out = ExpectedEdgeFilter().apply(_signal(SignalDirection.BUY), _tight_bars())
    assert out.direction is SignalDirection.HOLD
    assert out.reason_code == "NO_EDGE"
    assert out.strategy_id == "test"


def test_strong_move_passes() -> None:
    # wide bars: expected move (~1%) clears 2x default round-trip cost (0.28%)
    out = ExpectedEdgeFilter().apply(_signal(SignalDirection.BUY), _wide_bars())
    assert out.direction is SignalDirection.BUY


def test_zero_cost_always_passes() -> None:
    edge = ExpectedEdgeFilter(costs=CostModel(taker_fee_bps=0.0, spread_bps=0.0, slippage_bps=0.0))
    out = edge.apply(_signal(SignalDirection.SELL), _tight_bars())
    assert out.direction is SignalDirection.SELL


def test_too_short_window_not_rejected() -> None:
    # not enough bars to compute ATR -> cannot decide -> leave signal untouched
    out = ExpectedEdgeFilter().apply(_signal(SignalDirection.BUY), _tight_bars()[:5])
    assert out.direction is SignalDirection.BUY