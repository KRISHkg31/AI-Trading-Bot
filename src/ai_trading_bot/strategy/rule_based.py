"""Rule-based technical strategies (REQ-STR-01/02).

Each strategy turns a bar window into a :class:`Signal`. Direction, confidence,
and a machine-readable reason code are always set (REQ-SIG-01/06); suggested size
is left for the risk engine to size.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from ai_trading_bot.domain import Bar, Instrument, Signal, SignalDirection
from ai_trading_bot.features import indicators
from ai_trading_bot.strategy.base import Strategy, make_hold


def _closes(bars: Sequence[Bar]) -> pd.Series:
    return pd.Series([b.close for b in bars], dtype="float64")


def _conf(pct: float) -> float:
    """Map a 0..1 strength to a confidence clamped to [0.5, 0.95]."""

    return round(min(0.95, max(0.5, 0.5 + pct * 0.45)), 3)


class RSIMeanReversion(Strategy):
    """Buy oversold, sell overbought (counter-trend)."""

    id = "rsi_mean_reversion"
    version = "1.0.0"

    def __init__(self, period: int = 14, oversold: float = 30.0, overbought: float = 70.0) -> None:
        self.period, self.oversold, self.overbought = period, oversold, overbought

    def generate(self, instrument: Instrument, bars: Sequence[Bar], model_inputs=None) -> Signal:
        if len(bars) < self.period + 1:
            return make_hold(instrument, self.id)
        value = float(indicators.rsi(_closes(bars), self.period).iloc[-1])
        if value >= self.overbought:
            pct = min(1.0, (value - self.overbought) / (100.0 - self.overbought))
            return Signal(instrument.id, SignalDirection.SELL, _conf(pct), self.id,
                          reason_code="RSI_OVERBOUGHT")
        if value <= self.oversold:
            pct = min(1.0, (self.oversold - value) / self.oversold)
            return Signal(instrument.id, SignalDirection.BUY, _conf(pct), self.id,
                          reason_code="RSI_OVERSOLD")
        return make_hold(instrument, self.id)


class MACDCrossover(Strategy):
    """Trend-follow: trade the MACD line / signal-line cross."""

    id = "macd_crossover"
    version = "1.0.0"

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9) -> None:
        self.fast, self.slow, self.signal = fast, slow, signal

    def generate(self, instrument: Instrument, bars: Sequence[Bar], model_inputs=None) -> Signal:
        needed = self.slow + self.signal + 2
        if len(bars) < needed:
            return make_hold(instrument, self.id)
        macd_line, signal_line, _ = indicators.macd(_closes(bars), self.fast, self.slow, self.signal)
        up = macd_line.iloc[-2] <= signal_line.iloc[-2] and macd_line.iloc[-1] > signal_line.iloc[-1]
        down = macd_line.iloc[-2] >= signal_line.iloc[-2] and macd_line.iloc[-1] < signal_line.iloc[-1]
        if up:
            return Signal(instrument.id, SignalDirection.BUY, _conf(1.0), self.id,
                          reason_code="MACD_CROSS_UP")
        if down:
            return Signal(instrument.id, SignalDirection.SELL, _conf(1.0), self.id,
                          reason_code="MACD_CROSS_DOWN")
        return make_hold(instrument, self.id)


class EMATrend(Strategy):
    """Trend-follow on short/long EMA spread (no strict cross required)."""

    id = "ema_trend"
    version = "1.0.0"

    def __init__(self, short: int = 9, long: int = 21, min_gap_pct: float = 0.001) -> None:
        self.short, self.long, self.gap = short, long, min_gap_pct

    def generate(self, instrument: Instrument, bars: Sequence[Bar], model_inputs=None) -> Signal:
        if len(bars) < self.long + 1:
            return make_hold(instrument, self.id)
        closes = _closes(bars)
        ema_s = float(indicators.ema(closes, self.short).iloc[-1])
        ema_l = float(indicators.ema(closes, self.long).iloc[-1])
        spread = (ema_s - ema_l) / ema_l if ema_l else 0.0
        if spread >= self.gap:
            return Signal(instrument.id, SignalDirection.BUY, _conf(1.0), self.id,
                          reason_code="EMA_BULL")
        if spread <= -self.gap:
            return Signal(instrument.id, SignalDirection.SELL, _conf(1.0), self.id,
                          reason_code="EMA_BEAR")
        return make_hold(instrument, self.id)


class BollingerBreakout(Strategy):
    """Volatility breakout: close breaking a Bollinger band."""

    id = "bollinger_breakout"
    version = "1.0.0"

    def __init__(self, window: int = 20, num_std: float = 2.0) -> None:
        self.window, self.num_std = window, num_std

    def generate(self, instrument: Instrument, bars: Sequence[Bar], model_inputs=None) -> Signal:
        if len(bars) < self.window + 1:
            return make_hold(instrument, self.id)
        closes = _closes(bars)
        upper, _, lower = indicators.bollinger(closes, self.window, self.num_std)
        last_close = float(closes.iloc[-1])
        if last_close > float(upper.iloc[-1]):
            return Signal(instrument.id, SignalDirection.BUY, _conf(1.0), self.id,
                          reason_code="BB_BREAKOUT_UP")
        if last_close < float(lower.iloc[-1]):
            return Signal(instrument.id, SignalDirection.SELL, _conf(1.0), self.id,
                          reason_code="BB_BREAKOUT_DOWN")
        return make_hold(instrument, self.id)


BUILTIN_STRATEGIES: dict[str, Strategy] = {
    s.id: s
    for s in (RSIMeanReversion(), MACDCrossover(), EMATrend(), BollingerBreakout())
}