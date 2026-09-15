"""Shared test helpers."""

from __future__ import annotations

import numpy as np

from ai_trading_bot.domain import Bar, Timeframe


def bar_series(closes: np.ndarray, ts_0: int = 1_000_000_000_000_000_000,
               timeframe: Timeframe = Timeframe.M5) -> list[Bar]:
    """Build a list of canonical bars from a close-price path (1 min apart)."""
    step_ns = 60_000_000_000
    bars: list[Bar] = []
    for i, close in enumerate(float(c) for c in closes):
        o = close * 0.999
        h = close * 1.005
        low = close * 0.995
        bars.append(
            Bar(
                instrument_id="binance:BTC/USDT",
                timeframe=timeframe,
                ts_utc_ns=ts_0 + i * step_ns,
                open=o, high=h, low=low, close=close,
                volume=10.0, source="test",
            )
        )
    return bars