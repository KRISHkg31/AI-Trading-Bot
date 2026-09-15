"""Expected-edge filter (REQ-SIG-04).

Rejects signals whose expected reward is too small to clear round-trip
costs and slippage. The expected move is estimated as one ATR of
latest-bar data; the cost comes from the shared :class:`CostModel`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from ai_trading_bot.backtest.costs import CostModel
from ai_trading_bot.domain import Bar, Signal, SignalDirection
from ai_trading_bot.features import indicators

_ATR_DEFAULT_PERIOD = 14


@dataclass(slots=True)
class ExpectedEdgeFilter:
    costs: CostModel | None = None
    min_multiple: float = 2.0
    atr_period: int = _ATR_DEFAULT_PERIOD

    def apply(self, signal: Signal, bars: Sequence[Bar]) -> Signal:
        if signal.direction is SignalDirection.HOLD:
            return signal
        if not bars or len(bars) < self.atr_period + 1:
            return signal
        costs = self.costs or CostModel()
        px = bars[-1].close
        if px <= 0:
            return signal
        closes = pd.Series([b.close for b in bars], dtype="float64")
        highs = pd.Series([b.high for b in bars], dtype="float64")
        lows = pd.Series([b.low for b in bars], dtype="float64")
        atr_val = float(indicators.atr(highs, lows, closes, self.atr_period).iloc[-1])
        expected_move_pct = atr_val / px
        if not costs.edge_clear_cost(expected_move_pct, self.min_multiple):
            return Signal(
                instrument_id=signal.instrument_id,
                direction=SignalDirection.HOLD,
                confidence=0.0,
                strategy_id=signal.strategy_id,
                reason_code="NO_EDGE",
                suggested_size=None,
            )
        return signal