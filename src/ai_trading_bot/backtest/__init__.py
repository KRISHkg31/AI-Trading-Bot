"""Backtest package — cost-aware simulation, metrics, stress scenarios.

- :class:`CostModel` — realistic fees / spread / slippage (REQ-BT-02).
- :class:`BacktestEngine` — deterministic bar-driven simulator with
  walk-forward (REQ-BT-03) and stress scenarios (REQ-BT-05).
- :func:`compute_metrics` — standard performance & risk metrics (REQ-BT-06).
"""

from ai_trading_bot.backtest import scenarios
from ai_trading_bot.backtest.costs import CostModel
from ai_trading_bot.backtest.engine import BacktestEngine, BacktestResult
from ai_trading_bot.backtest.metrics import Metrics, TradeRecord, compute_metrics

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "CostModel",
    "Metrics",
    "TradeRecord",
    "compute_metrics",
    "scenarios",
]