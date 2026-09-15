"""Standardised performance & risk metrics (REQ-BT-06).

Every backtest run reports the same metrics from its equity curve and trades,
so results across strategies / folds / scenarios are comparable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

_NS_PER_YEAR = 365.25 * 24 * 3600 * 1e9


@dataclass(frozen=True, slots=True)
class TradeRecord:
    """One completed round-trip, exactly as the cost model saw it."""

    instrument_id: str
    direction: str  # "long" | "short"
    entry_ts_ns: int
    exit_ts_ns: int
    entry_px: float
    exit_px: float
    qty: float
    pnl_net_usd: float
    pnl_pct: float  # net PnL as % of entry notional
    fees_usd: float
    reason_exit: str  # STOP_LOSS | TAKE_PROFIT | SIGNAL | END_OF_DATA
    entry_bar: int
    exit_bar: int


@dataclass(frozen=True, slots=True)
class Metrics:
    n_bars: int
    n_trades: int
    total_return_pct: float
    ann_return_pct: float
    max_drawdown_pct: float  # negative, e.g. -12.4
    sharpe: float
    sortino: float
    calmar: float
    win_rate: float
    profit_factor: float
    avg_trade_pnl_pct: float
    fees_usd: float
    exposure_pct: float  # share of bars with an open position
    final_equity: float

    def summary(self) -> dict[str, object]:
        return {
            "n_bars": self.n_bars,
            "n_trades": self.n_trades,
            "total_return_pct": round(self.total_return_pct, 3),
            "ann_return_pct": round(self.ann_return_pct, 3),
            "max_drawdown_pct": round(self.max_drawdown_pct, 3),
            "sharpe": round(self.sharpe, 3),
            "sortino": round(self.sortino, 3),
            "calmar": round(self.calmar, 3),
            "win_rate": round(self.win_rate, 3),
            "profit_factor": round(self.profit_factor, 3),
            "avg_trade_pnl_pct": round(self.avg_trade_pnl_pct, 3),
            "fees_usd": round(self.fees_usd, 2),
            "exposure_pct": round(self.exposure_pct, 3),
            "final_equity": round(self.final_equity, 2),
        }


def _bars_in_position(trades: list[TradeRecord], n: int) -> int:
    """Union of trade bar intervals so back-to-back trades never double-count."""
    if not trades:
        return 0
    intervals = sorted((t.entry_bar, t.exit_bar) for t in trades)
    covered = 0
    s, e = intervals[0]
    for ns, ne in intervals[1:]:
        if ns <= e + 1:
            e = max(e, ne)
        else:
            covered += e - s + 1
            s, e = ns, ne
    covered += e - s + 1
    return min(covered, n)


def _annualisation(periods: int, span_ns: int, bars_per_year: int) -> float:
    """Periods-per-year implied by the actual time span, bounded to be sane."""
    if span_ns <= 0:
        return float(bars_per_year)
    return max(2.0, min(10 * bars_per_year, periods / (span_ns / _NS_PER_YEAR)))


def compute_metrics(
    equity: pd.Series,
    trades: Sequence[TradeRecord],
    initial_equity: float,
    bars_per_year: int = 105_120,
) -> Metrics:
    n = len(equity)
    if n == 0:
        return Metrics(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, initial_equity)

    final = float(equity.iloc[-1])
    total_return = final / initial_equity - 1.0 if initial_equity > 0 else 0.0
    span_ns = int(equity.index[-1] - equity.index[0])
    perks = _annualisation(n, span_ns, bars_per_year)

    series = equity.to_numpy(dtype="float64")
    running_max = np.maximum.accumulate(series)
    drawdowns = (series - running_max) / running_max
    max_dd = float(drawdowns.min()) if n else 0.0  # <= 0

    rets = pd.Series(series).pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    vol = float(rets.std()) if len(rets) > 1 else 0.0
    sharpe = float(rets.mean() / vol * np.sqrt(perks)) if vol > 0 else 0.0
    downside = rets[rets < 0]
    dvol = float(downside.std()) if len(downside) > 1 else 0.0
    sortino = float(rets.mean() / dvol * np.sqrt(perks)) if dvol > 0 else 0.0

    years = span_ns / _NS_PER_YEAR
    # Annualising against < ~1 hour of data is meaningless and can overflow.
    ann_return = (((final / initial_equity) ** (1 / years) - 1.0)
                  if years >= (1.0 / (24 * 365)) and final > 0 and initial_equity > 0 else 0.0)
    calmar = ann_return / abs(max_dd) if max_dd < 0 else 0.0

    t = list(trades)
    wins = [tr for tr in t if tr.pnl_net_usd > 0]
    losses = [tr for tr in t if tr.pnl_net_usd < 0]
    gross_win = sum(tr.pnl_net_usd for tr in wins)
    gross_loss = -sum(tr.pnl_net_usd for tr in losses)
    win_rate = len(wins) / len(t) if t else 0.0
    profit_factor = (float(gross_win / gross_loss) if gross_loss > 0
                     else (float("inf") if gross_win > 0 else 0.0))
    avg_trade_pct = float(np.mean([tr.pnl_pct for tr in t])) if t else 0.0
    fees = float(sum(tr.fees_usd for tr in t))

    in_position_bars = _bars_in_position(t, n)
    return Metrics(
        n_bars=n,
        n_trades=len(t),
        total_return_pct=total_return * 100.0,
        ann_return_pct=ann_return * 100.0,
        max_drawdown_pct=max_dd * 100.0,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_trade_pnl_pct=avg_trade_pct,
        fees_usd=fees,
        exposure_pct=(in_position_bars / n) * 100.0 if n else 0.0,
        final_equity=final,
    )