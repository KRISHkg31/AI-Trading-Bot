"""Cost-aware backtest engine (REQ-BT-01..07).

Bar-driven, deterministic, single-instrument simulator:

- Signals are computed on bars up to bar ``i-1`` and executed at the ``i``th
  bar's open (bar-open execution, no look-ahead).
- Fill prices move against the strategy by the model's fee + half-spread +
  slippage (REQ-BT-02).
- Position sizing is risk-based (risk % of equity / ATR stop); notional is
  capped to ``max_leverage`` x equity.
- Stops and targets are checked intrabar; if both would fill in the same bar
  the stop is assumed to trigger first (conservative).
- A ``cell`` is never opened while SL/TP ATR is unknown (warm-up).
- Walk-forward (REQ-BT-03) and stress scenarios (REQ-BT-05) share the exact
  same code path, so results are comparable.
- Every run records a reproducibility signature (REQ-BT-07).

Known, documented simplification: the *anti-whipsaw cooldown* uses wall-clock
time in the live path and is not applied here; the confidence filter and the
expected-edge filter are.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import pandas as pd

from ai_trading_bot.backtest.costs import CostModel
from ai_trading_bot.backtest.metrics import Metrics, TradeRecord, compute_metrics
from ai_trading_bot.config import AppConfig
from ai_trading_bot.domain import Bar, Instrument, Signal, SignalDirection
from ai_trading_bot.features import indicators
from ai_trading_bot.monitoring import get_logger
from ai_trading_bot.signal.edge import ExpectedEdgeFilter
from ai_trading_bot.strategy import BUILTIN_STRATEGIES, Strategy

_ATR_PERIOD = 14
logger = get_logger("backtest")
_MIN_STOP_PCT = 0.001  # cannot open a trade with a nearer-than-0.1% stop


@dataclass(frozen=True, slots=True)
class BacktestResult:
    method: str
    instrument_id: str
    metrics: Metrics
    equity: pd.Series  # indexed by ts_utc_ns
    trades: list[TradeRecord]
    signature: dict[str, Any]
    fold_bounds: list[tuple[int, int]] | None = None

    def report(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "instrument_id": self.instrument_id,
            "metrics": self.metrics.summary(),
            "n_trades": len(self.trades),
            "fold_bounds": self.fold_bounds,
            "signature": self.signature,
        }


class BacktestEngine:
    def __init__(
        self,
        config: AppConfig,
        strategies: dict[str, Strategy] | None = None,
    ) -> None:
        self.cfg = config
        self._costs = CostModel.from_dict(config.costs.model_dump())
        self._edge = ExpectedEdgeFilter(
            costs=self._costs, min_multiple=config.costs.min_edge_multiple
        )
        base = dict(BUILTIN_STRATEGIES)
        if strategies:
            base.update(strategies)
        self._strategies: dict[str, Strategy] = {}
        for sid in config.strategy.active:
            if sid in base:
                self._strategies[sid] = base[sid]

    # -- public API -------------------------------------------------------
    def run(
        self,
        instrument: Instrument,
        bars: Sequence[Bar],
        *,
        method: str = "plain",
        start_equity: float | None = None,
    ) -> BacktestResult:
        start_equity = start_equity if start_equity is not None else self.cfg.backtest.initial_equity
        equity, trades = self._run_series(instrument, bars, start_equity)
        metrics = compute_metrics(equity, trades, start_equity, self.cfg.backtest.bars_per_year)
        return BacktestResult(
            method=method,
            instrument_id=instrument.id,
            metrics=metrics,
            equity=equity,
            trades=trades,
            signature=self._signature(instrument, bars, start_equity, method),
        )

    def walk_forward(
        self,
        instrument: Instrument,
        bars: Sequence[Bar],
        *,
        n_folds: int | None = None,
        test_frac: float | None = None,
    ) -> BacktestResult:
        """Anchored out-of-sample walk-forward (REQ-BT-03)."""
        n = len(bars)
        n_folds = n_folds or self.cfg.backtest.walk_forward_folds
        test_frac = test_frac if test_frac is not None else self.cfg.backtest.walk_forward_test_frac
        test_len = max(1, int(n * test_frac))
        train = n - n_folds * test_len
        if train < _ATR_PERIOD + 4:
            raise ValueError(
                f"walk-forward needs train window > {_ATR_PERIOD + 4} bars "
                f"({n=} bars, {n_folds=} folds, {test_frac=})"
            )

        running = self.cfg.backtest.initial_equity
        parts: list[pd.Series] = []
        all_trades: list[TradeRecord] = []
        bounds: list[tuple[int, int]] = []
        for k in range(n_folds):
            test_start = k * test_len + train
            test_end = test_start + test_len
            test_bars = list(bars[test_start:test_end])
            sub = self.run(instrument, test_bars, method="walk_forward", start_equity=running)
            equity = sub.equity
            offset = test_start
            for tr in sub.trades:
                all_trades.append(replace(tr, entry_bar=tr.entry_bar + offset, exit_bar=tr.exit_bar + offset))
            bounds.append((test_start, test_end))
            if equity.empty:
                continue
            parts.append(equity)
            running = float(equity.iloc[-1])

        if not parts:
            raise ValueError("walk-forward produced no equity — no trades at all?")
        equity = pd.concat(parts)
        equity = equity.loc[~equity.index.duplicated(keep="first")].sort_index()
        metrics = compute_metrics(equity, all_trades, self.cfg.backtest.initial_equity,
                                  self.cfg.backtest.bars_per_year)
        return BacktestResult(
            method="walk_forward",
            instrument_id=instrument.id,
            metrics=metrics,
            equity=equity,
            trades=all_trades,
            signature=self._signature(instrument, bars, self.cfg.backtest.initial_equity, "walk_forward"),
            fold_bounds=bounds,
        )

    def run_scenario(self, instrument: Instrument, bars: Sequence[Bar], scenario_name: str,
                     transform) -> BacktestResult:
        """Stress-test a strategy against an adverse-regime transform (REQ-BT-05)."""
        stressed = transform(list(bars))
        return self.run(instrument, stressed, method=f"scenario:{scenario_name}")

    # -- core simulation --------------------------------------------------
    def _run_series(
        self, instrument: Instrument, bars: Sequence[Bar], start_equity: float
    ) -> tuple[pd.Series, list[TradeRecord]]:
        n = len(bars)
        if n == 0:
            empty = pd.Series(dtype="float64")
            return empty, []
        opens = pd.Series([b.open for b in bars], dtype="float64")
        highs = pd.Series([b.high for b in bars], dtype="float64")
        lows = pd.Series([b.low for b in bars], dtype="float64")
        closes = pd.Series([b.close for b in bars], dtype="float64")
        ts = pd.Index([b.ts_utc_ns for b in bars])
        atr = indicators.atr(highs, lows, closes, _ATR_PERIOD)
        atr_arr = atr.to_numpy(dtype="float64")

        bt = self.cfg.backtest
        min_conf = self.cfg.risk.min_confidence
        stop_mult = bt.stop_atr_mult
        rr = bt.take_profit_rr
        risk_pct = bt.risk_per_trade_pct / 100.0

        cash = start_equity
        pos: dict[str, float] | None = None
        equity_arr = pd.Series(index=ts, dtype="float64")
        trades: list[TradeRecord] = []

        def close_position(px: float, ts_ns: int, bar_i: int, reason: str) -> None:
            nonlocal cash, pos
            assert pos is not None
            abs_qty = abs(pos["qty"])
            notional = abs_qty * px
            fee = self._costs.fees_usd(notional, market=True)
            if pos["d"] > 0:
                cash += abs_qty * px - fee
            else:
                cash -= abs_qty * px + fee
            gross = pos["d"] * (px - pos["entry_px"]) * abs_qty
            pnl_net = gross - pos["entry_fee"] - fee
            pnl_pct = 100.0 * pnl_net / (pos["entry_px"] * abs_qty)
            trades.append(TradeRecord(
                instrument_id=instrument.id,
                direction="long" if pos["d"] > 0 else "short",
                entry_ts_ns=int(pos["entry_ts"]),
                exit_ts_ns=ts_ns,
                entry_px=pos["entry_px"],
                exit_px=px,
                qty=pos["qty"],
                pnl_net_usd=pnl_net,
                pnl_pct=pnl_pct,
                fees_usd=pos["entry_fee"] + fee,
                reason_exit=reason,
                entry_bar=int(pos["entry_bar"]),
                exit_bar=bar_i,
            ))
            pos = None

        def open_position(direction: int, open_px: float, bar_i: int) -> None:
            nonlocal cash, pos
            assert pos is None and bar_i >= 1
            atr_i = atr_arr[bar_i - 1]
            if not math.isfinite(atr_i):
                return
            stop_dist = max(atr_i * stop_mult, open_px * _MIN_STOP_PCT)
            entry_px = self._costs.fill_price(open_px, direction, market=True)
            qty = (cash * risk_pct) / stop_dist
            notional = qty * entry_px
            if notional > cash * bt.max_leverage:
                qty = (cash * bt.max_leverage) / entry_px
            entry_fee = self._costs.fees_usd(abs(qty) * entry_px, market=True)
            if direction > 0:
                cash -= abs(qty) * entry_px + entry_fee
            else:
                cash += abs(qty) * entry_px - entry_fee
            stop_px = entry_px - direction * stop_dist
            target_px = entry_px + direction * stop_dist * rr
            pos = {
                "d": float(direction), "qty": qty, "entry_px": entry_px,
                "entry_ts": ts[bar_i], "entry_bar": float(bar_i),
                "entry_fee": entry_fee, "stop": stop_px, "target": target_px,
            }

        for i in range(n):
            # 1. intrabar stop / target
            if pos is not None:
                d = pos["d"]
                hit_stop = (d > 0 and lows[i] <= pos["stop"]) or (d < 0 and highs[i] >= pos["stop"])
                hit_target = (d > 0 and highs[i] >= pos["target"]) or (d < 0 and lows[i] <= pos["target"])
                if hit_stop and hit_target:
                    close_position(pos["stop"], ts[i], i, "STOP_LOSS")  # conservative: stop first
                elif hit_stop:
                    close_position(pos["stop"], ts[i], i, "STOP_LOSS")
                elif hit_target:
                    close_position(pos["target"], ts[i], i, "TAKE_PROFIT")

            # 2. signal from bars[:i], act at bar i open
            if i >= 1:
                sig = self._signal(instrument, list(bars[:i]), min_conf)
                if pos is not None:
                    exit_on = (sig is not None and sig.direction is not sign_of_pos(pos))
                    if exit_on:
                        # close at open; cheap, bar-open exit avoids same-bar stop check
                        close_position(opens[i], ts[i], i, "SIGNAL")
                else:
                    if sig is not None and sig.direction in (SignalDirection.BUY, SignalDirection.SELL):
                        direction = 1 if sig.direction is SignalDirection.BUY else -1
                        open_position(direction, opens[i], i)
                        # freshly opened position can still stop out intrabar
                        if pos is not None:
                            d = pos["d"]
                            if (d > 0 and lows[i] <= pos["stop"]) or (d < 0 and highs[i] >= pos["stop"]):
                                close_position(pos["stop"], ts[i], i, "STOP_LOSS")

            # 3. mark-to-market
            equity_arr.iloc[i] = cash if pos is None else cash + pos["qty"] * closes[i]

        if pos is not None:
            close_position(closes[n - 1], ts[n - 1], n - 1, "END_OF_DATA")
            equity_arr.iloc[n - 1] = cash
        return equity_arr, trades

    def _signal(self, instrument: Instrument, bars: Sequence[Bar], min_conf: float) -> Signal | None:
        best: Signal | None = None
        for strategy in self._strategies.values():
            try:
                sig = strategy.generate(instrument, bars)
            except Exception as exc:  # noqa: BLE001 — defensive isolation of strategies
                logger.debug("strategy %s raised %s: %s", strategy.id, type(exc).__name__, exc)
                continue
            if (
                sig.direction in (SignalDirection.BUY, SignalDirection.SELL)
                and sig.confidence >= min_conf
                and (best is None or sig.confidence > best.confidence)
            ):
                best = sig
        if best is None:
            return None
        filtered = self._edge.apply(best, list(bars))
        if filtered.direction is SignalDirection.HOLD:
            return None
        return filtered

    def _signature(self, instrument: Instrument, bars: Sequence[Bar], start_equity: float,
                   method: str) -> dict[str, Any]:
        return {
            "method": method,
            "instrument_id": instrument.id,
            "strategy_versions": {sid: s.version for sid, s in self._strategies.items()},
            "strategy_active": list(self._strategies),
            "costs": self._costs.to_dict(),
            "backtest": {
                k: getattr(self.cfg.backtest, k)
                for k in ("initial_equity", "risk_per_trade_pct", "stop_atr_mult",
                          "take_profit_rr", "max_leverage", "bars_per_year", "seed")
            },
            "start_equity": start_equity,
            "n_bars": len(bars),
            "data_first_ts_ns": int(bars[0].ts_utc_ns) if bars else None,
            "data_last_ts_ns": int(bars[-1].ts_utc_ns) if bars else None,
            "data_source": bars[0].source if bars else None,
            "seed": self.cfg.backtest.seed,
        }


def sign_of_pos(pos: dict[str, float]) -> SignalDirection:
    return SignalDirection.BUY if pos["d"] > 0 else SignalDirection.SELL


__all__ = ["BacktestEngine", "BacktestResult", "CostModel", "Metrics", "TradeRecord", "sign_of_pos"]