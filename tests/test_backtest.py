"""Backtester tests: costs, metrics, engine, walk-forward, stress scenarios.

Uses deterministic synthetic bars so results are reproducible offline.
"""

from __future__ import annotations

import numpy as np

from ai_trading_bot.backtest import BacktestEngine, CostModel, compute_metrics, scenarios
from ai_trading_bot.backtest.metrics import TradeRecord
from ai_trading_bot.config import load_config
from ai_trading_bot.domain import instrument

from .helpers import bar_series


def _inst():
    return instrument("BTC/USDT")


# -- cost model -----------------------------------------------------------

def test_cost_model_math() -> None:
    cost = CostModel(taker_fee_bps=10.0, maker_fee_bps=4.0, spread_bps=4.0,
                     slippage_bps=2.0, fixed_usd=0.5)
    # side cost = (fee + spread/2 + slippage) / 1e4 = (10 + 2 + 2)/1e4
    assert cost.side_cost_pct(True) == 14.0 / 1e4
    assert cost.round_trip_cost_pct(True) == 28.0 / 1e4
    # buys fill higher, sells fill lower
    assert cost.fill_price(100.0, 1) > 100.0
    assert cost.fill_price(100.0, -1) < 100.0
    assert cost.fees_usd(1000.0) == 1000.0 * 14 / 1e4 + cost.fixed_usd


def test_cost_edge_clear() -> None:
    cost = CostModel(taker_fee_bps=10.0, spread_bps=4.0, slippage_bps=2.0)  # 0.28% round-trip
    assert not cost.edge_clear_cost(0.002, min_multiple=2.0)   # 0.2% < 0.56%
    assert cost.edge_clear_cost(0.006, min_multiple=2.0)       # 0.6% >= 0.56%
    assert cost.edge_clear_cost(0.003, min_multiple=1.0)       # 0.3% >= 0.28%


def test_cost_from_dict_filters_unknown_keys() -> None:
    assert CostModel.from_dict({"taker_fee_bps": 7.0, "bogus": 9}) == CostModel(taker_fee_bps=7.0)
    assert CostModel.from_dict(None) == CostModel()


# -- metrics --------------------------------------------------------------

def test_metrics_basic() -> None:
    import pandas as pd

    step_ns = 86_400 * 1_000_000_000  # one day per bar
    t0 = 1_000_000_000_000_000_000
    eq = pd.Series([100.0, 110.0, 120.0, 95.0, 105.0],
                   index=[t0 + i * step_ns for i in range(5)])
    trades = [
        TradeRecord("x", "long", t0, t0 + step_ns, 100, 110, 1, 9.5, 9.5, 0.5, "TAKE_PROFIT", 0, 1),
        TradeRecord("x", "long", t0, t0 + 2 * step_ns, 110, 120, 1, 9.5, 9.5, 0.5, "TAKE_PROFIT", 1, 2),
        TradeRecord("x", "short", t0, t0 + 3 * step_ns, 120, 95, -1, 24.5, 24.5, 0.5, "TAKE_PROFIT", 2, 3),
    ]
    m = compute_metrics(eq, trades, initial_equity=100.0)
    assert m.n_trades == 3
    assert m.win_rate == 1.0
    assert m.profit_factor > 1.0
    assert m.final_equity == 105.0
    assert m.ann_return_pct > 0.0
    assert m.max_drawdown_pct < 0.0  # the dip 120 -> 95 is a real drawdown
    assert m.exposure_pct > 0.0


# -- engine ---------------------------------------------------------------

def _engine(costs_override: dict[str, float] | None = None) -> BacktestEngine:
    cfg = load_config("dev")
    if costs_override:
        cfg = cfg.model_copy(update={
            "costs": cfg.costs.model_copy(update=costs_override),
        })
    return BacktestEngine(cfg)


def test_uptrend_produces_winning_long_trades() -> None:
    bars = bar_series(np.linspace(100, 200, 300))
    result = _engine().run(_inst(), bars)
    assert result.metrics.n_trades >= 1
    assert all(t.direction == "long" for t in result.trades)
    assert result.equity.iloc[-1] >= 10000.0
    assert result.metrics.final_equity >= 10_000.0
    assert result.metrics.fees_usd > 0.0  # costs were actually charged


def test_higher_costs_erode_returns() -> None:
    bars = bar_series(np.linspace(100, 200, 300))
    zero = {"taker_fee_bps": 0.0, "spread_bps": 0.0, "slippage_bps": 0.0}
    some = {"taker_fee_bps": 10.0, "spread_bps": 10.0, "slippage_bps": 5.0}  # 0.4% round-trip, edge still passes
    cheap = _engine(zero).run(_inst(), bars)
    expensive = _engine(some).run(_inst(), bars)
    assert cheap.metrics.n_trades >= 1 and expensive.metrics.n_trades >= 1  # both trade
    assert cheap.metrics.fees_usd == 0.0
    assert expensive.metrics.fees_usd > cheap.metrics.fees_usd
    assert expensive.metrics.total_return_pct < cheap.metrics.total_return_pct


def test_walk_forward_uses_out_of_sample_only() -> None:
    bars = bar_series(100 * (1 + 0.1 * np.sin(np.linspace(0, 20, 800))) + 10 * np.linspace(0, 1, 800))
    result = _engine().walk_forward(_inst(), bars, n_folds=3, test_frac=0.25)
    assert result.fold_bounds == [(200, 400), (400, 600), (600, 800)]
    assert result.method == "walk_forward"
    assert len(result.equity) == 600  # 3 x 200 OOS bars
    # every recorded trade ran in an OOS window (absolute bars 200..800)
    for t in result.trades:
        assert 200 <= t.entry_bar < 800


def test_walk_forward_requires_train_warmup() -> None:
    bars = bar_series(np.linspace(100, 120, 60))
    try:
        _engine().walk_forward(_inst(), bars, n_folds=4, test_frac=0.5)
        raise AssertionError("expected ValueError for degenerate walk-forward")
    except ValueError:
        pass


def test_stress_scenario_runs() -> None:
    bars = bar_series(np.linspace(100, 200, 400))
    result = _engine().run_scenario(_inst(), bars, "flash_crash", scenarios.flash_crash)
    assert result.method == "scenario:flash_crash"
    assert result.metrics.n_bars == 400


def test_signature_records_reproducibility() -> None:
    bars = bar_series(np.linspace(100, 120, 200))
    sig = _engine().run(_inst(), bars).signature
    assert sig.get("strategy_versions")
    assert sig["costs"]["taker_fee_bps"] == 5.0
    assert sig["seed"] == 42
    assert sig["n_bars"] == 200
    assert sig["data_source"] == "test"


# -- scenarios ------------------------------------------------------------

def test_scenario_ohlc_invariants_hold() -> None:
    bars = bar_series(np.linspace(100, 300, 500))
    for fn in (scenarios.flash_crash, scenarios.bear_market,
               scenarios.volatility_spike, scenarios.gap_down):
        out = fn(bars)
        assert len(out) == len(bars)
        for b in out:
            assert b.high >= max(b.open, b.close)
            assert b.low <= min(b.open, b.close)
            assert b.volume >= 0


def test_flash_crash_cuts_prices() -> None:
    bars = bar_series(np.linspace(100, 300, 500))
    base = bars[int(500 * 0.5) - 1].close
    out = scenarios.flash_crash(bars)
    new_closes = [b.close for b in out]
    assert min(new_closes) <= base * 0.72  # deepest point below a 28% haircut line
    assert out[-1].close > base * 0.5      # no total collapse


def test_gap_down_shifts_level_permanently() -> None:
    bars = bar_series(np.linspace(100, 300, 500))
    out = scenarios.gap_down(bars, at_frac=0.4, gap_pct=0.15)
    idx = int(500 * 0.4)
    offset = bars[idx].close * 0.15
    assert out[-1].close == bars[-1].close - offset


def test_volatility_spike_widens_range() -> None:
    bars = bar_series(np.linspace(100, 300, 500))
    start = int(500 * 0.5)
    out = scenarios.volatility_spike(bars, amplitude_mult=4.0)
    assert out[start].high - out[start].low == 4.0 * (bars[start].high - bars[start].low)