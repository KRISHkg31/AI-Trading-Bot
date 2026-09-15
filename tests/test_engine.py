
from ai_trading_bot.adapters.data import MarketDataAdapter
from ai_trading_bot.config import StrategyConfig, load_config
from ai_trading_bot.domain import AssetClass, Bar, Timeframe, instrument, utc_now_ns
from ai_trading_bot.engine import TradingEngine
from ai_trading_bot.persistence import BarStore, EventLog
from ai_trading_bot.risk import RiskGate


class FakeAdapter(MarketDataAdapter):
    """Deterministic offline adapter; can be made stale to test the data-quality kill."""

    def __init__(self, asset_class: AssetClass, stale: bool = False) -> None:
        self.asset_class = asset_class
        self.stale = stale

    def _stale_bar(self, inst, timeframe: Timeframe) -> Bar:
        age = 3600 * 1_000_000_000 if self.stale else 0
        return Bar(
            instrument_id=inst.id, timeframe=timeframe,
            ts_utc_ns=utc_now_ns() - age,
            open=100.0, high=101.0, low=99.0, close=100.5, volume=5.0, source="fake",
        )

    def fetch_bars(self, instrument, timeframe, start, end):
        bar = self._stale_bar(instrument, timeframe)
        self.last_update_ns = utc_now_ns() - (120 * 1_000_000_000 if self.stale else 0)
        return [bar]

    def health(self) -> dict[str, object]:
        return {"status": "stale" if self.stale else "ok"}


def _engine(tmp_path, stale: bool = False) -> TradingEngine:
    cfg = load_config("dev")
    inst = instrument("BTC/USDT")
    adapter = FakeAdapter(inst.asset_class, stale=stale)
    store = BarStore(base_dir=tmp_path / "raw")
    events = EventLog(db_path=tmp_path / "events.sqlite3")
    risk = RiskGate(cfg.risk)
    engine = TradingEngine(cfg, market_data={inst.asset_class: adapter},
                           store=store, eventlog=events, risk=risk)
    return engine


def test_engine_hold_path_and_heartbeat(tmp_path) -> None:
    engine = _engine(tmp_path)
    inst = instrument("BTC/USDT")
    engine.run_forever([inst], timeframe=Timeframe.M5, poll_interval_s=0.01, max_iterations=1)

    # Heartbeat is beating
    assert engine._heartbeat.age_seconds() < 5.0

    # Bars persisted
    bars = BarStore(base_dir=tmp_path / "raw").load_bars(inst.id, "5m")
    assert len(bars) == 1

    # A HOLD/NO_SIGNAL signal was audited (flat data -> no strategy fired)
    signals = engine._events.query("signal")
    assert len(signals) == 1
    assert signals[0]["payload"]["reason_code"] == "NO_SIGNAL"
    assert signals[0]["payload"]["direction"] == "hold"


def test_engine_data_quality_kill_flattens_risk(tmp_path) -> None:
    engine = _engine(tmp_path, stale=True)
    inst = instrument("BTC/USDT")
    engine.process_instrument(inst, Timeframe.M5)

    # Data-quality kill must flatten the risk gate (REQ-RSK-32)
    assert engine._risk._flattened is True


def test_engine_records_metrics_and_alerts_on_kill(tmp_path) -> None:
    """Phase 5: each pass emits a metric row; the data-quality kill fires a
    transition alert into the EventLog (REQ-MON-01/02, REQ-RSK-32)."""
    engine = _engine(tmp_path, stale=True)
    inst = instrument("BTC/USDT")
    engine.run_forever([inst], timeframe=Timeframe.M5, poll_interval_s=0.01,
                       max_iterations=2)

    metric_rows = engine._events.query("metric")
    assert len(metric_rows) == 2  # one self-contained snapshot per pass
    last = metric_rows[0]["payload"]
    assert last["risk_state"] == "SYSTEM_FLAT"  # stale data stuck -> killed
    assert last["n_positions"] == 0
    assert "equity" in last and "exposure_usd" in last

    # The kill fired one transition alert (data-quality kill -> SYSTEM_FLAT).
    halted = engine._events.query("risk_halted")
    assert len(halted) >= 1
    assert halted[0]["payload"]["reason"] == "SYSTEM_FLAT"


def test_engine_wires_ml_strategies_from_config(tmp_path) -> None:
    """When ml_* ids are in strategy.active the engine builds them against the
    configured registry dir; an empty registry must yield HOLD/MODEL_MISSING,
    never a crash (Phase 4, REQ-STR-04)."""
    from ai_trading_bot.domain import SignalDirection

    cfg = load_config("dev").model_copy(deep=True)
    cfg.strategy = StrategyConfig(active=["ml_momentum", "ml_mean_reversion"])
    cfg.models.model_dir = str(tmp_path / "models")  # hermetic: empty registry
    inst = instrument("BTC/USDT")
    adapter = FakeAdapter(inst.asset_class)
    engine = TradingEngine(
        cfg, market_data={inst.asset_class: adapter},
        store=BarStore(base_dir=tmp_path / "raw"),
        eventlog=EventLog(db_path=tmp_path / "events.sqlite3"),
        risk=RiskGate(cfg.risk),
    )
    assert "ml_momentum" in engine._strategies
    assert "ml_mean_reversion" in engine._strategies
    # Both are wired to the config's registry dir.
    for sid in ("ml_momentum", "ml_mean_reversion"):
        assert str(engine._strategies[sid]._repo._root) == cfg.models.model_dir
    # With no artifacts, inference degrades to HOLD/MODEL_MISSING, not a crash.
    bars = adapter.fetch_bars(inst, Timeframe.M5, None, None)
    sig = engine._strategies["ml_momentum"].generate(inst, bars * 80)
    assert sig.direction is SignalDirection.HOLD
    assert sig.reason_code == "MODEL_MISSING"
    # The whole loop still runs clean on the ML-only config.
    engine.process_instrument(inst, Timeframe.M5)


class FakeTrendAdapter(MarketDataAdapter):
    """30 steadily rising bars so EMATrend fires a real EMA_BULL BUY."""

    def __init__(self, asset_class: AssetClass) -> None:
        self.asset_class = asset_class
        self.last_update_ns = utc_now_ns()

    def fetch_bars(self, instrument, timeframe, start, end):
        base = utc_now_ns()
        step = 60 * 1_000_000_000  # 1m in ns
        bars = []
        for i in range(30):
            px = 100.0 + i
            bars.append(Bar(
                instrument_id=instrument.id, timeframe=timeframe,
                ts_utc_ns=base - (29 - i) * step,
                open=px, high=px + 0.5, low=px - 0.5, close=px + 0.25,
                volume=5.0, source="fake",
            ))
        return bars

    def health(self) -> dict[str, object]:
        return {"status": "ok"}


def test_engine_signal_flows_to_risk_and_paper(tmp_path) -> None:
    cfg = load_config("dev").model_copy(update={"strategy": StrategyConfig(active=["ema_trend"])})
    inst = instrument("BTC/USDT")
    adapter = FakeTrendAdapter(inst.asset_class)
    store = BarStore(base_dir=tmp_path / "raw")
    events = EventLog(db_path=tmp_path / "events.sqlite3")
    engine = TradingEngine(cfg, market_data={inst.asset_class: adapter},
                           store=store, eventlog=events)
    engine.process_instrument(inst, Timeframe.M5)

    # Strategy signal audited
    signals = events.query("signal")
    assert len(signals) == 1
    payload = signals[0]["payload"]
    assert payload["direction"] == "buy"
    assert payload["reason_code"] == "EMA_BULL"
    assert payload["confidence"] == 0.95  # max confidence 0.95 by design

    # Passed the risk gate with approval
    decisions = events.query("risk_decision")
    assert len(decisions) == 1
    assert decisions[0]["payload"]["approved"] is True


def test_engine_edge_filter_blocks_and_audits_no_edge(tmp_path) -> None:
    # Huge costs: ~1% ATR move cannot clear 2x round-trip cost -> REQ-SIG-04 blocks.
    cfg = load_config("dev").model_copy(update={
        "strategy": StrategyConfig(active=["ema_trend"]),
        "costs": cfg_costs_with_high_fees(),
    })
    inst = instrument("BTC/USDT")
    adapter = FakeTrendAdapter(inst.asset_class)
    events = EventLog(db_path=tmp_path / "events.sqlite3")
    engine = TradingEngine(cfg, market_data={inst.asset_class: adapter}, eventlog=events)
    engine.process_instrument(inst, Timeframe.M5)

    signals = events.query("signal")
    assert len(signals) == 1
    payload = signals[0]["payload"]
    assert payload["direction"] == "hold"
    assert payload["reason_code"] == "NO_EDGE"
    assert payload["strategy_id"] == "ema_trend"


def cfg_costs_with_high_fees():
    from ai_trading_bot.config import load_config

    base = load_config("dev")
    return base.costs.model_copy(update={"taker_fee_bps": 70.0, "spread_bps": 5.0, "slippage_bps": 5.0})