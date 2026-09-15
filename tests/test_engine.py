
from ai_trading_bot.adapters.data import MarketDataAdapter
from ai_trading_bot.config import load_config
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

    # A HOLD/NO_STRATEGY signal was audited
    signals = engine._events.query("signal")
    assert len(signals) == 1
    assert signals[0]["payload"]["reason_code"] == "NO_STRATEGY"
    assert signals[0]["payload"]["direction"] == "hold"


def test_engine_data_quality_kill_flattens_risk(tmp_path) -> None:
    engine = _engine(tmp_path, stale=True)
    inst = instrument("BTC/USDT")
    engine.process_instrument(inst, Timeframe.M5)

    # Data-quality kill must flatten the risk gate (REQ-RSK-32)
    assert engine._risk._flattened is True