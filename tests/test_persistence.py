import time

from ai_trading_bot.domain import (
    Bar,
    RiskDecision,
    Signal,
    SignalDirection,
    Timeframe,
    instrument,
    utc_now_ns,
)
from ai_trading_bot.persistence import BarStore, EventLog


def test_bar_store_roundtrip(tmp_path) -> None:
    store = BarStore(base_dir=tmp_path)
    btc = instrument("BTC/USDT")
    bar = Bar(
        instrument_id=btc.id, timeframe=Timeframe.M5, ts_utc_ns=utc_now_ns(),
        open=1.0, high=2.0, low=0.5, close=1.5, volume=10.0, source="binance",
    )
    store.append_bars([bar])
    out = store.load_bars(btc.id, "5m")
    assert len(out) == 1
    assert out[0].close == 1.5
    assert out[0].timeframe == Timeframe.M5

    # idempotent on duplicate timestamp (keep latest)
    dup = Bar(
        instrument_id=btc.id, timeframe=Timeframe.M5, ts_utc_ns=bar.ts_utc_ns,
        open=1.0, high=2.0, low=0.5, close=1.9, volume=10.0, source="binance",
    )
    store.append_bars([dup])
    out = store.load_bars(btc.id, "5m")
    assert len(out) == 1
    assert out[0].close == 1.9


def test_event_log_audit_trail(tmp_path) -> None:
    log = EventLog(db_path=tmp_path / "events.sqlite3")
    s = Signal(
        instrument_id="binance:BTC/USDT", direction=SignalDirection.BUY, confidence=0.7,
        strategy_id="trend_v1", reason_code="MACD_CROSS", created_utc_ns=utc_now_ns(),
    )
    sign = log.record_signal(s)
    dec = RiskDecision(False, "NO_STOP_LOSS", details=("stop required",))
    risk = log.record_risk_decision(s, dec)

    events = log.query("signal")
    assert events[0]["event_id"] == sign
    assert events[0]["payload"]["direction"] == "buy"

    risk_events = log.query("risk_decision")
    assert risk_events[0]["event_id"] == risk
    assert risk_events[0]["payload"]["approved"] is False
    assert risk_events[0]["payload"]["reason_code"] == "NO_STOP_LOSS"
    log.close()


def test_freshness_gate_detects_stale_data() -> None:
    from ai_trading_bot.adapters.data.base import DataStalenessError, assert_fresh

    now = time.time_ns()
    assert_fresh(now, stale_after_seconds=60, label="binance")  # fresh

    try:
        assert_fresh(now - 120 * 1_000_000_000, stale_after_seconds=60, label="binance")
        raise AssertionError("expected DataStalenessError")
    except DataStalenessError:
        pass