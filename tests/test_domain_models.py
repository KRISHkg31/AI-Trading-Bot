from ai_trading_bot.domain import (
    AccountState,
    AssetClass,
    Bar,
    Instrument,
    RiskDecision,
    Signal,
    SignalDirection,
    Timeframe,
)


def make_btc() -> Instrument:
    return Instrument(
        symbol="BTC/USDT", asset_class=AssetClass.CRYPTO, exchange="binance",
        base="BTC", quote="USDT", tick_size=0.01, lot_size=0.001,
    )


def test_instrument_id_is_exchange_prefixed() -> None:
    assert make_btc().id == "binance:BTC/USDT"


def test_bar_mid() -> None:
    bar = Bar(instrument_id="binance:BTC/USDT", timeframe=Timeframe.M5, ts_utc_ns=1,
              open=10.0, high=12.0, low=8.0, close=11.0, volume=100.0)
    assert bar.mid == 10.0


def test_signal_carries_required_elements() -> None:
    s = Signal(
        instrument_id="binance:BTC/USDT", direction=SignalDirection.BUY,
        confidence=0.7, strategy_id="trend_v1", reason_code="MACD_CROSS",
        suggested_size=0.5, horizon_seconds=300,
    )
    assert s.direction == SignalDirection.BUY
    assert s.confidence == 0.7
    assert s.horizon_seconds == 300
    assert s.reason_code == "MACD_CROSS"
    assert s.created_utc_ns > 0


def test_risk_decision_flags_approval() -> None:
    approved = RiskDecision(True, "OK", suggested_size=0.5)
    rejected = RiskDecision(False, "NO_STOP_LOSS")
    assert approved.approved is True
    assert rejected.approved is False


def test_account_state_drawdown_and_day_loss() -> None:
    acct = AccountState(equity=950.0, start_of_day_equity=1000.0, peak_equity=1100.0)
    assert acct.drawdown_from_peak_pct == 150.0 / 1100.0 * 100.0
    assert acct.day_loss_pct == 5.0