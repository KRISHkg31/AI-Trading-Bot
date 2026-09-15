"""Four-tier risk engine tests (REQ-RSK-*).

Covers: Tier 1 trade rules (stop, R:R, confidence, slippage, feasibility),
Tier 2 sizing + exposure caps (size-down semantics), Tier 3 halts (daily loss,
drawdown), Tier 4 system controls (kill switch, circuit breaker), and the
sizing math itself. A valid signal must never be rejected for being too large
--- it is scaled down to fit the caps instead.
"""

from __future__ import annotations

import time

from ai_trading_bot.config import load_config
from ai_trading_bot.domain import AccountState, Signal, SignalDirection
from ai_trading_bot.risk import RiskContext, RiskGate

_BTC = "binance:BTC/USDT"


def _limits(**overrides):
    return load_config("dev").risk.model_copy(update=overrides)


def _account(equity: float = 10_000.0, peak: float | None = None,
             sod: float | None = None) -> AccountState:
    return AccountState(
        equity=equity,
        start_of_day_equity=sod if sod is not None else equity,
        peak_equity=peak if peak is not None else equity,
    )


def _signal(**overrides) -> Signal:
    """A clean, risk-approvable BUY by default."""
    base: dict = {
        "instrument_id": _BTC,
        "direction": SignalDirection.BUY,
        "confidence": 0.9,
        "strategy_id": "test",
        "reason_code": "TEST",
        "reference_price": 130.0,
        "stop_loss_px": 128.0,    # 2.0 away
        "take_profit_px": 133.0,  # 3.0 away -> R:R exactly 1.5
        "risk_per_trade_pct": 1.0,
    }
    base.update(overrides)
    return Signal(**base)


def _gate(**limits_overrides) -> RiskGate:
    return RiskGate(_limits(**limits_overrides))


# -- Tier 1 ---------------------------------------------------------------


def test_hold_passthrough_approved() -> None:
    sig = Signal(instrument_id=_BTC, direction=SignalDirection.HOLD,
                 confidence=0.0, strategy_id="test", reason_code="NO_SIGNAL")
    decision = _gate().check(sig, _account())
    assert decision.approved is True
    assert decision.reason_code == "OK"


def test_ok_path_sizes_down_to_symbol_cap() -> None:
    # 1% risk / 2.0 stop -> 50 units = 6500 notional, far above the 10% symbol
    # cap (1000). The gate must approve and scale down to the cap, not reject.
    decision = _gate().check(_signal(), _account())
    assert decision.approved is True
    assert decision.reason_code == "OK"
    expected = 10_000.0 * 0.10 / 130.0  # symbol cap in units
    assert abs(decision.suggested_size - expected) / expected < 1e-9


def test_trade_without_stop_rejected() -> None:
    decision = _gate().check(_signal(stop_loss_px=None), _account())
    assert decision.approved is False
    assert decision.reason_code == "NO_STOP_LOSS"


def test_degenerate_stop_rejected() -> None:
    decision = _gate().check(_signal(stop_loss_px=130.0), _account())
    assert decision.approved is False
    assert decision.reason_code == "NO_STOP_LOSS"


def test_bad_risk_reward_rejected() -> None:
    decision = _gate().check(_signal(take_profit_px=130.5), _account())
    assert decision.approved is False
    assert decision.reason_code == "MIN_RISK_REWARD"


def test_exact_boundary_risk_reward_approved() -> None:
    # Take-profit at exactly stop_dist * min_risk_reward must pass (float-clean).
    decision = _gate().check(
        _signal(take_profit_px=130.0 + 2.0 * 1.5), _account()
    )
    assert decision.approved is True
    assert decision.reason_code == "OK"


def test_low_confidence_rejected() -> None:
    decision = _gate().check(_signal(confidence=0.3), _account())
    assert decision.approved is False
    assert decision.reason_code == "LOW_CONFIDENCE"


def test_slippage_too_large_rejected() -> None:
    # Market ref is 100 (from context); signal expects to fill at 100.5 -> 0.5%.
    decision = _gate().check(
        _signal(reference_price=100.5, stop_loss_px=98.5, take_profit_px=103.5),
        _account(),
        context=RiskContext(reference_price=100.0),
    )
    assert decision.approved is False
    assert decision.reason_code == "SLIPPAGE_TOO_LARGE"


def test_no_reference_price_rejected() -> None:
    decision = _gate().check(_signal(reference_price=None), _account())
    assert decision.approved is False
    assert decision.reason_code == "UNFEASIBLE"


# -- Tier 2 ---------------------------------------------------------------


def test_symbol_cap_already_exhausted_rejected() -> None:
    decision = _gate().check(
        _signal(), _account(), positions={_BTC: 1000.0},  # == 10% of equity
    )
    assert decision.approved is False
    assert decision.reason_code == "SYMBOL_EXPOSURE_LIMIT"


def test_total_cap_already_exhausted_rejected() -> None:
    decision = _gate().check(
        _signal(), _account(), positions={"binance:ETH/USDT": 6000.0},  # == 60%
    )
    assert decision.approved is False
    assert decision.reason_code == "TOTAL_EXPOSURE_LIMIT"


def test_size_down_respects_remaining_symbol_budget() -> None:
    # 5% already open on the symbol; a big signal sizes into the remaining 5%.
    decision = _gate().check(
        _signal(), _account(), positions={_BTC: 500.0},
    )
    assert decision.approved is True
    expected = 500.0 / 130.0  # remaining budget in units
    assert abs(decision.suggested_size - expected) / expected < 1e-9


def test_kelly_method_scales_risk_budget() -> None:
    gate = _gate(sizing_method="kelly", kelly_fraction=0.25)
    # Raw fractional size 50 -> kelly x0.25 -> 12.5; 10% symbol cap is 1000/130
    # = 7.69 units, so the cap still binds and wins here.
    decision = gate.check(_signal(), _account())
    expected = min(12.5, 1000.0 / 130.0)
    assert decision.approved is True
    assert abs(decision.suggested_size - expected) / expected < 1e-9


# -- sizing math ----------------------------------------------------------


def test_sizing_fractional_math() -> None:
    gate = _gate()
    size = gate._size_for_risk(_signal(), equity=10_000.0, ref=130.0, risk_pct=1.0)
    assert size == 100.0 / 2.0  # 50 units


def test_sizing_kelly_math() -> None:
    gate = _gate(sizing_method="kelly", kelly_fraction=0.25)
    size = gate._size_for_risk(_signal(), equity=10_000.0, ref=130.0, risk_pct=1.0)
    assert size == 12.5  # (10000*0.01*0.25) / 2.0 = 25.0 / 2.0


# -- Tier 3 ---------------------------------------------------------------


def test_daily_loss_halt_then_session_halted() -> None:
    gate = _gate(max_daily_loss_pct=3.0)
    # Equity down 4% from start-of-day -> halted.
    decision = gate.check(_signal(), _account(equity=9_600.0, sod=10_000.0))
    assert decision.approved is False
    assert decision.reason_code == "DAILY_LOSS_LIMIT"
    assert gate.halted == "DAILY_LOSS_LIMIT"

    # Even a fresh clean account is refused until manual review resumes the gate.
    decision = gate.check(_signal(), _account())
    assert decision.approved is False
    assert decision.reason_code == "SESSION_HALTED"


def test_drawdown_halt_and_resume() -> None:
    gate = _gate(max_drawdown_from_peak_pct=15.0)
    decision = gate.check(_signal(), _account(equity=8_400.0, peak=10_000.0))
    assert decision.approved is False
    assert decision.reason_code == "DRAWDOWN_LIMIT"
    assert gate.halted == "DRAWDOWN_LIMIT"

    gate.resume()
    assert gate.halted is None
    assert gate.check(_signal(), _account()).approved is True


# -- Tier 4 ---------------------------------------------------------------


def test_kill_switch_flattens_then_resume() -> None:
    gate = _gate()
    gate.kill()
    assert gate.flattened is True
    decision = gate.check(_signal(), _account())
    assert decision.approved is False
    assert decision.reason_code == "SYSTEM_FLAT"

    gate.resume()
    assert gate.check(_signal(), _account()).approved is True


def test_circuit_breaker_blocks_order_burst() -> None:
    gate = _gate(circuit_breaker_max_orders_per_minute=3, order_history_window_seconds=60)
    now = time.time_ns()
    for _ in range(3):
        gate.record_order(notional_usd=1000.0, instrument_id=_BTC, now_ns=now)
    decision = gate.check(_signal(), _account())
    assert decision.approved is False
    assert decision.reason_code == "CIRCUIT_BREAKER"


def test_circuit_breaker_expires_after_window() -> None:
    gate = _gate(circuit_breaker_max_orders_per_minute=3, order_history_window_seconds=60)
    old = time.time_ns() - 61 * 1_000_000_000  # outside the window
    for _ in range(3):
        gate.record_order(notional_usd=1000.0, instrument_id=_BTC, now_ns=old)
    decision = gate.check(_signal(), _account())
    assert decision.approved is True


def test_halted_reads_system_flat_first() -> None:
    gate = _gate()
    gate.kill()
    assert gate.halted == "SYSTEM_FLAT"