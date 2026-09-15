"""Execution layer tests (REQ-EXE-*).

Covers: paper fill math + idempotency, dry-run (no ledger effect), position
semantics (entry / no-stack / reversal / close), idempotent retry, the live
go-live gate, the execution router's mode selection, and the end-to-end engine
fix for repeated same-direction re-entries.
"""

from __future__ import annotations

from ai_trading_bot.adapters.broker import BrokerAdapter, Order, OrderStatus
from ai_trading_bot.adapters.broker.dry_run import DryRunAdapter
from ai_trading_bot.adapters.broker.live import BinanceBroker
from ai_trading_bot.adapters.broker.paper import PaperBroker
from ai_trading_bot.backtest.costs import CostModel
from ai_trading_bot.config import Credentials, load_config
from ai_trading_bot.domain import (
    AccountState,
    Instrument,
    Signal,
    SignalDirection,
    instrument,
)
from ai_trading_bot.execution import ExecutionEngine, ExecutionRouter
from ai_trading_bot.execution.base import LiveUnavailableError

_BTC = "binance:BTC/USDT"


def _paper_broker(price: float | None = 130.0) -> PaperBroker:
    costs = CostModel(taker_fee_bps=0.0, spread_bps=0.0, slippage_bps=0.0, fixed_usd=0.0)
    return PaperBroker(cost_model=costs, reference_price=lambda _id: price)


def _order(side: str = "buy", qty: float = 10.0, instrument_id: str = _BTC,
           created_ns: int = 0) -> Order:
    return Order(
        id="key-1", client_order_id="key-1", instrument_id=instrument_id,
        side=side, order_type="market", qty=qty, created_ns=created_ns,
        updated_ns=created_ns,
    )


# -- paper broker ---------------------------------------------------------


def test_paper_broker_fills_at_cost_adjusted_price() -> None:
    costs = CostModel(taker_fee_bps=100.0, spread_bps=0.0, slippage_bps=0.0)  # 1%
    broker = PaperBroker(cost_model=costs, reference_price=lambda _id: 100.0)
    inst = Instrument("BTC/USDT", _asset(), "binance", "BTC", "USDT")
    result = broker.submit_order(inst, _order(side="buy", qty=2.0))
    assert result.status == OrderStatus.FILLED
    assert abs(result.avg_fill_price - 101.0) < 1e-9  # 100 * (1 + 1%)
    assert result.filled_qty == 2.0


def test_paper_broker_idempotent_resubmit() -> None:
    broker = _paper_broker()
    inst = instrument("BTC/USDT")
    first = broker.submit_order(inst, _order())
    second = broker.submit_order(inst, _order())
    assert second is first  # same order object returned, no double fill
    assert len(broker._orders) == 1


def test_paper_broker_rejects_without_reference_price() -> None:
    broker = _paper_broker(price=None)
    inst = instrument("BTC/USDT")
    result = broker.submit_order(inst, _order())
    assert result.status == OrderStatus.REJECTED


# -- dry run --------------------------------------------------------------


def test_dry_run_accepts_without_fills() -> None:
    adapter = DryRunAdapter()
    inst = instrument("BTC/USDT")
    result = adapter.submit_order(inst, _order())
    assert result.status == OrderStatus.SUBMITTED
    assert result.filled_qty == 0.0
    assert adapter.accepted_count == 1


# -- execution engine: position semantics --------------------------------


def _engine(equity: float = 10_000.0, mode: str = "paper") -> tuple[ExecutionEngine, AccountState]:
    cfg = load_config("dev").model_copy(update={"execution": _exec_cfg(mode)})
    acct = AccountState(equity=equity, start_of_day_equity=equity, peak_equity=equity)
    router = ExecutionRouter(cfg, reference_price=lambda _id: 130.0)
    eng = ExecutionEngine(cfg, router, acct)
    return eng, acct


def _exec_cfg(mode: str):
    from ai_trading_bot.config import ExecutionConfig

    return ExecutionConfig(mode=mode, retry_max_attempts=3)


def _signal(direction: SignalDirection, conf: float = 0.9) -> Signal:
    return Signal(
        instrument_id=_BTC, direction=direction, confidence=conf,
        strategy_id="test", reason_code="TEST",
    )


def test_engine_entry_opens_position_and_marks_equity() -> None:
    eng, acct = _engine()
    res = eng.execute(instrument("BTC/USDT"), _signal(SignalDirection.BUY),
                      _approved(10.0), reference_price=130.0)
    assert res.action == "ENTRY"
    assert len(res.orders) == 1
    assert len(res.fills) == 1
    pos = acct.open_positions[_BTC]
    assert pos.qty == 10.0
    assert pos.direction is SignalDirection.BUY
    assert acct.equity < 10_000.0  # cash spent, MTM reflects invested notional


def test_engine_skips_already_positioned() -> None:
    eng, acct = _engine()
    eng.execute(instrument("BTC/USDT"), _signal(SignalDirection.BUY),
                _approved(10.0), reference_price=130.0)
    before = acct.open_positions[_BTC].qty
    res = eng.execute(instrument("BTC/USDT"), _signal(SignalDirection.BUY),
                      _approved(10.0), reference_price=130.0)
    assert res.action == "SKIP_ALREADY_POSITIONED"
    assert res.orders == []
    assert acct.open_positions[_BTC].qty == before  # no stacking


def test_engine_reversal_closes_then_opens() -> None:
    eng, acct = _engine()
    eng.execute(instrument("BTC/USDT"), _signal(SignalDirection.BUY),
                _approved(10.0), reference_price=130.0)
    res = eng.execute(instrument("BTC/USDT"), _signal(SignalDirection.SELL),
                      _approved(8.0), reference_price=130.0)
    assert res.action == "REVERSAL"
    assert len(res.orders) == 2  # close long + open short
    pos = acct.open_positions[_BTC]
    assert pos.direction is SignalDirection.SELL
    assert pos.abs_qty == 8.0


def test_engine_close_flattens_position() -> None:
    eng, acct = _engine()
    eng.execute(instrument("BTC/USDT"), _signal(SignalDirection.BUY),
                _approved(10.0), reference_price=130.0)
    res = eng.execute(instrument("BTC/USDT"), _signal(SignalDirection.CLOSE),
                      None, reference_price=130.0)
    assert res.action == "CLOSE"
    assert _BTC not in acct.open_positions
    assert acct.realized_pnl_total != 0.0 or abs(acct.equity - 10_000.0) < 1.0


def test_engine_hold_with_nothing_positions_is_noop() -> None:
    eng, _acct = _engine()
    res = eng.execute(instrument("BTC/USDT"), _signal(SignalDirection.CLOSE),
                      None, reference_price=130.0)
    assert res.action == "HOLD"


# -- idempotent retry -----------------------------------------------------


class _FlakyBroker(BrokerAdapter):
    """Fails the first N submits, then succeeds — a transient-network stand-in."""

    def __init__(self, fail_count: int = 1) -> None:
        self.fail_count = fail_count
        self.calls = 0
        self._orders: dict[str, Order] = {}
        self._accepted: list[str] = []

    def submit_order(self, instrument: Instrument, order: Order) -> Order:
        self.calls += 1
        if self.calls <= self.fail_count:
            from ai_trading_bot.execution.base import ExecutionError

            raise ExecutionError("transient network error")
        order.status = OrderStatus.FILLED
        order.avg_fill_price = 130.0
        order.filled_qty = order.qty
        self._orders[order.id] = order
        self._accepted.append(order.id)
        return order

    def cancel_order(self, order_id: str) -> bool:
        return False

    def get_order(self, order_id: str) -> Order:
        return self._orders[order_id]

    def reconcile(self, orders: list[Order]) -> list[Order]:
        return orders


def test_engine_retries_transient_failure_with_idempotent_key() -> None:

    class _BareRouter:
        def get(self, inst):  # pragma: no cover - simple stub
            return broker

    cfg = load_config("dev")
    acct = AccountState(equity=10_000.0, start_of_day_equity=10_000.0, peak_equity=10_000.0)
    broker = _FlakyBroker(fail_count=1)
    slept: list[float] = []
    eng = ExecutionEngine(cfg, _BareRouter(), acct,
                          sleep_fn=lambda s: slept.append(s))
    res = eng.execute(instrument("BTC/USDT"), _signal(SignalDirection.BUY),
                      _approved(5.0), reference_price=130.0)
    assert broker.calls == 2  # 1 fail + 1 success
    assert slept and slept[0] >= 0.5  # backoff applied
    assert len(res.fills) == 1
    assert len(broker._accepted) == 1  # single accepted order (idempotent)


# -- live gate ------------------------------------------------------------


def test_live_broker_gated_until_go_live() -> None:
    broker = BinanceBroker(Credentials(binance_api_key="x", binance_api_secret="y"),
                           environment="dev")
    try:
        broker.submit_order(instrument("BTC/USDT"), _order())
    except LiveUnavailableError:
        pass
    else:
        assert False, "gated live adapter must refuse outside environment=live"


def test_live_broker_gated_without_credentials_even_in_live_mode() -> None:
    # environment=live but no keys: the gate must still hold (Phase 6).
    broker = BinanceBroker(Credentials(), environment="live")
    try:
        broker.submit_order(instrument("BTC/USDT"), _order())
    except LiveUnavailableError:
        pass
    else:
        assert False, "live mode without credentials must refuse to trade"


def test_binance_live_order_flows_to_client_when_gate_clear() -> None:
    """Phase 6: with live mode + keys + a (fake) client, submit maps the
    vendor response into a canonical Order. The real exchange call itself is
    only ever exercised on the spot testnet first (docs/GO_LIVE.md)."""
    calls: list[dict] = []

    class FakeClient:
        def create_order(self, **kw: object) -> dict:
            calls.append(kw)
            return {"id": "vendor-1", "status": "FILLED", "price": "72000.0",
                    "filled": "1.0"}

    broker = BinanceBroker(Credentials(binance_api_key="k", binance_api_secret="s"),
                           environment="live", client=FakeClient())
    result = broker.submit_order(instrument("BTC/USDT"), _order(side="buy", qty=1.0))

    assert calls[0]["symbol"] == "BTCUSDT"
    assert calls[0]["side"] == "BUY"
    assert calls[0]["type"] == "MARKET"
    assert calls[0]["amount"] == 1.0
    assert result.status == OrderStatus.FILLED
    assert result.client_order_id == "vendor-1"
    assert result.filled_qty == 1.0


def test_binance_get_account_maps_balance() -> None:
    class FakeClient:
        def fetch_balance(self) -> dict:
            return {"total": {"USDT": 1234.5}, "free": {"USDT": 1000.0}}

    broker = BinanceBroker(Credentials(binance_api_key="k", binance_api_secret="s"),
                           environment="live", client=FakeClient())
    account = broker.get_account()
    assert account.equity == 1234.5
    assert account.cash == 1000.0


def test_router_modes_pick_adapter() -> None:
    for mode, expected in (("dry_run", DryRunAdapter), ("paper", PaperBroker)):
        cfg = load_config("dev").model_copy(
            update={"execution": _exec_cfg(mode)}
        )
        router = ExecutionRouter(cfg, reference_price=lambda _id: 130.0)
        adapter = router.get(instrument("BTC/USDT"))
        assert isinstance(adapter, expected)


def _approved(size: float):
    from ai_trading_bot.domain import RiskDecision

    return RiskDecision(True, "OK", suggested_size=size)


def _asset():
    from ai_trading_bot.domain import AssetClass

    return AssetClass.CRYPTO