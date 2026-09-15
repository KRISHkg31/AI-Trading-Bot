"""Paper broker: simulates fills offline through the shared cost model.

Phase 3's default. A paper order fills immediately at the given reference price
moved against the strategy by fee + half-spread + slippage (identical to the
backtest's :class:`~ai_trading_bot.backtest.costs.CostModel`), so paper results
are directly comparable to backtests (REQ-EXE-08, REQ-BT-02).

The reference price comes from a callable so the live loop can feed the latest
close and the engine never invents hard-coded prices.
"""

from __future__ import annotations

from ai_trading_bot.backtest.costs import CostModel
from ai_trading_bot.domain import Fill, Instrument, OrderSide

from .base import BrokerAdapter, Order, OrderStatus


class PaperBroker(BrokerAdapter):
    """Simulated broker that fills instantly at a cost-adjusted price."""

    def __init__(self, cost_model: CostModel | None = None,
                 reference_price: object | None = None) -> None:
        self._costs = cost_model or CostModel()
        self._ref = reference_price  # callable (instrument_id) -> float | None
        self._orders: dict[str, Order] = {}
        self._order_map: dict[str, str] = {}  # idempotency key -> order id
        self.last_fill: Fill | None = None

    def submit_order(self, instrument: Instrument, order: Order) -> Order:
        # Idempotency (REQ-EXE-09): resubmitting the same key returns the same
        # order with its existing fills, never a second position.
        if order.id in self._order_map:
            return self._orders[self._order_map[order.id]]

        ref = self._ref(instrument.id) if callable(self._ref) else self._ref
        if ref is None or ref <= 0:
            order.status = OrderStatus.REJECTED
            order.updated_ns = order.created_ns
            self._orders[order.id] = order
            self._order_map[order.id] = order.id
            return order

        side = 1 if order.side == OrderSide.BUY.value else -1
        fill_px = self._costs.fill_price(ref, side, market=True)
        fees = self._costs.fees_usd(order.qty * fill_px, market=True)

        order.status = OrderStatus.FILLED
        order.avg_fill_price = fill_px
        order.filled_qty = order.qty
        order.updated_ns = order.created_ns
        self._orders[order.id] = order
        self._order_map[order.id] = order.id

        self.last_fill = Fill(
            order_id=order.id,
            instrument_id=instrument.id,
            side=OrderSide(order.side),
            price=fill_px,
            qty=order.qty,
            fee_usd=fees,
            ts_utc_ns=order.created_ns,
            source="paper",
        )
        return order

    def cancel_order(self, order_id: str) -> bool:
        order = self._orders.get(order_id)
        if order is None or order.status in (OrderStatus.FILLED, OrderStatus.CANCELLED):
            return False
        order.status = OrderStatus.CANCELLED
        return True

    def get_order(self, order_id: str) -> Order:
        return self._orders[order_id]

    def reconcile(self, orders: list[Order]) -> list[Order]:
        return [self._orders.get(o.id, o) for o in orders]


__all__ = ["PaperBroker"]