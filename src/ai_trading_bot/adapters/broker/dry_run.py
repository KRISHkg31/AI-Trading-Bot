"""Dry-run adapter: validate and accept orders without any fill or account change.

``execution.mode: dry_run`` runs the full pipeline — signal, risk approval,
order construction, idempotency — but never touches the paper ledger. Used to
smoke-test wiring before even simulated money moves (REQ-EXE-07).
"""

from __future__ import annotations

from ai_trading_bot.domain import Instrument

from .base import BrokerAdapter, Order, OrderStatus


class DryRunAdapter(BrokerAdapter):
    """Accepts orders into a shadow store; records status transitions only."""

    def __init__(self) -> None:
        self._orders: dict[str, Order] = {}
        self.accepted_count = 0

    def submit_order(self, instrument: Instrument, order: Order) -> Order:
        order.status = OrderStatus.SUBMITTED
        order.updated_ns = order.created_ns
        self._orders[order.id] = order
        self.accepted_count += 1
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


__all__ = ["DryRunAdapter"]