"""Broker/execution adapter interface (Pluggable; REQ-EXE-*). Filled in Phase 3."

The risk engine is the only gate between signals and order submission (BR-01); an
adapter must never bypass it. Implementations: Binance (live + testnet), Alpaca
(paper/live), Zerodha (paper/live via sandbox).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ai_trading_bot.domain import AccountState, Instrument, utc_now_ns


class OrderStatus:
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class LiveUnavailableError(RuntimeError):
    """Raised when live trading is attempted before the Phase 6 go-live gate."""


@dataclass(slots=True)
class Order:
    id: str  # internal idempotency key
    client_order_id: str  # broker-side id
    instrument_id: str
    side: str  # buy | sell
    order_type: str  # market | limit | stop_loss | take_profit | stop_limit
    qty: float
    status: str = OrderStatus.PENDING
    limit_price: float | None = None
    stop_price: float | None = None
    avg_fill_price: float | None = None
    filled_qty: float = 0.0
    created_ns: int = field(default_factory=utc_now_ns)
    updated_ns: int = field(default_factory=utc_now_ns)


class BrokerAdapter:
    """Interface for placing and reconciling orders."""

    def submit_order(self, instrument: Instrument, order: Order) -> Order: ...
    def cancel_order(self, order_id: str) -> bool: ...
    def get_order(self, order_id: str) -> Order: ...
    def reconcile(self, orders: list[Order]) -> list[Order]: ...
    def get_account(self) -> AccountState: ...