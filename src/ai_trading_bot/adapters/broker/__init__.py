"""Broker adapters: the pluggable execution I/O layer (REQ-EXE-02).

- :class:`PaperBroker` — simulates fills through the shared cost model (default).
- :class:`DryRunAdapter` — validates orders without touching the ledger.
- Binance/Alpaca/Zerodha live adapters — gated until the Phase 6 rollout.

The risk engine is the only gate between signals and submission (BR-01); an
adapter must never bypass it.
"""

from ai_trading_bot.adapters.broker.base import (
    BrokerAdapter,
    LiveUnavailableError,
    Order,
    OrderStatus,
)

__all__ = ["BrokerAdapter", "LiveUnavailableError", "Order", "OrderStatus"]