"""Live broker adapters: Binance (crypto), Alpaca (US equities), Zerodha (IN).

One adapter per market (REQ-EXE-02), all implementing
:class:`~ai_trading_bot.adapters.broker.BrokerAdapter`. Built structurally in
Phase 3 but each *refuses to place a live order* unless the environment is
``live`` AND the matching credentials are present — the Phase 6 go-live gate
(REQ-EXE-10). Until then, calls raise :class:`LiveUnavailableError`, so no code
path can accidentally touch a real exchange.

The vendor HTTP calls are intentionally thin: they construct the vendor payload
and delegate to an optional client, which is only constructed once credentials
are supplied. This keeps the module importable and testable without network.
"""

from __future__ import annotations

from ai_trading_bot.config import Credentials
from ai_trading_bot.domain import AccountState, Instrument

from .base import BrokerAdapter, LiveUnavailableError, Order

_GATED_MESSAGE = "live trading is gated until the Phase 6 rollout review"


class _LiveBrokerBase(BrokerAdapter):
    """Shared gating + credential plumbing for live adapters."""

    vendor = "generic"

    def __init__(self, credentials: Credentials, environment: str,
                 client=None) -> None:
        self._creds = credentials
        self._environment = environment
        self._client = client  # vendor SDK; None until day-key supplied

    def _require_live(self) -> None:
        if self._environment != "live":
            raise LiveUnavailableError(_GATED_MESSAGE)
        if self._client is None:
            raise LiveUnavailableError(_GATED_MESSAGE)

    def submit_order(self, instrument: Instrument, order: Order) -> Order:
        self._require_live()  # raises before any side effect if gated
        # vendor payload + call are defined per vendor below
        return self._place(self._vendor_payload(instrument, order))

    def _vendor_payload(self, instrument: Instrument, order: Order) -> dict:
        raise NotImplementedError

    def _place(self, payload: dict) -> Order:
        raise NotImplementedError

    def cancel_order(self, order_id: str) -> bool:  # pragma: no cover - live only
        self._require_live()
        raise NotImplementedError

    def get_order(self, order_id: str) -> Order:  # pragma: no cover - live only
        self._require_live()
        raise NotImplementedError

    def reconcile(self, orders: list[Order]) -> list[Order]:  # pragma: no cover
        self._require_live()
        raise NotImplementedError

    def get_account(self) -> AccountState:  # pragma: no cover - live only
        self._require_live()
        raise NotImplementedError


class BinanceBroker(_LiveBrokerBase):
    """Crypto spot orders on Binance (uses the testnet when configured)."""

    vendor = "binance"

    def _vendor_payload(self, instrument: Instrument, order: Order) -> dict:
        symbol = instrument.symbol.replace("/", "")
        return {
            "symbol": symbol,
            "side": "BUY" if order.side == "buy" else "SELL",
            "type": "MARKET" if order.order_type == "market" else order.order_type.upper(),
            "quantity": str(order.qty),
            "testnet": bool(self._creds.binance_testnet),
            "newClientOrderId": order.id,
        }


class AlpacaBroker(_LiveBrokerBase):
    """US-equity orders on Alpaca (paper endpoint unless credentials say live)."""

    vendor = "alpaca"

    def _vendor_payload(self, instrument: Instrument, order: Order) -> dict:
        return {
            "symbol": instrument.symbol,
            "side": order.side,
            "type": order.order_type,
            "qty": str(order.qty),
            "time_in_force": "day",
            "client_order_id": order.id,
        }


class ZerodhaBroker(_LiveBrokerBase):
    """Indian-equity orders through Zerodha Kite Connect."""

    vendor = "zerodha"

    def _vendor_payload(self, instrument: Instrument, order: Order) -> dict:
        symbol = instrument.symbol.replace("/", "")
        return {
            "tradingsymbol": symbol,
            "transaction_type": "BUY" if order.side == "buy" else "SELL",
            "variety": "regular",
            "order_type": "MARKET" if order.order_type == "market" else order.order_type.upper(),
            "quantity": order.qty,
        }


_BROKERS = {
    "binance": BinanceBroker,
    "alpaca": AlpacaBroker,
    "zerodha": ZerodhaBroker,
}


def build_live_brokers(credentials: Credentials,
                       environment: str) -> dict[str, BrokerAdapter]:
    """Instantiate live broker adapters (none raises now; calls raise later)."""
    return {
        name: cls(credentials, environment)
        for name, cls in _BROKERS.items()
    }


__all__ = [
    "AlpacaBroker", "BinanceBroker", "ZerodhaBroker",
    "build_live_brokers",
]