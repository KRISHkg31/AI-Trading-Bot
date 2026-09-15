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

from .base import BrokerAdapter, LiveUnavailableError, Order, OrderStatus

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
    """Crypto spot orders on Binance (uses the testnet when configured).

    The ccxt client is built lazily from ``binance_api_key``/``binance_api_secret``
    only when the environment is ``live``; without those the gate stays closed
    (``LiveUnavailableError``), so a first live session must be on the spot
    testnet with ``BINANCE_TESTNET=TRUE`` — see ``docs/GO_LIVE.md``.
    """

    vendor = "binance"

    def _ensure_client(self) -> object | None:
        creds = self._creds
        if not (creds.binance_api_key and creds.binance_api_secret):
            return None
        import ccxt

        client = ccxt.binance({
            "apiKey": creds.binance_api_key,
            "secret": creds.binance_api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })
        if creds.binance_testnet:
            client.set_sandbox_mode(True)
        return client

    def _require_live(self) -> None:
        # Two independent gates, checked in this order: the environment must be
        # explicitly ``live`` AND live credentials must exist. Environment is
        # checked first so nothing (not even a client object) is built in
        # paper/backtest/dry-run.
        if self._environment != "live":
            raise LiveUnavailableError(_GATED_MESSAGE)
        if self._client is None:
            self._client = self._ensure_client()
        if self._client is None:
            raise LiveUnavailableError(_GATED_MESSAGE)

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

    def _place(self, payload: dict) -> Order:
        # Live path; exercised only on the testnet for the first real session.
        client = self._client
        resp = client.create_order(
            symbol=payload["symbol"],
            type=payload["type"],
            side=payload["side"],
            amount=float(payload["quantity"]),
            newClientOrderId=payload["newClientOrderId"],
        )
        return Order(
            id=payload["newClientOrderId"],
            client_order_id=str(resp.get("id")),
            instrument_id=payload["symbol"],
            side=payload["side"].lower(),
            order_type=payload["type"].lower(),
            qty=float(payload["quantity"]),
            status=OrderStatus.FILLED if resp.get("status") == "FILLED" else OrderStatus.SUBMITTED,
            avg_fill_price=float(resp["price"]) if resp.get("price") else None,
            filled_qty=float(resp.get("filled", 0.0) or 0.0),
        )

    def get_account(self) -> AccountState:
        self._require_live()
        bal = self._client.fetch_balance()
        total = float(bal.get("total", {}).get("USDT", 0.0) or 0.0)
        free = float(bal.get("free", {}).get("USDT", 0.0) or 0.0)
        return AccountState(equity=total, cash=free)


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