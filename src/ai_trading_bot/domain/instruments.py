"""Collection of market instruments used across strategies and backtests."""

from __future__ import annotations

from ai_trading_bot.domain.models import AssetClass, Instrument

INSTRUMENTS: dict[str, Instrument] = {
    "BTC/USDT": Instrument(
        symbol="BTC/USDT", asset_class=AssetClass.CRYPTO, exchange="binance",
        base="BTC", quote="USDT", currency="USD", tick_size=0.01, lot_size=0.001,
    ),
    "ETH/USDT": Instrument(
        symbol="ETH/USDT", asset_class=AssetClass.CRYPTO, exchange="binance",
        base="ETH", quote="USDT", currency="USD", tick_size=0.01, lot_size=0.001,
    ),
    "AAPL": Instrument(
        symbol="AAPL", asset_class=AssetClass.US_EQUITIES, exchange="alpaca",
        base="AAPL", quote="USD", currency="USD", tick_size=0.01, lot_size=1,
    ),
    "RELIANCE": Instrument(
        symbol="RELIANCE", asset_class=AssetClass.IN_EQUITIES, exchange="zerodha",
        base="RELIANCE", quote="INR", currency="INR", tick_size=0.05, lot_size=1,
    ),
}


def instrument(symbol: str) -> Instrument:
    if symbol not in INSTRUMENTS:
        raise KeyError(f"unknown instrument: {symbol}")
    return INSTRUMENTS[symbol]