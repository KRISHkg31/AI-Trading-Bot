"""Canonical, vendor-neutral domain model (REQ-DAT-03).

Every data source / broker adapter converts into these types, so strategies are
portable across exchanges. All timestamps are UTC integers (nanoseconds).
"""

from __future__ import annotations

import enum
import time
from collections.abc import Sequence
from dataclasses import dataclass, field


def utc_now_ns() -> int:
    """Current UTC time as integer nanoseconds since the epoch."""
    return time.time_ns()


class AssetClass(str, enum.Enum):
    CRYPTO = "crypto"
    US_EQUITIES = "us_equities"
    IN_EQUITIES = "in_equities"


class Timeframe(str, enum.Enum):
    TICK = "tick"
    S1 = "1s"
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    MO1 = "1M"


@dataclass(frozen=True, slots=True)
class Instrument:
    """A tradeable symbol in canonical form. ``symbol`` is vendor-neutral (e.g. ``BTC/USDT``)."""

    symbol: str
    asset_class: AssetClass
    exchange: str
    base: str
    quote: str
    currency: str = "USD"
    tick_size: float | None = None
    lot_size: float | None = None
    tradeable: bool = True

    @property
    def id(self) -> str:
        return f"{self.exchange}:{self.symbol}"


@dataclass(frozen=True, slots=True)
class Bar:
    """One OHLCV bar. ``ts_utc_ns`` is the bar close time."""

    instrument_id: str
    timeframe: Timeframe
    ts_utc_ns: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float | None = None
    source: str = ""  # vendor that produced the bar
    received_utc_ns: int | None = None  # when we ingested it (data-quality signal)

    @property
    def mid(self) -> float:
        return (self.high + self.low) / 2.0


@dataclass(frozen=True, slots=True)
class Tick:
    instrument_id: str
    ts_utc_ns: int  # exchange timestamp
    price: float
    size: float
    side: str | None = None  # 'buy' | 'sell' | None
    exchange_ts_ns: int | None = None
    received_utc_ns: int | None = None


class SignalDirection(str, enum.Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"


@dataclass(frozen=True, slots=True)
class Signal:
    """Decision emitted by a strategy/model, to be validated by the risk engine (REQ-SIG-*)."""

    instrument_id: str
    direction: SignalDirection
    confidence: float  # 0.0 .. 1.0
    strategy_id: str
    reason_code: str  # machine-readable, logged for audit
    suggested_size: float | None = None  # instrument units; None => risk engine sizes
    horizon_seconds: int | None = None
    model_version: str | None = None
    created_utc_ns: int = field(default_factory=utc_now_ns)
    # Trade-execution attributes validated by the risk gate (REQ-RSK-01..06).
    stop_loss_px: float | None = None
    take_profit_px: float | None = None
    risk_per_trade_pct: float | None = None
    reference_price: float | None = None  # where we expect to fill / enter near


@dataclass(frozen=True, slots=True)
class RiskDecision:
    """Output of the risk gate. No order is placed unless approved (BR-01)."""

    approved: bool
    reason_code: str  # "OK" or, e.g., "NO_STOP_LOSS" / "DAILY_LOSS_LIMIT"
    details: Sequence[str] = ()
    suggested_size: float | None = None


@dataclass(slots=True)
class Position:
    instrument_id: str
    qty: float
    avg_entry_price: float
    currency: str = "USD"


@dataclass(slots=True)
class AccountState:
    """Live account tracker maintained after every fill (REQ-RSK-26)."""

    equity: float
    currency: str = "USD"
    open_positions: dict[str, Position] = field(default_factory=dict)
    today_pnl: float = 0.0
    realized_pnl_total: float = 0.0
    start_of_day_equity: float | None = None
    peak_equity: float | None = None
    updated_utc_ns: int = field(default_factory=utc_now_ns)

    @property
    def drawdown_from_peak_pct(self) -> float:
        if not self.peak_equity or self.peak_equity <= 0:
            return 0.0
        return max(0.0, (self.peak_equity - self.equity) / self.peak_equity * 100.0)

    @property
    def day_loss_pct(self) -> float:
        base = self.start_of_day_equity or self.equity
        if base <= 0:
            return 0.0
        return (base - self.equity) / base * 100.0