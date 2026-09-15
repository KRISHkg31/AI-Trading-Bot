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


class OrderSide(str, enum.Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, enum.Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT = "take_profit"


@dataclass(slots=True)
class Position:
    """An open position. ``qty`` is signed: positive = long, negative = short.

    Risk and execution read direction from the sign, so a reversal is one
    transition (SELL to close a long opens a short net) rather than a stack.
    """

    instrument_id: str
    qty: float
    avg_entry_price: float
    currency: str = "USD"

    @property
    def direction(self) -> SignalDirection:
        if self.qty > 0:
            return SignalDirection.BUY
        if self.qty < 0:
            return SignalDirection.SELL
        return SignalDirection.HOLD

    @property
    def abs_qty(self) -> float:
        return abs(self.qty)

    def notional(self, price: float) -> float:
        """Gross exposure at ``price`` (always positive)."""
        return self.abs_qty * price


@dataclass(frozen=True, slots=True)
class Fill:
    """One completed (simulated or real) fill, appended to the audit trail."""

    order_id: str  # internal idempotency key
    instrument_id: str
    side: OrderSide
    price: float
    qty: float
    fee_usd: float
    ts_utc_ns: int = field(default_factory=utc_now_ns)
    source: str = ""  # "paper" | "dry_run" | broker id


@dataclass(slots=True)
class AccountState:
    """Live account tracker maintained after every fill (REQ-RSK-26)."""

    equity: float
    currency: str = "USD"
    cash: float | None = None  # None => 100% cash at construction (equity)
    open_positions: dict[str, Position] = field(default_factory=dict)
    today_pnl: float = 0.0
    realized_pnl_total: float = 0.0
    start_of_day_equity: float | None = None
    peak_equity: float | None = None
    updated_utc_ns: int = field(default_factory=utc_now_ns)

    def __post_init__(self) -> None:
        if self.cash is None:
            self.cash = self.equity
        if self.peak_equity is None:
            self.peak_equity = self.equity
        if self.start_of_day_equity is None:
            self.start_of_day_equity = self.equity

    def marked_equity(self, prices: dict[str, float]) -> float:
        """Mark-to-market equity = cash + sum of open-position valuations."""
        return self.cash + sum(
            pos.notional(prices[pid])
            for pid, pos in self.open_positions.items()
            if pid in prices
        )

    def apply_fill(self, fill: Fill) -> None:
        """Settle a fill into cash and open_positions (entry/exit/reversal).

        BUY fill: cash -= notional + fee. SELL fill: cash += notional - fee.

        Position book follows signed-quantity semantics:
          - no position: open with sign = side (long +, short -);
          - same direction as the existing position: add, reweight the average
            entry by cost;
          - opposite direction: close up to the existing size (realizing PnL on
            the closed leg at the original average entry) and, if the fill
            overflows the close, open a fresh opposite position at the fill
            price. A zero position is removed.
        """
        notional = fill.price * fill.qty
        pos = self.open_positions.get(fill.instrument_id)
        sign = 1.0 if fill.side is OrderSide.BUY else -1.0

        if fill.side is OrderSide.BUY:
            self.cash -= notional + fill.fee_usd
        else:
            self.cash += notional - fill.fee_usd

        if pos is None:
            self.open_positions[fill.instrument_id] = Position(
                fill.instrument_id, sign * fill.qty, fill.price, self.currency
            )
            self.updated_utc_ns = fill.ts_utc_ns
            return

        same_dir = (pos.qty > 0) == (sign > 0)
        if same_dir:
            total = pos.abs_qty + fill.qty
            avg = (pos.abs_qty * pos.avg_entry_price + notional) / total
            self.open_positions[fill.instrument_id] = Position(
                fill.instrument_id, sign * total, avg, self.currency
            )
        else:
            crossing = fill.qty >= pos.abs_qty
            closed = pos.abs_qty if crossing else fill.qty
            if pos.qty > 0:  # long closed by selling
                pnl = (fill.price - pos.avg_entry_price) * closed
            else:  # short covered by buying
                pnl = (pos.avg_entry_price - fill.price) * closed
            self.today_pnl += pnl
            self.realized_pnl_total += pnl
            if crossing:
                extra = fill.qty - pos.abs_qty
                if extra > 1e-12:
                    self.open_positions[fill.instrument_id] = Position(
                        fill.instrument_id, sign * extra, fill.price, self.currency
                    )
                else:
                    self.open_positions.pop(fill.instrument_id, None)
            else:
                new_qty = pos.qty + sign * fill.qty  # shrinks toward zero
                self.open_positions[fill.instrument_id] = Position(
                    fill.instrument_id, new_qty, pos.avg_entry_price, self.currency
                )
        self.updated_utc_ns = fill.ts_utc_ns

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