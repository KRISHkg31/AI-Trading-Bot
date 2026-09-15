"""Execution engine: one risk-approved signal in, settled orders out (REQ-EXE-*).

Owns everything between the risk gate and the broker adapter:

- **Position semantics** (single net exposure per symbol): ENTRY when flat,
  no-op when already aligned (no stacking), a full close + fresh entry on a
  reversal, and a market close on CLOSE. This is what makes the paper ledger
  consume budget — a second same-direction signal is skipped, not entered again.
- **Idempotent submission** (REQ-EXE-05/06/09): each logical order carries a
  deterministic idempotency key; a resubmission with the same key returns the
  same order/fills, never doubles a position. Transient failures retry with
  exponential backoff under ``execution.retry_max_attempts``.
- **Ledger + audit** (REQ-RSK-26): every fill is settled into
  :class:`~ai_trading_bot.domain.AccountState` and recorded as ``order``/``fill``
  events on the event log. Equity is marked-to-market from the last known price
  of every position, so risk halts (daily-loss / drawdown) see true equity.

Covers REQ-EXE-01 (order lifecycle), REQ-EXE-03 (account sync), REQ-EXE-04
(position management), REQ-EXE-07 (dry-run), REQ-EXE-08 (paper).
"""

from __future__ import annotations

import time
from collections.abc import Callable

from ai_trading_bot.adapters.broker import BrokerAdapter, Order, OrderStatus
from ai_trading_bot.backtest.costs import CostModel
from ai_trading_bot.config import AppConfig
from ai_trading_bot.domain import (
    AccountState,
    Fill,
    Instrument,
    OrderSide,
    Signal,
    SignalDirection,
    utc_now_ns,
)
from ai_trading_bot.execution.base import ExecutionError, ExecutionResult
from ai_trading_bot.persistence import EventLog

_EPOCHS_BETWEEN_KEYS = 5  # coordinate expiry: re-key within the same 5s bucket


class ExecutionEngine:
    def __init__(
        self,
        config: AppConfig,
        router,
        account: AccountState,
        eventlog: EventLog | None = None,
        now_ns_fn: Callable[[], int] = utc_now_ns,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.cfg = config
        self._router = router
        self._account = account
        self._events = eventlog or EventLog()
        self._now_ns = now_ns_fn
        self._sleep = sleep_fn
        self._costs = CostModel.from_dict(config.costs.model_dump())
        self._last_prices: dict[str, float] = {}  # instrument -> last known price
        self.submitted_keys: set[str] = set()

    # -- public ----------------------------------------------------------
    def execute(self, instrument: Instrument, signal: Signal, decision,
                reference_price: float) -> ExecutionResult:
        """Settle ``signal`` into the book through the router's adapter."""
        self._last_prices[instrument.id] = reference_price
        adapter = self._router.get(instrument)
        pos = self._account.open_positions.get(instrument.id)

        if signal.direction is SignalDirection.CLOSE:
            if pos is not None:
                res = self._close(instrument, adapter, pos, reference_price)
                self._sync_equity()
                return res
            return ExecutionResult(instrument.id, "HOLD", note="CLOSE but nothing held")

        if signal.direction is SignalDirection.HOLD:
            return ExecutionResult(instrument.id, "HOLD", note="signal is hold")

        # Single net position: no stacking (capital protection first).
        if pos is not None and pos.direction is signal.direction:
            return ExecutionResult(
                instrument.id, "SKIP_ALREADY_POSITIONED",
                note=f"already holding {signal.direction.value}",
            )

        size = (decision.suggested_size or 0.0) if decision else 0.0
        if size <= 0:
            return ExecutionResult(instrument.id, "HOLD", note="no size from risk decision")

        orders: list[Order] = []
        fills: list[Fill] = []
        action = "ENTRY"
        if pos is not None and pos.direction is not signal.direction:
            close_res = self._close(instrument, adapter, pos, reference_price)
            orders += close_res.orders
            fills += close_res.fills
            action = "REVERSAL"

        enter = self._enter(instrument, adapter, signal.direction, size, reference_price)
        if enter is not None:
            order, fill = enter
            orders.append(order)
            if fill is not None:
                fills.append(fill)
        self._sync_equity()
        if not fills:
            return ExecutionResult(instrument.id, action, orders, note="orders accepted")
        return ExecutionResult(instrument.id, action, orders, fills)

    # -- order building --------------------------------------------------
    def _build_order(self, instrument: Instrument, side: OrderSide, qty: float,
                     order_type: str = "market") -> Order:
        now = self._now_ns()
        key = self._idempotency_key(instrument, side, qty, now)
        return Order(
            id=key,
            client_order_id=key,
            instrument_id=instrument.id,
            side=side.value,
            order_type=order_type,
            qty=qty,
            status=OrderStatus.PENDING,
            created_ns=now,
            updated_ns=now,
        )

    def _idempotency_key(self, instrument: Instrument, side: OrderSide,
                         qty: float, now: int) -> str:
        """Deterministic per (instrument, side, size, time-bucket) so retries of
        the same logical order reuse the key while genuinely new orders differ."""
        bucket = now // (_EPOCHS_BETWEEN_KEYS * 1_000_000_000)
        compact = instrument.id.replace(":", "").replace("/", "")
        return f"{self.cfg.execution.idempotency_key_prefix}-{compact}-{side.value}-{int(qty * 1e6)}-{bucket}"

    def _submit(self, adapter: BrokerAdapter, instrument: Instrument, order: Order) -> Order:
        """Submit with exponential-backoff retry; resubmission stays idempotent."""
        max_attempts = self.cfg.execution.retry_max_attempts
        for attempt in range(max_attempts):
            try:
                result = adapter.submit_order(instrument, order)
            except ExecutionError:
                if attempt >= max_attempts - 1:
                    raise
                self._sleep(self._backoff_seconds(attempt))
                continue
            self.submitted_keys.add(order.id)
            self._events.record(
                "order",
                {
                    "order_id": order.id,
                    "instrument_id": instrument.id,
                    "side": order.side,
                    "order_type": order.order_type,
                    "qty": order.qty,
                    "status": result.status,
                    "avg_fill_price": result.avg_fill_price,
                },
                result.created_ns,
            )
            return result
        raise ExecutionError("order submission exhausted retries")

    def _backoff_seconds(self, attempt: int) -> float:
        return min(8.0, 0.5 * (2 ** attempt))

    # -- the two verbs ----------------------------------------------------
    def _enter(self, instrument: Instrument, adapter: BrokerAdapter,
               direction: SignalDirection, qty: float,
               reference_price: float) -> tuple[Order, Fill | None] | None:
        side = OrderSide.BUY if direction is SignalDirection.BUY else OrderSide.SELL
        order = self._build_order(instrument, side, qty)
        result = self._submit(adapter, instrument, order)
        fill = self._settle_fill(instrument, result)
        return (result, fill)

    def _close(self, instrument: Instrument, adapter: BrokerAdapter, pos,
               reference_price: float) -> ExecutionResult:
        # Close the whole position at market (reducing risk is always allowed).
        side = OrderSide.SELL if pos.qty > 0 else OrderSide.BUY
        order = self._build_order(instrument, side, pos.abs_qty)
        result = self._submit(adapter, instrument, order)
        fill = self._settle_fill(instrument, result)
        return ExecutionResult(
            instrument.id, "CLOSE", orders=[result],
            fills=[fill] if fill is not None else [],
        )

    def _settle_fill(self, instrument: Instrument, order: Order) -> Fill | None:
        """Turn a FILLED order into a ledger+audit Fill (No fill for dry-run)."""
        price, filled = order.avg_fill_price, order.filled_qty
        if order.status != OrderStatus.FILLED or not price or filled <= 0:
            return None
        fill = Fill(
            order_id=order.id,
            instrument_id=instrument.id,
            side=OrderSide(order.side),
            price=price,
            qty=filled,
            fee_usd=self._costs.fees_usd(filled * price, market=True),
            ts_utc_ns=order.updated_ns,
            source="paper",
        )
        self._account.apply_fill(fill)
        self._events.record(
            "fill",
            {
                "order_id": fill.order_id,
                "instrument_id": fill.instrument_id,
                "side": fill.side.value,
                "price": fill.price,
                "qty": fill.qty,
                "fee_usd": fill.fee_usd,
                "cash": self._account.cash,
            },
            fill.ts_utc_ns,
        )
        return fill

    def _sync_equity(self) -> None:
        """Mark the whole book to the last known price; track the peak."""
        self._account.equity = self._account.marked_equity(self._last_prices)
        peak = self._account.peak_equity or 0.0
        if self._account.equity > peak:
            self._account.peak_equity = self._account.equity


__all__ = ["ExecutionEngine"]