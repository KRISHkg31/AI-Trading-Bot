"""Execution-layer shared types (REQ-EXE-*).

An :class:`ExecutionEngine` turns one risk-approved signal into orders through a
:class:`~ai_trading_bot.adapters.broker.BrokerAdapter`, settles fills into the
:class:`~ai_trading_bot.domain.AccountState` ledger, and audits everything to the
event log. This module holds the result/error types only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ai_trading_bot.adapters.broker import LiveUnavailableError, Order


class ExecutionError(Exception):
    """Raised when a submitted order cannot be placed or reconciled."""


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """Outcome of executing one signal: what was done and what hit the book."""

    instrument_id: str
    action: str  # "ENTRY" | "CLOSE" | "REVERSAL" | "HOLD" | "SKIP_ALREADY_POSITIONED"
    orders: list[Order] = field(default_factory=list)
    fills: list[object] = field(default_factory=list)  # list[Fill]
    note: str = ""

    @property
    def traded(self) -> bool:
        return bool(self.orders)


__all__ = ["ExecutionError", "ExecutionResult", "LiveUnavailableError"]