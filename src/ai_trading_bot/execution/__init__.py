"""Execution layer: order management between the risk gate and the brokers.

- :class:`ExecutionEngine` — position semantics, idempotent submit/retry, ledger.
- :class:`ExecutionRouter` — picks the adapter for the instrument under the mode.
- result/error types in ``base``.
"""

from ai_trading_bot.execution.base import (
    ExecutionError,
    ExecutionResult,
    LiveUnavailableError,
)
from ai_trading_bot.execution.engine import ExecutionEngine
from ai_trading_bot.execution.router import ExecutionRouter

__all__ = [
    "ExecutionEngine",
    "ExecutionError",
    "ExecutionResult",
    "ExecutionRouter",
    "LiveUnavailableError",
]