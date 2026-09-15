"""The independent risk gate (REQ-RSK-*, BR-01).

No order is placed unless the risk engine approves. Layered checks:

  Tier 1  Trade-level     stop-loss required, min R:R, slippage, sanity (REQ-RSK-01..06)
  Tier 2  Position-level  sizing, risk-per-trade + symbol caps, correlation (REQ-RSK-11..15)
  Tier 3  Portfolio-level daily-loss halt, drawdown halt, exposure caps (REQ-RSK-21..26)
  Tier 4  System-level    circuit breaker, data-quality kill, kill switch, dead-man's (REQ-RSK-31..36)

Full implementation is Phase 2. This module defines the interface execution and the
core loop depend on, so every order path must pass through :meth:`RiskGate.check`.
"""

from __future__ import annotations

from ai_trading_bot.config import RiskLimits
from ai_trading_bot.domain import AccountState, RiskDecision, Signal


class RiskGate:
    """Independent gate between signal generation and execution."""

    def __init__(self, limits: RiskLimits) -> None:
        self._limits = limits
        self._flattened = False  # set True by kill switch / dead-man's switch

    def check(self, signal: Signal, account: AccountState,
              open_positions=None) -> RiskDecision:
        """Validate a signal against all applicable risk tiers.

        Phase 2 replaces the placeholder with the full four-tier engine.
        """
        if self._flattened:
            return RiskDecision(False, "SYSTEM_FLAT", details=("killed",))
        return RiskDecision(True, "OK", suggested_size=signal.suggested_size)

    def kill(self) -> None:
        """Manual / system kill switch: stop all new orders immediately (REQ-RSK-33)."""
        self._flattened = True

    def resume(self) -> None:
        self._flattened = False