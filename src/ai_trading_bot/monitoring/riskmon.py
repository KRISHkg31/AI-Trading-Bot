"""Transition-based risk alerts (REQ-MON-*).

Every tier-3/4 event (kill switch, circuit breaker, daily-loss/drawdown halt,
data-quality kill) must fire an immediate alert (REQ-MON-02). Rather than hook
every rejection inside the gate — which would couple the gate to chat plumbing —
a watcher *polls* the gate once per engine pass and alerts only on *transitions*:

  open -> halted/flattened   fires "RISK_HALTED"
  halted -> open             fires "RISK_RESUMED"

:meth:`RiskGate.check` only discovers a Tier-3 halt when a signal passes through
it; a daily-loss breach with no live signal would otherwise go silent. The
watcher therefore calls :meth:`RiskGate.assess` each poll so those halts are
registered and alerted even between signals.
"""

from __future__ import annotations

from ai_trading_bot.config import MonitoringConfig
from ai_trading_bot.domain import AccountState
from ai_trading_bot.monitoring.notify import Notifier
from ai_trading_bot.persistence import EventLog
from ai_trading_bot.risk import RiskGate

#: Human-readable short descriptions for the catalogued gate states.
_REASON_TEXT = {
    "SYSTEM_FLAT": "kill switch / data-quality kill engaged",
    "DAILY_LOSS_LIMIT": "daily loss limit breached",
    "DRAWDOWN_LIMIT": "drawdown from peak breached",
    "SESSION_HALTED": "session halted, manual review required",
    "CIRCUIT_BREAKER": "order burst limit exceeded",
}


class RiskAlertWatcher:
    """Alert on risk-state transitions, never on every poll."""

    def __init__(
        self,
        risk: RiskGate,
        events: EventLog,
        notifier: Notifier,
        monitor_cfg: MonitoringConfig | None = None,
    ) -> None:
        self._risk = risk
        self._events = events
        self._notifier = notifier
        self._cfg = monitor_cfg or MonitoringConfig()
        self._last_state: str = "OK"

    def _current_state(self, account: AccountState) -> str:
        # Kill switch outranks every other reason (matches RiskGate.halted).
        if self._risk.flattened:
            return "SYSTEM_FLAT"
        halted = self._risk.assess(account) or self._risk.halted
        return halted or "OK"

    def poll(self, account: AccountState) -> None:
        state = self._current_state(account)
        if state == self._last_state:
            return
        previous = self._last_state
        self._last_state = state

        if state == "OK":
            title, body = "RISK_RESUMED", f"risk gate reopened from {previous}"
            self._record("risk_resumed", state, body, previous)
        else:
            text = _REASON_TEXT.get(state, state)
            title, body = "RISK_HALTED", f"{text} (was {previous})"
            self._record("risk_halted", state, body, previous)
        self._notifier.notify(title, body)

    def _record(self, event_type: str, state: str, detail: str, prior: str) -> None:
        self._events.record(
            event_type,
            {"reason": state, "detail": detail, "prior_gate_state": prior},
        )


__all__ = ["RiskAlertWatcher"]