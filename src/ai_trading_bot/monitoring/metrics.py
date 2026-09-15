"""Periodic account/fit metrics snapshots (REQ-MON-01).

A thin recorder that appends one ``metric`` event per engine pass to the same
SQLite EventLog as the audit trail (the user-selected Phase 5 store). The row is
a self-contained snapshot — equity, cash, PnL, drawdown, exposure, active model
versions and a risk-state string — so an out-of-process watcher (the ``monitor``
CLI) can judge engine liveness without touching engine internals.
"""

from __future__ import annotations

from typing import Any

from ai_trading_bot.domain import AccountState, utc_now_ns
from ai_trading_bot.persistence import EventLog


class MetricsRecorder:
    """Appends one ``metric`` snapshot row per call; reader sees only rows."""

    def __init__(self, eventlog: EventLog, environment: str = "dev") -> None:
        self._events = eventlog
        self._environment = environment

    def record(
        self,
        *,
        account: AccountState,
        last_prices: dict[str, float] | None = None,
        model_versions: dict[str, str | None] | None = None,
        risk_state: str = "OK",
        iteration: int = 0,
        now_ns: int | None = None,
    ) -> str:
        """Write one snapshot row and return its event id."""
        prices = last_prices or {}
        exposure_usd = sum(
            abs(pos.qty) * prices.get(pid, pos.avg_entry_price)
            for pid, pos in account.open_positions.items()
        )
        payload: dict[str, Any] = {
            "environment": self._environment,
            "iteration": iteration,
            "ts_utc_ns": now_ns or utc_now_ns(),
            "equity": round(account.equity, 4),
            "cash": round(account.cash, 4),
            "peak_equity": round(account.peak_equity, 4),
            "drawdown_from_peak_pct": round(account.drawdown_from_peak_pct, 6),
            "day_loss_pct": round(account.day_loss_pct, 6),
            "realized_pnl_total": round(account.realized_pnl_total, 4),
            "today_pnl": round(account.today_pnl, 4),
            "n_positions": len(account.open_positions),
            "exposure_usd": round(exposure_usd, 4),
            "risk_state": risk_state,
            "model_versions": model_versions or {},
        }
        return self._events.record("metric", payload, ts_ns=payload["ts_utc_ns"])


def latest_metrics(eventlog: EventLog) -> dict[str, Any] | None:
    """Most recent ``metric`` row, or ``None`` if the engine never recorded one."""
    rows = eventlog.query("metric", limit=1)
    return rows[0]["payload"] if rows else None


def metrics_age_seconds(eventlog: EventLog) -> float | None:
    """Age of the latest heartbeat/metric row, or ``None`` if none exists yet."""
    row = latest_metrics(eventlog)
    if row is None:
        return None
    return (utc_now_ns() - row["ts_utc_ns"]) / 1e9


__all__ = ["MetricsRecorder", "latest_metrics", "metrics_age_seconds"]