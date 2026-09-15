"""Persistence layer (REQ-DAT-04).

- :class:`BarStore` persists canonical bars to Parquet (partitioned per
  instrument+timeframe), for replay, audit, and model retraining.
- :class:`EventLog` is an append-only SQLite journal for signals, risk decisions,
  and orders — the immutable audit trail (REQ-MON-03, REQ-NFR-14).
"""

from ai_trading_bot.persistence.store import BarStore, EventLog

__all__ = ["BarStore", "EventLog"]