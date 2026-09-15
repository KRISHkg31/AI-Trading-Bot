"""Bar and event persistence (REQ-DAT-04, REQ-NFR-14)."""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from ai_trading_bot.domain import Bar, RiskDecision, Signal, Timeframe, utc_now_ns


class BarStore:
    """Parquet-backed store of canonical bars, partitioned for efficient replay."""

    def __init__(self, base_dir: str | Path = "data/raw") -> None:
        self._root = Path(base_dir)
        self._root.mkdir(parents=True, exist_ok=True)

    def _path(self, instrument_id: str, timeframe: str) -> Path:
        safe = instrument_id.replace(":", "_").replace("/", "_")
        return self._root / f"{safe}__{timeframe}.parquet"

    def append_bars(self, bars: Sequence[Bar]) -> None:
        if not bars:
            return
        df = pd.DataFrame(
            [
                {
                    "instrument_id": b.instrument_id,
                    "timeframe": b.timeframe.value,
                    "ts_utc_ns": b.ts_utc_ns,
                    "open": b.open,
                    "high": b.high,
                    "low": b.low,
                    "close": b.close,
                    "volume": b.volume,
                    "quote_volume": b.quote_volume,
                    "source": b.source,
                    "received_utc_ns": b.received_utc_ns,
                }
                for b in bars
            ]
        )
        path = self._path(bars[0].instrument_id, bars[0].timeframe.value)
        if path.exists():
            existing = pd.read_parquet(path)
            df = pd.concat([existing, df]).drop_duplicates(
                subset=["ts_utc_ns"], keep="last"
            ).sort_values("ts_utc_ns")
        df.to_parquet(path, index=False)

    def load_bars(self, instrument_id: str, timeframe: str,
                  start_ns: int | None = None, end_ns: int | None = None) -> list[Bar]:
        path = self._path(instrument_id, timeframe)
        if not path.exists():
            return []
        df = pd.read_parquet(path)
        if start_ns is not None:
            df = df[df["ts_utc_ns"] >= start_ns]
        if end_ns is not None:
            df = df[df["ts_utc_ns"] <= end_ns]
        return [
            Bar(
                instrument_id=r.instrument_id,
                timeframe=Timeframe(r.timeframe) if isinstance(r.timeframe, str) else r.timeframe,
                ts_utc_ns=int(r.ts_utc_ns),
                open=float(r.open),
                high=float(r.high),
                low=float(r.low),
                close=float(r.close),
                volume=float(r.volume),
                quote_volume=None if pd.isna(r.quote_volume) else float(r.quote_volume),
                source=r.source,
                received_utc_ns=None if pd.isna(r.received_utc_ns) else int(r.received_utc_ns),
            )
            for r in df.itertuples()
        ]


class EventLog:
    """Append-only SQLite journal for the audit trail (signals, risk, orders, alerts)."""

    def __init__(self, db_path: str | Path = "data/events.sqlite3") -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                event_id   TEXT PRIMARY KEY,
                ts_utc_ns  INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                payload    TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def record(self, event_type: str, payload: dict[str, Any],
               ts_ns: int | None = None) -> str:
        event_id = str(uuid.uuid4())
        self._conn.execute(
            "INSERT INTO events (event_id, ts_utc_ns, event_type, payload) VALUES (?,?,?,?)",
            (event_id, ts_ns or utc_now_ns(), event_type, json.dumps(payload, default=str)),
        )
        self._conn.commit()
        return event_id

    def record_signal(self, signal: Signal) -> str:
        return self.record(
            "signal",
            {
                "instrument_id": signal.instrument_id,
                "direction": signal.direction.value,
                "confidence": signal.confidence,
                "strategy_id": signal.strategy_id,
                "reason_code": signal.reason_code,
                "suggested_size": signal.suggested_size,
                "model_version": signal.model_version,
            },
            signal.created_utc_ns,
        )

    def record_risk_decision(self, signal: Signal, decision: RiskDecision) -> str:
        return self.record(
            "risk_decision",
            {
                "instrument_id": signal.instrument_id,
                "strategy_id": signal.strategy_id,
                "approved": decision.approved,
                "reason_code": decision.reason_code,
                "details": list(decision.details),
                "suggested_size": decision.suggested_size,
            },
        )

    def query(self, event_type: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        sql = "SELECT event_id, ts_utc_ns, event_type, payload FROM events"
        params: tuple = ()
        if event_type:
            sql += " WHERE event_type = ?"
            params = (event_type,)
        sql += " ORDER BY ts_utc_ns DESC LIMIT ?"
        params = (*params, limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [
            {"event_id": e, "ts_utc_ns": t, "event_type": ty, "payload": json.loads(p)}
            for e, t, ty, p in rows
        ]

    def close(self) -> None:
        self._conn.close()