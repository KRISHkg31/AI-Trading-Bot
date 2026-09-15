"""Reproducible feature pipeline (REQ-STR-03).

Turns a window of canonical :class:`Bar` objects into a tidy DataFrame of
engineered features. Same code path in live engines and the backtester, so
live inference sees exactly the features a model was trained on.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from ai_trading_bot.domain import Bar
from ai_trading_bot.features import indicators

# Column order is deterministic; keep in sync with compute().
_FEATURE_COLUMNS = [
    "open", "high", "low", "close", "volume",
    "rsi_14", "sma_20", "ema_9",
    "macd_line", "macd_signal", "macd_hist",
    "bb_upper", "bb_mid", "bb_lower",
    "atr_14", "return_1", "vol_ratio_20",
]


class FeatureEngine:
    """Compute a fixed, versioned feature set from a bar window."""

    version = "1.0.0"

    def compute(self, bars: Sequence[Bar]) -> pd.DataFrame:
        if not bars:
            return pd.DataFrame(columns=_FEATURE_COLUMNS)
        df = (
            pd.DataFrame(
                {
                    "time": [b.ts_utc_ns for b in bars],
                    "open": [b.open for b in bars],
                    "high": [b.high for b in bars],
                    "low": [b.low for b in bars],
                    "close": [b.close for b in bars],
                    "volume": [b.volume for b in bars],
                }
            )
            .sort_values("time")
            .set_index("time")
        )

        close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]
        df["rsi_14"] = indicators.rsi(close, 14)
        df["sma_20"] = indicators.sma(close, 20)
        df["ema_9"] = indicators.ema(close, 9)
        df["macd_line"], df["macd_signal"], df["macd_hist"] = indicators.macd(close)
        df["bb_upper"], df["bb_mid"], df["bb_lower"] = indicators.bollinger(close)
        df["atr_14"] = indicators.atr(high, low, close, 14)
        df["return_1"] = close.pct_change()
        df["vol_ratio_20"] = vol / vol.rolling(20, min_periods=1).mean()
        return df

    def feature_names(self) -> list[str]:
        return list(_FEATURE_COLUMNS)

    def latest(self, bars: Sequence[Bar]) -> dict[str, Any]:
        row = self.compute(bars).tail(1)
        if row.empty:
            return {}
        return {str(k): v for k, v in row.iloc[-1].to_dict().items()}