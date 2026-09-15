import numpy as np

from ai_trading_bot.domain import utc_now_ns
from ai_trading_bot.features import FeatureEngine

from .helpers import bar_series


def test_feature_pipeline_columns_deterministic() -> None:
    engine = FeatureEngine()
    names = engine.feature_names()
    assert names == [
        "open", "high", "low", "close", "volume",
        "rsi_14", "sma_20", "ema_9",
        "macd_line", "macd_signal", "macd_hist",
        "bb_upper", "bb_mid", "bb_lower",
        "atr_14", "return_1", "vol_ratio_20",
    ]


def test_feature_pipeline_compute() -> None:
    bars = bar_series(np.linspace(100, 130, 120), ts_0=utc_now_ns())
    engine = FeatureEngine()
    df = engine.compute(bars)
    assert list(df.columns) == engine.feature_names()
    assert len(df) == 120
    # indicators need warmup; after it, series must be finite
    assert df["rsi_14"].dropna().iloc[-1] > 0  # present and not nan late in window
    assert df["bb_upper"].iloc[-1] >= df["bb_lower"].iloc[-1]


def test_feature_pipeline_empty() -> None:
    engine = FeatureEngine()
    assert engine.compute([]).empty
    assert engine.latest([]) == {}