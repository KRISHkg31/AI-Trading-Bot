import numpy as np
import pandas as pd

from ai_trading_bot.features import indicators


def test_sma_matches_rolling_mean() -> None:
    s = pd.Series(np.arange(1, 40, dtype=float))
    assert indicators.sma(s, 10).iloc[-1] == s.rolling(10).mean().iloc[-1]
    assert indicators.sma(s, 10).iloc[:9].isna().all()


def test_rsi_uptrend_goes_overbought() -> None:
    up = pd.Series(np.linspace(100, 200, 80))
    assert indicators.rsi(up, 14).iloc[-1] > 70  # strong uptrend -> overbought


def test_rsi_flat_avoids_extremes() -> None:
    flat = pd.Series(np.full(60, 50.0))
    value = indicators.rsi(flat, 14).iloc[-1]
    assert 0 < value < 100


def test_macd_structure() -> None:
    s = pd.Series(np.sin(np.linspace(0, 6, 200)) * 10 + 100)
    line, signal, hist = indicators.macd(s)
    assert (hist == line - signal).all()
    assert line.dropna().shape[0] > 0


def test_bollinger_ordering() -> None:
    s = pd.Series(np.random.default_rng(7).normal(100, 3, 100))
    upper, mid, lower = indicators.bollinger(s)
    assert (upper.iloc[20:] >= mid.iloc[20:]).all()
    assert (mid.iloc[20:] >= lower.iloc[20:]).all()


def test_atr_positive() -> None:
    n = 100
    close = pd.Series(np.linspace(100, 120, n) + np.sin(np.linspace(0, 5, n)))
    high = close + 1.0
    low = close - 1.0
    atr = indicators.atr(high, low, close, 14)
    assert atr.dropna().min() > 0.0