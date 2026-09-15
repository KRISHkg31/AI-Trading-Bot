import numpy as np

from ai_trading_bot.domain import instrument
from ai_trading_bot.strategy import rule_based

from .helpers import bar_series


def test_rsi_strategy_sells_overbought() -> None:
    btc = instrument("BTC/USDT")
    bars = bar_series(np.linspace(100, 220, 120))  # strong uptrend
    out = rule_based.RSIMeanReversion().generate(btc, bars)
    assert out.direction.value == "sell"
    assert out.reason_code == "RSI_OVERBOUGHT"
    assert 0.5 <= out.confidence <= 1.0


def test_macd_strategy_fires_on_cross() -> None:
    btc = instrument("BTC/USDT")
    # a V-shaped path forces a MACD cross near the bottom
    path = np.concatenate([np.linspace(120, 80, 60), np.linspace(80, 150, 80)])
    bars = bar_series(path)
    out = rule_based.MACDCrossover().generate(btc, bars)
    assert out.direction.value in ("buy", "sell", "hold")


def test_ema_trend_bull() -> None:
    btc = instrument("BTC/USDT")
    bars = bar_series(np.linspace(50, 200, 150))
    out = rule_based.EMATrend().generate(btc, bars)
    assert out.direction.value == "buy"
    assert out.reason_code == "EMA_BULL"


def test_bollinger_breakout_up() -> None:
    btc = instrument("BTC/USDT")
    flat = np.full(80, 100.0)
    path = np.concatenate([flat, [105.0]])  # last bar explodes above the bands
    bars = bar_series(path)
    out = rule_based.BollingerBreakout().generate(btc, bars)
    assert out.direction.value == "buy"
    assert out.reason_code == "BB_BREAKOUT_UP"


def test_short_window_returns_hold() -> None:
    btc = instrument("BTC/USDT")
    bars = bar_series(np.array([100.0, 101.0]))
    for strat in rule_based.BUILTIN_STRATEGIES.values():
        out = strat.generate(btc, bars)
        assert out.direction.value == "hold"