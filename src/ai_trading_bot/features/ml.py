"""ML feature pipeline: windowed (X, y) datasets with a shared train/serve path.

Phase 4 (ML strategies). The rule-based :class:`FeatureEngine` gives us an OHLCV
feature frame; this module derives the *model inputs* the ML strategies consume
and the supervised *labels* the retraining pipeline learns.

The golden rule is REQ-STR-03: **live inference must see exactly the features the
model was trained on.** Both the retrainer and the strategies call the same
:func:`compute_features` on the same canonical :class:`~ai_trading_bot.domain.Bar`
window, so train/serve skew cannot creep in by construction.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

from ai_trading_bot.domain import Bar

# Bumped whenever the model-input derivation changes; pinned per model artifact so
# a trained model always records *which* feature revision produced its inputs.
FEATURE_VERSION = "2.0.0"

# Momentum trades *continuation*: the target is direction of the forward return,
# the inputs emphasize trend. Mean-reversion trades *reversion* with the same
# forward-return target but inputs that emphasize short-term overextension.
_MOMENTUM_COLUMNS = [
    "return_3", "return_10", "return_20",      # multi-scale trend (px momentum)
    "ema_fast_gap", "macd_hist_norm",          # trend structure (px momentum)
    "vol_ratio_20", "atr_norm",                # activity regime (follow-through)
]
_REVERSION_COLUMNS = [
    "rsi_14", "return_1", "return_3",          # short-term overextension
    "rsi_overbought", "rsi_oversold",          # stretched (signed) levels
    "bb_dist", "atr_norm",                     # how far price is outside the bands
]
_ALL_COLUMNS = sorted({*_MOMENTUM_COLUMNS, *_REVERSION_COLUMNS})


def _closes(df: pd.DataFrame) -> pd.Series:
    return df["close"]


def compute_features(bars: Sequence[Bar]) -> pd.DataFrame:
    """Return the ML feature frame (indexed by bar time) for a bar window.

    Columns are a fixed, versioned set. Warm-up rows contain NaN and are dropped
    by :func:`build_dataset` / handled by callers at inference (a strategy sees a
    full window, so only its last row matters).
    """
    if not bars:
        return pd.DataFrame(columns=_ALL_COLUMNS)
    df = (
        pd.DataFrame(
            {
                "time": pd.to_datetime([b.ts_utc_ns for b in bars], unit="ns", utc=True),
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
    close, high, low, vol = (
        _closes(df), df["high"], df["low"], df["volume"],
    )
    atr = _atr(high, low, close, 14)
    ret = close.pct_change()

    out = pd.DataFrame(index=df.index)
    out["return_1"] = ret
    out["return_3"] = close.pct_change(3)
    out["return_10"] = close.pct_change(10)
    out["return_20"] = close.pct_change(20)
    ema_f = close.ewm(span=9, adjust=False).mean()
    ema_s = close.ewm(span=21, adjust=False).mean()
    out["ema_fast_gap"] = (ema_f - ema_s) / ema_s.replace(0.0, np.nan)
    macd_line = close.ewm(span=12, adjust=False).mean() - close.ewm(
        span=26, adjust=False
    ).mean()
    out["macd_hist_norm"] = macd_line / close.replace(0.0, np.nan)
    out["vol_ratio_20"] = vol / vol.rolling(20, min_periods=1).mean()
    out["atr_norm"] = atr / close.replace(0.0, np.nan)
    out["rsi_14"] = _rsi(close, 14)
    out["rsi_overbought"] = (out["rsi_14"] - 70.0) / 30.0
    out["rsi_oversold"] = (30.0 - out["rsi_14"]) / 30.0
    mid = close.rolling(20, min_periods=20).mean()
    std = close.rolling(20, min_periods=20).std(ddof=0)
    out["bb_dist"] = ((close - mid) / (2.0 * std.replace(0.0, np.nan)))
    out = out.loc[:, _ALL_COLUMNS]
    return out


def feature_columns(kind: str) -> list[str]:
    """Column subset a given kind of model consumes (kind = momentum|mean_reversion)."""
    if kind == "momentum":
        return list(_MOMENTUM_COLUMNS)
    if kind == "mean_reversion":
        return list(_REVERSION_COLUMNS)
    raise ValueError(f"unknown model kind {kind!r}")


def build_dataset(
    bars: Sequence[Bar],
    label_horizon_bars: int,
    *,
    kind: str = "momentum",
    min_abs_move: float = 0.001,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a supervised (X, y) dataset from a bar window.

    Labels come from the **forward** close ``label_horizon_bars`` bars ahead:
    ``y = +1`` when forward return > +``min_abs_move``, ``-1`` when below
    -``min_abs_move``, and ``0`` when the move is too small — a third, learnable
    "stay flat" class (a real bot should also learn to sit out sideways regimes).
    Rows without enough history (NaN features) or without a future window are
    dropped, so X and y are always in lockstep.
    """
    feats = compute_features(bars)
    if feats.empty:
        return np.empty((0, 0)), np.empty((0,))
    horiz = int(label_horizon_bars)
    # compute_features drops the raw close column, so recompute the target here,
    # aligned to the feature index (bars may arrive unordered).
    closes = pd.Series(
        {pd.Timestamp(b.ts_utc_ns, unit="ns", tz="UTC"): b.close for b in bars},
        dtype="float64",
    )
    feats = feats.assign(close=closes.loc[feats.index])
    fwd = feats["close"].shift(-horiz)
    ret = fwd / feats["close"] - 1.0
    y = np.where(ret > min_abs_move, 1.0, np.where(ret < -min_abs_move, -1.0, 0.0))
    keep = (
        np.isfinite(y)
        & np.isfinite(feats[feature_columns(kind)].to_numpy()).all(axis=1)
    )
    X = feats.loc[keep, feature_columns(kind)].to_numpy(dtype="float64")
    y = y[keep]
    return X, y.astype(int)


def last_features(bars: Sequence[Bar], kind: str) -> dict[str, float]:
    """Latest feature row as a feature-name -> value dict (live inference).

    Returns an empty dict during warm-up (no usable last row), so the calling
    strategy emits HOLD rather than predicting on NaNs.
    """
    feats = compute_features(bars)
    if feats.empty or len(feats) < 1:
        return {}
    row = feats.iloc[-1]
    values = {c: float(row[c]) for c in feature_columns(kind)}
    if any(math.isnan(v) for v in values.values()):
        return {}
    return values


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss
    out = 100.0 - (100.0 / (1.0 + rs))
    out = out.fillna(100.0)
    flat = avg_loss.eq(0.0) & avg_gain.eq(0.0)
    out = out.mask(flat, 50.0)
    return out.fillna(50.0)


__all__ = [
    "FEATURE_VERSION",
    "build_dataset",
    "compute_features",
    "feature_columns",
    "last_features",
]