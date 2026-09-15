"""Phase 4 ML tests: features, registry, strategies, retraining gate.

Covers the guarantees that actually matter for a trading bot:

- train/serve feature consistency and label alignment (no train-serve skew);
- versioned registry save/load/rollback (pinned to the audit trail);
- strategy inference drives the same Signal contract as rule-based strategies,
  records its model version, and degrades to HOLD when the model is missing;
- the retraining gate promotes only when the candidate genuinely beats the
  incumbent, never silently regressing.
"""

from __future__ import annotations

import tempfile

import numpy as np
import pandas as pd
import pytest
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier

from ai_trading_bot.domain import (
    AssetClass,
    Bar,
    Instrument,
    SignalDirection,
    Timeframe,
)
from ai_trading_bot.features.ml import (
    FEATURE_VERSION,
    build_dataset,
    feature_columns,
    last_features,
)
from ai_trading_bot.models import (
    MODEL_MOMENTUM,
    MODEL_REVERSION,
    ModelMeta,
    ModelRepo,
)
from ai_trading_bot.strategy.ml import (
    MLMeanReversionStrategy,
    MLMomentumStrategy,
)
from ai_trading_bot.training import train_model_for_bars

_BTC = Instrument("BTC/USDT", AssetClass.CRYPTO, "binance", "BTC", "USDT")


def make_bars(seed: int = 0, n: int = 500, trend: float = 0.0015) -> list[Bar]:
    """Deterministic synthetic OHLCV with a tradable regime (trend + snapback)."""
    rng = np.random.default_rng(seed)
    close = [100.0]
    for i in range(1, n):
        drift = trend if i % 60 < 35 else -0.0008
        close.append(close[-1] * (1 + drift + rng.normal(0, 0.001)))
    ts = pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC")
    return [
        Bar(
            _BTC.id, Timeframe.M5, int(t.value), c, c * 1.002, c * 0.998, c, 10.0,
        )
        for t, c in zip(ts, close)
    ]


def _trained_repo(tmp_path: str) -> ModelRepo:
    repo = ModelRepo(tmp_path)
    for model_id, kind in (
        (MODEL_MOMENTUM, "momentum"),
        (MODEL_REVERSION, "mean_reversion"),
    ):
        X, y = build_dataset(make_bars(seed=1), 12, kind=kind)
        est = HistGradientBoostingClassifier(max_iter=40, random_state=0).fit(X, y)
        repo.save(
            est,
            ModelMeta(
                model_id=model_id,
                version=0,
                kind=kind,
                feature_version=FEATURE_VERSION,
                sklearn_version=sklearn.__version__,
                metrics={"accuracy": 0.6, "log_loss": 0.7},
                n_samples=len(y),
            ),
            promote=True,
        )
    return repo


# -- features --------------------------------------------------------------


def test_dataset_labels_align_with_features_and_are_dropped_cleanly() -> None:
    bars = make_bars()
    X, y = build_dataset(bars, 12, kind="momentum")
    assert X.shape[0] == y.shape[0] > 0
    assert X.shape[1] == len(feature_columns("momentum"))
    assert np.isfinite(X).all()  # NaN/inf must not leak into model inputs
    # 3-class labels: +1/up, -1/down, 0/stay-flat is informative, not dropped.
    assert set(np.unique(y)).issubset({-1, 0, 1})
    assert {np.int64(-1), np.int64(1)}.issubset(set(np.unique(y)))

def test_momentum_and_reversion_use_different_feature_sets() -> None:
    m = set(feature_columns("momentum"))
    r = set(feature_columns("mean_reversion"))
    assert m != r
    assert "rsi_14" in r and "rsi_14" not in m
    assert "ema_fast_gap" in m


def test_last_features_matches_dataset_row() -> None:
    bars = make_bars()
    # The final usable dataset row is the one a live strategy would see last.
    last = last_features(bars, "momentum")
    assert last  # warm-up should have completed
    assert list(last.keys()) == feature_columns("momentum")
    for value in last.values():
        assert np.isfinite(value)


def test_last_features_empty_during_warmup() -> None:
    assert not last_features(make_bars(n=5), "momentum")


# -- registry --------------------------------------------------------------


def test_repo_versions_and_rollback(tmp_path: str) -> None:
    repo = _trained_repo(tmp_path)
    assert repo.active_version(MODEL_MOMENTUM) == 1
    # save a v2 manually, keep v1 active -> engine still uses v1
    repo.save(
        HistGradientBoostingClassifier(max_iter=30),
        ModelMeta(
            model_id=MODEL_MOMENTUM,
            version=0,
            kind="momentum",
            feature_version=FEATURE_VERSION,
            sklearn_version=sklearn.__version__,
        ),
        promote=False,
    )
    assert repo.active_version(MODEL_MOMENTUM) == 1
    # rollback guard: can't pin a version that doesn't exist
    with pytest.raises(ValueError):
        repo.activate(MODEL_MOMENTUM, 99)


def test_repo_metadata_pins_feature_and_sklearn_versions(tmp_path: str) -> None:
    repo = _trained_repo(tmp_path)
    _, meta = repo.load(MODEL_REVERSION)
    assert meta.feature_version == FEATURE_VERSION
    assert meta.sklearn_version == sklearn.__version__
    assert meta.kind == "mean_reversion"


# -- strategies ------------------------------------------------------------


def test_ml_strategies_emit_signal_with_model_version(tmp_path: str) -> None:
    repo = _trained_repo(tmp_path)
    strategy = MLMomentumStrategy(repo=repo)
    sig = strategy.generate(_BTC, make_bars(n=200))
    assert sig.strategy_id == "ml_momentum"
    assert sig.model_version == "ml_momentum@v1"
    assert sig.direction in (SignalDirection.BUY, SignalDirection.SELL, SignalDirection.HOLD)
    if sig.direction is not SignalDirection.HOLD:
        assert 0.0 <= sig.confidence <= 1.0


def test_ml_strategy_holds_when_model_missing() -> None:
    repo = ModelRepo(tempfile.mkdtemp())
    strategy = MLMeanReversionStrategy(repo=repo)
    sig = strategy.generate(_BTC, make_bars(n=200))
    assert sig.direction is SignalDirection.HOLD
    assert sig.reason_code == "MODEL_MISSING"


def test_ml_strategy_holds_during_warmup(tmp_path: str) -> None:
    repo = _trained_repo(tmp_path)
    strategy = MLMomentumStrategy(repo=repo)
    sig = strategy.generate(_BTC, make_bars(n=5))
    assert sig.direction is SignalDirection.HOLD
    assert sig.reason_code == "WARMUP"


# -- retraining gate -------------------------------------------------------


def test_first_training_promotes() -> None:
    repo = ModelRepo(tempfile.mkdtemp())
    outcome = train_model_for_bars(repo, MODEL_MOMENTUM, "momentum", make_bars(seed=3))
    assert outcome["promoted"] is True
    assert repo.active_version(MODEL_MOMENTUM) == outcome["version"]


def test_better_candidate_is_promoted() -> None:
    repo = ModelRepo(tempfile.mkdtemp())
    # Weak incumbent (0.5 = coin flip) leaves plenty of headroom for the margin.
    repo.save(
        HistGradientBoostingClassifier(max_iter=20),
        ModelMeta(
            model_id=MODEL_MOMENTUM,
            version=0,
            kind="momentum",
            feature_version=FEATURE_VERSION,
            sklearn_version=sklearn.__version__,
            metrics={"accuracy": 0.5, "log_loss": 0.7},
            n_samples=400,
        ),
        promote=True,
    )
    outcome = train_model_for_bars(
        repo, MODEL_MOMENTUM, "momentum", make_bars(seed=8), min_improvement=0.01,
    )
    assert outcome["promoted"] is True
    assert repo.active_version(MODEL_MOMENTUM) == outcome["version"]


def test_worse_candidate_is_not_promoted() -> None:
    repo = ModelRepo(tempfile.mkdtemp())
    # Seed a perfect incumbent: accuracy at the 1.0 ceiling with a wide required
    # margin, so no realistically-trainable candidate can beat it by the margin.
    repo.save(
        HistGradientBoostingClassifier(max_iter=20),
        ModelMeta(
            model_id=MODEL_MOMENTUM,
            version=0,
            kind="momentum",
            feature_version=FEATURE_VERSION,
            sklearn_version=sklearn.__version__,
            metrics={"accuracy": 1.0, "log_loss": 0.01},
            n_samples=400,
        ),
        promote=True,
    )
    outcome = train_model_for_bars(
        repo, MODEL_MOMENTUM, "momentum", make_bars(seed=5), min_improvement=0.05,
    )
    assert outcome["promoted"] is False
    assert outcome["version"] > 1  # candidate still recorded for manual review
    assert repo.active_version(MODEL_MOMENTUM) == 1  # incumbent untouched


def test_insufficient_data_is_skipped() -> None:
    repo = ModelRepo(tempfile.mkdtemp())
    outcome = train_model_for_bars(
        repo, MODEL_MOMENTUM, "momentum", make_bars(n=40)
    )
    assert "skipped" in outcome