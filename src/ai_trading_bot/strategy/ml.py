"""ML strategies (Phase 4): momentum and mean-reversion models behind the facade.

Both strategies speak the exact same :class:`~ai_trading_bot.domain.Signal`
contract as the rule-based ones, so the risk gate treats them identically
(confidence feeds REQ-RSK-05 min-confidence and Kelly sizing). Each is driven by
a versioned model artifact from the registry, and **every signal records the
model version that produced it** for the audit trail (REQ-MON-03).

Safety: if no model is trained yet, or inference is mid-warm-up, the strategy
returns HOLD ``MODEL_MISSING`` / ``WARMUP`` — it never invents a signal from a
missing or corrupt artifact, and never crashes the engine loop.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ai_trading_bot.domain import Bar, Instrument, Signal, SignalDirection
from ai_trading_bot.features.ml import (
    FEATURE_VERSION,
    feature_columns,
    last_features,
)
from ai_trading_bot.models import ModelRepo

# Signal-strength floor: never emit a direction the model isn't clear on.
min_probability = 0.55


class _MLStrategyBase:
    """Shared inference path for the two ML strategies."""

    id: str = "ml_base"
    kind: str = "momentum"  # subclasses pick their feature subset + artifact
    version: str = "2.0.0"

    def __init__(
        self,
        repo: ModelRepo | None = None,
        model_id: str | None = None,
        threshold: float = min_probability,
    ) -> None:
        self._repo = repo or ModelRepo()
        self._model_id = model_id or self.id
        self.threshold = threshold

    def _model_version_or_none(self) -> str | None:
        """Active artifact version as ``"ml_momentum@v3"``, or None when untrained."""
        version = self._repo.active_version(self._model_id)
        return None if version is None else f"{self._model_id}@v{version}"

    def generate(
        self,
        instrument: Instrument,
        bars: Sequence[Bar],
        controller_inputs: dict[str, object] | None = None,
    ) -> Signal:
        feats = last_features(bars, self.kind)
        if not feats:
            return Signal(
                instrument.id, SignalDirection.HOLD, 0.0, self.id, reason_code="WARMUP"
            )
        try:
            estimator, meta = self._repo.load(self._model_id)
        except FileNotFoundError:
            return Signal(
                instrument.id, SignalDirection.HOLD, 0.0, self.id,
                reason_code="MODEL_MISSING",
            )
        # The Last known model feature_version must match the live feature set —
        # a mismatch means serving on features the model was never trained on.
        if meta.feature_version != FEATURE_VERSION:
            return Signal(
                instrument.id, SignalDirection.HOLD, 0.0, self.id,
                reason_code="FEATURE_MISMATCH",
            )
        return self._signal_from_proba(
            instrument, self._predict_probability(estimator, feats), meta.version
        )

    def _predict_probability(self, estimator, feats: dict[str, float]) -> float:
        """P(class +1) using the exact same columns the model was trained on.

        Labels are {-1, 0, 1} (the 0 class is "stay flat" — informative, rarely
        triggers). predict_proba columns follow sorted classes, so the positive
        class is found by lookup, never hardcoded as index 1.
        """
        cols = feature_columns(self.kind)
        x = np.array([feats[c] for c in cols], dtype="float64").reshape(1, -1)
        if hasattr(estimator, "predict_proba"):
            proba = estimator.predict_proba(x)[0]
            classes = [int(c) for c in estimator.classes_]
            return float(proba[classes.index(1)])
        return float(estimator.decision_function(x)[0])

    def _signal_from_proba(
        self, instrument: Instrument, p_up: float, model_version: int
    ) -> Signal:
        if p_up >= self.threshold:
            return self._signal(instrument, SignalDirection.BUY, p_up, model_version)
        p_down = 1.0 - p_up
        if p_down >= self.threshold:
            return self._signal(instrument, SignalDirection.SELL, p_down, model_version)
        return Signal(
            instrument.id, SignalDirection.HOLD, 0.0, self.id,
            reason_code=f"LOW_CONF_{p_up:.2f}",
        )

    def _signal(
        self, instrument: Instrument, direction: SignalDirection,
        probability: float, model_version: int,
    ) -> Signal:
        # Map probability [threshold, 1] -> confidence [0.6, 0.95].
        conf = min(0.95, 0.6 + (probability - self.threshold) * 0.7)
        return Signal(
            instrument.id, direction, round(conf, 3), self.id,
            reason_code=direction.value.upper() + "_ML",
            model_version=f"{self._model_id}@v{model_version}",
            horizon_seconds=300,  # M5 bars are the ML trading cadence
        )


class MLMomentumStrategy(_MLStrategyBase):
    """Trade the model's read on the next horizon's direction (trend attach)."""

    id = "ml_momentum"
    kind = "momentum"


class MLMeanReversionStrategy(_MLStrategyBase):
    """Trade the model's read on short-term overextension reverting."""

    id = "ml_mean_reversion"
    kind = "mean_reversion"


def build_ml_strategies(repo: ModelRepo) -> dict[str, object]:
    """Instantiate both ML strategies against one registry (engine wiring)."""
    return {
        MLMomentumStrategy.id: MLMomentumStrategy(repo=repo),
        MLMeanReversionStrategy.id: MLMeanReversionStrategy(repo=repo),
    }


__all__ = [
    "MLMeanReversionStrategy",
    "MLMomentumStrategy",
    "build_ml_strategies",
]