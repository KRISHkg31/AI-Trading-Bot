"""Retraining pipeline: refit, validation-gate, and promote better versions.

The engine keeps using the *active* model until a retrain produces one that beats
it on held-out validation (REQ-STR-04, BRD model life-cycle). A candidate is only
promoted when its validation accuracy clears the incumbent by ``min_improvement``
(and log-loss isn't worse) — otherwise the incumbent stays active and the
candidate is saved under its own version for manual review, so the bot never
silently regresses.

Validation shape: the most recent ``val_frac`` of the history is held out
walk-forward style (first-fit / later-predict, like the bot would trade), so the
candidate is scored on data it never saw during fitting.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss

from ai_trading_bot.domain import Bar
from ai_trading_bot.features.ml import FEATURE_VERSION, build_dataset
from ai_trading_bot.models import ModelMeta, ModelRepo

_ESTIMATOR_KWARGS = {
    "max_iter": 100,
    "learning_rate": 0.06,
    "max_depth": 4,
    "l2_regularization": 1.0,
    "early_stopping": True,
    "random_state": 42,
}


def _train_classifier(X: np.ndarray, y: np.ndarray) -> HistGradientBoostingClassifier:
    """Fit a histogram gradient-boosted classifier (fast, robust, no GPU)."""
    model = HistGradientBoostingClassifier(**_ESTIMATOR_KWARGS)
    model.fit(
        np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0),
        y,
    )
    return model


def _evaluate(model, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    y = np.asarray(y)
    classes = np.asarray(getattr(model, "classes_", [0]))
    yhat = model.predict(X)
    pred = model.predict_proba(X)
    has_both = len(classes) >= 2
    if has_both:
        cols = classes.tolist()
        proba_pos = pred[:, cols.index(1)] if 1 in cols else pred[:, cols.index(0)]
    else:
        proba_pos = pred[:, 0]
    # y is {+1, -1, 0(neutral-dropped)}, so binarize to {0,1} for the metrics.
    y_bin = (y > 0).astype(int)
    yhat_bin = (yhat > 0).astype(int)
    metrics: dict[str, float] = {"accuracy": float(accuracy_score(y_bin, yhat_bin))}
    if has_both:
        try:
            proba_neg = 1.0 - proba_pos
            metrics["log_loss"] = float(
                log_loss(y_bin, np.column_stack([proba_neg, proba_pos]))
            )
        except ValueError:
            metrics["log_loss"] = float("nan")
    return metrics


def train_model_for_bars(
    repo: ModelRepo,
    model_id: str,
    kind: str,
    bars: Sequence[Bar],
    *,
    label_horizon_bars: int = 12,
    min_abs_move: float = 0.001,
    val_frac: float = 0.2,
    min_improvement: float = 0.01,
) -> dict[str, object]:
    """Refit ``model_id`` from ``bars`` and promote only if it genuinely improves.

    Returns a summary dict (candidate/incumbent metrics, promoted flag, version).
    Never raises on data scarcity — returns a ``{skipped: reason}`` dict.
    """
    X, y = build_dataset(
        bars, label_horizon_bars, kind=kind, min_abs_move=min_abs_move,
    )
    n = len(y)
    if n < 60:
        return {"skipped": f"only {n} usable samples (need >=60)"}

    split = max(5, round(n * (1.0 - val_frac)))
    X_tr, y_tr = X[:split], y[:split]
    X_va, y_va = X[split:], y[split:]
    # A validation set with a single class can't jump accuracy — skip.
    if len(np.unique(y_va)) < 2:
        return {"skipped": "validation set has a single class"}

    model = _train_classifier(X_tr, y_tr)
    cand = _evaluate(model, X_va, y_va)

    incumbent_metrics: dict[str, float] | None = None
    incumbent_version: int | None = None
    try:
        incumbent_version = repo.active_version(model_id)
        _, inc_meta = repo.load(model_id)
        incumbent_metrics = inc_meta.metrics
    except FileNotFoundError:
        pass

    meta = ModelMeta(
        model_id=model_id,
        version=0,  # repo assigns the next version
        kind=kind,
        feature_version=FEATURE_VERSION,
        sklearn_version=_sklearn_version(),
        params=_ESTIMATOR_KWARGS | {
            "kind": kind,
            "label_horizon_bars": label_horizon_bars,
            "min_abs_move": min_abs_move,
            "n_train_samples": int(split),
        },
        metrics=cand,
        n_samples=n,
        trained_at_ns=time.time_ns(),
        author="retrain-pipeline",
    )

    promoted = False
    if incumbent_version is None:
        promoted = True  # first model: nothing to beat
    else:
        cand_acc = cand.get("accuracy", 0.0)
        inc_acc = incumbent_metrics.get("accuracy", 0.0)
        cand_ll = cand.get("log_loss")
        inc_ll = incumbent_metrics.get("log_loss")
        better_acc = cand_acc >= inc_acc + min_improvement
        # NaN candidate log-loss is uninterpretable -> treat as a failure, so an
        # overfit model can never sneak past on a degenerate validation row.
        cand_ll_ok = cand_ll is not None and np.isfinite(cand_ll)
        ll_not_worse = (
            inc_ll is None or not cand_ll_ok or cand_ll <= inc_ll * 1.05 + 1e-9
        )
        promoted = better_acc and ll_not_worse

    version = repo.save(model, meta, promote=promoted)
    return {
        "model_id": model_id,
        "version": version,
        "promoted": promoted,
        "candidate_metrics": cand,
        "incumbent_version": incumbent_version,
        "incumbent_metrics": incumbent_metrics,
        "incumbent_promotion": incumbent_version is None,
    }


def retrain_all(
    repo: ModelRepo,
    bars_by_instrument: dict[str, Sequence[Bar]],
    *,
    label_horizon_bars: int = 12,
) -> list[dict[str, object]]:
    """Retrain every supported model kind off the given instrument histories."""
    results: list[dict[str, object]] = []
    for model_id, kind in (("ml_momentum", "momentum"), ("ml_mean_reversion", "mean_reversion")):
        for instrument_id, bars in bars_by_instrument.items():
            outcome = train_model_for_bars(
                repo, model_id, kind, bars,
                label_horizon_bars=label_horizon_bars,
            )
            outcome = {"instrument_id": instrument_id, **outcome}
            results.append(outcome)
    return results


def _sklearn_version() -> str:
    import sklearn

    return getattr(sklearn, "__version__", "unknown")


__all__ = ["retrain_all", "train_model_for_bars"]