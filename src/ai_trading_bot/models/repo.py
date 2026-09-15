"""Model registry: a versioned, auditable home for ML artifacts (BRD model-mgmt).

Every artifact is immutable once written:

- ``data/models/<model_id>/v<N>.joblib``  — the fitted estimator (joblib).
- ``data/models/<model_id>/v<N>.json``    — metadata pinned at creation time:
  feature_version, sklearn_version, params, metrics, training window, author.

The **active version** per model id is stored in ``data/models/<model_id>/active.json``.
The engine always loads the active version, so rolling back a bad deployment is a
one-field file flip (no code, no redeploy) — the Phase 6 go-live guardrail
(REQ-STR-04, REQ-NFR-* versioning).

Nothing here trains. :func:`save` ingests a fitted estimator + metric snapshot;
``training.pipeline`` decides *when* to promote a candidate as a new version.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib

# Known model ids so strategies and pipelines can quote them without magic strings.
MODEL_MOMENTUM = "ml_momentum"
MODEL_REVERSION = "ml_mean_reversion"
MODEL_IDS = (MODEL_MOMENTUM, MODEL_REVERSION)


@dataclass(frozen=True, slots=True)
class ModelMeta:
    """Immutable metadata describing one fitted version (pinned at write time)."""

    model_id: str
    version: int  # auto-incremented; 0 means "draft / not yet versioned"
    kind: str  # MODEL_MOMENTUM | MODEL_REVERSION human-readable
    feature_version: str  # which FeatureEngine / builder produced the inputs
    sklearn_version: str  # estimator reproducibility pin
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    n_samples: int = 0
    train_start_ns: int | None = None
    train_end_ns: int | None = None
    trained_at_ns: int = 0
    author: str = "pipeline"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelMeta:
        return cls(**data)


class ModelRepo:
    """File-backed registry of versioned model artifacts.

    Layout (one dir per model id)::

        data/models/ml_momentum/
          v1.joblib      v1.json      v2.joblib      v2.json
          active.json   # {"version": 2}
    """

    def __init__(self, base_dir: str | Path = "data/models") -> None:
        self._root = Path(base_dir)
        self._root.mkdir(parents=True, exist_ok=True)

    # -- paths -------------------------------------------------------------
    def _model_dir(self, model_id: str) -> Path:
        return self._root / model_id

    def _version_path(self, model_id: str, version: int) -> Path:
        return self._model_dir(model_id) / f"v{version}.joblib"

    def _meta_path(self, model_id: str, version: int) -> Path:
        return self._model_dir(model_id) / f"v{version}.json"

    def _active_path(self, model_id: str) -> Path:
        return self._model_dir(model_id) / "active.json"

    # -- version management ------------------------------------------------
    def next_version(self, model_id: str) -> int:
        """Auto-increment version for ``model_id`` (first save is v1)."""
        versions = self.versions(model_id)
        return (max(versions) + 1) if versions else 1

    def versions(self, model_id: str) -> list[int]:
        if not self._model_dir(model_id).exists():
            return []
        return sorted(
            int(p.stem[1:])
            for p in self._model_dir(model_id).glob("v*.joblib")
            if p.stem.startswith("v") and p.stem[1:].isdigit()
        )

    def active_version(self, model_id: str) -> int | None:
        path = self._active_path(model_id)
        if not path.exists():
            # No pin yet: default to the newest available version.
            versions = self.versions(model_id)
            return max(versions) if versions else None
        try:
            return int(json.loads(path.read_text(encoding="utf-8"))["version"])
        except (ValueError, KeyError):
            versions = self.versions(model_id)
            return max(versions) if versions else None

    def activate(self, model_id: str, version: int) -> None:
        """Pin the active version (the rollback lever). Refuses bad versions."""
        if version not in self.versions(model_id):
            raise ValueError(
                f"{model_id} has no version v{version}; have {self.versions(model_id)}"
            )
        self._model_dir(model_id).mkdir(parents=True, exist_ok=True)
        self._active_path(model_id).write_text(
            json.dumps({"version": version}, indent=2), encoding="utf-8"
        )

    # -- artifact I/O ------------------------------------------------------
    def save(
        self,
        estimator: Any,
        meta: ModelMeta,
        promote: bool = True,
    ) -> int:
        """Persist a fitted estimator + metadata as a new immutable version.

        Reports the assigned version. ``promote`` controls whether it becomes the
        active version immediately (training promotes only after it wins the
        incumbent comparison; model-less pipelines/backtests can look it up
        explicitly by version instead).
        """
        version = meta.version or self.next_version(meta.model_id)
        meta = ModelMeta(
            **{**meta.to_dict(), "version": version, "trained_at_ns": meta.trained_at_ns}
        )
        d = self._model_dir(meta.model_id)
        d.mkdir(parents=True, exist_ok=True)
        joblib.dump(estimator, self._version_path(meta.model_id, version))
        self._meta_path(meta.model_id, version).write_text(
            json.dumps(meta.to_dict(), indent=2, default=str), encoding="utf-8"
        )
        # Activate idempotently on promote — always write the pin so the active
        # version survives new unpromoted saves (lenient active_version() looks
        # like it tracks this, but it only falls back to newest when *our* pin is
        # absent, which would silently defeat rollback).
        if promote:
            self.activate(meta.model_id, version)
        return version

    def load(self, model_id: str, version: int | None = None) -> tuple[Any, ModelMeta]:
        """Load the estimator + metadata for a version (default: the active one).

        Raises ``FileNotFoundError`` when nothing is available — callers are
        expected to treat that as "model not yet trained" and emit HOLD, not crash.
        """
        version = version or self.active_version(model_id)
        if version is None:
            raise FileNotFoundError(f"no model for {model_id}")
        est = joblib.load(self._version_path(model_id, version))
        meta = ModelMeta.from_dict(
            json.loads(self._meta_path(model_id, version).read_text(encoding="utf-8"))
        )
        return est, meta

    def meta(self, model_id: str, version: int | None = None) -> ModelMeta:
        _, meta = self.load(model_id, version)
        return meta

    def list(self) -> list[dict[str, Any]]:
        """Human/CLI-friendly snapshot of every model version and the active pins."""
        rows: list[dict[str, Any]] = []
        for model_id in sorted(self._root.iterdir()):
            if not model_id.is_dir():
                continue
            active = self.active_version(model_id.name)
            for v in self.versions(model_id.name):
                meta = self.meta(model_id.name, v)
                rows.append(
                    {
                        "model_id": meta.model_id,
                        "version": meta.version,
                        "active": meta.version == active,
                        "kind": meta.kind,
                        "feature_version": meta.feature_version,
                        "sklearn_version": meta.sklearn_version,
                        "accuracy": meta.metrics.get("accuracy"),
                        "log_loss": meta.metrics.get("log_loss"),
                        "n_samples": meta.n_samples,
                        "trained_at": meta.trained_at_ns,
                    }
                )
        return rows


__all__ = [
    "MODEL_IDS",
    "MODEL_MOMENTUM",
    "MODEL_REVERSION",
    "ModelMeta",
    "ModelRepo",
]