from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from .base import Probe


class DiffInMeansProbe(Probe):
    """Supervised difference-in-means, a.k.a. the mass-mean probe.

    Direction = normalised(mean(X[y==1]) - mean(X[y==0])), with a 1-D logistic
    regression on the projected scalar for calibrated probabilities.

    This is the estimator the name conventionally refers to (Marks & Tegmark 2023),
    and it **uses labels**: the direction separates true statements from false ones.
    It deliberately has no `fit_paired`, so that contrastive data does not silently
    route it onto a different estimator — see ContrastDiffInMeansProbe for the
    label-free counterpart.
    """
    method = "diff_in_means"

    def __init__(self):
        self._direction: np.ndarray | None = None
        self._clf: LogisticRegression | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        diff = X[y == 1].mean(axis=0) - X[y == 0].mean(axis=0)
        norm = np.linalg.norm(diff)
        self._direction = diff / norm if norm > 0 else diff
        self._calibrate(X, y)

    def _calibrate(self, X: np.ndarray, y: np.ndarray) -> None:
        projections = X @ self._direction[:, None]
        self._clf = LogisticRegression(max_iter=1000)
        self._clf.fit(projections, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        projections = X @ self._direction[:, None]
        return self._clf.predict_proba(projections)[:, 1]

    @property
    def subspace(self) -> torch.Tensor:
        return torch.from_numpy(self._direction).float().unsqueeze(0)

    def state_dict(self) -> dict:
        return {"direction": self._direction, "clf": self._clf}

    @classmethod
    def from_state_dict(cls, d: dict) -> DiffInMeansProbe:
        probe = cls()
        probe._direction = d["direction"]
        probe._clf = d["clf"]
        return probe


class ContrastDiffInMeansProbe(DiffInMeansProbe):
    """Label-free difference in means between the two branches of a contrast pair.

    Direction = normalised(mean(X_base) - mean(X_cf)).  Where DiffInMeansProbe asks
    which way true statements sit relative to false ones, this asks which way one
    framing sits relative to the other, and needs no labels to do it.

    The two coincide only when the branch happens to indicate truth.  On the paper's
    datasets it does not — every pair contains one correct and one incorrect framing,
    so the branch is uninformative about truth and the two directions are unrelated.
    That gap is the point of the method, not a defect: it is the difference between a
    supervised reference point and a contrast-only one.
    """
    method = "contrast_diff_in_means"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError(
            "ContrastDiffInMeansProbe is contrast-only; use fit_paired(X_base, X_cf)."
        )

    def fit_paired(self, X_base: np.ndarray, X_cf: np.ndarray) -> None:
        direction = (X_base - X_cf).mean(axis=0)
        norm = np.linalg.norm(direction)
        self._direction = direction / norm if norm > 0 else direction
        X_all = np.vstack([X_base, X_cf])
        y_all = np.array([1] * len(X_base) + [0] * len(X_cf))
        self._calibrate(X_all, y_all)

    @classmethod
    def from_state_dict(cls, d: dict) -> "ContrastDiffInMeansProbe":
        probe = cls()
        probe._direction = d["direction"]
        probe._clf = d["clf"]
        return probe
