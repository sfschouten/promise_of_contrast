from __future__ import annotations

import warnings

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from .base import Probe


class LogisticProbe(Probe):
    method = "logistic"

    def __init__(self):
        self._clf = LogisticRegression(max_iter=1000)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            self._clf.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._clf.predict_proba(X)[:, 1]

    def predict_proba_matrix(self, X: np.ndarray) -> np.ndarray:
        return self._clf.predict_proba(X)

    @property
    def subspace(self) -> torch.Tensor:
        coef = torch.from_numpy(self._clf.coef_[0]).float()
        norm = coef.norm()
        if norm < 1e-12:
            raise ValueError("logistic probe has near-zero weight vector — no meaningful subspace")
        return (coef / norm).unsqueeze(0)  # (1, hidden_size)

    def state_dict(self) -> dict:
        return {"clf": self._clf}

    @classmethod
    def from_state_dict(cls, d: dict) -> LogisticProbe:
        probe = cls()
        probe._clf = d["clf"]
        return probe
