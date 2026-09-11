from __future__ import annotations

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from .base import Probe


class PCAProbe(Probe):
    """
    Fits PCA on training activations, then a logistic regression in the reduced space.
    subspace returns the top-k principal components as an orthonormal basis (k, hidden_size).
    """
    @property
    def method(self) -> str:
        return f"pca_{self.n_components}"

    def __init__(self, n_components: int = 16):
        self.n_components = n_components
        self._pca: PCA | None = None
        self._clf: LogisticRegression | None = None
        self._artifacts: dict[str, np.ndarray] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._pca = PCA(n_components=min(self.n_components, X.shape[0], X.shape[1]))
        X_reduced = self._pca.fit_transform(X)
        self._clf = LogisticRegression(max_iter=1000)
        self._clf.fit(X_reduced, y)
        self._artifacts = {
            "explained_variance_ratio": self._pca.explained_variance_ratio_,
            "explained_variance":       self._pca.explained_variance_,
        }

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._clf.predict_proba(self._pca.transform(X))[:, 1]

    @property
    def artifacts(self) -> dict[str, np.ndarray]:
        return self._artifacts

    @property
    def subspace(self) -> torch.Tensor:
        # PCA components are already orthonormal by construction.
        return torch.from_numpy(self._pca.components_).float()  # (k, hidden_size)

    def state_dict(self) -> dict:
        return {
            "pca": self._pca, "clf": self._clf,
            "n_components": self.n_components, "artifacts": self._artifacts,
        }

    @classmethod
    def from_state_dict(cls, d: dict) -> PCAProbe:
        probe = cls(n_components=d["n_components"])
        probe._pca = d["pca"]
        probe._clf = d["clf"]
        probe._artifacts = d.get("artifacts", {})
        return probe


class PairedPCAProbe(PCAProbe):
    """
    PCA over a *derived* matrix built from contrastive pairs, with a single principal
    component as the probe direction.

    ``style`` selects the matrix the PCA is fitted to:

        "subtract"  Xc − Xb   — the top component is the contrastive representation
                    clustering direction (CRC-TPC); components 0…k give the per-component
                    accuracies such analyses report.
        "concat"    [Xb; Xc]  — plain PCA of all activations, the usual baseline.
        "base_only" / "cf_only"  — one branch, for diagnosing which side carries variance.

    ``component_index`` picks which component becomes the direction, so component 2 can be
    probed on its own rather than as part of a top-k subspace.

    This is deliberately a subclass rather than a flag on ``PCAProbe``: ``fit_probe``
    dispatches on the *presence* of ``fit_paired``, so giving the parent one would silently
    reroute every existing ``pca_*`` run onto the paired path — which uses role-derived
    labels rather than the descriptor, and so would change results on datasets where both
    sides of a pair share a label (e.g. the cities grid's truth-preserving edges).
    """

    requires_pairs = True

    def __init__(self, style: str = "subtract", component_index: int = 0):
        super().__init__(n_components=component_index + 1)
        self.style = style
        self.component_index = component_index

    @property
    def method(self) -> str:
        return f"pca_pair_{self.style}_k{self.component_index}"

    def _matrix(self, X_base: np.ndarray, X_cf: np.ndarray) -> np.ndarray:
        if self.style == "subtract":
            return X_cf - X_base
        if self.style == "concat":
            return np.vstack([X_base, X_cf])
        if self.style == "base_only":
            return X_base
        if self.style == "cf_only":
            return X_cf
        raise ValueError(f"unknown style {self.style!r}")

    def fit_paired(self, X_base: np.ndarray, X_cf: np.ndarray) -> None:
        both = self._matrix(X_base, X_cf).astype(np.float64)
        self._pca = PCA(n_components=min(self.n_components, *both.shape))
        self._pca.fit(both)
        self._artifacts = {
            "explained_variance_ratio": self._pca.explained_variance_ratio_,
            "explained_variance":       self._pca.explained_variance_,
        }
        # Calibration head on the single chosen direction.  Labels are role-derived, so
        # this is meaningful only for framing-ordered pairs; raw_score is the
        # sign-of-projection alternative that needs no labels at all.
        X_all = np.vstack([X_base, X_cf])
        y_all = np.array([1] * len(X_base) + [0] * len(X_cf))
        self._clf = LogisticRegression(max_iter=1000)
        self._clf.fit(X_all @ self.subspace.numpy().T, y_all)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError(
            "PairedPCAProbe requires paired data; use fit_paired(X_base, X_cf)."
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._clf.predict_proba(X @ self.subspace.numpy().T)[:, 1]

    @property
    def subspace(self) -> torch.Tensor:
        """Just the chosen component, as a (1, hidden_size) basis."""
        comp = self._pca.components_[self.component_index]
        return torch.from_numpy(np.asarray(comp)).float().unsqueeze(0)

    def state_dict(self) -> dict:
        return {
            "pca": self._pca, "clf": self._clf,
            "n_components": self.n_components, "artifacts": self._artifacts,
            "style": self.style, "component_index": self.component_index,
        }

    @classmethod
    def from_state_dict(cls, d: dict) -> "PairedPCAProbe":
        probe = cls(style=d["style"], component_index=d["component_index"])
        probe._pca = d["pca"]
        probe._clf = d["clf"]
        probe._artifacts = d.get("artifacts", {})
        return probe
