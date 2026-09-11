from __future__ import annotations

import warnings

import numpy as np
import torch
from scipy.linalg import eigh as scipy_eigh
from sklearn.linear_model import LogisticRegression

from .base import Probe


class CrossCovarianceProbe(Probe):
    """
    Finds probe directions via eigendecomposition of the symmetrised cross-covariance:

        S = Xbᵀ Xc + Xcᵀ Xb      (shape hidden × hidden)

    Inputs are expected to be per-edge-type mean-centred (grand mean subtracted),
    as applied by fit_probe in base.py.  With that centering, the origin lies at the
    centroid of the entire cloud of (base, cf) pairs for this edge type, removing
    content confounds shared across all pairs of the same type.

    Negative eigenvalues correspond to directions of anti-covariance (base and cf
    project to opposite sides), which is the truth signal.  Positive eigenvalues
    are directions where both conditions co-vary (shared confounds).  The k most-negative eigenvectors are selected; a
    logistic regression on the projections gives calibrated probabilities.

    The eigendecomposition is performed in the support subspace of A = [Xb; Xc]
    (found via SVD), reducing the problem from O(hidden³) to O(rank(A)³).
    Since rank(A) ≤ 2n << hidden for typical interpretability datasets, this
    gives a large constant-factor speedup with no change to the result.

    ── Relation to the "difference of covariances" formulation ──────────────────
    Writing E = Xb + Xc, F = [Xb; Xc], and

        S_auto  = Σᵢ (xbᵢxbᵢᵀ + xcᵢxcᵢᵀ)        S_cross = Σᵢ (xbᵢxcᵢᵀ + xcᵢxbᵢᵀ)

    the same directions are often derived as the extreme eigenvectors of A_E − β·B with
    A_E = cov(E) = (S_auto + S_cross)/(n−1) and B = cov(F) = S_auto/(2n−1).  Since

        A_E − β·B = S_auto·[1/(n−1) − β/(2n−1)] + S_cross/(n−1),

    the S_auto term cancels exactly at β = (2n−1)/(n−1) ≈ 2, leaving S_cross/(n−1) —
    which is what this probe computes by default.  ``beta=None`` (the default) is
    therefore the β→2 member of that family, computed directly and without the O(1/n²)
    residual that passing a literal 2.0 would leave behind.

    Set ``beta`` to a number only to reproduce a *different* member of the family: at
    β = 1, for instance, the objective is S_cross/(n−1) + S_auto/(2n), i.e. the contrast
    term plus a total-variance term, which is a different estimator.

    Requires paired data (requires_pairs = True); fit() raises NotImplementedError.
    """
    requires_pairs = True

    def __init__(
        self,
        n_components: int = 1,
        support_threshold: float = 1e-6,
        beta: float | None = None,
    ):
        self.n_components = n_components
        self.support_threshold = support_threshold
        self.beta = beta
        self._components: np.ndarray | None = None  # (n_components, hidden_size)
        self._clf: LogisticRegression | None = None
        self._artifacts: dict[str, np.ndarray] = {}

    @property
    def method(self) -> str:
        # beta=None keeps the historical name, so probes already on disk keep their key.
        suffix = "" if self.beta is None else f"_b{self.beta:g}"
        return f"cross_covariance_{self.n_components}{suffix}"

    def fit_paired(self, X_base: np.ndarray, X_cf: np.ndarray) -> None:
        X_base = X_base.astype(np.float64)
        X_cf   = X_cf.astype(np.float64)
        n = len(X_base)
        Xb = X_base
        Xc = X_cf

        # Project into the support subspace of A = [Xb; Xc] before forming S.
        # rank(A) ≤ 2n << hidden, so this reduces the eigendecomposition from
        # O(hidden³) to O(rank³) with no change to the result.
        A = np.vstack([Xb, Xc])
        _, sing_vals, Vt = np.linalg.svd(A, full_matrices=False)
        V_r = Vt[sing_vals > sing_vals[0] * self.support_threshold]  # (r, hidden)
        Xb_r = Xb @ V_r.T                                            # (n, r)
        Xc_r = Xc @ V_r.T                                            # (n, r)
        S_cross = Xb_r.T @ Xc_r + Xc_r.T @ Xb_r                      # (r, r)
        if self.beta is None:
            S_r = S_cross / (n - 1)
        else:
            # A_E − β·B, with the ddof convention of that formulation.
            S_auto = Xb_r.T @ Xb_r + Xc_r.T @ Xc_r
            S_r = (S_auto + S_cross) / (n - 1) - self.beta * S_auto / (2 * n - 1)

        eigenvalues, eigenvectors = np.linalg.eigh(S_r)  # ascending order
        # Most-negative eigenvalues = maximal anti-covariance; this is the
        # truth signal (base and cf project to opposite sides of w).
        top_k = np.argsort(eigenvalues)[: self.n_components]
        components_r = eigenvectors[:, top_k].T              # (n_components, r)
        self._components = components_r @ V_r                # (n_components, hidden)
        self._artifacts = {"eigenvalues": eigenvalues}       # full spectrum, ascending

        X_all = np.vstack([X_base, X_cf])
        y_all = np.array([1] * n + [0] * n)
        projections = X_all @ self._components.T
        self._clf = LogisticRegression(max_iter=1000)
        self._clf.fit(projections, y_all)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError(
            "CrossCovarianceProbe requires paired data; use fit_paired(X_base, X_cf)."
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._clf.predict_proba(X.astype(np.float64) @ self._components.T)[:, 1]

    @property
    def artifacts(self) -> dict[str, np.ndarray]:
        return self._artifacts

    @property
    def subspace(self) -> torch.Tensor:
        return torch.from_numpy(self._components).float()  # (n_components, hidden_size)

    def state_dict(self) -> dict:
        return {
            "components":        self._components,
            "clf":               self._clf,
            "n_components":      self.n_components,
            "support_threshold": self.support_threshold,
            "beta":              self.beta,
            "artifacts":         self._artifacts,
        }

    @classmethod
    def from_state_dict(cls, d: dict) -> CrossCovarianceProbe:
        probe = cls(
            n_components=d["n_components"],
            support_threshold=d.get("support_threshold", 1e-6),
            beta=d.get("beta"),
        )
        probe._components = d["components"]
        probe._clf = d["clf"]
        probe._artifacts = d.get("artifacts", {})
        return probe



def tied_norm_violations(
    eigenvalues: np.ndarray,
    norms: np.ndarray,
    selected: np.ndarray,
    tie_rtol: float = 1e-3,
) -> list[tuple[int, int]]:
    """Positions in `selected` where a tie is broken against ascending ||w||.

    `GeneralizedCrossCovarianceProbe` selects the most-negative generalised eigenvalues,
    but on real activations a large set of directions reach the attainable minimum together
    and tie to several decimals.  What actually orders them is ||w||: Sigma-orthonormality
    fixes w^T Sigma w = 1, so the pooled variance along the unit direction w/||w|| is
    1/||w||^2, and shorter means more variance.  Empirically lambda and ||w|| come out
    rank-identical inside a tied run, which is what makes the selection deterministic and
    interpretable rather than an arbitrary basis of a degenerate eigenspace.

    That is an empirical regularity, not a theorem, so it is checked rather than assumed.
    Returns the (i, i+1) index pairs within a tied run where the norm decreases; empty
    means the ordering is consistent with the interpretation.

    `tie_rtol` is relative to the full eigenvalue range.
    """
    lam = np.asarray(eigenvalues)[selected]
    nrm = np.asarray(norms)
    span = float(np.ptp(eigenvalues)) or 1.0
    tol = tie_rtol * span
    return [(i, i + 1) for i in range(len(lam) - 1)
            if abs(lam[i + 1] - lam[i]) <= tol and nrm[i + 1] < nrm[i]]

class GeneralizedCrossCovarianceProbe(Probe):
    """
    Solves the generalised eigenproblem

        S w = λ Σ w

    where S = Xbᵀ Xc + Xcᵀ Xb  (symmetrised cross-covariance, per-pair-centred)
    and   Σ = Cov([Xb ; Xc])    (total covariance of all 2n samples pooled).

    Inputs are expected to be per-edge-type mean-centred (grand mean subtracted),
    as applied by fit_probe in base.py.  With that centering the generalised
    eigenvalue λ is proportional to the explained-variance ratio of the difference
    direction, making eigenvalues dimensionless and comparable across layers and
    datasets.

    Each eigenvector w maximises the ratio

        λ = wᵀ S w / wᵀ Σ w  =  2 · (Xb w)ᵀ(Xc w) / Var_total(w)

    so eigenvalues are dimensionless and comparable across layers and datasets.
    The k most-negative eigenvectors are selected: negative λ means base and cf
    project to opposite sides of w (the truth signal); positive λ means they
    co-vary (shared confounds).  A logistic regression on the projections gives
    calibrated probabilities.

    Two sources of ill-conditioning are handled explicitly:

    1. Rank deficiency (hidden_size > 2n, common in interpretability): the support
       of A = [Xb ; Xc] is found via SVD and the problem is solved in that subspace,
       where Σ has no zero eigenvalues by construction.

    2. Near-singularity within the support: a small regularisation is added to Σ_r
       before solving.  The amount is:

           ε = max(0, −λ_min(Σ_r)) + α · 1e-8 · ‖Σ_r‖_F

       The first term corrects any numerical negative eigenvalues; the second is a
       scale-invariant floor proportional to the matrix norm.

    Activations may arrive as float16; inputs are promoted to float64 before any
    linear algebra.  scipy.linalg.eigh solves the (symmetric) generalised problem
    and returns Σ-orthonormal eigenvectors lifted back to the full hidden space.

    Requires paired data (requires_pairs = True); fit() raises NotImplementedError.
    """
    requires_pairs = True

    def __init__(
        self,
        n_components: int = 1,
        support_threshold: float = 1e-6,
        alpha: float = 1.0,
    ):
        self.n_components      = n_components
        self.support_threshold = support_threshold
        self.alpha             = alpha
        self._components: np.ndarray | None = None  # (n_components, hidden_size)
        self._clf: LogisticRegression | None = None
        self._artifacts: dict[str, np.ndarray] = {}

    @property
    def method(self) -> str:
        return f"generalized_cross_covariance_{self.n_components}"

    def fit_paired(self, X_base: np.ndarray, X_cf: np.ndarray) -> None:
        # Promote to float64 — activations may arrive as float16.
        X_base = X_base.astype(np.float64)
        X_cf   = X_cf.astype(np.float64)

        n = len(X_base)

        Xb = X_base
        Xc = X_cf
        A  = np.vstack([Xb, Xc])                                           # (2n, hidden)
        S  = (Xb.T @ Xc + Xc.T @ Xb) / (n - 1)

        # ── Step 1: find support of A via SVD ──────────────────────────────
        _, sing_vals, Vt = np.linalg.svd(A, full_matrices=False)
        V_r = Vt[sing_vals > sing_vals[0] * self.support_threshold]   # (r, hidden)

        # ── Step 2: project S and Σ into support subspace ─────────────────
        S_r = V_r @ S @ V_r.T                                          # (r, r)
        A_proj = A @ V_r.T                                             # (2n, r)
        Sigma_r = (A_proj.T @ A_proj) / (2 * n - 1)                   # (r, r)

        # ── Step 3: regularise Σ_r ─────────────────────────────────────────
        # Correct any numerical negative eigenvalues, then add a small
        # scale-invariant floor so the problem stays well-conditioned.
        min_eig = np.linalg.eigvalsh(Sigma_r).min()
        eps = max(0.0, -min_eig) + self.alpha * 1e-8 * np.linalg.norm(Sigma_r)
        Sigma_r_reg = Sigma_r + eps * np.eye(Sigma_r.shape[0])

        # ── Step 4: solve generalised eigenproblem ─────────────────────────
        eigenvalues, eigenvectors = scipy_eigh(S_r, Sigma_r_reg)      # ascending
        # Most-negative λ = maximum anti-covariance per unit variance;
        # positive λ directions are shared confounds (co-vary across conditions).
        top_k = np.argsort(eigenvalues)[: self.n_components]
        components_r = eigenvectors[:, top_k].T                        # (n_components, r)

        # Lift back to full hidden space
        self._components = components_r @ V_r                          # (n_components, hidden)
        self._artifacts = {"eigenvalues": eigenvalues}                 # full spectrum, ascending

        # The selected components are only interpretable if, within a tied run of
        # eigenvalues, they are ordered by ||w|| — see tied_norm_violations.
        bad = tied_norm_violations(eigenvalues, np.linalg.norm(self._components, axis=1),
                                   top_k)
        if bad:
            warnings.warn(
                f"{self.method}: eigenvalues tie at positions {bad} but ||w|| decreases "
                f"there, so the selection is not ordered by variance within the tie. "
                f"Components from this fit are not interpretable in the usual way "
                f"(see tied_norm_violations).",
                RuntimeWarning, stacklevel=2)

        y_all = np.array([1] * n + [0] * n)
        projections = A @ self._components.T
        self._clf = LogisticRegression(max_iter=1000)
        self._clf.fit(projections, y_all)

    # A note on ordering, so this is not re-attempted.  lambda is a *fraction*: writing
    # each pair as symmetric plus antisymmetric parts, lambda = 2 - 4a where a is the
    # antisymmetric share of the variance along w.  Fractions saturate, and on real
    # activations a large set of directions reach the attainable maximum together
    # (a = 2/3, lambda = -2/3 on cities_grid) and are tied to four decimals.  Breaking
    # those ties by ||w|| was tried and is a strict no-op: within the tied set lambda and
    # ||w|| are already rank-identical (0.95, 3.37, 4.66, 8.08, ... in lambda order), so
    # sorting by norm reproduces the order argsort already gives.  The limitation is not
    # the ordering but the directions: maximal contrast *fraction* does not imply lying in
    # the span of the group means, so a noisy direction carrying a little contrast ties
    # with a genuine one.  CrossCovarianceProbe ranks by contrast *magnitude*, which does
    # not saturate, and recovers that span (principal angles 1.00/1.00/0.96 on
    # llama3_8b__cities_grid against 0.30/0.28/0.23 here).

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError(
            "GeneralizedCrossCovarianceProbe requires paired data; "
            "use fit_paired(X_base, X_cf)."
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._clf.predict_proba(
            X.astype(np.float64) @ self._components.T
        )[:, 1]

    @property
    def artifacts(self) -> dict[str, np.ndarray]:
        return self._artifacts

    @property
    def subspace(self) -> torch.Tensor:
        return torch.from_numpy(self._components).float()

    def state_dict(self) -> dict:
        return {
            "components":        self._components,
            "clf":               self._clf,
            "n_components":      self.n_components,
            "support_threshold": self.support_threshold,
            "alpha":             self.alpha,
            "artifacts":         self._artifacts,
        }

    @classmethod
    def from_state_dict(cls, d: dict) -> GeneralizedCrossCovarianceProbe:
        probe = cls(
            n_components=d["n_components"],
            support_threshold=d["support_threshold"],
            alpha=d["alpha"],
        )
        probe._components = d["components"]
        probe._clf = d["clf"]
        probe._artifacts = d.get("artifacts", {})
        return probe
