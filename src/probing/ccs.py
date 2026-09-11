from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression

from .base import Probe


# ── loss terms ────────────────────────────────────────────────────────────────
#
# Both are written over a pair of probabilities (p0, p1) for the two framings of the
# same input, and both are mean-reduced by the caller.

def _confidence_loss(p0: torch.Tensor, p1: torch.Tensor, kind: str) -> torch.Tensor:
    """Confidence (a.k.a. informative) term.

    "burns"   min(p0, p1)^2                        — Burns et al. (2023) as published.
    "min_sq"  min(p0, p1, 1-p0, 1-p1)^2            — the symmetric form of Farquhar et al.
              (2023).  Burns' version is minimised by pushing *both* probabilities up,
              which is not symmetric under relabelling; this one is.
    "none"    ablated.
    """
    if kind == "burns":
        return torch.minimum(p0, p1).pow(2)
    if kind == "min_sq":
        return torch.stack([p0, p1, 1 - p0, 1 - p1]).min(dim=0).values.pow(2)
    if kind == "none":
        return torch.zeros_like(p0)
    raise ValueError(f"unknown confidence loss {kind!r}")


def _consistency_loss(p0: torch.Tensor, p1: torch.Tensor, kind: str) -> torch.Tensor:
    """Consistency term: (p0 + p1 - 1)^2, i.e. the two framings' probabilities should
    sum to one.  "none" ablates it."""
    if kind == "default":
        return (p0 + p1 - 1).pow(2)
    if kind == "none":
        return torch.zeros_like(p0)
    raise ValueError(f"unknown consistency loss {kind!r}")


def _effective_w(V: torch.Tensor, G: torch.Tensor | None) -> torch.Tensor:
    """Alteration 1: restrict the search to unit vectors.

    Reproduces torch.nn.utils.parametrizations.weight_norm — the direction is v/||v||
    and only the scalar g carries magnitude — but in the batched (L, R, d) layout.
    """
    if G is None:
        return V
    return G * V / (V.norm(dim=-1, keepdim=True) + 1e-12)


def _svd_basis(X0: np.ndarray, X1: np.ndarray) -> np.ndarray:
    """Alteration 2: an orthonormal basis for the row space of the training data.

    Returns Vt of shape (r, d) with r = min(2n, d).  Since 2n is typically far below the
    hidden size, fitting in this basis is a *rank restriction* to the span of the data,
    not merely a rotation — which is also why it changes what weight decay does.
    """
    _, _, Vt = np.linalg.svd(np.vstack([X0, X1]), full_matrices=False)
    return Vt


class CCSProbe(Probe):
    """
    Contrast-Consistent Search (Burns et al. 2023), with the loss-term ablations and
    search-space alterations used to analyse it.

    Finds a linear direction by gradient descent on

        L = E[ consistency(p0, p1) + confidence(p0, p1) ]

    where p_i = sigma(w . h_i (+ b)) for a pair of framings (h0, h1).  The consistency
    term asks the two framings' probabilities to sum to one; the confidence term prevents
    the degenerate p == 0.5 solution.  Either can be ablated, and two alterations restrict
    where the direction may live:

        weight_norm=True   search only unit vectors (Alteration 1)
        svd_reduce=True    search only the span of the training data (Alteration 2)

    Defaults reproduce the original behaviour of this class exactly: Burns' confidence
    term, Adam, 10 random restarts keeping the lowest final loss, pooled z-scoring, a
    bias term, post-hoc sign fixing so base samples score above counterfactual ones, and
    a logistic calibration head.  Every option that departs from that appears in
    ``method``, so it gets its own cache and database key.

    For the unsupervised protocol used in the contrast-probing literature, set
    ``confidence="min_sq", optimizer="adamw", lr=1e-3, weight_decay=0.01, n_restarts=1,
    standardize=False, bias=False, calibrate=False, fix_sign=False`` and score with
    ``Probe.raw_score`` rather than ``predict_proba``.

    Requires paired data (requires_pairs = True); fit() raises NotImplementedError.
    """

    requires_pairs = True

    def __init__(
        self,
        n_restarts: int = 10,
        n_epochs: int = 1000,
        lr: float = 1e-2,
        weight_decay: float = 1e-4,
        seed: int = 0,
        device: str = "auto",
        confidence: str = "burns",
        consistency: str = "default",
        weight_norm: bool = False,
        svd_reduce: bool = False,
        optimizer: str = "adam",
        standardize: bool = True,
        bias: bool = True,
        calibrate: bool = True,
        fix_sign: bool = True,
    ):
        self.n_restarts = n_restarts
        self.n_epochs = n_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.seed = seed
        self.device = device  # "auto" → cuda if available else cpu
        self.confidence = confidence
        self.consistency = consistency
        self.weight_norm = weight_norm
        self.svd_reduce = svd_reduce
        self.optimizer = optimizer
        self.standardize = standardize
        self.bias = bias
        self.calibrate = calibrate
        self.fix_sign = fix_sign

        self._direction: np.ndarray | None = None  # (hidden_size,)
        self._clf: LogisticRegression | None = None
        self._artifacts: dict[str, np.ndarray] = {}

    # ── identity ──────────────────────────────────────────────────────────────

    @property
    def method(self) -> str:
        """Method name, encoding only the options that differ from the defaults.

        A plain default probe is ``"ccs"`` — unchanged from before these options existed,
        so probes already on disk keep their key.
        """
        opts = []
        if self.confidence != "burns":
            opts.append(f"conf={self.confidence}")
        if self.consistency != "default":
            opts.append(f"cons={self.consistency}")
        if self.svd_reduce:
            opts.append("svd")
        if self.weight_norm:
            opts.append("wn")
        if self.optimizer != "adam":
            opts.append(f"opt={self.optimizer}")
        if self.n_restarts != 10:
            opts.append(f"r={self.n_restarts}")
        if not self.standardize:
            opts.append("nostd")
        if not self.bias:
            opts.append("nobias")
        if not self.calibrate:
            opts.append("nocal")
        if not self.fix_sign:
            opts.append("nosign")
        return "ccs" if not opts else "ccs[" + ",".join(opts) + "]"

    def _torch_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _loss(self, W: torch.Tensor, B: torch.Tensor | None,
              h0: torch.Tensor, h1: torch.Tensor) -> torch.Tensor:
        """Per-restart loss.  W: (R, d), B: (R, 1) or None, h: (n, d) → (R,)."""
        logits0 = h0 @ W.t()
        logits1 = h1 @ W.t()
        if B is not None:
            logits0 = logits0 + B.t()
            logits1 = logits1 + B.t()
        p0, p1 = torch.sigmoid(logits0), torch.sigmoid(logits1)
        return (
            _consistency_loss(p0, p1, self.consistency).mean(dim=0)
            + _confidence_loss(p0, p1, self.confidence).mean(dim=0)
        )

    # ── fitting ───────────────────────────────────────────────────────────────

    def fit_paired(self, X_base: np.ndarray, X_cf: np.ndarray) -> None:
        X_base = X_base.astype(np.float32)
        X_cf = X_cf.astype(np.float32)

        basis = _svd_basis(X_base, X_cf).astype(np.float32) if self.svd_reduce else None
        if basis is not None:
            X_base, X_cf = X_base @ basis.T, X_cf @ basis.T

        if self.standardize:
            X_pool = np.vstack([X_base, X_cf])
            norm_std = X_pool.std(axis=0) + 1e-8
            norm_mean = X_pool.mean(axis=0)
            Xb_n = (X_base - norm_mean) / norm_std
            Xc_n = (X_cf - norm_mean) / norm_std
        else:
            norm_std = np.ones(X_base.shape[1], dtype=np.float32)
            Xb_n, Xc_n = X_base, X_cf

        device = self._torch_device()
        h0 = torch.tensor(Xb_n, device=device)
        h1 = torch.tensor(Xc_n, device=device)
        hidden_size = Xb_n.shape[1]
        R = self.n_restarts

        # CPU generator keeps weight init reproducible across devices; the
        # initialised tensor is then moved to the optimisation device.
        gen = torch.Generator()
        gen.manual_seed(self.seed)

        # All restarts are optimised in parallel: each restart's loss depends only on its
        # own row, so summing them gives independent gradients.  Drawing (R, d) at once
        # yields the same RNG sequence as R successive (d,) draws.
        V = nn.Parameter((torch.randn(R, hidden_size, generator=gen) * 0.01).to(device))
        params = [V]
        G = None
        if self.weight_norm:
            G = nn.Parameter(torch.ones(R, 1, device=device))
            params.append(G)
        B = None
        if self.bias:
            B = nn.Parameter(torch.zeros(R, 1, device=device))
            params.append(B)

        opt_cls = torch.optim.AdamW if self.optimizer == "adamw" else torch.optim.Adam
        opt = opt_cls(params, lr=self.lr, weight_decay=self.weight_decay)

        for _ in range(self.n_epochs):
            opt.zero_grad()
            self._loss(_effective_w(V, G), B, h0, h1).sum().backward()
            opt.step()

        with torch.no_grad():
            W = _effective_w(V, G)
            final_losses = self._loss(W, B, h0, h1)   # (R,)
        best_idx = int(torch.argmin(final_losses).item())
        best_w = W[best_idx].detach()
        best_b = B[best_idx].detach() if B is not None else torch.zeros(1, device=device)

        self._artifacts = {"restart_losses": final_losses.cpu().numpy()}

        # Map direction from normalised space back to original activation space:
        # h_n @ w_n  ==  h_orig @ (w_n / sigma)  +  const, so w_orig = w_n / sigma.
        w_orig = best_w.cpu().numpy() / norm_std

        if self.fix_sign:
            with torch.no_grad():
                p0 = torch.sigmoid(h0 @ best_w + best_b).mean().item()
                p1 = torch.sigmoid(h1 @ best_w + best_b).mean().item()
            if p0 < p1:
                w_orig = -w_orig

        if basis is not None:
            w_orig = w_orig @ basis   # lift back out of the data span

        # The raw magnitude, before normalisation, is worth keeping: it determines how
        # much of the fitted direction is leftover initialisation.  Gradients of a loss
        # in `h @ w` live in the row space of `h`, so any component of w0 outside that
        # span is never trained and AdamW's decoupled decay barely shrinks it
        # ((1 - lr*wd)^n ~ 0.99 here).  That residual is a fixed ~0.6 in norm, so the
        # fraction of the direction lying outside the data span -- and hence the ceiling
        # on lambda^K -- is set by how far |w| travelled.
        magnitude = float(np.linalg.norm(w_orig))
        self._artifacts["parameter_magnitude"] = np.array(magnitude)
        self._direction = w_orig / (magnitude + 1e-8)
        self._fit_head(X_base if basis is None else X_base @ basis,
                       X_cf if basis is None else X_cf @ basis)

    def _fit_head(self, X_base: np.ndarray, X_cf: np.ndarray) -> None:
        """Fit the logistic calibration head used by predict_proba.

        Skipped when calibrate=False, in which case predict_proba falls back to the
        uncalibrated raw_score.
        """
        if not self.calibrate:
            self._clf = None
            return
        X_all = np.vstack([X_base, X_cf])
        y_all = np.array([1] * len(X_base) + [0] * len(X_cf))
        self._clf = LogisticRegression(max_iter=1000)
        self._clf.fit(X_all @ self._direction[:, None], y_all)

    @classmethod
    def fit_paired_batched(
        cls,
        X_base: np.ndarray,
        X_cf: np.ndarray,
        *,
        seeds: list[int] | None = None,
        **kwargs,
    ) -> list[list["CCSProbe"]]:
        """Fit probes for many layers — and optionally many seeds — in one pass.

        X_base, X_cf have shape (L, n, d): the (already centered) base/counterfactual
        activations for L layers sharing the same n pairs.

        Returns an L x S grid of fitted probes.  With ``seeds=None`` there is a single
        column using ``kwargs["seed"]`` and the usual ``n_restarts`` restarts, of which
        the lowest-loss one is kept.  With ``seeds=[...]`` the restart axis *becomes* the
        seed axis: one initialisation per seed, all of them kept.  That is what a 30-seed
        sweep needs, and it costs one batched optimisation rather than 30 sequential fits.

        Equivalent to calling fit_paired once per (layer, seed): each layer gets its own
        pooled z-scoring, independent per-(layer, restart) gradients, and per-layer sign
        fixing and calibration.
        """
        proto = cls(**kwargs)
        X_base = np.ascontiguousarray(X_base, dtype=np.float32)
        X_cf = np.ascontiguousarray(X_cf, dtype=np.float32)
        L, n, d = X_base.shape

        sweep = seeds is not None
        seed_list = list(seeds) if sweep else [proto.seed]
        R = len(seed_list) if sweep else proto.n_restarts

        if proto.device == "auto":
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            dev = torch.device(proto.device)

        def _optimize(dev: torch.device):
            """Run the batched optimisation on `dev`.  Factored out so a CUDA OOM can
            retry on CPU.  Returns numpy arrays for everything the caller needs."""
            Xb = torch.tensor(X_base, device=dev)
            Xc = torch.tensor(X_cf, device=dev)

            basis = None
            if proto.svd_reduce:
                # Batched SVD over layers; (L, r, d) with r = min(2n, d).
                _, _, basis = torch.linalg.svd(torch.cat([Xb, Xc], dim=1),
                                               full_matrices=False)
                Xb = Xb @ basis.transpose(1, 2)
                Xc = Xc @ basis.transpose(1, 2)

            width = Xb.shape[-1]
            if proto.standardize:
                pool = torch.cat([Xb, Xc], dim=1)              # (L, 2n, width)
                mean = pool.mean(dim=1, keepdim=True)
                # unbiased=False to match numpy's population std used in fit_paired.
                std = pool.std(dim=1, keepdim=True, unbiased=False) + 1e-8
                h0, h1 = (Xb - mean) / std, (Xc - mean) / std
            else:
                std = torch.ones(L, 1, width, device=dev)
                h0, h1 = Xb, Xc

            if sweep:
                # One initialisation per seed, shared across layers so a layer sweep and
                # a seed sweep stay comparable.
                rows = []
                for s in seed_list:
                    g = torch.Generator()
                    g.manual_seed(int(s))
                    rows.append(torch.randn(width, generator=g) * 0.01)
                w0 = torch.stack(rows).to(dev)                        # (R, width)
            else:
                g = torch.Generator()
                g.manual_seed(proto.seed)
                w0 = (torch.randn(R, width, generator=g) * 0.01).to(dev)

            V = nn.Parameter(w0.unsqueeze(0).expand(L, -1, -1).contiguous())  # (L, R, width)
            params = [V]
            G = None
            if proto.weight_norm:
                G = nn.Parameter(torch.ones(L, R, 1, device=dev))
                params.append(G)
            B = None
            if proto.bias:
                B = nn.Parameter(torch.zeros(L, R, 1, device=dev))
                params.append(B)

            opt_cls = torch.optim.AdamW if proto.optimizer == "adamw" else torch.optim.Adam
            opt = opt_cls(params, lr=proto.lr, weight_decay=proto.weight_decay)

            def _loss() -> torch.Tensor:
                W = _effective_w(V, G)
                logits0 = torch.einsum("lnd,lrd->lnr", h0, W)
                logits1 = torch.einsum("lnd,lrd->lnr", h1, W)
                if B is not None:
                    bias = B.permute(0, 2, 1)                          # (L, 1, R)
                    logits0, logits1 = logits0 + bias, logits1 + bias
                p0, p1 = torch.sigmoid(logits0), torch.sigmoid(logits1)
                return (
                    _consistency_loss(p0, p1, proto.consistency).mean(dim=1)
                    + _confidence_loss(p0, p1, proto.confidence).mean(dim=1)
                )                                                      # (L, R)

            for _ in range(proto.n_epochs):
                opt.zero_grad()
                _loss().sum().backward()
                opt.step()

            with torch.no_grad():
                final = _loss()                                        # (L, R)
                W = _effective_w(V, G)                                 # (L, R, width)
                bias_t = B if B is not None else torch.zeros(L, R, 1, device=dev)
                score0 = torch.einsum("lnd,lrd->lnr", h0, W) + bias_t.permute(0, 2, 1)
                score1 = torch.einsum("lnd,lrd->lnr", h1, W) + bias_t.permute(0, 2, 1)
                flip = torch.sigmoid(score0).mean(dim=1) < torch.sigmoid(score1).mean(dim=1)

            return (
                std.squeeze(1).detach().cpu().numpy(),         # (L, width)
                W.detach().cpu().numpy(),                      # (L, R, width)
                np.where(flip.cpu().numpy(), -1.0, 1.0),       # (L, R)
                final.cpu().numpy(),                           # (L, R)
                None if basis is None else basis.cpu().numpy(),  # (L, r, d)
            )

        try:
            std_np, W_np, sign, final_np, basis_np = _optimize(dev)
        except torch.cuda.OutOfMemoryError:
            if dev.type != "cuda":
                raise
            torch.cuda.empty_cache()
            print("  [ccs] CUDA out of memory — falling back to CPU for this batched fit")
            std_np, W_np, sign, final_np, basis_np = _optimize(torch.device("cpu"))

        grid: list[list[CCSProbe]] = []
        for l in range(L):
            # Without a sweep, restarts are competing attempts and only the best is kept.
            cols = range(R) if sweep else [int(np.argmin(final_np[l]))]
            row: list[CCSProbe] = []
            for out_i, r in enumerate(cols):
                w = W_np[l, r] / std_np[l]
                if proto.fix_sign:
                    w = sign[l, r] * w
                if basis_np is not None:
                    w = w @ basis_np[l]
                direction = w / (np.linalg.norm(w) + 1e-8)

                probe = cls(**{**kwargs, "seed": seed_list[out_i] if sweep else proto.seed})
                probe._direction = direction
                probe._artifacts = {"restart_losses": final_np[l, [r]]}
                probe._fit_head(X_base[l], X_cf[l])
                row.append(probe)
            grid.append(row)
        return grid

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError("CCSProbe requires paired data; use fit_paired(X_base, X_cf).")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._clf is None:
            return self.raw_score(X)
        return self._clf.predict_proba(X.astype(np.float32) @ self._direction[:, None])[:, 1]

    @property
    def artifacts(self) -> dict[str, np.ndarray]:
        return self._artifacts

    @property
    def subspace(self) -> torch.Tensor:
        return torch.from_numpy(self._direction).float().unsqueeze(0)  # (1, hidden_size)

    # ── (de)serialisation ─────────────────────────────────────────────────────

    _OPTIONS = (
        "n_restarts", "n_epochs", "lr", "weight_decay", "seed", "device",
        "confidence", "consistency", "weight_norm", "svd_reduce", "optimizer",
        "standardize", "bias", "calibrate", "fix_sign",
    )

    def state_dict(self) -> dict:
        d = {name: getattr(self, name) for name in self._OPTIONS}
        d.update(direction=self._direction, clf=self._clf, artifacts=self._artifacts)
        return d

    @classmethod
    def from_state_dict(cls, d: dict) -> CCSProbe:
        # .get with defaults so probes written before these options existed still load.
        defaults = cls()
        probe = cls(**{name: d.get(name, getattr(defaults, name)) for name in cls._OPTIONS})
        probe._direction = d["direction"]
        probe._clf = d["clf"]
        probe._artifacts = d.get("artifacts", {})
        return probe
