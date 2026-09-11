"""How much of a probe direction lies in the top principal components of the data.

For a unit direction θ̂ and the principal directions v₁, v₂, … of an activation cloud,

    λ^K(θ) = ‖ (v₁·θ̂, …, v_K·θ̂) ‖ = sqrt( Σ_{i≤K} (vᵢ·θ̂)² )

is the fraction of θ̂'s length that lies in the span of the first K components.  It runs
from 0 (θ̂ orthogonal to the leading subspace) to 1 (θ̂ entirely inside it), and reaches
exactly 1 once K is the rank of the data.

It answers a specific question about unsupervised contrast probes: are they finding a
direction of their own, or just re-discovering the directions of largest variance?  A
curve that rises to ~1 by K = 2 says the probe is pinned to the top few components.
"""
from __future__ import annotations

import numpy as np
import torch

DEFAULT_KS = (1, 2, 4, 8, 16, 32, 64, 128)


def principal_directions(X: np.ndarray) -> torch.Tensor:
    """Right singular vectors of `X`, as rows, in descending singular-value order.

    No re-centering: the caller decides what "the data" means, and the contrast-probing
    convention is to take the activations exactly as the probe saw them.
    """
    t = torch.as_tensor(np.asarray(X), dtype=torch.float64)
    _, _, vh = torch.linalg.svd(t, full_matrices=False)
    return vh


def component_overlaps(direction: np.ndarray, pcs: torch.Tensor) -> torch.Tensor:
    """|vᵢ · θ̂| for each principal direction, given a precomputed basis."""
    theta = torch.as_tensor(np.asarray(direction), dtype=torch.float64)
    theta = theta / (theta.norm() + 1e-12)
    return (pcs @ theta).abs()


def lambda_k_from_pcs(
    direction: np.ndarray,
    pcs: torch.Tensor,
    ks: tuple[int, ...] = DEFAULT_KS,
) -> dict[int, float]:
    """λ^K against an already-computed basis.

    Callers scoring many probes against the same activations should take this route: the
    SVD dominates the cost and depends only on the data, so recomputing it per probe turns
    a seconds-long sweep into an hours-long one.
    """
    sims = component_overlaps(direction, pcs)
    return {k: float(sims[: min(k, sims.numel())].pow(2).sum().sqrt()) for k in ks}


def lambda_k(
    direction: np.ndarray,
    X: np.ndarray,
    ks: tuple[int, ...] = DEFAULT_KS,
) -> dict[int, float]:
    """λ^K for each K in `ks`, given a probe `direction` and the activations `X`.

    Ks larger than the number of available components are clipped, so the value at such K
    is the total overlap rather than an error.
    """
    return lambda_k_from_pcs(direction, principal_directions(X), ks)


def max_overlap(
    direction: np.ndarray,
    X: np.ndarray,
    ks: tuple[int, ...] = DEFAULT_KS,
) -> dict[int, float]:
    """The single largest |vᵢ·θ̂| among the first K components.

    Companion to λ^K: λ^K can be large because the direction is spread thinly over many
    components, whereas a large max says one component dominates.
    """
    return max_overlap_from_pcs(direction, principal_directions(X), ks)


def max_overlap_from_pcs(
    direction: np.ndarray,
    pcs: torch.Tensor,
    ks: tuple[int, ...] = DEFAULT_KS,
) -> dict[int, float]:
    """`max_overlap` against an already-computed basis."""
    sims = component_overlaps(direction, pcs)
    return {k: float(sims[: min(k, sims.numel())].max()) for k in ks}
