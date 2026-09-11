"""Unit tests for the CCS loss terms, the two search-space alterations, and the
batched fit (over layers and over seeds).

These pin down the pieces that a reproduction depends on and that are easy to get
subtly wrong: which minimum the confidence term takes, that weight-norm matches
PyTorch's own parametrisation, that the SVD alteration is a rank restriction, and that
the batched optimiser is equivalent to fitting one probe at a time.
"""
import numpy as np
import pytest
import torch

from src.probing.ccs import (
    CCSProbe,
    _confidence_loss,
    _consistency_loss,
    _effective_w,
    _svd_basis,
)


# ── loss terms ────────────────────────────────────────────────────────────────

def test_confidence_burns_is_min_of_two():
    p0 = torch.tensor([0.2, 0.9, 0.4])
    p1 = torch.tensor([0.7, 0.3, 0.6])
    got = _confidence_loss(p0, p1, "burns")
    assert torch.allclose(got, torch.tensor([0.2, 0.3, 0.4]) ** 2)


def test_confidence_min_sq_is_symmetric_farquhar_form():
    p0 = torch.tensor([0.2, 0.9, 0.4])
    p1 = torch.tensor([0.7, 0.3, 0.6])
    # min over p0, p1, 1-p0, 1-p1
    expected = torch.tensor([0.2, 0.1, 0.4]) ** 2
    assert torch.allclose(_confidence_loss(p0, p1, "min_sq"), expected)


def test_confidence_min_sq_is_relabelling_symmetric_but_burns_is_not():
    """Swapping p -> 1-p on both branches must leave the symmetric term unchanged."""
    p0, p1 = torch.tensor([0.9]), torch.tensor([0.8])
    sym = _confidence_loss(p0, p1, "min_sq")
    sym_flipped = _confidence_loss(1 - p0, 1 - p1, "min_sq")
    assert torch.allclose(sym, sym_flipped)

    burns = _confidence_loss(p0, p1, "burns")
    burns_flipped = _confidence_loss(1 - p0, 1 - p1, "burns")
    assert not torch.allclose(burns, burns_flipped)


def test_consistency_is_sum_to_one_penalty():
    p0 = torch.tensor([0.3, 0.5])
    p1 = torch.tensor([0.6, 0.5])
    assert torch.allclose(_consistency_loss(p0, p1, "default"),
                          torch.tensor([0.1, 0.0]) ** 2)


@pytest.mark.parametrize("kind,fn", [("none", _confidence_loss), ("none", _consistency_loss)])
def test_ablated_terms_are_zero(kind, fn):
    p0, p1 = torch.rand(5), torch.rand(5)
    assert torch.allclose(fn(p0, p1, kind), torch.zeros(5))


# ── Alteration 1: unit-vector search ──────────────────────────────────────────

def test_weight_norm_matches_pytorch_parametrisation():
    torch.manual_seed(0)
    d = 7
    v = torch.randn(1, 1, d)
    g = torch.rand(1, 1, 1) * 3

    ours = _effective_w(v, g)

    lin = torch.nn.Linear(d, 1, bias=False)
    lin = torch.nn.utils.parametrizations.weight_norm(lin)
    with torch.no_grad():
        lin.parametrizations.weight.original0.copy_(g.reshape(1, 1))
        lin.parametrizations.weight.original1.copy_(v.reshape(1, d))

    assert torch.allclose(ours.reshape(-1), lin.weight.reshape(-1), atol=1e-6)


def test_weight_norm_direction_has_magnitude_g():
    v = torch.tensor([[[3.0, 4.0]]])          # norm 5
    g = torch.tensor([[[2.0]]])
    assert torch.allclose(_effective_w(v, g).norm(), torch.tensor(2.0))


# ── Alteration 2: rank restriction ────────────────────────────────────────────

def test_svd_basis_is_a_rank_restriction_when_samples_are_scarce():
    rng = np.random.default_rng(0)
    n, d = 6, 40
    basis = _svd_basis(rng.normal(size=(n, d)), rng.normal(size=(n, d)))
    assert basis.shape == (2 * n, d), "basis should span only the 2n training rows"
    # Orthonormal rows.
    assert np.allclose(basis @ basis.T, np.eye(2 * n), atol=1e-8)
    # A vector outside the span is annihilated by projecting in and back out.
    x = rng.normal(size=d)
    x_span = (x @ basis.T) @ basis
    assert not np.allclose(x, x_span)


def test_svd_basis_is_a_pure_rotation_when_data_is_full_rank():
    rng = np.random.default_rng(1)
    n, d = 40, 6
    basis = _svd_basis(rng.normal(size=(n, d)), rng.normal(size=(n, d)))
    assert basis.shape == (d, d)
    x = rng.normal(size=d)
    assert np.allclose((x @ basis.T) @ basis, x, atol=1e-8)


# ── batched fitting ───────────────────────────────────────────────────────────

def _toy_pairs(n=64, d=12, seed=0):
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=d)
    direction /= np.linalg.norm(direction)
    shared = rng.normal(size=(n, d))
    offset = rng.normal(size=n)[:, None] * direction
    X0 = shared + offset
    X1 = shared - offset
    return X0.astype(np.float32), X1.astype(np.float32)


COMMON = dict(n_epochs=60, lr=1e-2, weight_decay=0.0, device="cpu")


def test_batched_single_layer_matches_eager_fit():
    X0, X1 = _toy_pairs()
    eager = CCSProbe(n_restarts=3, seed=7, **COMMON)
    eager.fit_paired(X0, X1)

    grid = CCSProbe.fit_paired_batched(X0[None], X1[None], n_restarts=3, seed=7, **COMMON)
    assert len(grid) == 1 and len(grid[0]) == 1
    cos = abs(float(np.dot(eager._direction, grid[0][0]._direction)))
    assert cos > 0.999, cos


def test_seed_sweep_matches_per_seed_eager_fits():
    """The sweep reuses the restart axis; each column must equal a standalone fit."""
    X0, X1 = _toy_pairs(seed=3)
    seeds = [0, 1, 2]
    grid = CCSProbe.fit_paired_batched(X0[None], X1[None], seeds=seeds,
                                       n_restarts=1, **COMMON)
    assert len(grid[0]) == len(seeds)
    for probe, s in zip(grid[0], seeds):
        assert probe.seed == s
        eager = CCSProbe(n_restarts=1, seed=s, **COMMON)
        eager.fit_paired(X0, X1)
        cos = abs(float(np.dot(eager._direction, probe._direction)))
        assert cos > 0.999, (s, cos)


def test_seed_sweep_actually_varies_the_solution():
    X0, X1 = _toy_pairs(seed=5)
    grid = CCSProbe.fit_paired_batched(X0[None], X1[None], seeds=[0, 1, 2, 3],
                                       n_restarts=1, n_epochs=5, lr=1e-2,
                                       weight_decay=0.0, device="cpu")
    dirs = np.stack([p._direction for p in grid[0]])
    off_diag = np.abs(dirs @ dirs.T)[np.triu_indices(len(dirs), k=1)]
    assert off_diag.min() < 0.9999, "different seeds produced identical directions"


def test_batched_layers_are_independent():
    """Two layers fitted in one pass must match the same two fitted separately."""
    X0a, X1a = _toy_pairs(seed=11)
    X0b, X1b = _toy_pairs(seed=12)
    grid = CCSProbe.fit_paired_batched(np.stack([X0a, X0b]), np.stack([X1a, X1b]),
                                       n_restarts=2, seed=4, **COMMON)
    for (X0, X1), row in zip([(X0a, X1a), (X0b, X1b)], grid):
        solo = CCSProbe(n_restarts=2, seed=4, **COMMON)
        solo.fit_paired(X0, X1)
        assert abs(float(np.dot(solo._direction, row[0]._direction))) > 0.999


@pytest.mark.parametrize("opts", [
    dict(weight_norm=True),
    dict(svd_reduce=True),
    dict(weight_norm=True, svd_reduce=True),
    dict(confidence="min_sq", consistency="none"),
    dict(confidence="none"),
    dict(optimizer="adamw", bias=False, standardize=False, calibrate=False, fix_sign=False),
])
def test_batched_matches_eager_across_variants(opts):
    X0, X1 = _toy_pairs(seed=21)
    kw = {**COMMON, "n_restarts": 2, "seed": 9, **opts}
    eager = CCSProbe(**kw)
    eager.fit_paired(X0, X1)
    grid = CCSProbe.fit_paired_batched(X0[None], X1[None], **kw)
    cos = abs(float(np.dot(eager._direction, grid[0][0]._direction)))
    assert cos > 0.99, (opts, cos)


# ── identity and round-tripping ───────────────────────────────────────────────

def test_default_method_name_is_unchanged():
    """Probes already on disk are keyed by "ccs"; the defaults must keep that name."""
    assert CCSProbe().method == "ccs"


def test_variant_options_appear_in_method_name():
    m = CCSProbe(confidence="min_sq", consistency="none",
                 weight_norm=True, svd_reduce=True).method
    assert m.startswith("ccs[") and "conf=min_sq" in m and "cons=none" in m
    assert "wn" in m and "svd" in m


def test_state_dict_round_trip_preserves_options():
    X0, X1 = _toy_pairs(seed=31)
    probe = CCSProbe(confidence="min_sq", weight_norm=True, n_restarts=1, **COMMON)
    probe.fit_paired(X0, X1)
    clone = CCSProbe.from_state_dict(probe.state_dict())
    assert clone.method == probe.method
    assert np.allclose(clone._direction, probe._direction)


def test_legacy_state_dict_without_new_options_still_loads():
    """State dicts written before the ablation options existed must round-trip."""
    X0, X1 = _toy_pairs(seed=41)
    probe = CCSProbe(**COMMON)   # defaults, so the legacy name is plain "ccs"
    probe.fit_paired(X0, X1)
    legacy = {k: v for k, v in probe.state_dict().items()
              if k in ("direction", "clf", "n_restarts", "n_epochs", "lr",
                       "weight_decay", "seed", "device", "artifacts")}
    clone = CCSProbe.from_state_dict(legacy)
    assert clone.method == "ccs"
    assert np.allclose(clone._direction, probe._direction)
