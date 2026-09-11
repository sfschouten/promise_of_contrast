"""Tests for the subspace-overlap metric λ^K."""
import numpy as np
import pytest

from src.probing.subspace_overlap import lambda_k, max_overlap, principal_directions


def _anisotropic(n=200, d=12, seed=0):
    """Data with a strongly dominant first component."""
    rng = np.random.default_rng(seed)
    basis = np.linalg.qr(rng.normal(size=(d, d)))[0]
    scales = np.array([50.0, 20.0] + [1.0] * (d - 2))
    return (rng.normal(size=(n, d)) * scales) @ basis.T, basis


def test_direction_along_the_top_component_has_overlap_one_at_k1():
    X, _ = _anisotropic()
    top = principal_directions(X)[0].numpy()
    got = lambda_k(top, X, ks=(1, 2, 4))
    assert got[1] == pytest.approx(1.0, abs=1e-8)
    assert got[4] == pytest.approx(1.0, abs=1e-8)


def test_direction_in_the_trailing_components_has_near_zero_overlap_at_small_k():
    X, _ = _anisotropic(seed=1)
    vh = principal_directions(X).numpy()
    got = lambda_k(vh[-1], X, ks=(1, 2, 4))
    assert got[1] < 1e-8 and got[4] < 1e-8


def test_lambda_k_reaches_one_at_full_rank():
    X, _ = _anisotropic(n=60, d=10, seed=2)
    rng = np.random.default_rng(3)
    got = lambda_k(rng.normal(size=10), X, ks=(10,))
    assert got[10] == pytest.approx(1.0, abs=1e-8)


def test_lambda_k_is_monotone_in_k():
    X, _ = _anisotropic(seed=4)
    rng = np.random.default_rng(5)
    got = lambda_k(rng.normal(size=12), X, ks=(1, 2, 4, 8, 12))
    values = [got[k] for k in (1, 2, 4, 8, 12)]
    assert all(a <= b + 1e-12 for a, b in zip(values, values[1:]))


def test_lambda_k_is_scale_invariant_in_the_direction():
    X, _ = _anisotropic(seed=6)
    rng = np.random.default_rng(7)
    theta = rng.normal(size=12)
    a = lambda_k(theta, X, ks=(2, 8))
    b = lambda_k(theta * -37.0, X, ks=(2, 8))
    assert a == pytest.approx(b, abs=1e-12)


def test_ks_beyond_the_available_components_are_clipped_not_an_error():
    X, _ = _anisotropic(n=20, d=8, seed=8)
    got = lambda_k(np.ones(8), X, ks=(4, 8, 128))
    assert got[128] == pytest.approx(got[8], abs=1e-12)


def test_max_overlap_never_exceeds_lambda_k():
    X, _ = _anisotropic(seed=9)
    rng = np.random.default_rng(10)
    theta = rng.normal(size=12)
    lam, mx = lambda_k(theta, X), max_overlap(theta, X)
    assert all(mx[k] <= lam[k] + 1e-12 for k in lam)


def test_raw_score_saturates_without_overflow():
    """Large projections must saturate cleanly, not raise a numpy overflow warning."""
    import warnings

    from src.probing.dim import DiffInMeansProbe

    probe = DiffInMeansProbe()
    probe._direction = np.array([1.0, 0.0])
    X = np.array([[1e4, 0.0], [-1e4, 0.0]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = probe.raw_score(X)
    assert got[0] == pytest.approx(1.0)
    assert got[1] == pytest.approx(0.0)


def test_precomputed_basis_gives_identical_results():
    """Scoring many probes against one activation set must reuse the basis, so the two
    routes have to agree exactly."""
    from src.probing.subspace_overlap import (
        lambda_k_from_pcs,
        max_overlap_from_pcs,
        principal_directions,
    )

    X, _ = _anisotropic(seed=11)
    pcs = principal_directions(X)
    rng = np.random.default_rng(12)
    for _ in range(3):
        theta = rng.normal(size=12)
        assert lambda_k(theta, X) == pytest.approx(lambda_k_from_pcs(theta, pcs))
        assert max_overlap(theta, X) == pytest.approx(max_overlap_from_pcs(theta, pcs))


def test_supervised_and_contrast_diff_in_means_are_different_estimators():
    """The two must not be confused: `diff_in_means` conventionally means the supervised
    mass-mean probe, and on contrastive data the branch says nothing about truth.

    Construct pairs where the branch is deliberately uninformative — each pair holds one
    true and one false framing, with the true one on either side — and check that the
    supervised direction recovers truth while the contrast direction does not.
    """
    import numpy as np

    from src.probing.dim import ContrastDiffInMeansProbe, DiffInMeansProbe

    rng = np.random.default_rng(0)
    n, d = 200, 16
    truth_dir = np.zeros(d); truth_dir[0] = 1.0
    branch_dir = np.zeros(d); branch_dir[1] = 1.0

    truth = rng.permutation(np.repeat([0, 1], n // 2))  # is branch 0 the correct framing?
                                                       # exactly balanced, so neither
                                                       # axis leaks into the other
    noise = lambda: rng.normal(scale=0.1, size=(n, d))
    # branch 0 carries -branch_dir, branch 1 carries +branch_dir; truth sits on truth_dir
    X_base = (2 * truth - 1)[:, None] * truth_dir - branch_dir + noise()
    X_cf = -(2 * truth - 1)[:, None] * truth_dir + branch_dir + noise()

    sup = DiffInMeansProbe()
    X_all = np.vstack([X_base, X_cf])
    y_all = np.concatenate([truth, 1 - truth])         # "is this framing correct?"
    sup.fit(X_all, y_all)

    con = ContrastDiffInMeansProbe()
    con.fit_paired(X_base, X_cf)

    s = sup.subspace[0].numpy()
    c = con.subspace[0].numpy()
    assert abs(s @ truth_dir) > 0.95, "supervised probe should recover the truth axis"
    assert abs(c @ branch_dir) > 0.95, "contrast probe should recover the branch axis"
    assert abs(s @ c) < 0.2, "the two estimators must not be treated as interchangeable"


def test_supervised_diff_in_means_does_not_expose_fit_paired():
    """fit_probe routes to fit_paired whenever it exists, so the supervised probe must
    not have one — otherwise contrastive data silently swaps the estimator underneath it.
    """
    from src.probing.dim import DiffInMeansProbe

    assert not hasattr(DiffInMeansProbe(), "fit_paired")
