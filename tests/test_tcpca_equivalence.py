"""The tuple-contrastive eigenproblem, and why it needs no separate implementation.

Contrast-probing work states the method as the eigendecomposition of

    cov(X⁺ + X⁻) − β · cov(X),        X = [X⁺ ; X⁻]

taking the *largest* eigenvector of β·cov(X) − cov(X⁺+X⁻).  `CrossCovarianceProbe`
instead takes the *most-negative* eigenvectors of the symmetrised cross-covariance
(Xbᵀ Xc + Xcᵀ Xb)/(n−1).  These tests show the two coincide exactly at β ≈ 2, so the
existing probe already implements the method — and pin down what changes when they
don't.
"""
import numpy as np
import pytest
import scipy.linalg

from src.probing.base import _center_paired
from src.probing.crosscov import CrossCovarianceProbe, GeneralizedCrossCovarianceProbe


def _reference_neutral_diff(N, P, beta):
    """Direct transcription of the difference-of-covariances formulation.

    Returns (intervene_basis, classify_basis), each (d, r) with eigenvalues ascending,
    so column -1 is the leading direction.
    """
    D, E = N - P, N + P
    F = np.concatenate([N, P], axis=0)
    _, _, Vt_f = np.linalg.svd(F, full_matrices=False)
    rD, rE, rF = D @ Vt_f.T, E @ Vt_f.T, F @ Vt_f.T

    def _sym_cov(x):
        c = np.cov(x, rowvar=False, ddof=1)
        return (c + c.T) / 2

    A_D, A_E, B = _sym_cov(rD), _sym_cov(rE), _sym_cov(rF)
    _, T = scipy.linalg.eigh(A_D - beta * B)
    _, C = scipy.linalg.eigh(beta * B - A_E)
    return Vt_f.T @ T, Vt_f.T @ C


def _toy(n=60, d=10, seed=0):
    """Paired activations with a shared component, a contrastive axis, and unequal
    branch means (so centering choices actually matter)."""
    rng = np.random.default_rng(seed)
    axis = rng.normal(size=d)
    axis /= np.linalg.norm(axis)
    shared = rng.normal(size=(n, d))
    swing = rng.normal(size=n)[:, None] * axis
    N = shared + swing + 0.5 * rng.normal(size=d)
    P = shared - swing - 0.5 * rng.normal(size=d)
    return N, P


def _pairs(n):
    """Minimal Sample stand-ins: _center_paired only reads descriptors."""
    from src.data.loader import Sample
    return [
        (Sample(id=f"b{i}", text="", descriptors={"dataset_name": "d"}),
         Sample(id=f"c{i}", text="", descriptors={"dataset_name": "d"}))
        for i in range(n)
    ]


def _cos(a, b):
    return abs(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))


def test_cross_covariance_equals_the_beta_two_member_of_the_family():
    """At β = (2n−1)/(n−1) the auto-covariance term cancels exactly."""
    N, P = _toy()
    n = len(N)
    beta_exact = (2 * n - 1) / (n - 1)

    _, classify = _reference_neutral_diff(N, P, beta_exact)

    Xb, Xc = _center_paired(_pairs(n), N.copy(), P.copy(), "midpoint")
    probe = CrossCovarianceProbe(n_components=1)
    probe.fit_paired(Xb, Xc)

    assert _cos(probe.subspace.numpy()[0], classify[:, -1]) > 1 - 1e-8


def test_intervene_and_classify_bases_coincide_when_branch_means_agree():
    """At β ≈ 2 the two bases reduce to

        β·B − A_E = (−S̃_cross + 2n·δδᵀ)/(n−1)
        A_D − β·B = (−S̃_cross − 2n·δδᵀ)/(n−1)

    with δ = (mean(N) − mean(P))/2 and S̃ computed per-branch-centred.  They are the same
    matrix exactly when δ = 0 — which is the case whenever the pipeline per-branch
    normalises activations before probing, as contrast-probing pipelines typically do.
    So the "two bases" such formulations expose are one basis in that setting.
    """
    N, P = _toy(seed=1)
    n = len(N)
    N = N - N.mean(0)          # delta = 0
    P = P - P.mean(0)
    intervene, classify = _reference_neutral_diff(N, P, (2 * n - 1) / (n - 1))
    assert _cos(intervene[:, -1], classify[:, -1]) > 1 - 1e-8


def test_intervene_and_classify_diverge_when_branch_means_differ():
    """Without per-branch normalisation the ±2n·δδᵀ term separates them, so which basis
    a result came from matters."""
    N, P = _toy(seed=1)
    n = len(N)
    intervene, classify = _reference_neutral_diff(N, P, (2 * n - 1) / (n - 1))
    assert _cos(intervene[:, -1], classify[:, -1]) < 0.999


def test_intervene_and_classify_diverge_at_beta_one():
    """At β = 1 the objective keeps a total-variance term, and the bases separate —
    which is why runs at β = 1 are a different estimator, not a rescaling."""
    N, P = _toy(seed=2)
    intervene, classify = _reference_neutral_diff(N, P, 1.0)
    assert _cos(intervene[:, -1], classify[:, -1]) < 0.99


def test_beta_one_probe_matches_the_beta_one_reference():
    N, P = _toy(seed=3)
    n = len(N)
    _, classify = _reference_neutral_diff(N, P, 1.0)

    Xb, Xc = _center_paired(_pairs(n), N.copy(), P.copy(), "midpoint")
    probe = CrossCovarianceProbe(n_components=1, beta=1.0)
    probe.fit_paired(Xb, Xc)
    assert _cos(probe.subspace.numpy()[0], classify[:, -1]) > 1 - 1e-6


def test_default_beta_is_unchanged_from_the_plain_cross_covariance():
    """beta=None must be bit-identical to the historical computation, so probes already
    on disk stay valid."""
    N, P = _toy(seed=4)
    n = len(N)
    Xb, Xc = _center_paired(_pairs(n), N.copy(), P.copy(), "midpoint")

    probe = CrossCovarianceProbe(n_components=3)
    probe.fit_paired(Xb, Xc)

    # The historical expression, computed independently.
    A = np.vstack([Xb, Xc]).astype(np.float64)
    _, sv, Vt = np.linalg.svd(A, full_matrices=False)
    V_r = Vt[sv > sv[0] * 1e-6]
    S = (Xb @ V_r.T).T @ (Xc @ V_r.T)
    S = (S + S.T) / (n - 1)
    eigvals, eigvecs = np.linalg.eigh(S)
    expected = (eigvecs[:, np.argsort(eigvals)[:3]].T @ V_r)

    for got, want in zip(probe.subspace.numpy(), expected):
        assert _cos(got, want) > 1 - 1e-8


def test_default_method_name_is_unchanged():
    assert CrossCovarianceProbe(n_components=8).method == "cross_covariance_8"
    assert CrossCovarianceProbe(n_components=8, beta=1).method == "cross_covariance_8_b1"


# ── centering is the difference that survives ────────────────────────────────

def test_midpoint_and_per_branch_centering_differ_by_a_diff_in_means_rank_one_term():
    """S_cross(midpoint) = S_cross(per_branch) − 2n·δδᵀ with δ = (mean(Xb)−mean(Xc))/2.

    So midpoint centering folds the diff-in-means direction into the contrast direction,
    and per-branch centering removes it.  Neither is wrong; they answer different
    questions, which is why the mode is configurable.
    """
    N, P = _toy(seed=5)
    n = len(N)
    pairs = _pairs(n)

    def s_cross(Xb, Xc):
        return Xb.T @ Xc + Xc.T @ Xb

    mid = s_cross(*_center_paired(pairs, N.copy(), P.copy(), "midpoint"))
    per = s_cross(*_center_paired(pairs, N.copy(), P.copy(), "per_branch"))
    delta = (N.mean(0) - P.mean(0)) / 2

    assert np.allclose(mid, per - 2 * n * np.outer(delta, delta), atol=1e-8)


def test_per_branch_centering_zeroes_each_branch_mean():
    N, P = _toy(seed=6)
    Xb, Xc = _center_paired(_pairs(len(N)), N.copy(), P.copy(), "per_branch")
    assert np.allclose(Xb.mean(0), 0, atol=1e-10)
    assert np.allclose(Xc.mean(0), 0, atol=1e-10)


def test_midpoint_centering_zeroes_the_pooled_mean_but_not_each_branch():
    N, P = _toy(seed=7)
    Xb, Xc = _center_paired(_pairs(len(N)), N.copy(), P.copy(), "midpoint")
    assert np.allclose((Xb.mean(0) + Xc.mean(0)) / 2, 0, atol=1e-10)
    assert not np.allclose(Xb.mean(0), 0, atol=1e-6)


def test_centering_none_is_a_no_op():
    N, P = _toy(seed=8)
    Xb, Xc = _center_paired(_pairs(len(N)), N.copy(), P.copy(), "none")
    assert np.array_equal(Xb, N) and np.array_equal(Xc, P)


def test_unknown_centering_mode_is_rejected():
    N, P = _toy(seed=9)
    with pytest.raises(ValueError):
        _center_paired(_pairs(len(N)), N, P, "sideways")


def _cov(A):
    """Population covariance about A's own mean."""
    C = A - A.mean(0)
    return C.T @ C / len(A)


def test_the_published_identity_holds_only_under_per_branch_centering():
    """cov(X+ + X-) - 2 cov(X_bar) = cov(X+,X-) + cov(X-,X+), with X_bar the concatenation.

    Covariance is translation-invariant, so cov(X+ + X-) does not care how the inputs were
    centred.  X_bar is a *concatenation*, though, and per-branch centring shifts its two
    halves by different vectors — which is not a translation, and so does change cov(X_bar).
    The identity is therefore exact only when the branch means already agree.
    """
    N, P = _toy(n=300, d=8, seed=21)
    n = len(N)
    pairs = _pairs(n)

    def s_cross(a, b):
        ca, cb = a - a.mean(0), b - b.mean(0)
        return (ca.T @ cb + cb.T @ ca) / len(a)

    target = s_cross(N, P)

    Xb, Xc = _center_paired(pairs, N.copy(), P.copy(), "per_branch")
    lhs = _cov(Xb + Xc) - 2 * _cov(np.vstack([Xb, Xc]))
    assert np.allclose(lhs, target, atol=1e-10), "identity should be exact here"

    Xb, Xc = _center_paired(pairs, N.copy(), P.copy(), "midpoint")
    lhs = _cov(Xb + Xc) - 2 * _cov(np.vstack([Xb, Xc]))
    assert not np.allclose(lhs, target, atol=1e-6), "identity should NOT be exact here"


def test_the_leftover_is_exactly_minus_two_delta_delta_transpose():
    """And the discrepancy is precisely the rank-one diff-in-means term."""
    N, P = _toy(n=300, d=8, seed=22)
    pairs = _pairs(len(N))

    def s_cross(a, b):
        ca, cb = a - a.mean(0), b - b.mean(0)
        return (ca.T @ cb + cb.T @ ca) / len(a)

    delta = (N.mean(0) - P.mean(0)) / 2
    Xb, Xc = _center_paired(pairs, N.copy(), P.copy(), "midpoint")
    lhs = _cov(Xb + Xc) - 2 * _cov(np.vstack([Xb, Xc]))
    assert np.allclose(lhs - s_cross(N, P), -2 * np.outer(delta, delta), atol=1e-10)


# ── GCC: the selection is only interpretable if ties are ordered by ||w|| ──────

def test_tied_norm_violations_flags_a_tie_broken_against_norm_order() -> None:
    from src.probing.crosscov import tied_norm_violations
    lam = np.array([-1.0, -1.0, -1.0, 2.0])          # first three tied
    assert tied_norm_violations(lam, np.array([1.0, 2.0, 3.0, 4.0]),
                                np.arange(4)) == []
    assert tied_norm_violations(lam, np.array([3.0, 1.0, 2.0, 4.0]),
                                np.arange(4)) == [(0, 1)]


def test_tied_norm_violations_ignores_untied_eigenvalues() -> None:
    """A norm decrease across a genuine eigenvalue gap is not a violation."""
    from src.probing.crosscov import tied_norm_violations
    lam = np.array([-1.0, 0.0, 1.0, 2.0])            # no ties
    assert tied_norm_violations(lam, np.array([3.0, 1.0, 2.0, 4.0]),
                                np.arange(4)) == []


def test_gcc_fit_orders_ties_by_norm_on_real_shaped_data() -> None:
    """The fit itself must not warn: the ordering assumption has to hold in practice."""
    import warnings
    from src.probing.crosscov import GeneralizedCrossCovarianceProbe
    rng = np.random.default_rng(0)
    shared = rng.normal(size=(120, 40))
    delta = rng.normal(size=(120, 40)) * 0.5
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)   # a warning fails the test
        probe = GeneralizedCrossCovarianceProbe(n_components=6)
        probe.fit_paired(shared + delta, shared - delta)
    assert probe.subspace.shape[0] == 6


# ── the N-ary formula the paper states ────────────────────────────────────────

def _nary_tuples(N: int, n: int = 400, d: int = 12, seed: int = 0):
    """N-ary tuples: a shared per-tuple component, per-slot offsets, and noise."""
    rng = np.random.default_rng(seed)
    shared = rng.normal(size=(n, d))
    slot = rng.normal(size=(N, d)) * 2.0
    return [shared + slot[i] + 0.3 * rng.normal(size=(n, d)) for i in range(N)]


def _paper_nary(Xs, centre: bool = True):
    """cov(sum_i X_i) - N cov(X*), with X* the concatenation of all N slot matrices."""
    if centre:
        Xs = [X - X.mean(0) for X in Xs]
    return (np.cov(sum(Xs), rowvar=False)
            - len(Xs) * np.cov(np.vstack(Xs), rowvar=False))


def _edge_pooled(Xs):
    """Sum of S_cross over all C(N,2) edges -- what beta=None computes on pooled edges."""
    import itertools
    Xs = [X - X.mean(0) for X in Xs]
    n, d = Xs[0].shape
    S = np.zeros((d, d))
    for i, j in itertools.combinations(range(len(Xs)), 2):
        S += Xs[i].T @ Xs[j] + Xs[j].T @ Xs[i]
    return S / (n - 1)


@pytest.mark.parametrize("N", [2, 3, 4, 7])
def test_edge_pooling_computes_the_papers_nary_formula(N):
    """Pooling all C(N,2) edges and taking beta=None *is* `cov(sum_i X_i) - N cov(X*)`.

    The paper states the N-ary objective directly; the pipeline expresses an N-tuple as
    its C(N,2) edges under one dataset name and runs the pairwise probe over them.  Those
    are the same operator: expanding cov(sum_i X_i) leaves sum_i cov(X_i) plus every
    cross term, and `N cov(X*)` is exactly the sum of the per-slot autocovariances, so
    the difference is the sum of cross-covariances over unordered pairs -- the edge sum.
    They agree up to the O(1/n) ddof term, and span the same subspace.
    """
    Xs = _nary_tuples(N)
    A, B = _paper_nary(Xs), _edge_pooled(Xs)

    scale = np.trace(A.T @ B) / np.trace(B.T @ B)
    assert abs(scale - 1.0) < 0.01
    assert np.linalg.norm(A - scale * B) / np.linalg.norm(A) < 1e-3

    wa = np.linalg.eigh(A)[1][:, :3]
    wb = np.linalg.eigh(B)[1][:, :3]
    assert np.linalg.svd(wa.T @ wb, compute_uv=False).min() > 0.999


def test_the_nary_formula_needs_per_slot_centering():
    """`N cov(X*)` is the sum of per-slot autocovariances only once each slot is centred.

    Left uncentred, cov(X*) also absorbs the between-slot mean differences and the
    difference stops being the contrastive object.  How badly depends on where the slot
    offsets happen to fall, so this averages over seeds rather than pinning one: centred,
    the leading subspaces agree to better than 0.999 every time; uncentred they do not.
    This is why the paper protocol centres per branch.
    """
    def overlap(Xs, centre):
        A, B = _paper_nary(Xs, centre=centre), _edge_pooled(Xs)
        wa = np.linalg.eigh(A)[1][:, :3]
        wb = np.linalg.eigh(B)[1][:, :3]
        return np.linalg.svd(wa.T @ wb, compute_uv=False).min()

    tuples = [_nary_tuples(4, seed=s) for s in range(8)]
    centred = [overlap(Xs, True) for Xs in tuples]
    uncentred = [overlap(Xs, False) for Xs in tuples]

    assert min(centred) > 0.999
    assert max(uncentred) < 0.9          # never recovers the subspace
    assert float(np.mean(uncentred)) < 0.5


# ── one decomposition behind CRC-TPC, tcPCA and whitened tcPCA ────────────────

def _centred_pair_cloud(n: int = 600, d: int = 10, seed: int = 3):
    """Pairs with a loud shared axis, a quiet contrastive axis, and noise; per-branch centred."""
    rng = np.random.default_rng(seed)
    shared = rng.normal(size=(n, 1)) * 6.0 * np.eye(d)[0]            # loud, same in both
    signal = rng.choice([-1.0, 1.0], size=(n, 1)) * 1.0 * np.eye(d)[1]  # quiet, opposite
    loud_contrast = rng.normal(size=(n, 1)) * 3.0 * np.eye(d)[2]      # loud, partly opposite
    Xp = shared + signal + loud_contrast + rng.normal(size=(n, d)) * 0.5
    Xm = shared - signal - 0.4 * loud_contrast + rng.normal(size=(n, d)) * 0.5
    return Xp - Xp.mean(0), Xm - Xm.mean(0)


def test_pair_mean_and_half_difference_decompose_the_pooled_covariance():
    """With m = (x+ + x-)/2 and d = (x+ - x-)/2 (branches centred separately):
    cov(X±) = cov(m) + cov(d), cov(X+ + X-) = 4 cov(m), and S_cross = 2n (cov(m) - cov(d))."""
    Xp, Xm = _centred_pair_cloud()
    n = len(Xp)
    m, d = (Xp + Xm) / 2, (Xp - Xm) / 2
    cov = lambda A: A.T @ A / len(A)                                   # population, zero mean
    pooled = cov(np.vstack([Xp, Xm]))
    assert np.allclose(pooled, cov(m) + cov(d))
    assert np.allclose(cov(Xp + Xm), 4 * cov(m))
    S_cross = Xp.T @ Xm + Xm.T @ Xp
    assert np.allclose(S_cross, 2 * n * (cov(m) - cov(d)))


def test_three_methods_rank_directions_by_three_objectives():
    """CRC-TPC maximises cov(d); tcPCA maximises cov(d) - cov(m); whitened tcPCA maximises
    (cov(d) - cov(m)) / (cov(d) + cov(m)).  So CRC-TPC = tcPCA objective + total variance,
    which is why it prefers loud directions, and whitening turns the absolute difference
    into a bounded, scale-free share."""
    Xp, Xm = _centred_pair_cloud()
    m, d = (Xp + Xm) / 2, (Xp - Xm) / 2
    cov = lambda A: A.T @ A / len(A)
    Cm, Cd = cov(m), cov(d)

    top = lambda M: np.linalg.eigh(M)[1][:, -1]
    align = lambda u, v: abs(u @ v) / (np.linalg.norm(u) * np.linalg.norm(v))

    # CRC-TPC: top principal component of the differences X- - X+ = -2d.
    crc = top(cov(Xm - Xp))
    assert align(crc, top(Cd)) > 0.999

    # tcPCA, from the probe itself.
    tc = CrossCovarianceProbe(n_components=1)
    tc.fit_paired(Xp, Xm)
    w_tc = tc._components[0]
    assert align(w_tc, top(Cd - Cm)) > 0.999

    # Whitened tcPCA, from the probe itself: the generalised problem (Cd - Cm) w = mu (Cd + Cm) w.
    import scipy.linalg
    gcc = GeneralizedCrossCovarianceProbe(n_components=1)
    gcc.fit_paired(Xp, Xm)
    w_g = gcc._components[0]
    _, vecs = scipy.linalg.eigh(Cd - Cm, Cd + Cm)
    assert align(w_g, vecs[:, -1]) > 0.999

    # On this cloud the absolute objectives agree with each other and not with the ratio.
    # Axis 2 is loud and only partly contrastive (cov(d) = 4.41, cov(m) = 0.81); axis 1 is
    # quiet and purely contrastive (cov(d) = 1, cov(m) = 0).  CRC-TPC and tcPCA both take
    # axis 2, because 4.41 - 0.81 > 1 - 0: tcPCA's correction is absolute, so a large
    # direction still wins.  Only whitening, which scores the share, takes axis 1.
    assert np.argmax(np.abs(crc)) == 2
    assert np.argmax(np.abs(w_tc)) == 2
    assert np.argmax(np.abs(w_g)) == 1
