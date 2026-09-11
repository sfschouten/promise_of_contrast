"""The canonical-view geometry, checked against cases with known exact answers."""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

# The geometry lives with the figure that uses it, not in src/probing: nothing in the
# training path imports it, and while it sat under src/probing/ every edit invalidated
# `train_probes` and `evaluate_probes` for every run.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                       / "scripts" / "reproductions" / "promise_of_contrast"))

from tetrahedron import (
    camera_basis,
    edge_lengths,
    overview_view,
    opposite_edge_pairings,
    project,
    squareness,
    view_axis,
    view_to_elev_azim,
)

# The regular tetrahedron inscribed in a cube.  Every edge has length 2*sqrt(2) and each
# pair of opposite edges is perpendicular, so all three canonical views must be squares.
REGULAR = {
    "a": np.array([1.0, 1.0, 1.0]),
    "b": np.array([1.0, -1.0, -1.0]),
    "c": np.array([-1.0, 1.0, -1.0]),
    "d": np.array([-1.0, -1.0, 1.0]),
}


def test_three_pairings_and_each_vertex_used_once() -> None:
    pairings = opposite_edge_pairings(list(REGULAR))
    assert len(pairings) == 3
    for p in pairings:
        assert not set(p.edge_a) & set(p.edge_b)
        assert set(p.edge_a) | set(p.edge_b) == set(REGULAR)
    # the three pairings are distinct as unordered partitions
    parts = {frozenset([frozenset(p.edge_a), frozenset(p.edge_b)]) for p in pairings}
    assert len(parts) == 3


def test_projection_is_always_a_parallelogram() -> None:
    """True for *any* tetrahedron — this is the geometric fact, not a finding."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        means = {k: rng.normal(size=3) for k in "abcd"}
        for pairing in opposite_edge_pairings(list(means)):
            basis = camera_basis(view_axis(means, pairing))
            a0, a1 = (project(means[k][None, :], basis)[0] for k in pairing.edge_a)
            b0, b1 = (project(means[k][None, :], basis)[0] for k in pairing.edge_b)
            # diagonals bisect each other <=> the midpoints coincide
            np.testing.assert_allclose((a0 + a1) / 2, (b0 + b1) / 2, atol=1e-10)


def test_regular_tetrahedron_projects_to_a_square_from_every_canonical_view() -> None:
    for pairing in opposite_edge_pairings(list(REGULAR)):
        m = squareness(REGULAR, pairing)
        assert m["diagonal_ratio"] == pytest.approx(1.0, abs=1e-10)
        assert m["diagonal_angle_deg"] == pytest.approx(90.0, abs=1e-8)
        assert m["side_ratio"] == pytest.approx(1.0, abs=1e-10)
        assert m["max_side_error"] == pytest.approx(0.0, abs=1e-10)


def test_squareness_detects_a_non_square() -> None:
    """Stretching one opposite-edge pair breaks the equal-diagonal condition only."""
    means = dict(REGULAR)
    means["a"] = REGULAR["a"] * np.array([1.0, 3.0, 1.0])
    means["b"] = REGULAR["b"] * np.array([1.0, 3.0, 1.0])
    worst = min(squareness(means, p)["diagonal_ratio"]
                for p in opposite_edge_pairings(list(means)))
    assert worst < 0.9, "a stretched tetrahedron must not read as a square"


def test_view_axis_is_the_semantic_contrast_direction() -> None:
    """The midpoint axis equals the difference of the two group means it separates."""
    pairing = opposite_edge_pairings(list(REGULAR))[0]
    mid_a = (REGULAR[pairing.edge_a[0]] + REGULAR[pairing.edge_a[1]]) / 2
    mid_b = (REGULAR[pairing.edge_b[0]] + REGULAR[pairing.edge_b[1]]) / 2
    expected = (mid_b - mid_a) / np.linalg.norm(mid_b - mid_a)
    np.testing.assert_allclose(view_axis(REGULAR, pairing), expected, atol=1e-12)


def test_camera_basis_is_orthonormal_and_preserves_lengths() -> None:
    rng = np.random.default_rng(1)
    for _ in range(10):
        basis = camera_basis(rng.normal(size=3))
        np.testing.assert_allclose(basis @ basis.T, np.eye(3), atol=1e-10)
        assert np.linalg.det(basis) == pytest.approx(1.0, abs=1e-10)
        # a vector already in the image plane keeps its length under projection
        v = basis[0] * 2.5 + basis[1] * 1.5
        assert np.linalg.norm(project(v[None, :], basis)[0]) == pytest.approx(
            np.linalg.norm(v), abs=1e-10)


def test_camera_basis_survives_a_degenerate_up_hint() -> None:
    view = np.array([0.0, 0.0, 1.0])
    basis = camera_basis(view, up_hint=view)          # parallel hint must not blow up
    np.testing.assert_allclose(basis @ basis.T, np.eye(3), atol=1e-10)
    np.testing.assert_allclose(basis[2], view, atol=1e-12)


def test_all_six_edges_reported() -> None:
    lengths = edge_lengths(REGULAR)
    assert len(lengths) == 6
    assert set(lengths) == set(combinations(REGULAR, 2))
    for v in lengths.values():
        assert v == pytest.approx(2 * np.sqrt(2), abs=1e-10)


def test_overview_view_keeps_clear_of_every_canonical_axis() -> None:
    """The overview must not accidentally be one of the collapsing directions.

    For a regular tetrahedron the unconstrained optimum *is* a canonical axis, so this
    also pins the reason the constraint exists.
    """
    d = overview_view(REGULAR, min_angle_deg=20.0, n_directions=2048)
    for pairing in opposite_edge_pairings(list(REGULAR)):
        cos = abs(float(np.dot(d, view_axis(REGULAR, pairing))))
        assert cos <= np.cos(np.radians(20.0)) + 1e-9


def test_overview_view_still_separates_all_four_points() -> None:
    d = overview_view(REGULAR, min_angle_deg=20.0, n_directions=2048)
    proj = project(np.stack([REGULAR[k] for k in REGULAR]), camera_basis(d))
    gaps = [np.linalg.norm(proj[i] - proj[j]) for i, j in combinations(range(4), 2)]
    assert min(gaps) > 0.5, "the overview must not collapse two vertices onto each other"


def test_view_to_elev_azim_round_trips() -> None:
    for view in (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]),
                 np.array([0.3, -0.5, 0.8])):
        view = view / np.linalg.norm(view)
        elev, azim = view_to_elev_azim(view)
        e, a = np.radians(elev), np.radians(azim)
        back = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
        np.testing.assert_allclose(back, view, atol=1e-10)


def test_contrast_axes_are_the_group_mean_differences() -> None:
    from tetrahedron import axis_angles, contrast_axes
    axes = contrast_axes(REGULAR)
    assert len(axes) == 3
    # For the regular tetrahedron the three axes are mutually perpendicular.
    for angle in axis_angles(REGULAR).values():
        assert angle == pytest.approx(90.0, abs=1e-8)


def test_axis_angles_detect_non_orthogonal_contrasts() -> None:
    from tetrahedron import axis_angles
    skewed = dict(REGULAR)
    skewed["d"] = REGULAR["d"] + np.array([1.4, 0.0, 0.0])
    assert min(axis_angles(skewed).values()) < 85.0
