"""Canonical viewing directions for a four-group contrast design.

A 2x2 design gives four group means; in any three-dimensional projection they are the
vertices of a tetrahedron.  Choosing a camera angle by hand invites the obvious objection
that the picture was fitted to the claim, so this module derives the angles from the
geometry instead.

## The one fact the figures rest on

A tetrahedron has three pairs of **opposite edges** — pairs sharing no vertex.  Project
along the axis joining the midpoints of one such pair and those midpoints coincide, so the
four projected points are symmetric about a common centre: the quadrilateral's diagonals
bisect each other, which makes it a **parallelogram**.

That is true of *every* tetrahedron, so "it looks like a parallelogram from the side" is
not evidence of anything.  The claim with content is that the parallelogram is a **square**,
and a parallelogram whose diagonals bisect each other is a square exactly when those
diagonals are

  * equal in length     -> the two opposite edges have equal length, and
  * perpendicular       -> the two opposite edges are perpendicular in projection.

Both are properties of the data, and `squareness` reports them as numbers rather than
leaving them to the eye.

## Why the three views are the semantic ones

For the 2x2 <pc, pi, nc, ni> grid (polarity x content) the three pairings are exactly the
three contrasts of the design, because the midpoint axis of a pairing is the difference of
the two group means it separates:

    (pc,pi) | (nc,ni)   axis = mean(pos)     - mean(neg)         collapses negation
    (pc,nc) | (pi,ni)   axis = mean(correct) - mean(incorrect)   collapses base
    (pc,ni) | (pi,nc)   axis = mean(true)    - mean(false)       collapses truth

So each canonical view answers "what is left once this contrast is projected out?".
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np


@dataclass(frozen=True)
class Pairing:
    """One pair of opposite edges, and the view that collapses it."""

    edge_a: tuple[str, str]
    edge_b: tuple[str, str]

    @property
    def label(self) -> str:
        return f"({','.join(self.edge_a)}) | ({','.join(self.edge_b)})"


def opposite_edge_pairings(labels: list[str]) -> list[Pairing]:
    """The three ways to split four vertices into two disjoint edges.

    Order is deterministic: it follows `combinations` over the labels as given, and the
    partner edge is whatever two labels are left.
    """
    if len(labels) != 4:
        raise ValueError(f"a tetrahedron has four vertices, got {len(labels)}")
    if len(set(labels)) != 4:
        raise ValueError(f"vertex labels must be distinct: {labels}")

    seen: list[Pairing] = []
    for edge in combinations(labels, 2):
        rest = tuple(l for l in labels if l not in edge)
        if any(set(edge) == set(p.edge_b) for p in seen):
            continue                      # already recorded as the partner of an earlier edge
        seen.append(Pairing(edge_a=edge, edge_b=rest))
    return seen


def view_axis(means: dict[str, np.ndarray], pairing: Pairing) -> np.ndarray:
    """Unit vector from the midpoint of `edge_a` to the midpoint of `edge_b`.

    Looking along this axis is what makes the two midpoints coincide.
    """
    mid_a = (means[pairing.edge_a[0]] + means[pairing.edge_a[1]]) / 2
    mid_b = (means[pairing.edge_b[0]] + means[pairing.edge_b[1]]) / 2
    d = mid_b - mid_a
    n = np.linalg.norm(d)
    if n == 0:
        raise ValueError(
            f"opposite edges {pairing.label} share a midpoint; the view is undefined")
    return d / n


def camera_basis(view: np.ndarray, up_hint: np.ndarray | None = None) -> np.ndarray:
    """A right-handed orthonormal basis whose third row is the viewing direction.

    Rows 0 and 1 span the image plane, so `points @ basis[:2].T` is the orthographic
    projection.  `up_hint` only fixes the in-plane rotation; it never affects lengths or
    angles, and a hint parallel to the view is replaced rather than allowed to degenerate.
    """
    w = view / np.linalg.norm(view)
    hint = np.zeros_like(w) if up_hint is None else np.asarray(up_hint, dtype=float)
    if hint.shape != w.shape or np.linalg.norm(hint) == 0 or \
            abs(np.dot(hint / np.linalg.norm(hint), w)) > 1 - 1e-8:
        # any vector not parallel to w will do; pick the axis w leans on least
        hint = np.eye(len(w))[int(np.argmin(np.abs(w)))]
    u = hint - np.dot(hint, w) * w
    u /= np.linalg.norm(u)
    v = np.cross(w, u)
    return np.vstack([u, v, w])


def project(points: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Orthographic projection onto the image plane of `basis` (an (n, 2) array)."""
    return np.asarray(points, dtype=float) @ basis[:2].T


def squareness(means: dict[str, np.ndarray], pairing: Pairing) -> dict[str, float]:
    """How close the projected parallelogram is to a square.

    The two projected opposite edges are the parallelogram's diagonals, so:

      `diagonal_ratio`      shorter/longer, 1.0 when the opposite edges are equal length
      `diagonal_angle_deg`  angle between them, 90 when they are perpendicular
      `side_ratio`          shorter/longer side of the parallelogram, 1.0 for a rhombus
      `max_side_error`      largest relative deviation of a side from the mean side

    A square needs `diagonal_ratio` = 1 and `diagonal_angle_deg` = 90; the side figures
    are the same statement read off the sides instead, and are the easier ones to eyeball
    against the plot.
    """
    view = view_axis(means, pairing)
    basis = camera_basis(view)
    a0, a1 = (project(means[k][None, :], basis)[0] for k in pairing.edge_a)
    b0, b1 = (project(means[k][None, :], basis)[0] for k in pairing.edge_b)

    p, q = a0 - a1, b0 - b1                      # the two diagonals
    lp, lq = float(np.linalg.norm(p)), float(np.linalg.norm(q))
    cos = float(np.dot(p, q) / (lp * lq)) if lp and lq else 0.0
    angle = float(np.degrees(np.arccos(np.clip(abs(cos), -1.0, 1.0))))

    # Sides of the parallelogram are the half-sums and half-differences of the diagonals.
    s1, s2 = np.linalg.norm((p + q) / 2), np.linalg.norm((p - q) / 2)
    sides = np.array([s1, s2, s1, s2], dtype=float)
    mean_side = float(sides.mean())

    return {
        "diagonal_ratio":     min(lp, lq) / max(lp, lq) if max(lp, lq) else 0.0,
        "diagonal_angle_deg": angle,
        "side_ratio":         float(min(s1, s2) / max(s1, s2)) if max(s1, s2) else 0.0,
        "max_side_error":     float(np.abs(sides - mean_side).max() / mean_side)
                              if mean_side else 0.0,
        "diagonal_a":         lp,
        "diagonal_b":         lq,
    }


def contrast_axes(means: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The three opposite-edge midpoint axes, keyed by `Pairing.label`.

    Each is the difference of the two group means the pairing separates, so its norm is
    the size of that contrast and the angles between them say whether the design's three
    contrasts are carried by independent directions.  This is the quantity behind the
    square: the projected parallelogram is a rectangle when the two *remaining* axes are
    perpendicular, and a square only when they are also equal in length.
    """
    out = {}
    for pairing in opposite_edge_pairings(list(means)):
        mid_a = (means[pairing.edge_a[0]] + means[pairing.edge_a[1]]) / 2
        mid_b = (means[pairing.edge_b[0]] + means[pairing.edge_b[1]]) / 2
        out[pairing.label] = mid_b - mid_a
    return out


def axis_angles(means: dict[str, np.ndarray]) -> dict[tuple[str, str], float]:
    """Pairwise angles between the three contrast axes, in degrees (90 = orthogonal)."""
    axes = contrast_axes(means)
    out = {}
    for a, b in combinations(axes, 2):
        u = axes[a] / np.linalg.norm(axes[a])
        v = axes[b] / np.linalg.norm(axes[b])
        out[(a, b)] = float(np.degrees(np.arccos(np.clip(abs(float(u @ v)), -1.0, 1.0))))
    return out


def edge_lengths(means: dict[str, np.ndarray]) -> dict[tuple[str, str], float]:
    """All six edge lengths of the tetrahedron, for reporting alongside the views."""
    return {e: float(np.linalg.norm(means[e[0]] - means[e[1]]))
            for e in combinations(means, 2)}


def _fibonacci_sphere(n: int) -> np.ndarray:
    """`n` roughly equidistributed unit vectors — a deterministic direction sweep."""
    i = np.arange(n, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.column_stack([np.cos(theta) * np.sin(phi),
                            np.sin(theta) * np.sin(phi),
                            np.cos(phi)])


def overview_view(
    means: dict[str, np.ndarray],
    min_angle_deg: float = 20.0,
    n_directions: int = 4096,
) -> np.ndarray:
    """A viewing direction for the panel that shows the tetrahedron *as* a tetrahedron.

    Criterion: maximise the smallest projected distance between vertices — the view that
    hides the least — subject to sitting at least `min_angle_deg` away from all three
    canonical axes.

    The constraint is not decoration.  For a regular tetrahedron the *unconstrained*
    optimum is exactly a canonical axis: looking down an opposite-edge midpoint axis gives
    a square of side s and diagonal s*sqrt(2), whose minimum pairwise distance s beats
    every skew view.  So "the most informative direction" and "the direction that collapses
    a contrast" coincide there, and an unconstrained search would hand back the very view
    the overview is meant to contrast with.
    """
    axes = [view_axis(means, p) for p in opposite_edge_pairings(list(means))]
    cos_limit = np.cos(np.radians(min_angle_deg))
    pts = np.stack([means[k] for k in means])

    best, best_score = None, -np.inf
    for d in _fibonacci_sphere(n_directions):
        if any(abs(float(np.dot(d, a))) > cos_limit for a in axes):
            continue                       # too close to a collapsing direction
        proj = project(pts, camera_basis(d))
        score = min(np.linalg.norm(proj[i] - proj[j])
                    for i, j in combinations(range(len(proj)), 2))
        if score > best_score:
            best, best_score = d, score
    if best is None:
        raise ValueError(
            f"no direction is more than {min_angle_deg} deg from all three canonical axes")
    return best


def view_to_elev_azim(view: np.ndarray) -> tuple[float, float]:
    """Matplotlib `view_init` angles placing the camera on the +`view` side of the scene.

    Only for 3-D axes; the 2-D panels project explicitly and do not need it.
    """
    x, y, z = view / np.linalg.norm(view)
    return float(np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))), float(np.degrees(np.arctan2(y, x)))
