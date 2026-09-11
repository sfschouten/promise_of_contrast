"""Fixed-camera stills of the <pc, pi, nc, ni> tetrahedron on the leading tcPCs.

The figure shows truth, base proposition and negation together on the first three tcPCs.
A 3-D scatter would do that but leave the reader trusting whatever camera angle the
authors happened to like.  This module derives the
angles instead: the four group means form a tetrahedron, a tetrahedron has three pairs of
opposite edges, and looking down the axis joining the midpoints of one pair collapses
exactly one contrast of the 2x2 design.  See `tetrahedron.py` beside this file for the geometry
and the reason a parallelogram is guaranteed while a *square* is not.

Panels: the three canonical views plus one overview that keeps clear of all three, so the
tetrahedron is visible before it is flattened.  The camera is orthographic and the axes are
locked to tcPC 0/1/2 throughout — only the camera moves between panels.

Each panel fills a cube, which means the three axes are scaled independently.  That is a
deliberate choice: the tick labels still show the true range of every axis, so the real
proportions remain readable, but a shape on the page is not a shape in the data.  Every
claim about squareness accordingly comes from `squareness()` and lands in
`tetrahedron_views.csv` and the panel subtitles, never from the drawn outline.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from src.activations.cache import load_sample_cache
from src.data.graph import load_family_as_pairs
from src.probing import Manifest, load_probes, probe_key
from tetrahedron import (
    axis_angles,
    camera_basis,
    contrast_axes,
    edge_lengths,
    opposite_edge_pairings,
    overview_view,
    project,
    squareness,
    view_axis,
    view_to_elev_azim,
)

from _common import activations_dir, model_name, probes_dir

# The 2x2 grid.  Truth is the XOR of polarity and content: pc and ni are true.
NODES = ["pc", "pi", "nc", "ni"]
NODE_SET = set(NODES)
NODE_LABELS = {
    "pc": "pos / correct  (T)",
    "pi": "pos / incorrect (F)",
    "nc": "neg / correct  (F)",
    "ni": "neg / incorrect (T)",
}
NODE_COLORS = {"pc": "#1b6ca8", "ni": "#5aa9e6", "pi": "#b3122e", "nc": "#e3736a"}

# Which contrast each opposite-edge pairing collapses.  Keyed by the frozen partition so
# the naming does not depend on the order `opposite_edge_pairings` happens to return.
CONTRAST_OF = {
    frozenset([frozenset(["pc", "pi"]), frozenset(["nc", "ni"])]): "negation",
    frozenset([frozenset(["pc", "nc"]), frozenset(["pi", "ni"])]): "base",
    frozenset([frozenset(["pc", "ni"]), frozenset(["pi", "nc"])]): "truth",
}

RUN = "cities_grid"
DATASET = "cities_grid_all"
POSITION = "last"

# One page per contrastive basis.
#
# The generalised basis earns its page.  Its *eigenvalues* are uninformative here -- a
# large set of directions reach the attainable maximum together (lambda = -2/3, i.e. an
# antisymmetric variance share of 2/3) and tie to four decimals -- but the selection is
# not thereby arbitrary.  Sigma-orthonormality fixes w^T Sigma w = 1, so ||w|| encodes the
# pooled variance along the unit direction as 1/||w||^2, and within the tied set lambda and
# ||w|| are rank-identical.  The ordering is therefore deterministic and data-determined:
# maximal contrast fraction first, then maximal variance.
#
# What it buys is legibility.  In the tuple-contrastive basis the three contrasts differ in
# raw magnitude by 5.4x on llama2 (negation 3.09, base 1.50, truth 0.57), so a single view
# cannot show all three at a readable scale.  Whitening equalises them, which is why this
# page often shows the <pc, pi, nc, ni> structure more cleanly.  The cost is that shape on
# the page is no longer distance in the data, so the squareness numbers in
# tetrahedron_views.csv -- not the drawn outline -- remain the evidence.
METHODS = {
    "cross_covariance_8":             "tcPC",
    "generalized_cross_covariance_8": "GCC",
}


# In each canonical view one of the two surviving contrasts is put straight up, so the
# square the view shows sits on its side rather than its corner.
UP_OF = {"negation": "truth", "base": "truth", "truth": "base"}


def angle_between(means: dict[str, np.ndarray], a: str, b: str) -> float:
    """Angle in degrees between two named contrast axes (90 = independent directions)."""
    name = {p.label: _contrast_name(p) for p in opposite_edge_pairings(NODES)}
    for (la, lb), v in axis_angles(means).items():
        if {name[la], name[lb]} == {a, b}:
            return v
    raise KeyError(f"no axis pair {a}/{b}")


def _contrast_name(pairing) -> str:
    return CONTRAST_OF.get(
        frozenset([frozenset(pairing.edge_a), frozenset(pairing.edge_b)]), "?")


def projected_points(model: str, layer: int, method: str = "cross_covariance_8",
                     split: str = "eval") -> tuple[np.ndarray, np.ndarray]:
    """Held-out samples' activations in the first three tcPC coordinates.

    Same convention as `report_subspace_scatter.py`: coordinates are
    `X @ probe.subspace.T`, so nothing is re-fitted here and the axes are tcPC 0/1/2.
    Returns `(coords, nodes)` with one row per sample.
    """
    pdir = probes_dir(model, RUN)
    manifest = Manifest.load(pdir)
    probes = load_probes(manifest.params_file)

    entry = next((e for e in manifest.entries
                  if e.dataset == DATASET and e.method == method
                  and str(e.train_position) == POSITION and e.layer == layer), None)
    if entry is None:
        raise LookupError(
            f"no {method} probe for {DATASET} at layer {layer}, position {POSITION}")
    probe = probes[probe_key(entry.dataset, entry.layer, entry.train_position,
                             entry.method, entry.descriptor, entry.seed)]
    W = probe.subspace.float().numpy()[:3]           # (3, hidden)

    samples = [s for s in load_family_as_pairs("cities_grid")
               if s.descriptors.get("dataset_name") == DATASET]
    # Only cities the probe was not fitted on (the split is by city), unless asked otherwise.
    held_out = set(entry.test_ids)
    if split == "eval" and held_out:
        samples = [s for s in samples if s.id in held_out]
    cache_dir, mname = activations_dir(model, RUN), model_name(model)

    coords, nodes, seen = [], [], set()
    for s in samples:
        node = s.descriptors.get("node")
        if node not in NODE_SET or s.id in seen:
            continue                                  # each node appears in several edges
        seen.add(s.id)
        vec = load_sample_cache(cache_dir, mname, s).get((layer, POSITION))
        if vec is None:
            continue
        coords.append(vec.float().numpy() @ W.T)
        nodes.append(node)

    if not coords:
        raise LookupError(f"no cached activations for {DATASET} at layer {layer}")
    return np.asarray(coords), np.asarray(nodes)


def group_means(model: str, layer: int,
                method: str = "cross_covariance_8") -> tuple[dict[str, np.ndarray], int]:
    """The four group means, and the number of samples behind them."""
    coords, nodes = projected_points(model, layer, method)
    means = {}
    for k in NODES:
        mask = nodes == k
        if not mask.any():
            raise LookupError(f"no samples for group {k} at layer {layer}")
        means[k] = coords[mask].mean(axis=0)
    return means, len(coords)


def view_table(means: dict[str, np.ndarray]) -> pd.DataFrame:
    """Squareness of each canonical view, as numbers rather than an impression.

    `side_ratio` is the one to read: it is the ratio of the two contrast magnitudes that
    survive the view, so 1.0 is a square and anything else is a rectangle with that
    aspect.  `diagonal_angle_deg` near 90 says the two surviving contrasts are carried by
    perpendicular directions, which is the substantive claim; the aspect ratio then only
    says how large one contrast is relative to the other.
    """
    axes = contrast_axes(means)
    pairings = opposite_edge_pairings(NODES)
    rows = []
    for pairing in pairings:
        m = squareness(means, pairing)
        # compare on label: Pairing is a value object, and identity would never match
        others = [p for p in pairings if p.label != pairing.label]
        rows.append({
            "collapses": _contrast_name(pairing),
            "opposite_edges": pairing.label,
            "collapsed_axis_norm": round(float(np.linalg.norm(axes[pairing.label])), 4),
            "surviving": " x ".join(_contrast_name(p) for p in others),
            "surviving_norms": " x ".join(
                f"{float(np.linalg.norm(axes[p.label])):.3f}" for p in others),
            "side_ratio": round(m["side_ratio"], 4),
            "diagonal_ratio": round(m["diagonal_ratio"], 4),
            "diagonal_angle_deg": round(m["diagonal_angle_deg"], 2),
        })
    return pd.DataFrame(rows)


def axis_table(means: dict[str, np.ndarray]) -> pd.DataFrame:
    """Magnitude of each contrast axis and the angles between them.

    The headline numbers: near-90 angles mean the eigendecomposition — which never saw a
    label — put negation, content and truth on independent directions.
    """
    axes = contrast_axes(means)
    name = {p.label: _contrast_name(p) for p in opposite_edge_pairings(NODES)}
    angles = {(name[a], name[b]): v for (a, b), v in axis_angles(means).items()}
    rows = []
    for label, vec in axes.items():
        norm = float(np.linalg.norm(vec))
        rows.append({
            "contrast": name[label],
            "norm": round(norm, 4),
            "unit_vector_tcpc_0_1_2": np.round(vec / norm, 3).tolist(),
            **{f"angle_to_{other}": round(v, 2)
               for (a, b), v in angles.items()
               for other in [b if a == name[label] else a]
               if name[label] in (a, b)},
        })
    return pd.DataFrame(rows)


def edge_table(means: dict[str, np.ndarray]) -> pd.DataFrame:
    lengths = edge_lengths({k: means[k] for k in NODES})
    mean_len = float(np.mean(list(lengths.values())))
    return pd.DataFrame([
        {"edge": f"{a}-{b}", "length": round(v, 4),
         "relative_to_mean": round(v / mean_len, 4)}
        for (a, b), v in lengths.items()
    ])


# Depth cues.  The camera is orthographic, so there is no perspective to convey depth;
# adding perspective instead would bend the very lengths and angles the figure exists to
# measure.  A mild size ramp plus matplotlib's own depth shading does the job and moves
# nothing.  In a canonical view the depth axis *is* the collapsed contrast, so these cues
# show precisely the dimension the projection removes.
POINT_SIZE = (4.5, 9.0)


def _depth_scale(depth: np.ndarray) -> np.ndarray:
    """Map depth to [0, 1], 1 = nearest the camera.  Flat input maps to the mid-tone."""
    lo, hi = float(depth.min()), float(depth.max())
    if hi - lo < 1e-12:
        return np.full_like(depth, 0.5)
    return (depth - lo) / (hi - lo)


def _roll_for(ax, elev: float, azim: float, center: np.ndarray, up: np.ndarray) -> float:
    """Camera roll (degrees) that puts `up` straight up on screen.

    Needed for views looking almost straight down the vertical axis: there matplotlib has
    no natural "up" and the azimuth alone spins the picture, so a square lands on its corner.
    """
    from mpl_toolkits.mplot3d import proj3d
    best, best_err = 0.0, np.inf
    for roll in np.arange(-180.0, 180.0, 1.0):
        ax.view_init(elev=elev, azim=azim, roll=roll)
        M = ax.get_proj()
        (x0, y0, _), (x1, y1, _) = (proj3d.proj_transform(*p, M) for p in (center, center + up))
        err = abs(np.degrees(np.arctan2(x1 - x0, y1 - y0)))
        if err < best_err:
            best, best_err = float(roll), err
    return best


def _panel(ax, means, coords, nodes, view, title, subtitle,
           comp: str = "tcPC", up: np.ndarray | None = None, loose: bool = False,
           axis_names: tuple[str, str, str] | None = None) -> None:
    """One 3-D panel viewed along `view`.

    The projection is set to orthographic and the box aspect to the data ranges, so one
    unit is the same length on every axis and the on-screen geometry is the geometry of
    the data — the property the canonical views depend on.  What the 3-D axes add over a
    flat projection is the frame: panes, gridlines and a labelled tcPC 2 axis, so the
    reader can see that the flat-looking views really are a solid seen edge-on.
    """
    ax.set_proj_type("ortho")
    elev, azim = view_to_elev_azim(view)
    ax.view_init(elev=elev, azim=azim)

    w = view / np.linalg.norm(view)
    depth = _depth_scale(coords @ w)
    sizes = POINT_SIZE[0] + (POINT_SIZE[1] - POINT_SIZE[0]) * depth
    if loose:
        sizes = sizes * 0.35                       # printed ~1.5 in wide
    order = np.argsort(depth)                     # far first, near painted over them
    ax.scatter(coords[order, 0], coords[order, 1], coords[order, 2],
               s=sizes[order], c=[NODE_COLORS[n] for n in nodes[order]],
               alpha=0.35, linewidths=0, depthshade=True, zorder=2)

    for a, b in ((x, y) for i, x in enumerate(NODES) for y in NODES[i + 1:]):
        seg = np.vstack([means[a], means[b]])
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color="0.2", lw=1.2, alpha=0.9, zorder=4)
    for k in NODES:
        m = means[k]
        ax.scatter([m[0]], [m[1]], [m[2]], s=34 if loose else 110, marker="D",
                   color=NODE_COLORS[k], edgecolor="#222222", linewidth=0.8 if loose else 1.1,
                   depthshade=False, zorder=5)
        label = ax.text(m[0], m[1], m[2], f"  {k}" if loose else f"     {k}",
                        fontsize=7 if loose else 10, color="#111111", zorder=6)
        # a white halo, or the label vanishes into its own cluster
        label.set_path_effects([pe.withStroke(linewidth=2.6, foreground="white")])

    # One unit must be one unit on every axis, or the angles on screen are not the angles
    # in the data.  Ranges are padded so the panes do not clip the cloud.
    lo, hi = coords.min(axis=0), coords.max(axis=0)
    pad = 0.06 * np.maximum(hi - lo, 1e-9)
    lo, hi = lo - pad, hi + pad
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    # Box aspect follows the data ranges, so one unit is one unit on every axis.  It is
    # only normalised and zoomed, never equalised: forcing a cube would rescale the axes
    # independently and the aspect ratios the figure reports would stop matching it.
    # The box is a cube, so each axis is scaled independently and the panel fills its
    # subplot.  The *tick scales* then carry the true proportions — tcPC 0 spanning 7 units
    # against tcPC 2's 1.75 is legible from the labels — so nothing is hidden, but shape on
    # the page is no longer distance in the data.  The square/rectangle test therefore lives
    # in the printed numbers and in tetrahedron_views.csv, not in the drawn outline.
    ax.set_box_aspect((1, 1, 1), zoom=1.0 if loose else 1.03)

    names = axis_names or (f"{comp} 0", f"{comp} 1", f"{comp} 2")
    lab_size, tick_size = (6.5, 5.5) if loose else (7.5, 5.5)
    ax.set_xlabel(names[0], fontsize=lab_size, labelpad=-4 if loose else -2)
    ax.set_ylabel(names[1], fontsize=lab_size, labelpad=-4 if loose else -2)
    ax.set_zlabel(names[2], fontsize=lab_size, labelpad=-4 if loose else -2)
    ax.tick_params(labelsize=tick_size, pad=-2 if loose else 0)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(3 if loose else 4))
    # An axis pointing at the camera is foreshortened to nothing: its tick labels pile up
    # into a smear and its name floats over the middle of the plot.  Drop both.
    w_unit = view / np.linalg.norm(view)
    for axis, basis_vec, setter in zip((ax.xaxis, ax.yaxis, ax.zaxis), np.eye(3),
                                       (ax.set_xlabel, ax.set_ylabel, ax.set_zlabel)):
        if abs(float(basis_vec @ w_unit)) > 0.9:
            axis.set_ticklabels([])
            setter("")
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0.04)
        pane.pane.set_edgecolor("0.8")

    if up is not None:
        center = np.array([(lo[i] + hi[i]) / 2 for i in range(3)])
        u = np.asarray(up, float)
        u = u - (u @ w_unit) * w_unit              # the part of `up` visible from this camera
        if np.linalg.norm(u) > 1e-9:
            ax.view_init(elev=elev, azim=azim,
                         roll=_roll_for(ax, elev, azim, center, u / np.linalg.norm(u)))
    if title or subtitle:
        ax.set_title(title, fontsize=10.5, pad=22)
        # inside the axes: anything placed below them lands on the next row's title, and
        # tight_layout reserves no space for text2D
        ax.text2D(0.5, 0.005, subtitle, transform=ax.transAxes, ha="center", va="bottom",
                  fontsize=7.6, color="0.35")


def _page(fig_title: str, means, coords, nodes, views, n, layer, comp):
    """One figure: three canonical views plus an overview, for a single basis."""
    axes_by_name = {_contrast_name(p): contrast_axes(means)[p.label]
                    for p in opposite_edge_pairings(NODES)}
    pairings = {_contrast_name(p): p for p in opposite_edge_pairings(NODES)}
    panel_views = [view_axis(means, pairings[row["collapses"]])
                   for _, row in views.iterrows()] + [overview_view(means)]

    fig, axs = plt.subplots(2, 2, figsize=(11.0, 10.2),
                            subplot_kw={"projection": "3d"})
    for ax, (_, row), vw in zip(axs.flat, views.iterrows(), panel_views):
        horiz, vert = row["surviving"].split(" x ")
        between = angle_between(means, horiz, vert)
        _panel(ax, means, coords, nodes, vw, up=axes_by_name[UP_OF[row["collapses"]]],
               title=
               f"depth is $\\bf{{{row['collapses']}}}$", subtitle=
               f"{horiz} x {vert}  ({row['surviving_norms']}),  "
               f"aspect {row['side_ratio']:.2f},  {between:.0f}° apart",
               comp=comp)
    _panel(axs.flat[3], means, coords, nodes, panel_views[3],
           "depth is $\\bf{oblique}$ (clear of all three axes)",
           f"layer {layer}, {n} samples, {comp} 0-2", comp=comp)

    handles = [Line2D([], [], marker="D", linestyle="none", markersize=7,
                      markerfacecolor=NODE_COLORS[k], markeredgecolor="#222222",
                      label=f"{k} — {NODE_LABELS[k]}") for k in NODES]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(fig_title, fontsize=11)
    fig.tight_layout(rect=(0, 0.045, 1, 0.93), h_pad=2.5, w_pad=1.5)
    return fig


def build(model: str, out_dir: Path, layer: int = 15) -> list[str]:
    """Write the stills and the numbers behind them; returns the filenames written.

    One page per contrastive basis, so the tuple-contrastive geometry can be compared
    against its variance-normalised sibling without hunting through two files.
    """
    view_rows, axis_rows, edge_rows = [], [], []
    pdf_path = out_dir / "fig_cities_grid_tetrahedron.pdf"

    with PdfPages(pdf_path) as pdf:
        for method, comp in METHODS.items():
            # A basis the run did not fit should cost its page, not the paper build.
            try:
                coords, nodes = projected_points(model, layer, method)
            except LookupError as exc:
                print(f"  skipping {comp} page: {exc}")
                continue
            means = {k: coords[nodes == k].mean(axis=0) for k in NODES}
            n = len(coords)
            views = view_table(means)

            fig = _page(
                f"{model}: the <pc, pi, nc, ni> tetrahedron on the first three {comp}s\n"
                "each canonical view looks down the axis that collapses one contrast; "
                "nearer points are larger",
                means, coords, nodes, views, n, layer, comp)
            pdf.savefig(fig)
            plt.close(fig)

            for df, sink in ((views, view_rows), (axis_table(means), axis_rows),
                             (edge_table(means), edge_rows)):
                sink.append(df.assign(method=method, basis=comp))

    if not view_rows:
        raise LookupError(
            f"no contrastive basis available for {model} at layer {layer}: "
            f"tried {', '.join(METHODS)}")

    written = [pdf_path.name]
    for name, rows in (("tetrahedron_views.csv", view_rows),
                       ("tetrahedron_axes.csv", axis_rows),
                       ("tetrahedron_edges.csv", edge_rows)):
        pd.concat(rows, ignore_index=True).to_csv(out_dir / name, index=False)
        written.append(name)
    return written



def _overflow(fig, ax) -> tuple[float, float, float, float]:
    """How far (figure fractions) drawn content spills past the left, right, bottom, top edge.

    Axis names are measured from their extents -- a name placed entirely off the canvas
    leaves no pixels to find -- and everything else from the rendered pixels touching an edge.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    left = right = bottom = top = 0.0
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        label = axis.label
        if not label.get_text() or not label.get_visible():
            continue
        e = label.get_window_extent(renderer)
        left, right = max(left, -e.x0 / W), max(right, (e.x1 - W) / W)
        bottom, top = max(bottom, -e.y0 / H), max(top, (e.y1 - H) / H)
    rgba = np.asarray(fig.canvas.buffer_rgba())
    ink = (rgba[..., :3] < 245).any(axis=-1) & (rgba[..., 3] > 0)
    step = 0.02
    if ink[:, 0].any():
        left = max(left, step)
    if ink[:, -1].any():
        right = max(right, step)
    if ink[-1, :].any():
        bottom = max(bottom, step)
    if ink[0, :].any():
        top = max(top, step)
    return left, right, bottom, top


def _ink_box(fig, pad: float = 0.02, return_touch: bool = False):
    """Bounding box (inches) of what is actually drawn, labels included.

    `bbox_inches="tight"` misses 3-D axis labels, so the box is taken from the rendered
    pixels: everything that is not background, plus `pad` inches.
    """
    from matplotlib.transforms import Bbox
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    ink = (rgba[..., :3] < 245).any(axis=-1) & (rgba[..., 3] > 0)
    ys, xs = np.nonzero(ink)
    h, dpi = ink.shape[0], fig.dpi
    touches = (xs.min() == 0 or ys.min() == 0
               or xs.max() == ink.shape[1] - 1 or ys.max() == h - 1)
    if return_touch:
        return touches
    return Bbox([[xs.min() / dpi - pad, (h - ys.max() - 1) / dpi - pad],
                 [(xs.max() + 1) / dpi + pad, (h - ys.min()) / dpi + pad]])


FILE_TAG = {"tcPC": "tcpca", "GCC": "wtcpca"}
DISPLAY = {"tcPC": "tcPC", "GCC": "g-tcPC"}   # axis names in the paper


def build_loose(model: str, out_dir: Path, layer: int = 15) -> list[str]:
    """One file per view and basis, no titles, plus one legend file.

    Written to `out_dir/panels/` for the paper to assemble with subfigure: the three
    canonical views (depth is negation / base / truth) and the oblique overview, for tcPCA
    and whitened tcPCA.  The numbers a caption may want (aspect, angle) are in
    tetrahedron_views.csv.
    """
    pdir = out_dir / "panels"
    pdir.mkdir(exist_ok=True)
    written = []
    for method, comp in METHODS.items():
        try:
            coords, nodes = projected_points(model, layer, method)
        except LookupError as exc:
            print(f"  loose {comp}: skipped ({exc})")
            continue
        means = {k: coords[nodes == k].mean(axis=0) for k in NODES}
        pairings = {_contrast_name(p): p for p in opposite_edge_pairings(NODES)}
        caxes = {name: contrast_axes(means)[p.label] for name, p in pairings.items()}
        views = {name: view_axis(means, p) for name, p in pairings.items()}
        views["oblique"] = overview_view(means)
        shown = DISPLAY.get(comp, comp)
        figs = {}
        for vname, vw in views.items():
            # A fixed canvas with the axes inset: 3-D axis labels are drawn outside the axes
            # box, and a tight bounding box does not see them, so they need a margin.
            # The canonical views drop their axis names -- two of the three axes are seen
            # edge-on anyway -- so their box can fill the panel.  The oblique view keeps them,
            # with a margin to draw them in.  300 dpi keeps the crop box precise.
            named = vname == "oblique"
            # Every view: the same 1.5 in axes box on a 2 in canvas, so the oblique view's
            # names land in the margin instead of off the canvas, and the shared crop below
            # keeps all four at one scale.
            fig = plt.figure(figsize=(2.0, 2.0), dpi=300)
            # the oblique view's names (left and bottom) need more room than the margin
            ax = fig.add_axes([0.2, 0.16, 0.72, 0.72] if named else [0.125, 0.125, 0.75, 0.75],
                              projection="3d")
            _panel(ax, means, coords, nodes, vw, "", "", comp=comp, loose=True,
                   up=caxes[UP_OF[vname]] if vname in UP_OF else None,
                   axis_names=(f"{shown} 1", f"{shown} 2", f"{shown} 3") if named else ("", "", ""))
            # Labels of a 3-D axis land wherever the camera puts them; if anything reaches
            # the canvas edge it would be cut, so shrink the axes about their centre and redraw.
            # If content spills off one side, move the axes away from it; only if it spills
            # off opposite sides at once, shrink.  Moving keeps the plot at full size.
            for _ in range(12):
                left, right, bottom, top = _overflow(fig, ax)
                if max(left, right, bottom, top) == 0:
                    break
                x0, y0, w, h = ax.get_position().bounds
                dx = (left if not right else 0) - (right if not left else 0)
                dy = (bottom if not top else 0) - (top if not bottom else 0)
                if left and right:
                    x0, w = x0 + 0.03 * w, 0.94 * w
                if bottom and top:
                    y0, h = y0 + 0.03 * h, 0.94 * h
                ax.set_position([x0 + 1.05 * dx, y0 + 1.05 * dy, w, h])
            else:
                import warnings
                warnings.warn(f"{comp}/{vname}: content still touches the canvas edge",
                              RuntimeWarning)
            figs[vname] = fig
        # The three canonical views are cropped to one shared box -- the union of their drawn
        # content -- so they print at one scale with matching text and point sizes.  The
        # oblique view, which carries the axis names, is cropped to its own content and can
        # be sized independently in LaTeX.
        from matplotlib.transforms import Bbox
        canonical = Bbox.union([_ink_box(f) for v, f in figs.items() if v != "oblique"])
        for vname, fig in figs.items():
            name = f"tetrahedron_{FILE_TAG[comp]}_{vname}.pdf"
            box = _ink_box(fig) if vname == "oblique" else canonical
            fig.savefig(pdir / name, bbox_inches=box, dpi=300)
            plt.close(fig)
            written.append(f"panels/{name}")

    handles = [Line2D([], [], marker="D", linestyle="none", markersize=5,
                      markerfacecolor=NODE_COLORS[k], markeredgecolor="#222222",
                      label=f"{k}: {NODE_LABELS[k]}") for k in NODES]
    # Printed at 0.8 of the text width (~5 in).  The canvas is wider than the four entries
    # need and the save crops to them, so nothing is cut off.
    fig = plt.figure(figsize=(7.0, 0.4))
    fig.legend(handles=handles, loc="center", ncol=4, frameon=False, fontsize=7)
    fig.savefig(pdir / "tetrahedron_legend.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    written.append("panels/tetrahedron_legend.pdf")
    return written
