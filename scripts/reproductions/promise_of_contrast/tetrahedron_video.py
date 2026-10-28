"""Animated views of the negation x truth tetrahedron (the project site's video).

Three camera paths:

- `tour` (default, the one on the site): the three square views -- each looking down one
  contrast axis, so that contrast collapses -- then a free tumble back to the first.
- `turntable`: the camera circles one contrast axis, which stays vertical on screen.
  Because the three contrast axes are (near-)orthogonal, the circle passes through the two
  views that collapse the other contrasts; the video pauses briefly at each.
- `wander`: a closed loop that tumbles continuously.

Every path loops seamlessly.  The MP4 is H.264 / yuv420p, which PowerPoint also plays.
Needs `ffmpeg` on the PATH.

    python scripts/reproductions/promise_of_contrast/tetrahedron_video.py

Rendering is a plain 2-D orthographic projection, done here rather than with mplot3d, so
the rotation axis is exact and one unit is one unit on screen in every frame.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]
DEFAULT_OUT = (ROOT / "outputs" / "reproductions" / "promise_of_contrast" / "llama2_7b"
               / "tetrahedron_tour.mp4")

from tetrahedron import contrast_axes, opposite_edge_pairings  # noqa: E402
from tetrahedron_figure import (NODE_COLORS, NODE_LABELS, NODES,  # noqa: E402
                                _contrast_name, projected_points)

MODELS = {"llama2_7b": "Llama-2-7B", "llama3_8b": "Llama-3.1-8B"}
BASES = {"generalized_cross_covariance_8": "g-tcPC", "cross_covariance_8": "tcPC"}
METHOD_NAME = {"generalized_cross_covariance_8": "g-tcPCA", "cross_covariance_8": "tcPCA"}
PLOT_W, TX = 0.66, 0.68             # plot width and side-panel left edge, figure fraction
CONTRAST_LABEL = {"negation": "negation", "base": "content", "truth": "truth"}


def named_axes(means):
    names = {p.label: _contrast_name(p) for p in opposite_edge_pairings(NODES)}
    return {names[k]: v for k, v in contrast_axes(means).items()}


def frame_angles(fps: int, seconds: float, hold: float, stops: list[float]) -> np.ndarray:
    """Camera angle per frame: constant speed, with a pause at each angle in `stops`."""
    n = int(round(fps * seconds))
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    hold_n = int(round(fps * hold))
    out = []
    for t in theta:
        out.append(t)
        for s in stops:
            if abs(np.angle(np.exp(1j * (t - s)))) < np.pi / n:   # nearest frame to the stop
                out.extend([s] * hold_n)
    return np.asarray(out)


def unit_of(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def wander_path(u: np.ndarray, e1: np.ndarray, e2: np.ndarray, n: int, elev: float,
                turns: int = 2, waves: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """A closed camera loop: `turns` circuits about `u` while the elevation swings
    +-`elev` `waves` times.  Returns per-frame view directions and screen-up vectors.

    Frames are spaced evenly in arc length, so the camera moves at constant speed.  Up is
    parallel-transported along the path (no fixed vertical, so the motion reads as a
    tumble); the roll it accumulates over the loop is spread evenly over all frames so the
    last frame meets the first.
    """
    def d_of(s):
        az = s
        el = elev * np.sin(waves / turns * s)
        return (np.cos(el) * np.cos(az))[:, None] * e1 + (np.cos(el) * np.sin(az))[:, None] * e2 \
            + np.sin(el)[:, None] * u

    fine = np.linspace(0, 2 * np.pi * turns, 20000, endpoint=False)
    dense = d_of(fine)
    seg = np.linalg.norm(np.diff(np.vstack([dense, dense[:1]]), axis=0), axis=1)
    arc = np.concatenate([[0], np.cumsum(seg)])[:-1]
    s = np.interp(np.linspace(0, arc[-1] + seg[-1], n, endpoint=False), arc, fine)
    d = d_of(s)

    ups = np.empty_like(d)
    up = u - (u @ d[0]) * d[0]
    up /= np.linalg.norm(up)
    for i in range(n):
        up = up - (up @ d[i]) * d[i]               # transport: drop the component along d
        up /= np.linalg.norm(up)
        ups[i] = up
    # Holonomy: carried once round the loop, up comes back rolled by some angle.
    end = ups[-1] - (ups[-1] @ d[0]) * d[0]
    end /= np.linalg.norm(end)
    roll = np.arctan2(np.cross(ups[0], end) @ d[0], ups[0] @ end)
    for i in range(n):                             # undo it a little at every frame
        a = -roll * (i + 1) / n
        side = np.cross(d[i], ups[i])
        ups[i] = np.cos(a) * ups[i] + np.sin(a) * side
    return d, ups


def _cam(d: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Camera matrix, rows (right, up, toward viewer): a proper rotation, world -> screen."""
    d = d / np.linalg.norm(d)
    up = up - (up @ d) * d
    up /= np.linalg.norm(up)
    return np.vstack([np.cross(d, up), up, -d])


def _ease(x: np.ndarray) -> np.ndarray:
    """Smootherstep: zero velocity and acceleration at both ends."""
    return x * x * x * (x * (6 * x - 15) + 10)


def tour_path(n: np.ndarray, b: np.ndarray, t: np.ndarray, fps: int, hold: float,
              move: float, tumble: float) -> tuple[np.ndarray, np.ndarray]:
    """The three square views in turn, then a free tumble back to the first.

    `n, b, t` are the negation, content and truth axes, orthonormalised.  Each square is
    held for `hold` seconds, with a `move`-second eased swing between them; the tumble is a
    C2 rotation spline through oblique views that comes to rest on the first square, so the
    video loops.
    """
    from scipy.spatial.transform import Rotation, RotationSpline, Slerp

    def rot(dv, uv):
        B = np.vstack([n, b, t])                     # contrast coordinates -> data space
        return Rotation.from_matrix(_cam(np.asarray(dv, float) @ B, np.asarray(uv, float) @ B))

    squares = [rot((0, 0, 1), (1, 0, 0)),            # down truth:    negation up, content across
               rot((0, 1, 0), (1, 0, 0)),            # down content:  negation up, truth across
               rot((1, 0, 0), (0, -1, 0))]           # down negation: content up, truth across
    # Oblique stops for the tumble, chosen so successive ones are 60-100 degrees apart and
    # the axis of turning changes at each.  Corner-ish directions (1,1,1) show the
    # tetrahedron's three-fold face-on views.
    obliques = [rot((1, 0.9, -0.5), (-0.4, 1, 0.9)),
                rot((0.6, -0.8, -1), (1, 0.6, -0.3)),
                rot((-0.9, -1, 0.3), (0.3, -0.6, 1)),
                rot((-1, 0.5, 0.9), (0.8, 0.9, 0.6))]

    frames = []
    hold_n, move_n = int(round(fps * hold)), int(round(fps * move))
    for i, sq in enumerate(squares):
        frames += [sq] * hold_n
        if i + 1 < len(squares):
            sl = Slerp([0, 1], Rotation.concatenate([sq, squares[i + 1]]))
            frames += list(sl(_ease((np.arange(move_n) + 1) / (move_n + 1))))

    keys = [squares[-1]] + obliques + [squares[0]]
    gaps = [np.degrees((keys[i].inv() * keys[i + 1]).magnitude()) for i in range(len(keys) - 1)]
    times = np.concatenate([[0], np.cumsum(gaps)])
    spline = RotationSpline(times, Rotation.concatenate(keys))
    tum_n = int(round(fps * tumble))
    # Constant speed, with eased ramps at either end so the tumble leaves the last square
    # and settles on the first at rest: speed is a smootherstep up over the first `ramp`
    # of the time and down over the last, and progress is its integral.
    ramp = 0.2
    x = np.linspace(0, 1, 4001)
    speed = _ease(np.clip(x / ramp, 0, 1)) * _ease(np.clip((1 - x) / ramp, 0, 1))
    prog = np.concatenate([[0], np.cumsum((speed[1:] + speed[:-1]) / 2)])
    u = np.interp(np.arange(1, tum_n + 1) / tum_n, x, prog / prog[-1])
    frames += list(spline(u * times[-1]))[:-1]       # the last frame is the video's first

    M = np.stack([f.as_matrix() for f in frames])
    return -M[:, 2], M[:, 1]


def render(model: str, method: str, spin_about: str, out: Path, fps: int, seconds: float,
           hold: float, width: int, height: int, dpi: int, path: str = "turntable",
           elev: float = 50.0, move: float = 1.2, gif: bool = False) -> None:
    coords, nodes = projected_points(model, 15, method)
    means = {k: coords[nodes == k].mean(axis=0) for k in NODES}
    center = np.mean([means[k] for k in NODES], axis=0)
    coords = coords - center
    means = {k: v - center for k, v in means.items()}
    axes = named_axes(means)

    up = axes[spin_about] / np.linalg.norm(axes[spin_about])
    others = [c for c in ("negation", "base", "truth") if c != spin_about]
    # In-plane basis: e1 is the first other contrast, made exactly perpendicular to `up`.
    e1 = axes[others[0]] - (axes[others[0]] @ up) * up
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    # Camera looks along d(theta); depth collapses a contrast when d is parallel to it.
    stops = []
    for c in others:
        a = axes[c] - (axes[c] @ up) * up
        phi = np.arctan2(a @ e2, a @ e1)
        stops += [phi % (2 * np.pi), (phi + np.pi) % (2 * np.pi)]
    # Start a little before the first square, so the opening frames already turn.
    start = (stops[0] - np.radians(25)) % (2 * np.pi)
    if path == "turntable":
        angles = frame_angles(fps, seconds, hold,
                              [(s - start) % (2 * np.pi) for s in stops]) + start
        views = np.cos(angles)[:, None] * e1 + np.sin(angles)[:, None] * e2
        ups = np.tile(up, (len(views), 1))
        # Fixed screen extent: the horizontal reach is bounded by the radius about the
        # spin axis, the vertical by the extent along it.
        radial = np.linalg.norm(coords - np.outer(coords @ up, up), axis=1).max()
        vert = np.abs(coords @ up).max()
    elif path == "tour":
        B = np.linalg.svd(np.vstack([unit_of(axes[c]) for c in ("negation", "base", "truth")]))
        O = B[0] @ B[2]                              # nearest orthonormal basis to the axes
        if np.linalg.det(O) < 0:
            O[2] = -O[2]
        views, ups = tour_path(O[0], O[1], O[2], fps, hold, move, seconds)
        radial = vert = np.linalg.norm(coords, axis=1).max()
    else:
        # Start just off the first square view, as the turntable does.
        c0, s0 = np.cos(start), np.sin(start)
        f1, f2 = c0 * e1 + s0 * e2, -s0 * e1 + c0 * e2
        views, ups = wander_path(up, f1, f2, int(round(fps * seconds)), np.radians(elev))
        # No fixed vertical, so any direction may end up horizontal: bound by the sphere.
        radial = vert = np.linalg.norm(coords, axis=1).max()
    # Captions name a collapse view whenever the camera is near one, on either path.
    unit = {c: axes[c] / np.linalg.norm(axes[c]) for c in ("negation", "base", "truth")}
    pad = 1.08

    comp = BASES[method]
    w_in, h_in = width / dpi, height / dpi
    tmp = Path(tempfile.mkdtemp(prefix="spin_", dir=out.parent))
    fig = plt.figure(figsize=(w_in, h_in), dpi=dpi, facecolor="white")
    for i, (d, up) in enumerate(zip(views, ups)):
        fig.clf()
        ax = fig.add_axes([0.0, 0.0, PLOT_W, 1.0])
        # d is the viewing direction, into the scene
        right = np.cross(d, up)                        # right-handed: right x up = toward viewer
        P = np.vstack([right, up])
        xy = coords @ P.T
        depth = -(coords @ d)                          # larger = nearer the camera
        dn = (depth + radial) / (2 * radial)
        order = np.argsort(depth)
        cols = np.array([NODE_COLORS[n] for n in nodes])
        ax.scatter(xy[order, 0], xy[order, 1], s=(10 + 22 * dn[order]) * (dpi / 100) ** 2 / 2.2,
                   c=cols[order], alpha=0.42, linewidths=0, zorder=2)
        mxy = {k: means[k] @ P.T for k in NODES}
        # Depth cues: nearer edges and vertices are heavier, and painted later.
        mdepth = {k: (-(means[k] @ d) + radial) / (2 * radial) for k in NODES}
        edges = [(a, b) for a_i, a in enumerate(NODES) for b in NODES[a_i + 1:]]
        for a, b in sorted(edges, key=lambda e: mdepth[e[0]] + mdepth[e[1]]):
            near = (mdepth[a] + mdepth[b]) / 2
            ax.plot([mxy[a][0], mxy[b][0]], [mxy[a][1], mxy[b][1]], color="0.15",
                    lw=0.8 + 2.6 * near, alpha=0.45 + 0.5 * near, zorder=3,
                    solid_capstyle="round")
        for k in sorted(NODES, key=lambda k: mdepth[k]):
            ax.scatter(*mxy[k], s=90 + 120 * mdepth[k], marker="D", color=NODE_COLORS[k],
                       edgecolor="#222", linewidth=1.3, zorder=4)
            t = ax.annotate(k, mxy[k], xytext=(9, 7), textcoords="offset points", fontsize=15,
                            color="#111", zorder=5, weight="bold")
            t.set_path_effects([pe.withStroke(linewidth=3.5, foreground="white")])
        r = radial * pad
        v = vert * pad
        aspect = (PLOT_W * w_in) / h_in
        half_w = max(r, v * aspect)
        ax.set_xlim(-half_w, half_w)
        ax.set_ylim(-half_w / aspect, half_w / aspect)
        ax.set_aspect("equal")
        ax.axis("off")

        if path == "turntable":
            # The spin axis, drawn as a vertical arrow at the left edge.
            x0 = -half_w * 0.94
            ax.annotate("", xy=(x0, v * 0.85), xytext=(x0, -v * 0.85),
                        arrowprops=dict(arrowstyle="-|>", color="0.45", lw=1.6), zorder=1)
            ax.text(x0 + half_w * 0.03, 0, f"{CONTRAST_LABEL[spin_about]} axis\n(rotation axis)",
                    rotation=90, va="center", ha="left", fontsize=12, color="0.4")

        # Orientation gizmo: the three basis directions as seen from this camera.
        gx, gy, gl = half_w * 0.80, -half_w / aspect * 0.78, half_w * 0.09
        for j, e in enumerate(np.eye(3)):
            p = e @ P.T
            ax.annotate("", xy=(gx + gl * p[0], gy + gl * p[1]), xytext=(gx, gy),
                        arrowprops=dict(arrowstyle="-|>", color="0.55", lw=1.1,
                                        shrinkA=0, shrinkB=0), zorder=1)
            seen = float(np.clip((np.linalg.norm(p) - 0.35) / 0.4, 0, 1))
            if seen > 0:                             # an axis pointing at us has no room for a label
                ax.text(gx + gl * 1.45 * p[0], gy + gl * 1.45 * p[1], f"{comp} {j}",
                        fontsize=9, color="0.45", ha="center", va="center", alpha=seen)

        # Side panel: title, legend, and the caption at the collapse views.
        fig.text(TX, 0.92, f"{MODELS[model]}, layer 15", fontsize=17, weight="bold", color="#111")
        fig.text(TX, 0.875, f"{METHOD_NAME[method]}, first three components",
                 fontsize=13, color="0.35")
        for j, k in enumerate(NODES):
            yy = 0.76 - j * 0.058
            fig.patches.append(plt.Rectangle((TX + 0.005, yy - 0.012), 0.014, 0.025,
                                             transform=fig.transFigure, color=NODE_COLORS[k]))
            label = NODE_LABELS[k].replace("pos", "affirmed").replace("neg", "negated")
            fig.text(TX + 0.028, yy, f"{k}   {' '.join(label.split())}", fontsize=12, va="center",
                     color="#222")
        off = {c: np.arccos(min(1.0, abs(float(d @ w)))) for c, w in unit.items()}
        c = min(off, key=off.get)
        alpha = float(np.clip(1 - off[c] / np.radians(14), 0, 1))
        if alpha > 0:
            rest = [CONTRAST_LABEL[o] for o in ("negation", "base", "truth") if o != c]
            fig.text(TX, 0.40, f"Looking down the\n{CONTRAST_LABEL[c]} axis", fontsize=16,
                     weight="bold", color="#111", alpha=alpha, va="top", linespacing=1.3)
            fig.text(TX, 0.29, f"The {CONTRAST_LABEL[c]} contrast collapses;\n"
                                 f"{rest[0]} × {rest[1]} form a square.",
                     fontsize=12, color="0.3", alpha=alpha, va="top", linespacing=1.4)
        fig.savefig(tmp / f"f{i:04d}.png", dpi=dpi, facecolor="white")
    plt.close(fig)

    # H.264 High profile, yuv420p, MP4 with the index up front: the format PowerPoint
    # (Windows and Mac) recommends.  No audio track.
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
                    "-i", str(tmp / "f%04d.png"), "-c:v", "libx264", "-profile:v", "high",
                    "-level", "4.1", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "slow",
                    "-movflags", "+faststart", str(out)], check=True)
    if gif:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
                        "-i", str(tmp / "f%04d.png"), "-vf",
                        "fps=20,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=stats_mode=full[p];"
                        "[b][p]paletteuse=dither=sierra2_4a", str(out.with_suffix(".gif"))],
                       check=True)
    for f in tmp.iterdir():
        f.unlink()
    tmp.rmdir()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="llama2_7b", choices=MODELS)
    ap.add_argument("--method", default="generalized_cross_covariance_8", choices=BASES)
    ap.add_argument("--path", default="tour", choices=["turntable", "wander", "tour"],
                    help="turntable: one fixed axis; wander: a closed tumbling loop; "
                         "tour: the three squares, then a tumble")
    ap.add_argument("--move", type=float, default=1.2, help="tour: seconds between squares")
    ap.add_argument("--elev", type=float, default=50.0, help="wander: elevation swing, degrees")
    ap.add_argument("--spin-about", default="negation", choices=["negation", "base", "truth"])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--gif", action="store_true", help="also write a GIF next to the MP4")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seconds", type=float, default=10.0, help="one revolution (turntable, excluding pauses), the whole loop (wander), "
                         "or the tumble (tour)")
    ap.add_argument("--hold", type=float, default=1.0, help="pause at each collapse (square) view")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--dpi", type=int, default=160)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    render(args.model, args.method, args.spin_about, args.out, args.fps, args.seconds,
           args.hold, args.width, args.height, args.dpi, args.path, args.elev,
           args.move, args.gif)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
