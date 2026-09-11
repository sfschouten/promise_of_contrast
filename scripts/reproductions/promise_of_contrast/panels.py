"""Column-width, single-layer projection panels for the paper.

The report figures (`projections.py`) stack every layer of every eigenvector pair on one
tall page -- right for exploring, too big for a column.  These draw one layer, the first two
components, held-out samples only, with the contrast pairs as faint lines.  Each file has
two files: tcPCA and generalized tcPCA (g-tcPCA), so either can go in the paper.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.activations.cache import load_sample_cache
from src.data.graph import load_family_as_pairs
from src.probing import Manifest, load_probes, probe_key

from _common import activations_dir, model_name, probes_dir

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]
TRUTH = {"t": "true", "f": "false", "n": "neither"}

METHODS = {"cross_covariance_8": "tcPCA", "generalized_cross_covariance_8": "g-tcPCA"}
FILE_TAG = {"tcPCA": "tcpca", "g-tcPCA": "wtcpca"}

PANELS = {
    "days_of_week": dict(run="days_of_week", act_run="days_of_week",
                                   family="days_of_week", dataset="days_of_week_circular",
                                   colour=lambda s: s.descriptors.get("target"), order=DAYS,
                                   cmap="hsv", size=(1.5, 1.35)),
    "months_of_year": dict(run="months_of_year", act_run="months_of_year",
                                     family="months_of_year", dataset="months_of_year_circular",
                                     colour=lambda s: s.descriptors.get("target"), order=MONTHS,
                                     cmap="hsv", size=(1.5, 1.35)),
    "trinary": dict(run="tot_city_tv_aff", act_run="tot", family="tot",
                              dataset=None, compound="tot_city_tv_aff",
                              colour=lambda s: TRUTH.get(s.id.rsplit("_", 2)[-2]),
                              order=list(TRUTH.values()), cmap=None, size=(2.7, 2.0)),
}
TRUTH_COLOURS = {"true": "#d62728", "false": "#1f77b4", "neither": "#2ca02c"}


# The layer sweep projects the same runs many times; load each store and file once.
_PROBES: dict = {}
_SAMPLES: dict = {}
_ACTS: dict = {}


def _store(pdir: Path):
    if pdir not in _PROBES:
        manifest = Manifest.load(pdir)
        _PROBES[pdir] = (manifest, load_probes(manifest.params_file))
    return _PROBES[pdir]


def _project(model: str, spec: dict, method: str, layer: int, position: str,
             split: str = "eval"):
    pdir = probes_dir(model, spec["run"])
    if not (pdir / "manifest.json").exists():
        raise LookupError(f"no probes at {pdir}")
    manifest, probes = _store(pdir)
    entry_ds = spec["dataset"] or f"{model}__{spec['compound']}"
    entry = next((e for e in manifest.entries if e.dataset == entry_ds and e.method == method
                  and e.layer == layer and str(e.train_position) == position), None)
    if entry is None:
        raise LookupError(f"no {method} probe for {entry_ds} at layer {layer}/{position}")
    probe = probes[probe_key(entry.dataset, entry.layer, entry.train_position,
                             entry.method, entry.descriptor, entry.seed)]
    W = probe.subspace.float().numpy()[:2]

    if spec["family"] not in _SAMPLES:
        _SAMPLES[spec["family"]] = load_family_as_pairs(spec["family"]) or []
    samples = _SAMPLES[spec["family"]]
    if spec["dataset"]:
        samples = [s for s in samples if s.descriptors.get("dataset_name") == spec["dataset"]]
    else:
        params = yaml.safe_load((ROOT / "params.yaml").read_text())
        keep = set(params["compounds"][spec["compound"]].get("datasets") or [])
        if keep:
            samples = [s for s in samples if s.descriptors.get("dataset_name") in keep]
    test = set(entry.test_ids)
    if test and split != "all":
        samples = [s for s in samples if (s.id in test) == (split == "eval")]

    cache_dir, mname = activations_dir(model, spec["act_run"]), model_name(model)
    coords, labels, index, pairs = [], [], {}, {}
    for s in samples:
        if s.id not in index:
            key = (str(cache_dir), s.id)
            if key not in _ACTS:
                _ACTS[key] = load_sample_cache(cache_dir, mname, s)
            vec = _ACTS[key].get((layer, position))
            if vec is None:
                continue
            index[s.id] = len(coords)
            coords.append(vec.float().numpy() @ W.T)
            labels.append(spec["colour"](s))
        pid = s.descriptors.get("pair_id")
        if pid is not None:
            pairs.setdefault(pid, []).append(index[s.id])
    segments = [v for v in pairs.values() if len(v) == 2]
    return np.asarray(coords), np.asarray(labels, dtype=object), segments


def _colours(order, spec):
    import matplotlib.pyplot as plt
    if spec["cmap"]:
        cmap = plt.get_cmap(spec["cmap"])
        return {o: cmap(i / len(order)) for i, o in enumerate(order)}
    return TRUTH_COLOURS


def _draw(ax, coords, labels, segments, spec, comp):
    from matplotlib.collections import LineCollection
    import matplotlib.patheffects as pe

    small = spec["size"][0] < 2.0
    if segments:
        seg = np.stack([coords[[a, b]] for a, b in segments[:3000]])
        ax.add_collection(LineCollection(seg, colors="0.6", linewidths=0.15, alpha=0.06, zorder=1))
    order = [o for o in spec["order"] if o in set(labels)]
    colours = _colours(order, spec)
    means = {}
    for o in order:
        pts = coords[labels == o]
        ax.scatter(pts[:, 0], pts[:, 1], s=1.2 if small else 2.5, alpha=0.5, linewidths=0,
                   color=colours[o], zorder=2)
        means[o] = pts.mean(0)
    # class means joined in order and closed: the calendar loop, or the trinary triangle
    cyc = np.stack([means[o] for o in order] + [means[order[0]]])
    ax.plot(cyc[:, 0], cyc[:, 1], color="#222", lw=0.6, zorder=3)
    for o in order:
        ax.scatter(*means[o], s=12 if small else 22, marker="D", color=colours[o],
                   edgecolors="#222", linewidths=0.4, zorder=4)
        # the means are labelled directly: a legend does not fit a 1.5 in panel
        t = ax.annotate(o[:3] if spec["cmap"] else o, means[o], xytext=(2, 2),
                        textcoords="offset points", fontsize=5.5 if small else 6.5, zorder=5)
        t.set_path_effects([pe.withStroke(linewidth=1.5, foreground="white")])
    ax.set_xlabel(f"{comp} 1", fontsize=6.5 if small else 8, labelpad=1)
    ax.set_ylabel(f"{comp} 2", fontsize=6.5 if small else 8, labelpad=1)
    ax.tick_params(labelsize=5.5 if small else 7, pad=1, length=2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def _nearest_centroid_accuracy(train, test) -> float:
    (Xtr, ytr), (Xte, yte) = train, test
    classes = sorted(set(ytr) & set(yte))
    if not classes:
        return float("nan")
    C = np.stack([Xtr[ytr == c].mean(0) for c in classes])
    pred = np.array(classes, dtype=object)[np.argmin(((Xte[:, None, :] - C[None]) ** 2).sum(-1), axis=1)]
    return float((pred == yte).mean())


def _cyclic_order(X, y, order) -> float:
    """1.0 when the class means go round their centre in the natural order (either way)."""
    present = [o for o in order if o in set(y)]
    if len(present) < 3:
        return float("nan")
    M = np.stack([X[y == o].mean(0) for o in present])
    d = M - M.mean(0)
    seq = [present[i] for i in np.argsort(np.arctan2(d[:, 1], d[:, 0]))]
    k = seq.index(present[0])
    seq = seq[k:] + seq[:k]
    return float(seq == present or seq == [present[0]] + present[:0:-1])


def sweep(model: str, position: str = "last", layers: list[int] | None = None) -> pd.DataFrame:
    """Per figure, basis and layer: how cleanly the classes separate in the plotted plane.

    `accuracy` is nearest-centroid accuracy on held-out samples, with centroids fitted on
    the training samples, in the first two components -- the plane the figure shows.
    `cyclic` (days, months) is 1 when the class means go round in calendar order.
    """
    params = yaml.safe_load((ROOT / "params.yaml").read_text())
    layers = layers or list(params["activations"]["layers"])
    rows = []
    for name, spec in PANELS.items():
        for method, comp in METHODS.items():
            for layer in layers:
                try:
                    tr = _project(model, spec, method, layer, position, split="train")
                    te = _project(model, spec, method, layer, position, split="eval")
                except (LookupError, KeyError):
                    continue
                rows.append({"figure": name, "basis": comp, "layer": layer,
                             "accuracy": _nearest_centroid_accuracy(tr[:2], te[:2]),
                             "cyclic": _cyclic_order(te[0], te[1], spec["order"]) if spec["cmap"] else np.nan})
    return pd.DataFrame(rows)


def choose(df: pd.DataFrame, default: int = 15) -> dict[tuple[str, str], int]:
    """Best layer per (figure, basis).

    For the calendar features the figure's claim is that the classes go round in order, so
    only layers where they do are eligible (when any is); among those, the highest held-out
    accuracy wins, and ties go to the layer nearest the paper's block 15.
    """
    out = {}
    for (fig, basis), g in df.groupby(["figure", "basis"]):
        if g["cyclic"].notna().any() and (g["cyclic"] == 1).any():
            g = g[g["cyclic"] == 1]
        g = g.assign(dist=(g["layer"] - default).abs())
        best = g.sort_values(["accuracy", "dist"], ascending=[False, True]).iloc[0]
        out[(fig, basis)] = int(best["layer"])
    return out


def write_loose_sweep(df: pd.DataFrame, chosen: dict, out_dir: Path) -> list[str]:
    """One small panel per figure: held-out accuracy against layer, both bases."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(exist_ok=True)
    written = []
    colour = {"tcPCA": "#1f77b4", "g-tcPCA": "#9467bd"}
    for name, g in df.groupby("figure"):
        fig, ax = plt.subplots(figsize=(2.0, 1.3))        # printed at 0.32 of the text width
        for basis, gb in g.groupby("basis"):
            gb = gb.sort_values("layer")
            ax.plot(gb["layer"], gb["accuracy"], "-o", ms=2, lw=1, color=colour.get(basis), label=basis)
            best = chosen.get((name, basis))
            if best is not None:
                ax.scatter([best], gb.loc[gb["layer"] == best, "accuracy"], s=30, facecolors="none",
                           edgecolors=colour.get(basis), linewidths=1, zorder=4)
        ax.axvline(15, color="0.75", lw=0.7, ls="--")
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("layer (decoder block)", fontsize=6.5, labelpad=1)
        ax.set_ylabel("held-out accuracy", fontsize=6.5, labelpad=1)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6, frameon=False, loc="lower right")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        fig.tight_layout()
        fname = f"nonbinary_sweep_{name}.pdf"
        fig.savefig(out_dir / fname)
        plt.close(fig)
        written.append(fname)
    return written


def build(model: str, out_dir: Path, layer: int = 15, position: str = "last",
          layers: dict[tuple[str, str], int] | None = None) -> list[str]:
    """Write `panels/<figure>_<basis>.pdf` for every figure and basis available."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pdir = out_dir / "panels"
    pdir.mkdir(exist_ok=True)
    written = []
    for name, spec in PANELS.items():
        for method, comp in METHODS.items():
            try:
                at = (layers or {}).get((name, comp), layer)
                coords, labels, segments = _project(model, spec, method, at, position)
            except (LookupError, KeyError) as exc:
                print(f"  {name} [{comp}]: skipped ({exc})")
                continue
            fig, ax = plt.subplots(figsize=spec["size"])
            _draw(ax, coords, labels, segments, spec, comp)
            fig.tight_layout()
            fname = f"{name}_{FILE_TAG[comp]}.pdf"
            fig.savefig(pdir / fname)
            plt.close(fig)
            written.append(f"panels/{fname}")
    return written
