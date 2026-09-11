"""Eigenvalue spectra and the activation-strength tables.

Two things are read straight off the fitted probes rather than the results database:

* **Spectra** — a tuple-contrastive probe stores its full eigenvalue spectrum as an
  artifact.  The interesting end is the most negative: those are the directions along
  which the two framings of a pair sit on opposite sides.  Plotted by magnitude, largest
  first, so a single dominant bar means one clearly contrast-consistent direction and a
  flat profile means the contrast is spread over many.

* **Activation strengths** — projecting held-out samples onto the leading direction and
  ranking them shows *what* the direction responds to.  Reading the extremes is how you
  find out whether a "truth" direction is actually tracking something else.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.activations.cache import load_sample_cache
from src.data.loader import load_jsonl
from src.probing import Manifest, load_probes, probe_key
from src.probing.base import _center_paired, _extract_pairs

from _common import DATASETS, POSITION_LABELS


# ── eigenvalue spectra ────────────────────────────────────────────────────────

def collect_spectra(probes_dir: Path, catalogue: dict, method_label: str = "tcPCA-8",
                    top_k: int = 10) -> pd.DataFrame:
    """Top-`top_k` contrast eigenvalues per (dataset, position), by |λ| at the negative end."""
    manifest = Manifest.load(probes_dir)
    probes = load_probes(manifest.params_file)
    wanted = {m for m, i in catalogue.items() if i["label"] == method_label}

    rows = []
    for entry in manifest.entries:
        if entry.method not in wanted:
            continue
        probe = probes.get(probe_key(entry.dataset, entry.layer, entry.train_position,
                                     entry.method, entry.descriptor, entry.seed))
        eigenvalues = (probe.artifacts.get("eigenvalues") if probe else None)
        if eigenvalues is None:
            continue
        # Ascending order, so the most contrast-consistent directions are at the front.
        magnitudes = np.abs(np.sort(np.asarray(eigenvalues)))[:top_k]
        for i, value in enumerate(magnitudes):
            rows.append({
                "dataset": entry.dataset,
                "position": POSITION_LABELS.get(str(entry.train_position),
                                                str(entry.train_position)),
                "layer": entry.layer,
                "index": i,
                "eigenvalue": float(value),
            })
    return pd.DataFrame(rows)


def spectrum_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Turn "the spectrum looks diffuse" into two numbers per (dataset, position).

    `gap` is lambda_0 / lambda_1: how far the leading contrast direction stands out from
    the next one.  `participation` is (sum L)^2 / sum L^2 over the retained eigenvalues —
    the standard participation ratio, read as "how many directions are effectively
    carrying the contrast".  It is 1 when one direction carries everything and approaches
    the number of eigenvalues when they are equal.

    A single dominant direction means the contrast isolates one feature; a flat spectrum
    means it does not, and is where an unsupervised probe has several equally good
    solutions to choose between.
    """
    rows = []
    for (ds, pos), g in df.groupby(["dataset", "position"]):
        values = g.sort_values("index")["eigenvalue"].to_numpy(dtype=float)
        if len(values) < 2 or values[0] <= 0:
            continue
        rows.append({
            "dataset": ds,
            "position": pos,
            "lambda_0": values[0],
            "lambda_1": values[1],
            "gap": values[0] / values[1] if values[1] > 0 else float("inf"),
            "participation": float(values.sum() ** 2 / (values ** 2).sum()),
        })
    return pd.DataFrame(rows).sort_values(["position", "gap"], ascending=[True, False])


def plot_spectra(pages, out_path: Path, position: str = "answer",
                 datasets: list[str] | None = None, n_cols: int = 3) -> None:
    """One page per (title, spectra) pair.

    Bases are kept on separate pages rather than overlaid because their eigenvalues are
    not in the same units: the tuple-contrastive ones are in activation units, while the
    generalised ones are explained-variance ratios (`S w = lambda Sigma w`).  Sharing an
    axis would invite a comparison that is not meaningful.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    if isinstance(pages, pd.DataFrame):          # single-page callers
        pages = [("", pages)]
    rendered = [(t, d[d["position"] == position]) for t, d in pages]
    rendered = [(t, d) for t, d in rendered if not d.empty]
    if not rendered:
        return

    with PdfPages(out_path) as pdf:
        for title, sub in rendered:
            page_datasets = [d for d in (datasets or sorted(sub["dataset"].unique()))
                             if d in set(sub["dataset"])]
            if not page_datasets:
                continue
            n_rows = (len(page_datasets) + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 2.4 * n_rows),
                                     squeeze=False)
            for ax in axes.flat:
                ax.set_visible(False)
            for i, ds in enumerate(page_datasets):
                ax = axes[i // n_cols][i % n_cols]
                ax.set_visible(True)
                sel = sub[sub["dataset"] == ds].sort_values("index")
                # Stems, largest first — the shape is the message, not the absolute scale.
                ax.stem(sel["index"], sel["eigenvalue"], basefmt=" ")
                ax.set_xticks(sel["index"])
                ax.set_xticklabels([f"$\\lambda_{{{j:02d}}}$" for j in sel["index"]],
                                   fontsize=6, rotation=90)
                ax.set_title(ds, fontsize=9)
                ax.grid(alpha=.2, axis="y", linewidth=.5)
                ax.set_ylim(bottom=0)
            if title:
                fig.suptitle(title, fontsize=10)
            fig.tight_layout(rect=(0, 0, 1, 0.96 if title else 1))
            pdf.savefig(fig)
            plt.close(fig)


# ── activation strengths ──────────────────────────────────────────────────────

def activation_strengths(model_name: str, probes_dir: Path, act_dir: Path,
                         samples_path: Path, catalogue: dict, dataset: str,
                         method_label: str = "tcPCA-1", position: str = "answer",
                         centering: str = "per_branch") -> pd.DataFrame:
    """Rank the held-out samples of `dataset` by projection onto the leading direction.

    Returns one row per sample with its activation, its rank as a fraction, and the text
    that produced it — the extremes of that ranking are what a qualitative read of the
    direction is based on.
    """
    manifest = Manifest.load(probes_dir)
    probes = load_probes(manifest.params_file)
    samples_by_id = {s.id: s for s in load_jsonl(str(samples_path))}
    wanted = {m for m, i in catalogue.items() if i["label"] == method_label}
    want_pos = {p for p, lab in POSITION_LABELS.items() if lab == position} | {position}

    for entry in manifest.entries:
        if (entry.method not in wanted or entry.dataset != dataset
                or str(entry.train_position) not in want_pos):
            continue
        probe = probes.get(probe_key(entry.dataset, entry.layer, entry.train_position,
                                     entry.method, entry.descriptor, entry.seed))
        if probe is None:
            continue
        pairs = _extract_pairs([samples_by_id[i] for i in entry.test_ids
                                if i in samples_by_id])
        if not pairs:
            continue

        akey = (entry.layer, entry.train_position)
        caches = {s.id: load_sample_cache(act_dir, model_name, s)
                  for pair in pairs for s in pair}
        Xb = np.stack([caches[b.id][akey].float().numpy() for b, _ in pairs])
        Xc = np.stack([caches[c.id][akey].float().numpy() for _, c in pairs])
        Xb, Xc = _center_paired(pairs, Xb, Xc, centering)

        direction = probe.subspace.numpy()[0]
        direction = direction / (np.linalg.norm(direction) + 1e-12)

        records = []
        for side, X, which in ((0, Xb, "base"), (1, Xc, "cf")):
            for j, (pair, activation) in enumerate(zip(pairs, X @ direction)):
                s = pair[side]
                records.append({
                    "activation": float(activation),
                    "branch": s.descriptors.get("branch", side),
                    "role": which,
                    "label": s.descriptors.get("label"),
                    "truth": s.descriptors.get("truth"),
                    "question": s.metadata.get("question", ""),
                    "answer": s.metadata.get("answer", ""),
                    "text": s.text,
                })
        df = pd.DataFrame(records).sort_values("activation").reset_index(drop=True)
        df["rel_rank"] = df.index / max(len(df) - 1, 1)
        return df
    return pd.DataFrame()


def extremes(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """The `n` most and least activating samples, tagged so they read as one table."""
    if df.empty:
        return df
    top = df.tail(n).iloc[::-1].assign(end="high")
    bottom = df.head(n).assign(end="low")
    return pd.concat([top, bottom], ignore_index=True)


# ── per-seed accuracy ─────────────────────────────────────────────────────────

def plot_seed_accuracy(evals: pd.DataFrame, catalogue: dict, out_path: Path,
                       label: str = "CCS", datasets: list[str] | None = None,
                       title: str | None = None) -> bool:
    """Every seed's accuracy as its own point, per dataset and token position.

    Table 1 reports mean +/- sd, which is only informative when the seeds cluster.  Where
    CCS lands in more than one basin the mean sits between them and describes no run that
    actually happened -- `imdb` on the period token is the clear case.  Showing the seeds
    themselves is the honest version of the same measurement, and makes the multi-basin
    datasets visible rather than inferable from a large sd.

    Returns False if no matching rows exist, so the caller can skip the file.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wanted = {m for m, i in catalogue.items() if i.get("label") == label}
    sub = evals[evals["method"].isin(wanted) & evals["seed"].notna()]
    if sub.empty:
        return False

    order = [d for d in (datasets or DATASETS) if d in set(sub["dataset"])]
    order += sorted(set(sub["dataset"]) - set(order))
    positions = [p for p in ("answer", "period") if p in set(sub["position"])]

    # One column per dataset, the two token positions offset within it.
    colours = {"answer": "#1f77b4", "period": "#d62728"}
    offsets = {p: (i - (len(positions) - 1) / 2) * 0.26 for i, p in enumerate(positions)}
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(3.0, 2.1))      # printed at ~0.49 of the text width
    for col, ds in enumerate(order):
        for pos in positions:
            cell = sub[(sub["dataset"] == ds) & (sub["position"] == pos)]
            if cell.empty:
                continue
            acc = cell["acc"].to_numpy(dtype=float)
            x = col + offsets[pos] + rng.uniform(-0.055, 0.055, size=len(acc))
            ax.scatter(x, acc, s=4, alpha=0.55, linewidths=0,
                       color=colours[pos], zorder=3)
            ax.scatter([col + offsets[pos]], [acc.mean()], marker="_", s=70,
                       linewidths=1.2, color=colours[pos], zorder=4)

    ax.axhline(0.5, color="0.6", lw=0.8, ls="--", zorder=1)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=35, ha="right", fontsize=6.5)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("accuracy per seed")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    handles = [plt.Line2D([], [], marker="o", ls="", color=colours[p], label=p)
               for p in positions]
    ax.legend(handles=handles, loc="lower left", frameon=False, ncol=2, handletextpad=0.1)
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return True


# ── spectrum shape against CCS reliability ────────────────────────────────────

def gap_vs_instability(evals: pd.DataFrame, catalogue: dict,
                       summary: pd.DataFrame, basis: str = "tcPCA-8") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per (dataset, position): the leading eigen-gap and how much CCS varies over seeds.

    The paper reads the spectra by eye and notes that diffuse spectra coincide with
    unreliable CCS.  This makes that a number: Spearman's rho between the gap
    lambda_0/lambda_1 (and the participation ratio) and the CCS seed spread, over every
    dataset x position cell.  Returns (cells, correlations).
    """
    from scipy.stats import spearmanr

    ccs = {m for m, i in catalogue.items() if i.get("label") == "CCS"}
    per_seed = evals[evals["method"].isin(ccs)].groupby(["dataset", "position"])["acc"]
    spread = pd.DataFrame({"ccs_sd": per_seed.std(ddof=0),
                           "ccs_range": per_seed.max() - per_seed.min(),
                           "ccs_max": per_seed.max()}).reset_index()
    spec = summary[summary["basis"] == basis][["dataset", "position", "gap", "participation"]]
    cells = spread.merge(spec, on=["dataset", "position"])

    rows = []
    for x in ("gap", "participation"):
        for y in ("ccs_sd", "ccs_range", "ccs_max"):
            rho, p = spearmanr(cells[x], cells[y])
            rows.append({"spectrum": x, "ccs": y, "rho": rho, "p": p, "n": len(cells)})
    return cells, pd.DataFrame(rows)


def plot_gap_vs_instability(cells: pd.DataFrame, corr: pd.DataFrame, out_path: Path,
                            title: str | None = None) -> None:
    """Seed spread of CCS against the leading eigen-gap, one point per dataset x position.

    Only the unstable cells (sd >= `LABEL_MIN_SD`) are named: the stable ones sit in a
    pile at sd ~ 0 where no labelling stays legible, and the caption says so.  Labels are
    placed greedily, most unstable first, at the first offset that collides with nothing.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    markers = {"answer": "o", "period": "s"}
    colours = {"answer": "#1f77b4", "period": "#d62728"}
    fig, ax = plt.subplots(figsize=(3.0, 2.2))      # printed at column width
    for pos, grp in cells.groupby("position"):
        ax.scatter(grp["gap"], grp["ccs_sd"], marker=markers.get(pos, "o"), s=14,
                   color=colours.get(pos, "0.3"), label=pos, zorder=3)
    # Log axis, but labelled with plain numbers: matplotlib's "2x10^0" minor labels collide.
    ax.set_xscale("log")
    lo, hi = cells["gap"].min(), cells["gap"].max()
    ax.xaxis.set_major_locator(FixedLocator(
        [t for t in (1, 1.5, 2, 3, 5, 7, 10, 15, 20, 30) if lo / 1.2 <= t <= hi * 1.2]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel(r"leading eigen-gap $\lambda_0/\lambda_1$ (tcPCA)")
    ax.set_ylabel("CCS accuracy, sd over seeds")
    r = corr[(corr["spectrum"] == "gap") & (corr["ccs"] == "ccs_sd")].iloc[0]
    p = "< .001" if r["p"] < 0.001 else f"= {r['p']:.3f}".replace("0.", ".")
    # Above the axes: the top of the plot is where the unstable (labelled) datasets sit.
    rho = ax.text(1.0, 1.02, rf"Spearman $\rho$ = {r['rho']:+.2f}  ($p$ {p})",
                  transform=ax.transAxes, ha="right", va="bottom")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    # Right-hand middle is empty on both models: stable cells hug sd ~ 0 at large gaps.
    leg = ax.legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.4),
                    handletextpad=0.2, borderaxespad=0.2)
    if title:
        ax.set_title(title, fontsize=8)
    fig.tight_layout()

    # Greedy label placement in display space, after the layout is final.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pts = ax.transData.transform(cells[["gap", "ccs_sd"]].to_numpy())
    pad = 3.0                                          # marker half-size, px at 100 dpi
    obstacles = [rho.get_window_extent(renderer), leg.get_window_extent(renderer)]
    box = ax.get_window_extent(renderer)
    offsets = [(3, 0, "left", "center"), (-3, 0, "right", "center"), (0, 3, "center", "bottom"),
               (0, -3, "center", "top"), (3, 3, "left", "bottom"), (-3, 3, "right", "bottom"),
               (3, -3, "left", "top"), (-3, -3, "right", "top")]

    def hits(bb, own: int) -> int:
        n = sum(bb.overlaps(o) for o in obstacles)
        n += sum(1 for j, (x, y) in enumerate(pts) if j != own
                 and bb.x0 - pad < x < bb.x1 + pad and bb.y0 - pad < y < bb.y1 + pad)
        return n + (not (box.x0 <= bb.x0 and bb.x1 <= box.x1 + 20 and box.y0 <= bb.y0 and bb.y1 <= box.y1))

    for i in cells.index[cells["ccs_sd"] >= LABEL_MIN_SD].sort_values(
            key=lambda idx: -cells.loc[idx, "ccs_sd"]):
        own = cells.index.get_loc(i)
        best = None
        for dx, dy, ha, va in offsets:
            t = ax.annotate(cells.loc[i, "dataset"], (cells.loc[i, "gap"], cells.loc[i, "ccs_sd"]),
                            xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                            fontsize=6, color="0.35")
            bb = t.get_window_extent(renderer)
            score = hits(bb, own)
            if best is None or score < best[0]:
                if best:
                    best[1].remove()
                best = (score, t, bb)
            else:
                t.remove()
            if score == 0:
                break
        obstacles.append(best[2])
    fig.savefig(out_path)
    plt.close(fig)


LABEL_MIN_SD = 0.01          # CCS seed sd above which a dataset is named in the gap plot


def write_loose_spectra(df: pd.DataFrame, out_dir: Path, tag: str, position: str,
                        datasets: list[str]) -> list[str]:
    """One small stem plot per dataset, no title (the paper's subcaption names it)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(exist_ok=True)
    written = []
    for ds in datasets:
        sel = df[(df["dataset"] == ds) & (df["position"] == position)].sort_values("index")
        if sel.empty:
            continue
        # Printed at 0.32 of the text width (~2.0 in): thin stems, small heads.
        fig, ax = plt.subplots(figsize=(2.0, 1.3))
        markers, stems, _ = ax.stem(sel["index"], sel["eigenvalue"], basefmt=" ")
        plt.setp(markers, markersize=2.2)
        plt.setp(stems, linewidth=0.7)
        ax.set_xticks(sel["index"])
        ax.set_xticklabels([f"$\\lambda_{{{j}}}$" for j in sel["index"]], fontsize=6)
        ax.grid(alpha=.2, axis="y", linewidth=.5)
        ax.set_ylim(bottom=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        fig.tight_layout()
        name = f"eigenvalues_{tag}_{position}_{ds}.pdf"
        fig.savefig(out_dir / name)
        plt.close(fig)
        written.append(name)
    return written
