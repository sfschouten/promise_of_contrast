"""Quantitative checks that back qualitative claims in the paper.

  copa_components   accuracy of each of the first tcPCs on COPA (the case study asks
                    whether the *second* tcPC carries truth; the pipeline only scores the first)
  crc_variance      how loud (variance) and how pure (intra-pair share) the directions of
                    CRC-TPC, tcPCA and whitened tcPCA are -- the claim that CRC-TPC fails
                    because it finds high-variance directions
  answer_vs_period  what differs between the two token positions

All scoring follows the pwl protocol: held-out pairs, per-branch centring, a pair called
correctly when the true side projects higher, orientation fixed on the training split.
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

POSITIONS = {"answer": "second_to_last", "period": "last"}


class _Store:
    """Probes, samples and activations for one model, loaded once."""

    def __init__(self, model_name: str, probes_dir: Path, act_dir: Path,
                 samples_path: Path, catalogue: dict):
        self.model_name, self.act_dir, self.catalogue = model_name, act_dir, catalogue
        self.manifest = Manifest.load(probes_dir)
        self.probes = load_probes(self.manifest.params_file)
        self.samples = load_jsonl(str(samples_path))
        self._cache: dict[str, dict] = {}

    def entry(self, label: str, dataset: str, position: str):
        wanted = {m for m, i in self.catalogue.items() if i.get("label") == label}
        pos = POSITIONS.get(position, position)
        for e in self.manifest.entries:
            if e.method in wanted and e.dataset == dataset and str(e.train_position) == pos:
                probe = self.probes.get(probe_key(e.dataset, e.layer, e.train_position,
                                                  e.method, e.descriptor, e.seed))
                if probe is not None:
                    return e, probe
        return None, None

    def arrays(self, entry, split: str, centering: str = "per_branch"):
        """(pairs, Xb, Xc) for the entry's train or eval split, centred as in fitting."""
        test = set(entry.test_ids)
        ds = [s for s in self.samples if s.descriptors.get("dataset_name") == entry.dataset]
        pairs = _extract_pairs(ds) or []
        pairs = [p for p in pairs if (p[0].id in test) == (split == "eval")]
        key = (entry.layer, entry.train_position)
        vec = lambda s: self._load(s)[key].float().numpy()
        Xb = np.stack([vec(b) for b, _ in pairs])
        Xc = np.stack([vec(c) for _, c in pairs])
        Xb, Xc = _center_paired(pairs, Xb, Xc, centering)
        return pairs, Xb, Xc

    def clear(self) -> None:
        """Drop cached activations; called between datasets to bound memory."""
        self._cache.clear()

    def _load(self, s):
        if s.id not in self._cache:
            self._cache[s.id] = load_sample_cache(self.act_dir, self.model_name, s)
        return self._cache[s.id]


def _pair_accuracy(pairs, Xb, Xc, W) -> np.ndarray:
    """Per direction: fraction of pairs where the true side projects higher."""
    base_true = np.array([b.descriptors["label"] == 1 for b, _ in pairs])
    return (((Xb @ W.T) > (Xc @ W.T)) == base_true[:, None]).mean(0)


def copa_components(store: _Store, k: int = 4, dataset: str = "copa") -> pd.DataFrame:
    """Accuracy and eigenvalue of each of the first `k` tcPCs, per token position."""
    rows = []
    for position in POSITIONS:
        entry, probe = store.entry("tcPCA-8", dataset, position)
        if entry is None:
            continue
        W = probe.subspace.float().numpy()[:k]
        acc_tr = _pair_accuracy(*store.arrays(entry, "train"), W)
        acc_ev = _pair_accuracy(*store.arrays(entry, "eval"), W)
        acc = np.where(acc_tr >= 0.5, acc_ev, 1 - acc_ev)
        eig = -np.sort(probe.artifacts["eigenvalues"])[:k]
        rows += [{"dataset": dataset, "position": position, "component": i + 1,
                  "accuracy": float(acc[i]), "eigenvalue": float(eig[i])} for i in range(k)]
    return pd.DataFrame(rows)


def crc_variance(store: _Store, datasets: list[str] | None = None) -> pd.DataFrame:
    """Loudness and purity of each method's direction, on held-out pairs.

    loudness  variance of X± along w, relative to the largest principal variance of X±
    purity    cov(d) / (cov(d) + cov(m)) along w, with m the pair mean and d the
              half-difference: 1 = all of the direction's variance is intra-pair (contrast),
              0.5 = none of it is
    """
    methods = {"CRC-TPC": "PC1", "tcPCA": "tcPCA-1", "g-tcPCA": "GCC-1"}
    rows = []
    for ds in datasets or DATASETS:
        store.clear()
        for position in POSITIONS:
            ref, _ = store.entry("tcPCA-1", ds, position)
            if ref is None:
                continue
            pairs, Xb, Xc = store.arrays(ref, "eval")
            X = np.vstack([Xb, Xc])
            top_var = float(np.linalg.svd(X, compute_uv=False)[0] ** 2 / len(X))
            m, d = (Xb + Xc) / 2, (Xb - Xc) / 2
            for name, label in methods.items():
                entry, probe = store.entry(label, ds, position)
                if entry is None:
                    continue
                w = probe.subspace.float().numpy()[0]
                w = w / np.linalg.norm(w)
                var = float(np.mean((X @ w) ** 2))
                vm, vd = float(np.mean((m @ w) ** 2)), float(np.mean((d @ w) ** 2))
                acc = _pair_accuracy(pairs, Xb, Xc, w[None, :])[0]
                acc_tr = _pair_accuracy(*store.arrays(entry, "train"), w[None, :])[0]
                rows.append({"dataset": ds, "position": position, "method": name,
                             "loudness": var / top_var, "purity": vd / (vd + vm),
                             "accuracy": float(acc if acc_tr >= 0.5 else 1 - acc)})
    return pd.DataFrame(rows)


def token_norms(store: _Store, layer: int = 15, datasets: list[str] | None = None) -> pd.DataFrame:
    """Mean activation norm at each token position, per dataset (all samples)."""
    rows = []
    for ds in datasets or DATASETS:
        store.clear()
        samples = [s for s in store.samples if s.descriptors.get("dataset_name") == ds]
        for position, key in POSITIONS.items():
            norms = [float(np.linalg.norm(store._load(s)[(layer, key)].float().numpy()))
                     for s in samples if (layer, key) in store._load(s)]
            if norms:
                rows.append({"dataset": ds, "position": position,
                             "mean_norm": float(np.mean(norms)), "n": len(norms)})
    return pd.DataFrame(rows)


def answer_vs_period(evals: pd.DataFrame, catalogue: dict, summary: pd.DataFrame,
                     norms: pd.DataFrame) -> pd.DataFrame:
    """One row per dataset: accuracies at both positions, eigen-gaps, and the norm ratio."""
    label_of = {m: i.get("label") for m, i in catalogue.items()}
    ev = evals.assign(label=evals["method"].map(label_of))
    rows = []
    for ds in [d for d in DATASETS if d in set(ev["dataset"])]:
        row = {"dataset": ds}
        for pos in POSITIONS:
            at = ev[(ev["dataset"] == ds) & (ev["position"] == pos)]
            ccs = at[at["label"] == "CCS"]["acc"]
            row[f"CCS med ({pos})"] = float(ccs.median()) if len(ccs) else np.nan
            for lab in ("tcPCA-1", "logistic"):
                v = at[at["label"] == lab]["acc"]
                row[f"{lab} ({pos})"] = float(v.iloc[0]) if len(v) else np.nan
            g = summary[(summary["dataset"] == ds) & (summary["position"] == pos)
                        & (summary["basis"] == "tcPCA-8")]["gap"]
            row[f"gap ({pos})"] = float(g.iloc[0]) if len(g) else np.nan
            nrm = norms[(norms["dataset"] == ds) & (norms["position"] == pos)]["mean_norm"]
            row[f"norm ({pos})"] = float(nrm.iloc[0]) if len(nrm) else np.nan
        row["norm ratio period/answer"] = row["norm (period)"] / row["norm (answer)"]
        rows.append(row)
    return pd.DataFrame(rows)


def plot_answer_vs_period(df: pd.DataFrame, out_path: Path, title: str | None = None) -> None:
    """Left: tcPCA and CCS-median accuracy at both positions.  Right: eigen-gap at both."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(df))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for lab, mk in (("tcPCA-1", "o"), ("CCS med", "s")):
        a1.scatter(x - 0.12, df[f"{lab} (answer)"], marker=mk, color="#1f77b4", s=18,
                   label=f"{lab}, answer")
        a1.scatter(x + 0.12, df[f"{lab} (period)"], marker=mk, color="#d62728", s=18,
                   facecolors="none", label=f"{lab}, period")
    a1.axhline(0.5, color="0.6", lw=0.7, ls="--")
    a1.set_ylabel("accuracy"); a1.set_ylim(0.4, 1.02)
    a2.bar(x - 0.2, df["gap (answer)"], width=0.4, color="#1f77b4", label="answer")
    a2.bar(x + 0.2, df["gap (period)"], width=0.4, color="#d62728", label="period")
    a2.set_ylabel(r"eigen-gap $\lambda_0/\lambda_1$")
    for a in (a1, a2):
        a.set_xticks(x); a.set_xticklabels(df["dataset"], rotation=40, ha="right", fontsize=7)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    a1.legend(frameon=False, fontsize=6, ncol=2, loc="lower left")
    a2.legend(frameon=False, fontsize=7)
    if title:
        fig.suptitle(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_layer_sweep(df: pd.DataFrame, out_path: Path, chosen_layer: int = 15,
                     title: str | None = None) -> None:
    """Accuracy against layer, one page per token position, one panel per dataset."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    style = {"tcPCA-1": ("tcPCA", "#1f77b4", "-"), "GCC-1": ("g-tcPCA", "#9467bd", "-"),
             "logistic": ("logistic (supervised)", "0.45", ":"), "CCS": ("CCS (median, range)", "#d62728", "-")}
    with PdfPages(out_path) as pdf:
        for position in [p for p in POSITION_LABELS.values() if p in set(df["position"].map(
                lambda v: POSITION_LABELS.get(v, v)))]:
            sub = df[df["position"].map(lambda v: POSITION_LABELS.get(v, v)) == position]
            names = [d for d in DATASETS if d in set(sub["dataset"])]
            fig, axes = plt.subplots(3, 3, figsize=(7.2, 6.4), sharex=True, sharey=True)
            for ax, ds in zip(axes.flat, names):
                at = sub[sub["dataset"] == ds]
                for method, (lab, col, ls) in style.items():
                    m = at[at["method"] == method]
                    if m.empty:
                        continue
                    g = m.groupby("layer")["acc"]
                    ax.plot(g.median().index, g.median().values, ls, color=col, lw=1.2,
                            label=lab, marker="o", ms=2)
                    if method == "CCS":
                        ax.fill_between(g.min().index, g.min().values, g.max().values,
                                        color=col, alpha=0.15, lw=0)
                ax.axvline(chosen_layer, color="0.7", lw=0.8, ls="--")
                ax.axhline(0.5, color="0.85", lw=0.6)
                ax.set_title(ds, fontsize=8)
                ax.set_ylim(0.35, 1.02)
            for ax in axes.flat[len(names):]:
                ax.axis("off")
            for ax in axes[-1]:
                ax.set_xlabel("layer (decoder block)", fontsize=7)
            for ax in axes[:, 0]:
                ax.set_ylabel("accuracy", fontsize=7)
            axes.flat[0].legend(fontsize=6, frameon=False, loc="lower right")
            fig.suptitle(f"{title + ' - ' if title else ''}{position} token "
                         f"(dashed: block {chosen_layer}, used in the paper)", fontsize=9)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


SWEEP_STYLE = {"tcPCA-1": ("tcPCA", "#1f77b4", "-"), "GCC-1": ("g-tcPCA", "#9467bd", "-"),
               "logistic": ("logistic (supervised)", "0.45", ":"),
               "CCS": ("CCS (median, range)", "#d62728", "-")}


def write_loose_layer_sweep(df: pd.DataFrame, out_dir: Path, chosen_layer: int = 15) -> list[str]:
    """One small panel per (dataset, position), plus a shared legend file."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    out_dir.mkdir(exist_ok=True)
    pos_label = df["position"].map(lambda v: POSITION_LABELS.get(v, v))
    written = []
    for position in ("answer", "period"):
        for ds in [d for d in DATASETS if d in set(df["dataset"])]:
            at = df[(pos_label == position) & (df["dataset"] == ds)]
            if at.empty:
                continue
            fig, ax = plt.subplots(figsize=(2.0, 1.3))    # printed at 0.32 of the text width
            for method, (lab, col, ls) in SWEEP_STYLE.items():
                m = at[at["method"] == method]
                if m.empty:
                    continue
                g = m.groupby("layer")["acc"]
                ax.plot(g.median().index, g.median().values, ls, color=col, lw=1.1,
                        marker="o", ms=1.8)
                if method == "CCS":
                    ax.fill_between(g.min().index, g.min().values, g.max().values,
                                    color=col, alpha=0.15, lw=0)
            ax.axvline(chosen_layer, color="0.7", lw=0.8, ls="--")
            ax.axhline(0.5, color="0.85", lw=0.6)
            ax.set_ylim(0.35, 1.02)
            ax.tick_params(labelsize=6)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            fig.tight_layout()
            name = f"layer_sweep_{position}_{ds}.pdf"
            fig.savefig(out_dir / name)
            plt.close(fig)
            written.append(name)
    handles = [Line2D([], [], color=c, ls=ls, lw=1.2, label=l) for l, c, ls in SWEEP_STYLE.values()]
    handles.append(Line2D([], [], color="0.7", ls="--", lw=0.8, label=f"block {chosen_layer} (paper)"))
    fig = plt.figure(figsize=(5.6, 0.22))          # printed at 0.9 of the text width
    fig.legend(handles=handles, loc="center", ncol=5, frameon=False, fontsize=7)
    fig.savefig(out_dir / "layer_sweep_legend.pdf")
    plt.close(fig)
    return written + ["layer_sweep_legend.pdf"]


def write_loose_answer_vs_period(df: pd.DataFrame, out_dir: Path) -> list[str]:
    """The two halves of plot_answer_vs_period as separate files."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(exist_ok=True)
    x = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(3.0, 2.0))      # printed at 0.48 of the text width
    for lab, mk in (("tcPCA-1", "o"), ("CCS med", "s")):
        ax.scatter(x - 0.12, df[f"{lab} (answer)"], marker=mk, color="#1f77b4", s=16,
                   label=f"{lab.replace('-1', '')}, answer")
        ax.scatter(x + 0.12, df[f"{lab} (period)"], marker=mk, color="#d62728", s=16,
                   facecolors="none", label=f"{lab.replace('-1', '')}, period")
    ax.axhline(0.5, color="0.6", lw=0.7, ls="--")
    ax.set_ylabel("accuracy"); ax.set_ylim(0.4, 1.02)
    ax.set_xticks(x); ax.set_xticklabels(df["dataset"], rotation=40, ha="right", fontsize=6)
    ax.tick_params(axis="y", labelsize=6)
    ax.legend(frameon=False, fontsize=5.5, ncol=2, loc="lower left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(out_dir / "answer_vs_period_accuracy.pdf"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.0, 2.0))
    ax.bar(x - 0.2, df["gap (answer)"], width=0.4, color="#1f77b4", label="answer")
    ax.bar(x + 0.2, df["gap (period)"], width=0.4, color="#d62728", label="period")
    ax.set_ylabel(r"eigen-gap $\lambda_0/\lambda_1$", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(df["dataset"], rotation=40, ha="right", fontsize=6)
    ax.tick_params(axis="y", labelsize=6)
    ax.legend(frameon=False, fontsize=6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(out_dir / "answer_vs_period_gap.pdf"); plt.close(fig)
    return ["answer_vs_period_accuracy.pdf", "answer_vs_period_gap.pdf"]


def copa_pairs(strengths: pd.DataFrame) -> pd.DataFrame:
    """One row per held-out COPA prompt: both choices and their activations on the first tcPC.

    `strengths` is `figures.activation_strengths` (one row per sample).  The two samples of a
    prompt share its `question`; `label` marks the correct choice.
    """
    import re
    rows = []
    for q, g in strengths.groupby("question"):
        if len(g) != 2:
            continue
        by = {int(str(a).strip()[-1]): r for a, r in zip(g["answer"], g.itertuples())}
        if set(by) != {1, 2}:
            continue
        prem = re.search(r"‘‘‘(.*?)’’’", q)
        c1 = re.search(r"Choice 1: (.*?) Choice 2:", q)
        c2 = re.search(r"Choice 2: (.*?) Q:", q)
        if not (prem and c1 and c2):
            continue
        rows.append({
            "question": q, "premise": prem.group(1), "choice1": c1.group(1), "choice2": c2.group(1),
            "kind": "cause" if "the cause" in q.split("Q:")[-1] else "effect",
            "a1": float(by[1].activation), "a2": float(by[2].activation),
            "correct": 1 if int(by[1].label) == 1 else 2,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["gap"] = (df["a1"] - df["a2"]).abs()
        df["peak"] = df[["a1", "a2"]].abs().max(axis=1)
    return df


def write_loose_layer_sweep_mean(df: pd.DataFrame, out_dir: Path, chosen_layer: int = 15) -> list[str]:
    """Accuracy against layer averaged over the datasets, one panel per token.

    CCS is summarised per dataset by its median over seeds first.  The per-dataset curves
    stay in layer_sweep.csv (and on the companion site); the paper shows only this summary.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(exist_ok=True)
    pos_label = df["position"].map(lambda v: POSITION_LABELS.get(v, v))
    written = []
    for position in ("answer", "period"):
        at = df[pos_label == position]
        if at.empty:
            continue
        fig, ax = plt.subplots(figsize=(3.0, 1.9))          # printed at 0.48 of the text width
        for method, (lab, col, ls) in SWEEP_STYLE.items():
            m = at[at["method"] == method]
            if m.empty:
                continue
            curve = m.groupby(["dataset", "layer"])["acc"].median().groupby("layer").mean()
            ax.plot(curve.index, curve.values, ls, color=col, lw=1.2, marker="o", ms=2.5, label=lab)
        ax.axvline(chosen_layer, color="0.7", lw=0.8, ls="--")
        ax.set_xlabel("layer (decoder block)")
        ax.set_ylabel("mean accuracy, 9 datasets")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        fig.tight_layout()
        name = f"layer_sweep_mean_{position}.pdf"
        fig.savefig(out_dir / name)
        plt.close(fig)
        written.append(name)
    return written
