"""λ^K: how much of each CCS solution lies in the leading principal subspace.

For every fitted CCS variant, at every (dataset, token position, seed), this projects the
learned direction onto the principal directions of the held-out activations and reports
the cumulative overlap λ^K for K = 1, 2, 4, … 128.

The comparison that matters is between the full objective and its two ablations: if
dropping the confidence term collapses λ^K towards zero while keeping it pushes λ^K
towards one, then that term is what pins the solution to the high-variance directions.

Activations are centred exactly as they were at fit time, so the principal directions are
the ones the probe actually saw.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.activations.cache import load_sample_cache
from src.data.loader import load_jsonl
from src.probing import Manifest, load_probes, probe_key
from src.probing.base import _center_paired, _extract_pairs
from src.probing.subspace_overlap import (
    DEFAULT_KS,
    lambda_k_from_pcs,
    max_overlap_from_pcs,
    principal_directions,
)

from _common import POSITION_LABELS


def _eval_pairs(samples_by_id: dict, test_ids: list[str]):
    """Rebuild the held-out pairs from the manifest's test ids."""
    kept = [samples_by_id[i] for i in test_ids if i in samples_by_id]
    return _extract_pairs(kept)


def compute(model_alias: str, model_name: str, probes_dir: Path, act_dir: Path,
            samples_path: Path, catalogue: dict, centering: str = "per_branch",
            ks: tuple[int, ...] = DEFAULT_KS) -> pd.DataFrame:
    manifest = Manifest.load(probes_dir)
    probes = load_probes(manifest.params_file)
    samples = load_jsonl(str(samples_path))
    samples_by_id = {s.id: s for s in samples}

    # Activations are read once per (dataset, position) and reused across seeds — the
    # principal directions do not depend on which probe we are scoring.
    cache: dict[str, dict] = {}

    def _acts(sample_id: str):
        if sample_id not in cache:
            cache[sample_id] = load_sample_cache(act_dir, model_name, samples_by_id[sample_id])
        return cache[sample_id]

    pcs_cache: dict[tuple, np.ndarray] = {}
    rows = []
    for entry in manifest.entries:
        info = catalogue.get(entry.method)
        if info is None or info["kind"] != "ccs":
            continue
        probe = probes.get(probe_key(entry.dataset, entry.layer, entry.train_position,
                                     entry.method, entry.descriptor, entry.seed))
        if probe is None:
            continue

        key = (entry.dataset, entry.layer, entry.train_position)
        if key not in pcs_cache:
            pairs = _eval_pairs(samples_by_id, entry.test_ids)
            if not pairs:
                continue
            akey = (entry.layer, entry.train_position)
            try:
                Xb = np.stack([_acts(b.id)[akey].float().numpy() for b, _ in pairs])
                Xc = np.stack([_acts(c.id)[akey].float().numpy() for _, c in pairs])
            except KeyError:
                continue
            Xb, Xc = _center_paired(pairs, Xb, Xc, centering)
            # The basis depends only on the activations, so it is computed once per
            # (dataset, layer, position) and reused across every probe and seed.
            pcs_cache[key] = principal_directions(np.vstack([Xb, Xc]))
        pcs = pcs_cache[key]

        direction = probe.subspace.numpy()[0]
        lam = lambda_k_from_pcs(direction, pcs, ks=ks)
        mx = max_overlap_from_pcs(direction, pcs, ks=ks)
        for k in ks:
            rows.append({
                "model": model_alias,
                "method": info["label"],
                "dataset": entry.dataset,
                "position": POSITION_LABELS.get(str(entry.train_position),
                                                str(entry.train_position)),
                "layer": entry.layer,
                "seed": entry.seed,
                "K": k,
                "lambda_k": lam[k],
                "max_overlap": mx[k],
            })
    return pd.DataFrame(rows)


# The three objectives Figure 1/6 shows, in the configuration it uses.
#
# They are the *weight-normalised* probes, for every objective, whereas Tables 1-2 vary
# weight-norm, because it is alteration a1 there.  Reusing the ablation probes for this
# figure would put lambda^128 at 0.74; with weight_norm it is 0.95-0.96.
PAPER_OBJECTIVES = ("CCS+a1", "L_conf+a1", "L_cons+a1")

# Display names.  Plain CCS / L_conf / L_cons would hide that all three are
# weight-normalised -- and so differ from Table 2's CCS and L_conf.
PAPER_OBJECTIVE_LABELS = {"CCS+a1": "CCS+a1", "L_conf+a1": "L_conf+a1",
                          "L_cons+a1": "L_cons+a1"}


def plot(df: pd.DataFrame, out_path: Path, datasets: list[str] | None = None,
         title: str = "", methods: tuple[str, ...] | None = None,
         rename: dict[str, str] | None = None, positions: list[str] | None = None,
         panel_size: tuple[float, float] = (4.0, 2.5), legend: bool = True,
         titles: bool = True) -> None:
    """One panel per (dataset, position); a line per CCS variant, averaged over seeds.

    `methods` restricts and orders the objectives plotted; None keeps all of them, with
    CCS / L_conf / L_cons first.  The restricted form is what matches the published
    figure — seven lines per panel is unreadable at print size, and the ablations beyond
    the two single-term objectives are the subject of Table 2 rather than this figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    if df.empty:
        return
    mean = (df.groupby(["method", "dataset", "position", "K"])["lambda_k"]
              .mean().reset_index())
    datasets = datasets or sorted(mean["dataset"].unique())
    positions = [p for p in (positions or ("answer", "period")) if p in set(mean["position"])]
    available = set(mean["method"])
    if methods is not None:
        methods = [m for m in methods if m in available]
        if not methods:
            return
        mean = mean[mean["method"].isin(methods)].copy()
        if rename:
            mean["method"] = mean["method"].map(lambda m: rename.get(m, m))
            methods = [rename.get(m, m) for m in methods]
    else:
        methods = [m for m in PAPER_OBJECTIVES if m in available]
        methods += sorted(available - set(methods))

    with PdfPages(out_path) as pdf:
        fig, axes = plt.subplots(len(datasets), len(positions),
                                 figsize=(panel_size[0] * len(positions),
                                          panel_size[1] * len(datasets)),
                                 squeeze=False, sharex=True, sharey=True)
        for r, ds in enumerate(datasets):
            for c, pos in enumerate(positions):
                ax = axes[r][c]
                for method in methods:
                    sel = mean[(mean["dataset"] == ds) & (mean["position"] == pos)
                               & (mean["method"] == method)].sort_values("K")
                    if sel.empty:
                        continue
                    ax.plot(range(len(sel)), sel["lambda_k"], marker="o",
                            label=method, linewidth=1.1, markersize=2.5)
                    ax.set_xticks(range(len(sel)))
                    ax.set_xticklabels(sel["K"].astype(int))
                ax.set_ylim(0, 1)
                ax.grid(alpha=.25, linewidth=.5)
                if titles:
                    ax.set_title(f"{pos} | {ds}", fontsize=9)
                if c == 0:
                    ax.set_ylabel("$\\lambda^K$")
                if r == len(datasets) - 1:
                    ax.set_xlabel("K")
        handles, labels = axes[0][0].get_legend_handles_labels()
        single = len(datasets) * len(positions) == 1
        if handles and single and legend:
            axes[0][0].legend(handles, labels, frameon=False, loc="center right")
        elif handles and legend:
            fig.legend(handles, labels, loc="center right", title="objective",
                       frameon=False)
        if title:
            fig.suptitle(title, fontsize=10)
        fig.tight_layout(rect=(0, 0, 1.0 if single else 0.88, 0.98 if title else 1.0))
        pdf.savefig(fig)
        plt.close(fig)
