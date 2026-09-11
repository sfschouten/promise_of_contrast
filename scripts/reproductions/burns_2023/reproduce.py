"""
Reproduction of a figure from:
  Burns, C., Ye, H., Klein, D., & Steinhardt, J. (2023). Discovering Latent
  Knowledge in Language Models Without Supervision. ICLR 2023. arXiv:2212.03827.

Figure produced (with LLaMA-3.1-8B in place of the paper's models):
  fig2_transfer_matrix.pdf — CCS transfer-accuracy matrix (cf. Burns et al. Fig 2).

A CCS probe is trained (unsupervised) on each source dataset and applied — with
NO refit of the direction or calibration head — to every target dataset.  Each
cell reports pair accuracy: the fraction of target contrast pairs where the probe
ranks the true side above the false side (p_base > p_cf).  The target is
re-centred by its own midpoint before scoring (see eval_probe_transfer), matching
the per-dataset centering used at train time.  The diagonal is in-distribution.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.activations.cache import load_sample_cache
from src.data.loader import Sample, load_jsonl
from src.probing import Manifest, eval_probe_transfer, load_probes

# Stable row/column order (Burns' 10-dataset suite + hellaswag as configured here).
CCS_ORDER = [
    "imdb", "amazon_polarity", "ag_news", "dbpedia_14", "rte", "qnli",
    "boolq", "copa", "hellaswag", "piqa", "story_cloze",
]

POSITION = "last"
DESCRIPTOR = "label"
METHOD = "ccs"


def _best_layer(layers: list[int], target: int = 14) -> int:
    """Highest available layer ≤ target; falls back to the largest layer.

    target=14: empirically the best-transferring layer for CCS on this model
    (highest mean in-distribution + off-diagonal pair accuracy in a layer sweep).
    """
    eligible = [l for l in layers if l <= target]
    return max(eligible) if eligible else max(layers)


def _load_activations(
    samples: list[Sample], cache_dir: Path, model_name: str, layer: int,
) -> dict[str, dict]:
    """Build {sample.id: {(layer, POSITION): tensor}} for the chosen layer only."""
    activations: dict[str, dict] = {}
    for s in samples:
        cache = load_sample_cache(cache_dir, model_name, s)
        v = cache.get((layer, POSITION))
        if v is not None:
            activations[s.id] = {(layer, POSITION): v}
    return activations


def compute_transfer_matrix(
    samples: list[Sample],
    activations: dict[str, dict],
    datasets: list[str],
    probes: dict[tuple, object],
    fit_infos: dict[str, object],
    layer: int,
) -> np.ndarray:
    """N×N pair-accuracy matrix: rows = trained-on (source), cols = evaluated-on."""
    by_ds = {d: [s for s in samples if s.descriptors.get("dataset_name") == d]
             for d in datasets}
    n = len(datasets)
    mat = np.full((n, n), np.nan)
    for i, src in enumerate(datasets):
        probe = probes.get((src, layer, POSITION, METHOD, DESCRIPTOR))
        info = fit_infos.get(src)
        if probe is None or info is None:
            print(f"  [warn] no CCS probe for source {src!r} at layer {layer} — skipping row")
            continue
        for j, tgt in enumerate(datasets):
            res = eval_probe_transfer(
                probe, info, activations, by_ds[tgt], POSITION, eval_dataset=tgt,
            )
            if res is not None and res.pair_accuracy is not None:
                mat[i, j] = res.pair_accuracy
        print(f"  {src}: transfer row computed")
    return mat


def _plot_page(pdf, mat: np.ndarray, datasets: list[str], layer: int,
               title: str, cbar_label: str, cmap: str, vmin: float, vmax: float) -> float:
    """Render one matrix page (with off-diagonal mean margins) into `pdf`.
    Returns the mean off-diagonal value."""
    n = len(datasets)
    off = mat.copy()
    np.fill_diagonal(off, np.nan)
    row_mean = np.nanmean(off, axis=1)               # transfer from source i
    col_mean = np.nanmean(off, axis=0)               # transfer to target j
    overall = np.nanmean(off)

    disp = np.full((n + 1, n + 1), np.nan)
    disp[:n, :n] = mat
    disp[:n, n] = row_mean
    disp[n, :n] = col_mean
    disp[n, n] = overall

    fig, ax = plt.subplots(figsize=(1.0 * (n + 2), 1.0 * (n + 2)))
    im = ax.imshow(disp, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")

    labels = datasets + ["mean"]
    ax.set_xticks(range(n + 1))
    ax.set_yticks(range(n + 1))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Evaluated on (target)", fontsize=9)
    ax.set_ylabel("Trained on (source)", fontsize=9)

    mid = (vmin + vmax) / 2
    for r in range(n + 1):
        for c in range(n + 1):
            v = disp[r, c]
            if np.isnan(v):
                continue
            ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(v - mid) > 0.28 and v < mid + 0.28 else "black")

    for i in range(n):
        ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                               edgecolor="red", lw=1.2))
    ax.axhline(n - 0.5, color="black", lw=1.0)
    ax.axvline(n - 0.5, color="black", lw=1.0)

    ax.set_title(title, fontsize=9)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    return overall


def _plot_matrix(mat: np.ndarray, datasets: list[str], layer: int, out_path: Path) -> None:
    """Two-page PDF: sign-invariant CCS accuracy (headline) + raw signed matrix.

    CCS is unsupervised and its loss is symmetric to flipping the direction's sign,
    so the polarity of "true" vs "false" is arbitrary per probe.  The principled,
    Burns-comparable metric is therefore the sign-invariant max(p, 1-p); the raw
    signed pair accuracy is kept as a second page to expose cross-dataset sign
    disagreement (near-0 cells are perfect separations with flipped polarity).
    """
    sign_inv = np.maximum(mat, 1.0 - mat)
    with PdfPages(out_path) as pdf:
        off_si = _plot_page(
            pdf, sign_inv, datasets, layer,
            title=(f"CCS transfer accuracy (sign-invariant) — LLaMA-3.1-8B, layer {layer}\n"
                   f"max(p, 1-p); red = in-distribution.  cf. Burns et al. (2023) Fig 2"),
            cbar_label="accuracy  max(p, 1-p)", cmap="viridis", vmin=0.5, vmax=1.0,
        )
        off_raw = _plot_page(
            pdf, mat, datasets, layer,
            title=(f"CCS transfer — raw signed pair accuracy — layer {layer}\n"
                   f"p(base > cf); ~0 = perfect separation, flipped sign"),
            cbar_label="pair accuracy (base > cf)", cmap="RdBu_r", vmin=0.0, vmax=1.0,
        )
    print(f"  → {out_path}  (mean off-diagonal: sign-invariant {off_si:.3f}, raw {off_raw:.3f})")


def make_fig2(params: dict, out_dir: Path) -> None:
    model_name = params["models"]["llama3_8b"]["name"]
    cache_dir = Path("data/activations/llama3_8b__ccs")
    probes_dir = Path(params["output"]["probes_dir"]) / "llama3_8b__ccs"
    samples_path = "data/processed/samples_ccs.jsonl"
    layer = _best_layer(params["activations"]["layers"])

    if not Path(samples_path).exists():
        print(f"  Samples not found ({samples_path}) — skipping Fig 2")
        return
    if not (probes_dir / "manifest.json").exists():
        print(f"  CCS probes not found ({probes_dir}) — run train_probes@llama3_8b__ccs first")
        return

    samples = load_jsonl(samples_path)
    present = {s.descriptors.get("dataset_name") for s in samples}
    datasets = [d for d in CCS_ORDER if d in present]
    # Include any extra datasets not in the canonical order, for completeness.
    datasets += sorted(d for d in present if d and d not in CCS_ORDER)
    print(f"Fig 2: layer={layer}, model={model_name}, datasets={datasets}")

    activations = _load_activations(samples, cache_dir, model_name, layer)
    print(f"  Loaded activations for {len(activations)} samples")

    manifest = Manifest.load(probes_dir)
    probes = load_probes(manifest.params_file)
    fit_infos = {
        e.dataset: e for e in manifest.entries
        if e.layer == layer and str(e.train_position) == POSITION
        and e.method == METHOD and e.descriptor == DESCRIPTOR
    }

    mat = compute_transfer_matrix(samples, activations, datasets, probes, fit_infos, layer)
    _plot_matrix(mat, datasets, layer, out_dir / "fig2_transfer_matrix.pdf")


def main() -> None:
    params = yaml.safe_load(Path("params.yaml").read_text())
    out_dir = Path(params["output"]["reproductions_dir"]) / "burns_2023"
    out_dir.mkdir(parents=True, exist_ok=True)

    make_fig2(params, out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
