"""
Eigenvalue-spectrum report for cross-covariance probes.

Produces two multi-page PDFs:
  eigenspectra_heatmap.pdf   — heatmap of top-K most-negative eigenvalues × layer
  eigenspectra_ridgeline.pdf — per-layer normalised drop-off (ridgeline bars)

Each PDF has one page per (family, descriptor, train_position, method).
Within each page all datasets belonging to that family are stacked vertically.

Run from the project root:
    python scripts/report_eigenspectra.py [--top-k 50] [--threshold 0] [--output-dir path/]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.probing import Manifest, load_probes


def _load_data(probes_dir: Path, method_prefix: str) -> pd.DataFrame:
    """Return eigenvalue rows read directly from probe .pt files."""
    rows = []
    for run_dir in sorted(probes_dir.iterdir()):
        if not run_dir.is_dir() or not (run_dir / "manifest.json").exists():
            continue
        manifest = Manifest.load(run_dir)
        try:
            probes = load_probes(manifest.params_file)
        except Exception as exc:
            print(f"  [{run_dir.name}] skipping — could not load probes: {exc}")
            continue

        entry_index: dict[tuple, str] = {
            (e.dataset, e.layer, str(e.train_position), e.method, e.descriptor):
                e.family or manifest.dataset or "unknown"
            for e in manifest.entries
        }

        for key, probe in probes.items():
            # Seed-swept probes append the init seed; ignore it here, spectra are
            # deterministic given the data.
            dataset_name, layer, position, method, descriptor = key[:5]
            if not method.startswith(method_prefix):
                continue
            eigenvalues = probe.artifacts.get("eigenvalues")
            if eigenvalues is None:
                continue
            family = entry_index.get(
                (dataset_name, layer, str(position), method, descriptor), "unknown"
            )
            rows.append({
                "dataset":        dataset_name,
                "layer":          layer,
                "train_position": str(position),
                "method":         method,
                "descriptor":     descriptor,
                "values":         list(eigenvalues),
                "family":         family,
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Strip the _N suffix so k=1 and k=16 variants aren't plotted twice.
    df["method"] = df["method"].str.replace(r"_\d+$", "", regex=True)
    df = df.drop_duplicates(subset=["dataset", "layer", "train_position", "method", "descriptor"])
    return df


def _plot_ridgeline(
    ax: plt.Axes,
    layers: list[int],
    matrix: np.ndarray,
    title: str,
    ridge_scale: float = 0.75,
) -> None:
    K = matrix.shape[1]
    ranks = np.arange(K)

    for i, layer in enumerate(layers):
        v = matrix[i]
        denom = abs(v[0]) if abs(v[0]) > 1e-12 else 1.0
        v_norm = v / denom

        y_base = float(i)
        y_line = y_base + v_norm * ridge_scale

        ax.axhline(y_base, color="gray", lw=0.3, alpha=0.4, zorder=1)
        ax.vlines(ranks, y_base, y_line, color="steelblue", lw=0.8, alpha=0.85, zorder=2)

    ax.set_xlabel("Eigenvalue rank (0 = most negative)")
    ax.set_ylabel("Layer")
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers, fontsize=7)
    ax.set_xlim(-0.5, K - 0.5)
    ax.set_ylim(len(layers) - 0.5, -ridge_scale - 0.2)
    ax.set_title(title, fontsize=9)


def _make_heatmap_page(
    ax_heat: plt.Axes,
    ax_count: plt.Axes,
    layers: list[int],
    matrix: np.ndarray,
    threshold: float,
    title: str,
) -> None:
    K = matrix.shape[1]
    counts = (matrix < threshold).sum(axis=1)

    lim = max(float(np.abs(matrix).max()), 1e-9)
    im = ax_heat.imshow(
        matrix.T, aspect="auto", origin="upper",
        cmap="RdBu_r", interpolation="nearest",
        extent=[-0.5, len(layers) - 0.5, K - 0.5, -0.5],
    )
    im.set_clim(-lim, lim)
    plt.colorbar(im, ax=ax_heat, label="Eigenvalue (λ)", fraction=0.03, pad=0.02)
    ax_heat.set_xticks(range(len(layers)))
    ax_heat.set_xticklabels(layers, fontsize=7, rotation=45, ha="right")
    ax_heat.set_yticks(range(0, K, max(1, K // 10)))
    ax_heat.set_ylabel("Eigenvalue rank (0 = most negative)")
    ax_heat.set_title(title, fontsize=9)

    ax_count.bar(range(len(layers)), counts, color="steelblue", alpha=0.85)
    ax_count.set_xticks(range(len(layers)))
    ax_count.set_xticklabels(layers, fontsize=7, rotation=45, ha="right")
    ax_count.set_ylabel(f"# λ < {threshold:g}")
    ax_count.set_xlabel("Layer")
    mean_count = float(counts.mean())
    ax_count.axhline(mean_count, color="darkorange", linewidth=1.2, linestyle="--",
                     label=f"mean = {mean_count:.1f}")
    ax_count.legend(fontsize=8)


def _build_dataset_data(
    page_df: pd.DataFrame,
    top_k: int,
    datasets: list[str],
) -> list[tuple[str, list[int], np.ndarray]]:
    result = []
    for dataset in datasets:
        gdf = page_df[page_df["dataset"] == dataset].sort_values("layer")
        layers = gdf["layer"].tolist()
        K = min(top_k, max(len(v) for v in gdf["values"]))
        matrix = np.array([
            np.sort(np.array(v, dtype=float))[:K]
            for v in gdf["values"]
        ])
        result.append((dataset, layers, matrix))
    return result


def _write_heatmap_page(
    pdf: PdfPages,
    dataset_data: list[tuple[str, list[int], np.ndarray]],
    all_layers: list[int],
    threshold: float,
    suptitle: str,
) -> None:
    HEIGHT_PER_DATASET = 4.5
    n = len(dataset_data)
    heatmap_width = max(8, len(all_layers) * 0.45)
    fig, axes = plt.subplots(
        n * 2, 1,
        figsize=(heatmap_width, n * HEIGHT_PER_DATASET),
        gridspec_kw={"height_ratios": [3, 1] * n},
    )
    axes = np.asarray(axes).flatten()

    for i, (dataset, layers, matrix) in enumerate(dataset_data):
        _make_heatmap_page(
            axes[i * 2], axes[i * 2 + 1],
            layers, matrix, threshold,
            f"{dataset}  —  top-{matrix.shape[1]} most-negative eigenvalues × layer",
        )
        if i < n - 1:
            axes[i * 2 + 1].set_xlabel("")

    fig.suptitle(suptitle, fontsize=10, y=1.002)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _write_ridgeline_page(
    pdf: PdfPages,
    dataset_data: list[tuple[str, list[int], np.ndarray]],
    suptitle: str,
) -> None:
    HEIGHT_PER_DATASET = 4.5
    n = len(dataset_data)
    ridge_width = max(6, dataset_data[0][2].shape[1] * 0.12)
    fig, axes = plt.subplots(
        n, 1,
        figsize=(ridge_width, n * HEIGHT_PER_DATASET),
        squeeze=False,
    )

    for i, (dataset, layers, matrix) in enumerate(dataset_data):
        _plot_ridgeline(
            axes[i, 0], layers, matrix,
            title=f"{dataset}  —  normalised eigenvalue drop-off by layer",
        )
        if i < n - 1:
            axes[i, 0].set_xlabel("")

    fig.suptitle(suptitle, fontsize=10, y=1.002)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=50,
                        help="Number of most-negative eigenvalues to show per layer (default: 50)")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Count eigenvalues below this value (default: 0)")
    parser.add_argument("--method-filter", default="cross_covariance",
                        help="Probe method prefix to include (default: cross_covariance).")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: <reports_dir>)")
    args = parser.parse_args()

    params = yaml.safe_load(Path("params.yaml").read_text())
    probes_dir = Path(params["output"]["probes_dir"])
    out_dir = Path(args.output_dir) if args.output_dir else Path(params["output"]["reports_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_data(probes_dir, args.method_filter)

    if df.empty:
        print(f"No eigenvalue artifacts found for method prefix '{args.method_filter}'.")
        return

    heatmap_path = out_dir / "eigenspectra_heatmap.pdf"
    ridgeline_path = out_dir / "eigenspectra_ridgeline.pdf"

    page_groups = df.groupby(["family", "descriptor", "train_position", "method"], sort=True)
    n_pages = page_groups.ngroups
    print(f"Writing {n_pages} page(s) each to:")
    print(f"  {heatmap_path}")
    print(f"  {ridgeline_path}")

    with PdfPages(heatmap_path) as heat_pdf, PdfPages(ridgeline_path) as ridge_pdf:
        for (family, descriptor, train_pos, method), page_df in page_groups:
            datasets = sorted(page_df["dataset"].unique())
            all_layers = sorted(page_df["layer"].unique())
            suptitle = f"family={family}  |  descriptor={descriptor}  |  train={train_pos}  |  {method}"

            dataset_data = _build_dataset_data(page_df, args.top_k, datasets)

            _write_heatmap_page(heat_pdf, dataset_data, all_layers, args.threshold, suptitle)
            _write_ridgeline_page(ridge_pdf, dataset_data, suptitle)

            print(f"  {family} | {descriptor} | train={train_pos} | {method}  ({len(datasets)} datasets)")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
