"""
Summary heatmap: maximum accuracy and AUC across layers, by dataset and method.

Produces a single-page PDF with two heatmaps side by side:
  left  — max accuracy (across all layers and eval positions)
  right — max AUC

Rows = datasets grouped by family; columns = probe methods.
Dashed lines separate families; family names are annotated on the left.

Run from the project root:
    python scripts/report_max_accuracy.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, str(Path(__file__).parent.parent))

# Display order and abbreviations for method names.
METHOD_ORDER = [
    "logistic",
    "diff_in_means",
    "contrast_diff_in_means",
    "pca_1",
    "pca_16",
    "cross_covariance_1",
    "cross_covariance_16",
    "generalized_cross_covariance_1",
    "generalized_cross_covariance_16",
]
METHOD_ABBREV = {
    "logistic":                        "LR",
    "diff_in_means":                   "DiM",
    "contrast_diff_in_means":          "cDiM",
    "pca_1":                           "PCA-1",
    "pca_16":                          "PCA-16",
    "cross_covariance_1":              "CC-1",
    "cross_covariance_16":             "CC-16",
    "generalized_cross_covariance_1":  "GCC-1",
    "generalized_cross_covariance_16": "GCC-16",
}


def _load_data(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return conn.execute("""
        SELECT
            pe.dataset,
            pe.method,
            COALESCE(pr.family, 'unknown') AS family,
            MAX(pe.accuracy) AS max_accuracy,
            MAX(pe.auc)      AS max_auc
        FROM probe_evals pe
        LEFT JOIN (
            SELECT DISTINCT run_id, dataset, family
            FROM probe_runs
        ) pr ON pe.run_id = pr.run_id AND pe.dataset = pr.dataset
        GROUP BY pe.dataset, pe.method, pr.family
    """).df()


def _annot(pivot: pd.DataFrame) -> np.ndarray:
    """Build annotation array: formatted values where present, blank where NaN."""
    def fmt(v):
        return f"{v:.2f}" if pd.notna(v) else ""
    return np.vectorize(fmt)(pivot.values)


def _draw_heatmap(ax, pivot, annot, title, sep_positions):
    sns.heatmap(
        pivot, ax=ax,
        cmap="YlOrRd", vmin=0.5, vmax=1.0,
        annot=annot, fmt="",
        annot_kws={"size": 6},
        linewidths=0.3, linecolor="#e0e0e0",
        cbar_kws={"label": title, "shrink": 0.5},
        mask=pivot.isna(),
    )
    ax.set_title(title, fontsize=10, pad=6)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=7, rotation=45)
    plt.setp(ax.get_xticklabels(), ha="right")
    ax.tick_params(axis="y", labelsize=7, rotation=0)
    for sep in sep_positions:
        ax.axhline(sep, color="#444444", linewidth=1.0, linestyle="--")


def main() -> None:
    params  = yaml.safe_load(Path("params.yaml").read_text())
    db_path = params["output"]["results_db"]
    out_dir = Path(params["output"]["reports_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "max_accuracy.pdf"

    conn = duckdb.connect(db_path)
    df   = _load_data(conn)
    conn.close()

    if df.empty:
        print("No eval results found.")
        return

    # Dataset row order: sorted by family then dataset name.
    dataset_order = (
        df[["family", "dataset"]]
        .drop_duplicates()
        .sort_values(["family", "dataset"])
        ["dataset"]
        .tolist()
    )

    # Method column order: follow METHOD_ORDER, append unknowns at the end.
    present_methods = df["method"].unique().tolist()
    method_order = [m for m in METHOD_ORDER if m in present_methods]
    method_order += sorted(m for m in present_methods if m not in METHOD_ORDER)
    col_labels = [METHOD_ABBREV.get(m, m) for m in method_order]

    df["method_label"] = df["method"].map(lambda m: METHOD_ABBREV.get(m, m))

    # Build pivots.
    def make_pivot(metric):
        return (
            df.pivot_table(index="dataset", columns="method_label",
                           values=metric, aggfunc="max")
            .reindex(index=dataset_order, columns=col_labels)
        )

    piv_acc = make_pivot("max_accuracy")
    piv_auc = make_pivot("max_auc")

    # Family membership per row (for separator lines and labels).
    family_by_ds = (
        df[["dataset", "family"]].drop_duplicates().set_index("dataset")["family"]
    )
    families = [family_by_ds.get(ds, "") for ds in dataset_order]
    sep_positions = [i for i in range(1, len(dataset_order))
                     if families[i] != families[i - 1]]

    # Figure dimensions.
    n_rows    = len(dataset_order)
    n_cols    = len(col_labels)
    row_h     = 0.38   # inches per dataset row
    col_w     = 0.72   # inches per method column
    fam_col_w = 2.8    # left column: family label (left) + dataset name (right)
    heat_w    = n_cols * col_w
    heat_h    = n_rows * row_h
    fig_w     = fam_col_w + 2 * heat_w + 2.2   # 2.2 for colorbars + gaps
    fig_h     = heat_h + 2.0                    # 2.0 for title + x labels

    # Three columns: row labels | accuracy heatmap | AUC heatmap.
    fig, (ax_fam, ax_acc, ax_auc) = plt.subplots(
        1, 3,
        figsize=(fig_w, fig_h),
        gridspec_kw={"width_ratios": [fam_col_w, heat_w, heat_w], "wspace": 0.05},
    )
    fig.suptitle(
        "Max accuracy / AUC across layers — per dataset and method",
        fontsize=11, y=1.0,
    )

    _draw_heatmap(ax_acc, piv_acc, _annot(piv_acc), "Max accuracy", sep_positions)
    _draw_heatmap(ax_auc, piv_auc, _annot(piv_auc), "Max AUC",      sep_positions)
    # Row labels live in ax_fam; suppress them on both heatmaps.
    ax_acc.set_yticklabels([])
    ax_auc.set_yticklabels([])

    # Row-label column: family name (left-aligned) | dataset name (right-aligned).
    # Both sub-columns share ax_fam's y range so positions align with heatmap rows.
    ax_fam.set_xlim(0, 1)
    ax_fam.set_ylim(0, n_rows)
    ax_fam.invert_yaxis()
    ax_fam.axis("off")
    # Subtle divider between the two sub-columns.
    ax_fam.axvline(0.42, color="#dddddd", linewidth=0.7, zorder=0)

    # Dataset names — right sub-column.
    for i, ds in enumerate(dataset_order):
        ax_fam.text(0.99, i + 0.5, ds,
                    ha="right", va="center",
                    fontsize=6.5, color="#222222")

    # Family group labels — left sub-column, with separator lines at boundaries.
    cur_fam, fam_start = None, 0
    for i, ds in enumerate(dataset_order):
        fam = families[i]
        if fam != cur_fam:
            if cur_fam is not None:
                ax_fam.text(0.02, (fam_start + i) / 2, cur_fam,
                            ha="left", va="center",
                            fontsize=7, style="italic", color="#333333")
                ax_fam.axhline(i, color="#888888", linewidth=0.6, linestyle="--")
            cur_fam, fam_start = fam, i
    # last family
    ax_fam.text(0.02, (fam_start + n_rows) / 2, cur_fam,
                ha="left", va="center",
                fontsize=7, style="italic", color="#333333")

    plt.tight_layout()
    with PdfPages(out_path) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print(f"Done. {out_path}")


if __name__ == "__main__":
    main()
