"""
Classification accuracy report.

Produces one PDF per probe method, written to the reports directory as
accuracies_{method}.pdf.  Each PDF has one page per
(family, descriptor, train_position); all datasets in the family are stacked
on that page (HEIGHT_PER_DATASET inches each).  Each dataset block shows:
  - Accuracy heatmap:  x = layer,  y = eval_position,  colour = accuracy
  - AUC heatmap:       same axes,  colour = AUC

Run from the project root:
    python scripts/report_accuracies.py [--metric accuracy|auc|both]
"""
from __future__ import annotations

import argparse
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


def _load_data(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    rows = conn.execute("""
        SELECT pe.dataset, pe.layer, pe.train_position, pe.eval_position,
               pe.method, pe.descriptor, pe.accuracy, pe.auc,
               pr.family
        FROM probe_evals pe
        LEFT JOIN (
            SELECT DISTINCT run_id, dataset, family
            FROM probe_runs
        ) pr ON pe.run_id = pr.run_id AND pe.dataset = pr.dataset
        ORDER BY pe.dataset, pe.descriptor, pe.train_position, pe.method, pe.layer, pe.eval_position
    """).fetchall()

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows, columns=[
        "dataset", "layer", "train_position", "eval_position",
        "method", "descriptor", "accuracy", "auc", "family",
    ])


def _heatmap_block(
    axes: list[plt.Axes],
    gdf: pd.DataFrame,
    metrics: list[str],
    dataset: str,
) -> None:
    """Fill one row of axes with heatmaps, one per metric."""
    def _coerce(v):
        try:
            return int(v)
        except (ValueError, TypeError):
            return v

    gdf = gdf.copy()
    gdf["eval_position"] = gdf["eval_position"].map(_coerce)
    layers = sorted(gdf["layer"].unique())
    eval_pos = sorted(gdf["eval_position"].unique(),
                      key=lambda x: (isinstance(x, str), x))

    for ax, metric in zip(axes, metrics):
        try:
            pivot = gdf.pivot_table(
                index="eval_position", columns="layer",
                values=metric, aggfunc="mean",
            ).reindex(index=eval_pos, columns=layers)
        except Exception:
            ax.set_visible(False)
            continue

        sns.heatmap(
            pivot, ax=ax,
            cmap="viridis", vmin=0.5, vmax=1.0,
            cbar_kws={"label": metric.capitalize(), "shrink": 0.8},
            linewidths=0,
        )
        ax.set_title(f"{dataset}  —  {metric}", fontsize=9)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Eval position")
        ax.tick_params(axis="x", labelsize=7, rotation=45)
        ax.tick_params(axis="y", labelsize=7, rotation=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metric", default="both", choices=["accuracy", "auc", "both"],
        help="Which metric(s) to show (default: both)",
    )
    args = parser.parse_args()

    metrics = (
        ["accuracy", "auc"] if args.metric == "both"
        else [args.metric]
    )
    n_metrics = len(metrics)

    params = yaml.safe_load(Path("params.yaml").read_text())
    db_path = params["output"]["results_db"]
    out_dir = Path(params["output"]["reports_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(db_path)
    df = _load_data(conn)
    conn.close()

    if df.empty:
        print("No eval results found.")
        return

    df["family"] = df["family"].fillna("unknown")

    HEIGHT_PER_DATASET = 3.5
    COL_WIDTH = 6.0

    for method, method_df in df.groupby("method", sort=True):
        safe_name = str(method).replace(" ", "_").replace("/", "_")
        out_path = out_dir / f"accuracies_{safe_name}.pdf"

        page_groups = method_df.groupby(
            ["family", "descriptor", "train_position"], sort=True
        )
        print(f"Writing {page_groups.ngroups} page(s) → {out_path}")

        with PdfPages(out_path) as pdf:
            for (family, descriptor, train_pos), page_df in page_groups:
                datasets = sorted(page_df["dataset"].unique())
                n = len(datasets)

                fig, axes = plt.subplots(
                    n, n_metrics,
                    figsize=(COL_WIDTH * n_metrics, n * HEIGHT_PER_DATASET),
                    squeeze=False,
                )

                for i, dataset in enumerate(datasets):
                    gdf = page_df[page_df["dataset"] == dataset]
                    _heatmap_block(list(axes[i]), gdf, metrics, dataset)

                fig.suptitle(
                    f"method={method}  |  family={family}  |  descriptor={descriptor}  |  train={train_pos}",
                    fontsize=10, y=1.002,
                )
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                print(f"  {family} | {descriptor} | train={train_pos}  ({n} datasets)")

    print("\nDone.")


if __name__ == "__main__":
    main()
