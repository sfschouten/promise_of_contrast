"""Reproduction of:
  Probing Without Labels: Contrast-only Reproductions of Interpretability Results.

Produces, for one model, into outputs/reproductions/promise_of_contrast/<model>/:

  table1_ccs_accuracy.csv        CCS accuracy per dataset, both token positions, over seeds
  table2_loss_ablations.csv      the seven CCS objectives on the datasets CCS handles
  table3_tcpca_vs_ccs.csv        tcPCA against the spread of CCS solutions, and CRC-TPC
  table5_principal_components.csv  accuracy of principal components 1-5 of X_cf - X_base
  tables.pdf                     the same four, rendered for reading
  fig_eigenvalues.pdf            contrast eigenvalue spectra per dataset
  eigenvalue_summary.csv         per dataset: the leading gap and participation ratio
  fig_lambda_k.pdf               how far each objective's solution sits inside the
                                 leading principal subspace
  lambda_k.csv                   the underlying per-seed values
  fig_cities_grid_projection.pdf   the 2x2 negation x content grid on the leading components
  fig_trinary_projection.pdf       true / false / neither
  fig_days_of_week_projection.pdf  the seven weekdays
  copa_activation_strengths.csv  every held-out COPA sample ranked by projection
  copa_extremes.csv              the most and least activating of them

All accuracies are uncalibrated: (p_base + 1 - p_cf)/2 thresholded at 0.5 against the
sample-level truth, with no calibration head and no sign flip.  Values below 0.5 are
reported as they are; for an unsupervised probe that is a result, not an error.

Run from the project root:
    python scripts/reproductions/promise_of_contrast/reproduce.py --model llama2_7b
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import _style
import diagnostics
import figures
import layer_sweep
import lambda_k as lambda_k_mod
import projections
import tetrahedron_figure
import panels
import pools
import tables
from _common import (
    ABLATION_DATASETS,
    DATASETS,
    PROFILE,
    activations_dir,
    eval_db_path,
    method_catalogue,
    model_name,
    output_dir,
    probes_dir,
)


def _render_tables(named: list[tuple[str, pd.DataFrame]], out_path: Path,
                   title: str) -> None:
    """One page per table, rendered as text so the PDF is readable next to the paper."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(out_path) as pdf:
        for name, df in named:
            if df.empty:
                continue
            height = 1.2 + 0.32 * (len(df) + 1)
            fig, ax = plt.subplots(figsize=(min(2.0 + 1.5 * len(df.columns), 16), height))
            ax.axis("off")
            ax.set_title(f"{title}\n{name}", fontsize=10, loc="left")
            table = ax.table(cellText=df.astype(str).values,
                             colLabels=list(df.columns), loc="center", cellLoc="center")
            table.auto_set_font_size(False)
            table.set_fontsize(7)
            table.scale(1, 1.25)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama2_7b",
                        help="model alias from params.yaml:models")
    parser.add_argument("--skip-activations", action="store_true",
                        help="tables only; skip anything that reads the activation cache")
    args = parser.parse_args()
    _style.apply()   # figures are drawn at their printed size; see _style.py

    alias = args.model
    out = output_dir(alias)
    catalogue = method_catalogue()
    db = eval_db_path(alias)

    if not db.exists():
        print(f"no evaluation results at {db} — run evaluate_probes for {alias}__{PROFILE}",
              file=sys.stderr)
        return 1

    # ── tables ────────────────────────────────────────────────────────────────
    evals = tables.load_evals(db)
    if evals.empty:
        print(f"{db} has no rows with train_position == eval_position", file=sys.stderr)
        return 1

    t1 = tables.table_ccs_accuracy(evals, catalogue)
    t2 = tables.table_loss_ablations(evals, catalogue, position="answer")
    t3 = tables.table_tcpca_vs_ccs(evals, catalogue, position="period")
    t5 = tables.table_principal_components(evals, catalogue)
    t_sign = tables.table_sign_instability(evals, catalogue)

    named = [
        ("Table 1 — CCS accuracy (mean ± sd over seeds)", t1),
        ("Table 2 — loss-term ablations and alterations (answer token)", t2),
        ("Table 3 — tcPCA vs the spread of CCS solutions (period token)", t3),
        ("Table 5 — accuracy of individual principal components", t5),
        ("Sign instability — how often the learned orientation came out backwards", t_sign),
    ]
    for fname, df in zip(
        ["table1_ccs_accuracy", "table2_loss_ablations",
         "table3_tcpca_vs_ccs", "table5_principal_components", "sign_instability"],
        [t1, t2, t3, t5, t_sign],
    ):
        df.to_csv(out / f"{fname}.csv", index=False)
    _render_tables(named, out / "tables.pdf", title=f"{alias} ({model_name(alias)})")
    # The paper \inputs these, so its tables are exactly what this run produced.
    params = yaml.safe_load((ROOT / "params.yaml").read_text())
    for fname, text in (
        ("table1.tex", tables.latex_table1(t1)),
        ("table2.tex", tables.latex_table2(t2)),
        ("table3.tex", tables.latex_table3(t3)),
        ("table5.tex", tables.latex_table5(t5)),
        ("table_protocol.tex", tables.latex_protocol(tables.protocol_rows(params, PROFILE, alias))),
    ):
        (out / fname).write_text(text)
    for name, df in named:
        print(f"\n{name}\n{df.to_string(index=False)}")

    # The examples behind every number above: pool sizes, and the ids themselves, rebuilt from the split and checked against what was evaluated.
    ids = pools.pool_ids()
    sizes = pools.pool_sizes(ids)
    mismatched = pools.check_against_evals(sizes, evals)
    if mismatched:
        print("rebuilt pools disagree with the evaluated ones:\n  " + "\n  ".join(mismatched),
              file=sys.stderr)
        return 1
    ids.to_csv(out / "pool_ids.csv", index=False)
    sizes.to_csv(out / "pools.csv", index=False)
    print(f"\nPools (pairs)\n{sizes.to_string(index=False)}")

    # Table 1's mean +/- sd hides whether the seeds cluster.  This is the same numbers
    # with every seed drawn, which is where the multi-basin datasets show themselves.
    if figures.plot_seed_accuracy(
            evals, catalogue, out / "fig_seed_accuracy.pdf", label="CCS",
            datasets=DATASETS,
            title=None):
        print(f"\nWrote {out / 'fig_seed_accuracy.pdf'}")

    # Every CCS seed's accuracy, for the companion site's per-seed view.
    ccs_methods = {m for m, i in catalogue.items() if i.get("label") == "CCS"}
    evals[evals["method"].isin(ccs_methods)][["dataset", "position", "seed", "acc"]].to_csv(
        out / "ccs_seed_accuracy.csv", index=False)

    if args.skip_activations:
        print(f"\nWrote tables to {out}")
        return 0

    # ── figures and qualitative tables (need the activation cache) ────────────
    # One page per contrastive basis.  The generalised eigenvalues are explained-variance
    # ratios rather than activation units, so they get their own page rather than sharing
    # an axis with the tuple-contrastive spectra.
    BASES = [("tcPCA-8", "tuple-contrastive"), ("GCC-8", "generalised (variance-normalised)")]
    frames, pages = [], []
    for label, blurb in BASES:
        df = figures.collect_spectra(probes_dir(alias), catalogue, method_label=label)
        if df.empty:
            print(f"  no {label} probes — skipping that eigenvalue page")
            continue
        frames.append(df.assign(basis=label))
        pages.append((f"Contrast eigenvalue spectra, {blurb} — {alias}, answer token", df))

    if frames:
        spectra = pd.concat(frames, ignore_index=True)
        spectra.to_csv(out / "eigenvalues.csv", index=False)
        summary = pd.concat(
            [figures.spectrum_summary(f).assign(basis=f["basis"].iloc[0]) for f in frames],
            ignore_index=True)
        summary.to_csv(out / "eigenvalue_summary.csv", index=False)
        print("\nSpectrum shape (answer token) — how concentrated the contrast is")
        print(summary[summary["position"] == "answer"].to_string(
            index=False, float_format=lambda v: f"{v:8.2f}"))
        figures.plot_spectra(pages, out / "fig_eigenvalues.pdf",
                             position="answer", datasets=DATASETS)
        figures.plot_spectra(pages, out / "fig_eigenvalues_period.pdf",
                             position="period", datasets=DATASETS)
        # Loose per-dataset spectra for the paper (Figure 2 and Appendix H).
        # The whitened spectra are flat on every binary dataset, so only tcPCA's are drawn.
        for frame in frames:
            if frame["basis"].iloc[0] != "tcPCA-8":
                continue
            for pos in ("answer", "period"):
                figures.write_loose_spectra(frame, out / "panels", "tcpca", pos, DATASETS)
        # The spectrum-reliability link as a number, not a reading.
        cells, corr = figures.gap_vs_instability(evals, catalogue, summary)
        cells.to_csv(out / "gap_vs_instability.csv", index=False)
        corr.to_csv(out / "gap_vs_instability_corr.csv", index=False)
        figures.plot_gap_vs_instability(cells, corr, out / "fig_gap_vs_instability.pdf")

    lam = lambda_k_mod.compute(
        alias, model_name(alias), probes_dir(alias), activations_dir(alias),
        ROOT / "data" / "processed" / f"samples_{PROFILE}.jsonl", catalogue,
    )
    if not lam.empty:
        lam.to_csv(out / "lambda_k.csv", index=False)
        lambda_k_mod.plot(lam, out / "fig_lambda_k.pdf", datasets=ABLATION_DATASETS,
                          title=f"Subspace overlap of the learned direction — {alias}")
        # Figure 1/6: the three headline objectives, weight-normalised.
        lambda_k_mod.plot(
            lam, out / "fig_lambda_k_paper.pdf", datasets=ABLATION_DATASETS,
            title=f"Subspace overlap of the learned direction — {alias}",
            methods=lambda_k_mod.PAPER_OBJECTIVES,
            rename=lambda_k_mod.PAPER_OBJECTIVE_LABELS)
        # Loose panels for the paper (Figure 1 is imdb/answer; Appendix C is the grid).
        (out / "panels").mkdir(exist_ok=True)
        for ds in ABLATION_DATASETS:
            for pos in ("answer", "period"):
                lambda_k_mod.plot(
                    lam, out / "panels" / f"lambda_k_{ds}_{pos}.pdf", datasets=[ds],
                    positions=[pos], methods=lambda_k_mod.PAPER_OBJECTIVES,
                    rename=lambda_k_mod.PAPER_OBJECTIVE_LABELS, panel_size=(3.0, 1.9),
                    legend=(pos == "answer" and ds in ("imdb", "comparisons")), titles=False)

    strengths = figures.activation_strengths(
        model_name(alias), probes_dir(alias), activations_dir(alias),
        ROOT / "data" / "processed" / f"samples_{PROFILE}.jsonl", catalogue,
        dataset="copa", method_label="tcPCA-1", position="answer",
    )
    if not strengths.empty:
        strengths.to_csv(out / "copa_activation_strengths.csv", index=False)
        figures.extremes(strengths).to_csv(out / "copa_extremes.csv", index=False)
        copa_pairs = diagnostics.copa_pairs(strengths)
        copa_pairs.to_csv(out / "copa_pairs.csv", index=False)
        (out / "table_copa_examples_full.tex").write_text(tables.latex_copa_examples_full(copa_pairs))
        (out / "table_copa_examples_main.tex").write_text(tables.latex_copa_examples_main(copa_pairs))

    # Numbers behind claims the paper made qualitatively.
    store = diagnostics._Store(model_name(alias), probes_dir(alias), activations_dir(alias),
                               ROOT / "data" / "processed" / f"samples_{PROFILE}.jsonl",
                               catalogue)
    copa = diagnostics.copa_components(store)
    copa.to_csv(out / "copa_components.csv", index=False)
    (out / "table_copa_components.tex").write_text(tables.latex_copa_components(copa))
    crc = diagnostics.crc_variance(store)
    crc.to_csv(out / "crc_variance.csv", index=False)
    (out / "table_crc_variance.tex").write_text(tables.latex_crc_variance(crc))
    if frames:
        norms = diagnostics.token_norms(store)
        avp = diagnostics.answer_vs_period(evals, catalogue, summary, norms)
        avp.to_csv(out / "answer_vs_period.csv", index=False)
        diagnostics.plot_answer_vs_period(avp, out / "fig_answer_vs_period.pdf", title=alias)
        diagnostics.write_loose_answer_vs_period(avp, out / "panels")
    del store

    # Accuracy against layer, if the sweep has been run (it is its own DVC stage).
    sweep_csv = layer_sweep.out_path(alias)
    if sweep_csv.exists():
        sweep = pd.read_csv(sweep_csv)
        sweep.to_csv(out / "layer_sweep.csv", index=False)
        diagnostics.plot_layer_sweep(sweep, out / "fig_layer_sweep.pdf", title=alias)
        diagnostics.write_loose_layer_sweep(sweep, out / "panels")
        diagnostics.write_loose_layer_sweep_mean(sweep, out / "panels")
    else:
        print(f"  no layer sweep at {sweep_csv} - skipping fig_layer_sweep.pdf")

    print("\nProjection figures")
    projections.build(alias, out)

    # The 3-D view, with derived camera angles rather than a hand-picked one: see
    # scripts/reproductions/promise_of_contrast/tetrahedron_figure.py.
    print("\nTetrahedron views")
    for name in tetrahedron_figure.build(alias, out):
        print(f"  {name}")
    for name in tetrahedron_figure.build_loose(alias, out):
        print(f"  {name}")

    # The non-binary figures pick their layer by how cleanly the classes separate in the
    # plane they show (held-out nearest-centroid accuracy), not by convention.
    print("\nLayer choice for the non-binary figures")
    nb = panels.sweep(alias)
    nb.to_csv(out / "nonbinary_layers.csv", index=False)
    chosen = panels.choose(nb)
    pd.DataFrame([{"figure": f, "basis": b, "layer": l} for (f, b), l in chosen.items()]
                 ).to_csv(out / "nonbinary_layers_chosen.csv", index=False)
    panels.write_loose_sweep(nb, chosen, out / "panels")
    for (f, b), l in sorted(chosen.items()):
        print(f"  {f:16s} {b:8s} layer {l}")

    print("\nColumn-width projection panels")
    for name in panels.build(alias, out, layers=chosen):
        print(f"  {name}")

    print(f"\nWrote {len(list(out.iterdir()))} files to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
