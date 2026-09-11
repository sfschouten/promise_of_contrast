"""The paper build target: assert that everything the paper claims still reproduces.

This is deliberately downstream of both `reproduce_promise_of_contrast` stages, so
`dvc repro check_paper` pulls in exactly the paper's chain — data, activations, probes,
evaluations, tables, figures — and nothing auxiliary.  A clean `dvc repro check_paper` is
the single command that answers "does the paper still build?".

It checks that every expected output exists, for both models, then writes a short
report.  Failures are fatal, so the stage goes red rather than
producing a reassuring file nobody reads.

Run from the project root:
    python scripts/reproductions/promise_of_contrast/check_paper.py
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from _common import PROFILE, output_dir  # noqa: E402

MODELS = ["llama2_7b", "llama3_8b"]

# What a complete reproduction produces, per model.
EXPECTED = [
    "table1_ccs_accuracy.csv",
    "table2_loss_ablations.csv",
    "table3_tcpca_vs_ccs.csv",
    "table5_principal_components.csv",
    "sign_instability.csv",
    "eigenvalues.csv",
    "eigenvalue_summary.csv",
    "lambda_k.csv",
    "copa_activation_strengths.csv",
    "copa_extremes.csv",
    "tables.pdf",
    "fig_eigenvalues.pdf",
    "fig_lambda_k.pdf",
    "fig_lambda_k_paper.pdf",
    "fig_cities_grid_projection.pdf",
    "fig_trinary_projection.pdf",
    "fig_days_of_week_projection.pdf",
    "fig_cities_grid_tetrahedron.pdf",
    "tetrahedron_views.csv",
    "tetrahedron_axes.csv",
    "tetrahedron_edges.csv",
    "fig_seed_accuracy.pdf",
    "pools.csv",
    "pool_ids.csv",
    "table1.tex",
    "table2.tex",
    "table3.tex",
    "table5.tex",
    "table_protocol.tex",
    "gap_vs_instability.csv",
    "gap_vs_instability_corr.csv",
    "fig_gap_vs_instability.pdf",
    "copa_components.csv",
    "crc_variance.csv",
    "table_crc_variance.tex",
    "table_copa_components.tex",
    "copa_pairs.csv",
    "table_copa_examples_full.tex",
    "table_copa_examples_main.tex",
    "answer_vs_period.csv",
    "fig_answer_vs_period.pdf",
    "layer_sweep.csv",
    "fig_layer_sweep.pdf",
    "fig_eigenvalues_period.pdf",
    "fig_months_of_year_projection.pdf",
    "panels/lambda_k_imdb_answer.pdf",
    "panels/eigenvalues_tcpca_answer_amazon.pdf",
    "panels/tetrahedron_wtcpca_oblique.pdf",
    "panels/tetrahedron_tcpca_negation.pdf",
    "panels/tetrahedron_legend.pdf",
    "panels/trinary_tcpca.pdf",
    "panels/days_of_week_tcpca.pdf",
    "panels/months_of_year_tcpca.pdf",
    "panels/layer_sweep_answer_imdb.pdf",
    "panels/layer_sweep_legend.pdf",
    "panels/layer_sweep_mean_answer.pdf",
    "panels/layer_sweep_mean_period.pdf",
    "ccs_seed_accuracy.csv",
    "panels/answer_vs_period_accuracy.pdf",
    "panels/answer_vs_period_gap.pdf",
    "nonbinary_layers.csv",
    "nonbinary_layers_chosen.csv",
    "panels/nonbinary_sweep_days_of_week.pdf",
    "panels/nonbinary_sweep_months_of_year.pdf",
    "panels/nonbinary_sweep_trinary.pdf",
]


def main() -> int:
    lines: list[str] = []
    failures: list[str] = []

    def record(title: str, ok: bool, detail: str = "") -> None:
        lines.append(f"- {'PASS' if ok else 'FAIL'} — {title}")
        if detail:
            lines.append(f"  {detail}")
        if not ok:
            failures.append(title)

    # every expected artefact exists, for both models
    for model in MODELS:
        out = output_dir(model)
        missing = [name for name in EXPECTED if not (out / name).exists()]
        record(f"{model}: {len(EXPECTED) - len(missing)}/{len(EXPECTED)} outputs present",
               not missing,
               f"missing: {', '.join(missing)}" if missing else "")

    report = output_dir(MODELS[0]).parent / "CHECKS.md"
    report.write_text(
        "# Paper build check\n\n"
        f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n\n"
        + "\n".join(lines)
        + ("\n\nAll checks passed.\n" if not failures else
           "\n\n**Failed:** " + "; ".join(failures) + "\n")
    )
    print("\n".join(lines))
    print(f"\n-> {report}")
    if failures:
        print(f"\n{len(failures)} check(s) failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
