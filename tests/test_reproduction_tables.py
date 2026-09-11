"""Tests for the Probing Without Labels table builders.

These run on a synthetic results frame, so they check the table *shapes and wiring* —
which methods map to which columns, how seeds are aggregated, how missing combinations
are rendered — without needing a trained sweep.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPRO = Path(__file__).resolve().parents[1] / "scripts/reproductions/promise_of_contrast"
sys.path.insert(0, str(REPRO))

import tables  # noqa: E402
from _common import method_catalogue  # noqa: E402


@pytest.fixture(scope="module")
def catalogue():
    return method_catalogue()


def _label_to_method(catalogue, label):
    return next(m for m, i in catalogue.items() if i["label"] == label)


def _frame(catalogue, datasets=("comparisons", "amazon"), rng_seed=0):
    """A synthetic probe_evals frame covering every configured method."""
    rng = np.random.default_rng(rng_seed)
    rows = []
    for method, info in catalogue.items():
        for ds in datasets:
            for pos in ("answer", "period"):
                for seed in info["seeds"]:
                    rows.append({
                        "dataset": ds, "layer": 15,
                        "train_position": pos, "eval_position": pos, "position": pos,
                        "method": method, "seed": seed,
                        "acc": float(rng.uniform(0.4, 1.0)),
                        "calibrated_accuracy": 0.5, "auc": 0.5, "n_eval": 100,
                        "pair_accuracy": 0.5, "pos_prob_mean": 0.5, "neg_prob_mean": 0.5,
                    })
    return pd.DataFrame(rows)


def test_ccs_table_has_one_row_per_dataset_and_both_positions(catalogue):
    df = _frame(catalogue)
    out = tables.table_ccs_accuracy(df, catalogue)
    assert list(out["dataset"]) == ["comparisons", "amazon"]
    assert {"answer (%)", "period (%)"} <= set(out.columns)
    assert (out["n_answer"] == 30).all(), "all 30 seeds should be aggregated"


def test_ccs_cells_report_mean_and_spread(catalogue):
    df = _frame(catalogue)
    method = _label_to_method(catalogue, "CCS")
    sub = df[(df["method"] == method) & (df["dataset"] == "comparisons")
             & (df["position"] == "answer")]
    cell = tables.table_ccs_accuracy(df, catalogue).iloc[0]["answer (%)"]
    mean, sd = cell.split(" ± ")
    assert int(mean) == round(100 * sub["acc"].mean())
    assert int(sd) == round(100 * sub["acc"].std(ddof=0))


def test_ablation_table_has_the_seven_objectives_in_order(catalogue):
    out = tables.table_loss_ablations(_frame(catalogue), catalogue, position="answer")
    assert list(out.columns) == [
        "dataset", "CCS", "L_conf", "L_cons", "L_cons+a1", "L_cons+a2",
        "L_cons+a1+a2", "CCS+a1+a2",
    ]


def test_tcpca_table_summarises_ccs_by_spread_not_mean(catalogue):
    df = _frame(catalogue)
    out = tables.table_tcpca_vs_ccs(df, catalogue, position="period")
    assert {"CCS min", "CCS med", "CCS max", "CRC-TPC", "tcPCA"} <= set(out.columns)
    row = out[out["dataset"] == "comparisons"].iloc[0]
    assert int(row["CCS min"]) <= int(row["CCS med"]) <= int(row["CCS max"])


def test_principal_component_table_covers_five_components_at_both_positions(catalogue):
    out = tables.table_principal_components(_frame(catalogue), catalogue)
    assert [c for c in out.columns if c.startswith("PC")] == \
        ["PC1", "PC2", "PC3", "PC4", "PC5"]
    assert set(out["token"]) == {"answer", "period"}
    assert len(out) == 2 * 2   # 2 datasets x 2 positions


def test_missing_combinations_render_as_a_dash_rather_than_failing(catalogue):
    df = _frame(catalogue)
    df = df[df["method"] != _label_to_method(catalogue, "PC3")]
    out = tables.table_principal_components(df, catalogue)
    assert (out["PC3"] == "—").all()
    assert (out["PC1"] != "—").all()


def test_below_chance_accuracies_survive_into_the_table(catalogue):
    """No max(acc, 1-acc) anywhere: a sub-0.5 unsupervised result must be reported."""
    df = _frame(catalogue)
    method = _label_to_method(catalogue, "CCS")
    mask = (df["method"] == method) & (df["dataset"] == "comparisons")
    df.loc[mask, "acc"] = 0.2
    cell = tables.table_ccs_accuracy(df, catalogue).iloc[0]["answer (%)"]
    assert cell.startswith("20"), cell


def test_only_matched_train_and_eval_positions_are_loaded(tmp_path, catalogue):
    """A probe is scored at the position it was trained on; cross-position rows are
    written to the database but must not reach the tables."""
    import duckdb

    from src.results.db import get_connection, init_schema

    db = tmp_path / "evals.duckdb"
    conn = get_connection(str(db))
    init_schema(conn)
    conn.executemany(
        """INSERT INTO probe_evals
           (eval_id, run_id, dataset, layer, train_position, eval_position, method,
            descriptor, accuracy, auc, n_eval, pair_accuracy, seed,
            uncalibrated_accuracy, pos_prob_mean, neg_prob_mean)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [("e", "r", "cities", 15, "last", "last", "logistic", "label",
          .5, .5, 10, .5, None, 0.9, .5, .5),
         ("e", "r", "cities", 15, "last", "second_to_last", "logistic", "label",
          .5, .5, 10, .5, None, 0.1, .5, .5)],
    )
    conn.close()

    got = tables.load_evals(db)
    assert len(got) == 1
    assert got.iloc[0]["acc"] == pytest.approx(0.9)
