from pathlib import Path

import duckdb
import numpy as np

from ..probing.base import ProbeFitInfo, ProbeEvalResult


def get_connection(db_path: str | Path) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(db_path))


def init_schema(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS probe_runs (
            run_id         VARCHAR,
            model          VARCHAR,
            dataset        VARCHAR,
            family         VARCHAR,
            layer          INTEGER,
            train_position VARCHAR,
            method         VARCHAR,
            descriptor     VARCHAR,
            n_train        INTEGER,
            n_test         INTEGER,
            n_pairs        INTEGER,
            params_file    VARCHAR,
            created_at     TIMESTAMP DEFAULT current_timestamp
        )
    """)
    conn.execute("ALTER TABLE probe_runs ADD COLUMN IF NOT EXISTS family VARCHAR")
    # Probe-initialisation seed; NULL for deterministic probes and for anything fitted
    # before seed sweeps existed.
    conn.execute("ALTER TABLE probe_runs ADD COLUMN IF NOT EXISTS seed INTEGER")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS probe_evals (
            eval_id        VARCHAR,
            run_id         VARCHAR,
            dataset        VARCHAR,
            layer          INTEGER,
            train_position VARCHAR,
            eval_position  VARCHAR,
            method         VARCHAR,
            descriptor     VARCHAR,
            accuracy       DOUBLE,
            auc            DOUBLE,
            n_eval         INTEGER,
            pair_accuracy  DOUBLE,
            created_at     TIMESTAMP DEFAULT current_timestamp
        )
    """)
    conn.execute(
        "ALTER TABLE probe_evals ADD COLUMN IF NOT EXISTS dataset VARCHAR"
    )
    conn.execute("ALTER TABLE probe_evals ADD COLUMN IF NOT EXISTS seed INTEGER")
    # Unsupervised CCS-style scoring: threshold (p_base + 1 - p_cf)/2 at 0.5 against the
    # sample-level truth, with no calibration head and no sign disambiguation, so a probe
    # may legitimately score below 0.5.  `accuracy` remains the calibrated number.
    for col in ("uncalibrated_accuracy", "pos_prob_mean", "neg_prob_mean"):
        conn.execute(f"ALTER TABLE probe_evals ADD COLUMN IF NOT EXISTS {col} DOUBLE")
    # Whether the unsupervised orientation had to be flipped, decided on the train split.
    conn.execute("ALTER TABLE probe_evals ADD COLUMN IF NOT EXISTS sign_flipped BOOLEAN")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS probe_artifacts (
            run_id         VARCHAR,
            dataset        VARCHAR,
            layer          INTEGER,
            train_position VARCHAR,
            method         VARCHAR,
            descriptor     VARCHAR,
            name           VARCHAR,
            values         DOUBLE[],
            created_at     TIMESTAMP DEFAULT current_timestamp
        )
    """)
    conn.execute(
        "ALTER TABLE probe_artifacts ADD COLUMN IF NOT EXISTS dataset VARCHAR"
    )
    conn.execute("ALTER TABLE probe_artifacts ADD COLUMN IF NOT EXISTS seed INTEGER")


def insert_probe_runs(
    conn: duckdb.DuckDBPyConnection,
    entries: list[ProbeFitInfo],
    run_id: str,
    model: str,
    params_file: str,
    dataset: str = "",
    family: str = "",
) -> None:
    rows = [
        (run_id, model, e.dataset or dataset, e.family or family,
         e.layer, str(e.train_position), e.method, e.descriptor,
         e.n_train, len(e.test_ids), e.n_pairs, params_file, e.seed)
        for e in entries
    ]
    if not rows:
        return
    conn.executemany(
        """INSERT INTO probe_runs
           (run_id, model, dataset, family, layer, train_position, method, descriptor,
            n_train, n_test, n_pairs, params_file, seed)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )


def insert_probe_evals(
    conn: duckdb.DuckDBPyConnection,
    results: list[ProbeEvalResult],
    eval_id: str,
    run_id: str,
) -> None:
    rows = [
        (eval_id, run_id, r.dataset,
         r.layer, str(r.train_position), str(r.eval_position),
         r.method, r.descriptor, r.accuracy, r.auc, r.n_eval, r.pair_accuracy,
         r.seed, r.uncalibrated_accuracy, r.pos_prob_mean, r.neg_prob_mean,
         r.sign_flipped)
        for r in results
    ]
    if not rows:
        return
    conn.executemany(
        """INSERT INTO probe_evals
           (eval_id, run_id, dataset, layer, train_position, eval_position,
            method, descriptor, accuracy, auc, n_eval, pair_accuracy,
            seed, uncalibrated_accuracy, pos_prob_mean, neg_prob_mean, sign_flipped)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )


def insert_probe_artifacts(
    conn: duckdb.DuckDBPyConnection,
    entries: list[ProbeFitInfo],
    run_id: str,
) -> None:
    """Insert per-probe artifact arrays (eigenvalues, explained variance, …)."""
    rows = [
        (run_id, e.dataset, e.layer, str(e.train_position), e.method, e.descriptor,
         name, arr.tolist(), e.seed)
        for e in entries
        for name, arr in e.artifacts.items()
        if isinstance(arr, np.ndarray) and arr.ndim == 1
    ]
    if not rows:
        return
    conn.executemany(
        """INSERT INTO probe_artifacts
           (run_id, dataset, layer, train_position, method, descriptor, name, values,
            seed)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
