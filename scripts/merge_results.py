"""
Merge per-run eval result files from data/eval_results/ into data/results.duckdb.

Each evaluate_probes stage writes data/eval_results/<run_key>.duckdb.
This stage combines them all into the single results_db used by downstream reports.
"""
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.results.db import get_connection, init_schema


def main():
    params  = yaml.safe_load(Path("params.yaml").read_text())
    op      = params["output"]
    src_dir = Path(op["eval_results_dir"])
    dst_path = Path(op["results_db"])

    src_files = sorted(src_dir.glob("*.duckdb"))
    if not src_files:
        print("No per-run eval files found — nothing to merge")
        return

    # Delete and recreate to avoid duplicate rows on re-runs.
    if dst_path.exists():
        dst_path.unlink()

    conn = get_connection(str(dst_path))
    init_schema(conn)

    for src_path in src_files:
        # Per-run files written before a schema change are missing the newer columns, and
        # DVC caches evaluate_probes so they are not necessarily regenerated.  Migrate each
        # source in place first, then copy by explicit column name rather than SELECT *,
        # so a column added later can never shift the mapping.
        src_conn = get_connection(str(src_path))
        init_schema(src_conn)
        src_conn.close()

        alias = "src_" + src_path.stem.replace("-", "_").replace(".", "_")
        conn.execute(f"ATTACH '{src_path}' AS {alias} (READ_ONLY)")
        for table in ("probe_runs", "probe_artifacts", "probe_evals"):
            cols = [r[0] for r in conn.execute(f"DESCRIBE {table}").fetchall()]
            col_list = ", ".join(f'"{c}"' for c in cols)
            conn.execute(
                f"INSERT INTO {table} ({col_list}) SELECT {col_list} FROM {alias}.{table}"
            )
        conn.execute(f"DETACH {alias}")
        print(f"  merged {src_path.name}")

    counts = {
        t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        for t in ("probe_runs", "probe_evals")
    }
    conn.close()
    print(f"Done — {counts['probe_runs']} probe_runs, {counts['probe_evals']} probe_evals → {dst_path}")


if __name__ == "__main__":
    main()
