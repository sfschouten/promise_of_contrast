"""Compare freshly built results against the ones committed to this repository.

`outputs/reproductions/promise_of_contrast/` is committed, and rebuilding the paper
overwrites it in place, so after `run_paper.sh` the committed numbers are still available
from git.  This script reads them from a git ref (default `HEAD`) and compares them with
the files on disk.

Bit-identical numbers are not expected on different hardware: activations are stored in
bf16/fp16, and a different GPU or batch size perturbs them at ~1e-2, which can move a
borderline example or send a CCS seed into the other basin.  So there are three tiers:

  * **exact** — the example pools.  These depend only on the data and the seeds, never on
    the GPU, so any difference means the datasets were not rebuilt faithfully.
  * **headline** — the paper's tables.  A cell fails if it moves by more than
    `--tolerance` percentage points, or by more than twice its own seed spread where the
    table reports one (a bimodal cell cannot be pinned down more tightly than that).
    The CCS min/max columns are seed extremes and are reported, not judged.
  * **reported** — every other CSV: the size of the differences is printed, nothing fails.

Run from the project root, after rebuilding:
    python scripts/reproductions/promise_of_contrast/compare_to_committed.py [--ref HEAD]
"""
from __future__ import annotations

import argparse
import io
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = Path("outputs/reproductions/promise_of_contrast")
MODELS = ["llama2_7b", "llama3_8b"]

EXACT = {"pools.csv", "pool_ids.csv"}
HEADLINE = {"table1_ccs_accuracy.csv", "table2_loss_ablations.csv",
            "table3_tcpca_vs_ccs.csv", "table5_principal_components.csv",
            "sign_instability.csv"}
SEED_EXTREMES = {"CCS min", "CCS max"}

_PM = re.compile(r"^\s*(-?[\d.]+)\s*±\s*(-?[\d.]+)\s*$")


def committed(ref: str, path: Path) -> pd.DataFrame | None:
    r = subprocess.run(["git", "show", f"{ref}:{path.as_posix()}"],
                       cwd=ROOT, capture_output=True, text=True)
    return pd.read_csv(io.StringIO(r.stdout)) if r.returncode == 0 else None


def _cell(v) -> tuple[float, float | None] | None:
    """(value, seed sd) for a numeric or `mean ± sd` cell; None for text."""
    if isinstance(v, (int, float, np.number)):
        return (float(v), None) if pd.notna(v) else None
    m = _PM.match(str(v))
    if m:
        return float(m.group(1)), float(m.group(2))
    try:
        return float(v), None
    except ValueError:
        return None


def _same_structure(a: pd.DataFrame, b: pd.DataFrame) -> str | None:
    if list(a.columns) != list(b.columns):
        return "columns differ"
    if a.shape != b.shape:
        return f"shape {b.shape} vs committed {a.shape}"
    return None


def compare_headline(ref: pd.DataFrame, new: pd.DataFrame, tol: float) -> tuple[list, list]:
    """(failures, notes): cells beyond tolerance, and seed-extreme cells that moved."""
    fails, notes = [], []
    label = new.columns[0]
    for (_, r0), (_, r1) in zip(ref.iterrows(), new.iterrows()):
        for col in new.columns[1:]:
            c0, c1 = _cell(r0[col]), _cell(r1[col])
            if c0 is None or c1 is None:
                if str(r0[col]) != str(r1[col]):
                    fails.append(f"{r1[label]} / {col}: {r0[col]!r} -> {r1[col]!r}")
                continue
            d = abs(c1[0] - c0[0])
            bound = max(tol, 2 * max(c0[1] or 0, c1[1] or 0))
            if d <= bound:
                continue
            where = f"{r1[label]} / {col}: {r0[col]} -> {r1[col]} (|d| {d:g} > {bound:g})"
            (notes if col in SEED_EXTREMES else fails).append(where)
    return fails, notes


def numeric_diff(ref: pd.DataFrame, new: pd.DataFrame) -> str:
    num = [c for c in new.columns if pd.api.types.is_numeric_dtype(new[c])
           and pd.api.types.is_numeric_dtype(ref[c])]
    if not num:
        text_same = ref.astype(str).equals(new.astype(str))
        return "identical" if text_same else "text differs"
    d = (new[num].to_numpy(float) - ref[num].to_numpy(float))
    d = np.abs(d[~np.isnan(d)])
    if not d.size or d.max() == 0:
        return "identical"
    return f"max |d| {d.max():.3g}, median |d| {np.median(d):.3g}, " \
           f"{(d > 1e-9).mean():.0%} of cells differ"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="HEAD", help="git ref holding the committed results")
    ap.add_argument("--tolerance", type=float, default=3.0,
                    help="headline-table tolerance, in percentage points")
    args = ap.parse_args()

    failures: list[str] = []
    for model in MODELS:
        print(f"\n== {model} (against {args.ref})")
        for path in sorted((ROOT / OUT / model).glob("*.csv")):
            rel = path.relative_to(ROOT)
            ref = committed(args.ref, rel)
            if ref is None:
                print(f"  {path.name:34s} not committed at {args.ref} — skipped")
                continue
            new = pd.read_csv(path)
            bad = _same_structure(ref, new)
            if path.name in EXACT:
                ok = bad is None and ref.astype(str).equals(new.astype(str))
                print(f"  {path.name:34s} {'identical' if ok else 'DIFFERS — ' + (bad or 'values')}")
                if not ok:
                    failures.append(f"{model}/{path.name}: example pools differ")
                continue
            if bad:
                print(f"  {path.name:34s} {bad}")
                if path.name in HEADLINE:
                    failures.append(f"{model}/{path.name}: {bad}")
                continue
            if path.name in HEADLINE:
                fails, notes = compare_headline(ref, new, args.tolerance)
                status = "within tolerance" if not fails else f"{len(fails)} cell(s) out"
                print(f"  {path.name:34s} {status}; {numeric_diff_headline(ref, new)}")
                for f in fails:
                    print(f"      FAIL {f}")
                    failures.append(f"{model}/{path.name}: {f}")
                for n in notes:
                    print(f"      note (seed extreme) {n}")
            else:
                print(f"  {path.name:34s} {numeric_diff(ref, new)}")

    print()
    if failures:
        print(f"{len(failures)} difference(s) beyond tolerance", file=sys.stderr)
        return 1
    print("All headline numbers agree with the committed results within tolerance.")
    return 0


def numeric_diff_headline(ref: pd.DataFrame, new: pd.DataFrame) -> str:
    """Largest movement of any headline value, in the table's own units."""
    ds = [abs(c1[0] - c0[0])
          for col in new.columns[1:]
          for v0, v1 in zip(ref[col], new[col])
          if (c0 := _cell(v0)) is not None and (c1 := _cell(v1)) is not None]
    return f"max |d| {max(ds):g}" if ds and max(ds) else "identical"


if __name__ == "__main__":
    raise SystemExit(main())
