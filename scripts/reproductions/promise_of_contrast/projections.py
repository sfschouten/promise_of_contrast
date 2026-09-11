"""The projection figures: activations plotted on the leading contrastive components.

These reuse `scripts/report_subspace_scatter.py` rather than re-implementing the plot —
it already handles stratified sampling, pair connector lines, the colour/shade encoding
of a 2x2 factorial, and the cyclic palette the days-of-the-week figure needs.  All this
module does is ask it for the specific pages the paper uses.

Three figures, each a different tuple size:

    cities grid   4-tuple <pos/correct, pos/incorrect, neg/correct, neg/incorrect>,
                  where truth is the XOR of negation and content
    trinary       3-tuple <true, false, neither>
    days of week  7-tuple, one node per weekday

A run that has no probes yet is skipped rather than treated as an error, so the tables
still build on a partially-computed pipeline.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCATTER = ROOT / "scripts" / "report_subspace_scatter.py"

# name → (selector flag, run/compound suffix, extra args)
#
# The 2-D report pairs consecutive eigenvectors on its own (EV 0 vs 1, EV 2 vs 3, ...),
# so the only control needed is how many to show.  `--components` is for the 3-D mode and
# expects triples.
FIGURES = {
    "fig_cities_grid_projection": (
        "--runs", "cities_grid",
        ["--datasets", "cities_grid_all", "--max-vectors", "4"],
    ),
    "fig_trinary_projection": (
        "--compounds", "tot_city_tv_aff",
        ["--max-vectors", "2"],
    ),
    "fig_days_of_week_projection": (
        "--runs", "days_of_week",
        ["--max-vectors", "2"],
    ),
    "fig_months_of_year_projection": (
        "--runs", "months_of_year",
        ["--max-vectors", "2"],
    ),
}


def _target_exists(model: str, suffix: str) -> bool:
    return (ROOT / "data" / "probes" / f"{model}__{suffix}" / "manifest.json").exists()


def build(model: str, out_dir: Path) -> list[str]:
    """Render every projection figure available for `model`; return what was written."""
    written = []
    for name, (flag, suffix, extra) in FIGURES.items():
        if not _target_exists(model, suffix):
            print(f"  {name}: no probes for {model}__{suffix} — skipped")
            continue
        cmd = [sys.executable, str(SCATTER), flag, f"{model}__{suffix}",
               "--output-dir", str(out_dir), "--output-name", name, *extra]
        result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  {name}: FAILED\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}")
            continue
        written.append(name)
        print(f"  {name}: ok")
    return written
