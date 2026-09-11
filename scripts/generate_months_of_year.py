"""
Months-of-the-year modular-arithmetic dataset (Engels et al. 2024).

Produces one complete graph per arithmetic problem, e.g.
"Let's do some month of the year math. Three months from January is {month}", with
a node per candidate month.  Edges incident to the correct answer are truth
pairs; the rest are circular pairs for cross-covariance.  See src/data/sources/temporal.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.sources.temporal import generate


def main():
    generate("months_of_year", id_prefix="moy")


if __name__ == "__main__":
    main()
