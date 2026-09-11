"""Unit tests for per-contrast-type centering in paired probe fitting.

Verifies that `_grouped_midpoints` centres each (dataset_name, center_key) group
independently — the mechanism that lets a single dataset_name (e.g. the temporal
`*_circular` set) hold many "pair types" that are each mean-centred on their own.

Self-contained (no model / pipeline run).  Run from the project root:
    pytest tests/test_centering.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.loader import Sample
from src.probing.base import _center_group_keys, _grouped_midpoints


def _pair(ds: str, center_key: str | None, vb, vc):
    """A (base, cf) pair carrying dataset_name / optional center_key descriptors."""
    d = {"dataset_name": ds}
    if center_key is not None:
        d["center_key"] = center_key
    base = Sample(id=f"b_{ds}_{center_key}_{vb[0]}", text="", descriptors=dict(d))
    cf   = Sample(id=f"c_{ds}_{center_key}_{vc[0]}", text="", descriptors=dict(d))
    return (base, cf), np.asarray(vb, float), np.asarray(vc, float)


def test_center_key_splits_groups_within_one_dataset_name():
    # Two pair types under ONE dataset_name, each offset to a different location.
    # Group A clouds around +10, group B around -10; per-type centering must
    # remove each offset independently (single-group centering would not).
    specs = [
        _pair("circ", "0_1", [10, 10], [12, 8]),
        _pair("circ", "0_1", [11, 9],  [13, 7]),
        _pair("circ", "2_3", [-10, -10], [-8, -12]),
        _pair("circ", "2_3", [-9, -11],  [-7, -13]),
    ]
    pairs = [s[0] for s in specs]
    Xb = np.stack([s[1] for s in specs])
    Xc = np.stack([s[2] for s in specs])

    keys = _center_group_keys(pairs)
    assert keys == [("circ", "0_1"), ("circ", "0_1"), ("circ", "2_3"), ("circ", "2_3")]

    mm = _grouped_midpoints(pairs, Xb, Xc)
    cb, cc = Xb - mm, Xc - mm

    # Within each group the base+cf midpoint is now exactly the origin.
    for g in [slice(0, 2), slice(2, 4)]:
        midpoint = (cb[g].mean(0) + cc[g].mean(0)) / 2
        assert np.allclose(midpoint, 0.0, atol=1e-9)

    # Group A's centred points stay near the origin, NOT pulled toward B's offset:
    # a single global midpoint would be ~0 here too (A and B are symmetric), so
    # check the discriminating property — each group keeps its own spread, the
    # cross-group offset (~20) is gone.
    assert np.abs(cb[0:2]).max() < 5
    assert np.abs(cb[2:4]).max() < 5


def test_absent_center_key_collapses_to_dataset_name():
    # No center_key ⇒ behaviour identical to grouping by dataset_name alone.
    specs = [
        _pair("dsA", None, [5, 5], [7, 3]),
        _pair("dsA", None, [6, 4], [8, 2]),
        _pair("dsB", None, [-5, -5], [-3, -7]),
    ]
    pairs = [s[0] for s in specs]
    Xb = np.stack([s[1] for s in specs])
    Xc = np.stack([s[2] for s in specs])

    assert _center_group_keys(pairs) == [("dsA", ""), ("dsA", ""), ("dsB", "")]

    mm = _grouped_midpoints(pairs, Xb, Xc)
    # dsA rows share one midpoint; dsB its own.
    assert np.allclose(mm[0], mm[1])
    assert not np.allclose(mm[0], mm[2])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
