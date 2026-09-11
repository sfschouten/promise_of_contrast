"""Structural tests for Trilemma-of-Truth data generation and train/test splits.

Run from the project root:
    pytest tests/test_tot_structure.py -v
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import pytest

def _find_project_root() -> Path:
    """Walk up from this file to find the directory that contains data/."""
    p = Path(__file__).resolve()
    for _ in range(8):
        p = p.parent
        if (p / "data").exists():
            return p
    return Path(__file__).resolve().parent.parent

ROOT = _find_project_root()
sys.path.insert(0, str(ROOT))

RAW_PAIRS   = ROOT / "data" / "raw" / "tot.jsonl"
PROC_SAMPLES = ROOT / "data" / "processed" / "samples_tot.jsonl"

# Expected (base_ml, cf_ml) per edge type.
_EDGE_ML = {
    "tf_aff": (1, 0), "tf_neg": (1, 0),
    "tn_aff": (1, 2), "tn_neg": (1, 2),
    "fn_aff": (0, 2), "fn_neg": (0, 2),
    "na_t":   (1, 1),
    "na_f":   (0, 0),
    "na_n":   (2, 2),
}

# NA edges have mixed negation (base=neg, cf=aff).
_NA_EDGES = {"na_t", "na_f", "na_n"}


def _edge_type(dataset_name: str) -> str:
    """Extract the edge suffix from a dataset_name like 'tot_city_tf_aff'."""
    for edge in _EDGE_ML:
        if dataset_name.endswith(f"_{edge}"):
            return edge
    return ""


@pytest.fixture(scope="module")
def raw_pairs():
    if not RAW_PAIRS.exists():
        pytest.skip(f"{RAW_PAIRS} not found")
    # tot.jsonl is a *graphs* file (generate_tot.py emits Graph objects); materialize
    # its typed edges into the ContrastivePairs these tests assert over.
    from src.data.graph import load_graphs_jsonl, materialize_pairs
    return materialize_pairs(load_graphs_jsonl(str(RAW_PAIRS)))


@pytest.fixture(scope="module")
def proc_samples():
    if not PROC_SAMPLES.exists():
        pytest.skip(f"{PROC_SAMPLES} not found")
    from src.data.loader import load_jsonl
    return load_jsonl(str(PROC_SAMPLES))


# ── Test 1: vertex labels in raw pairs ───────────────────────────────────────

def test_raw_pairs_vertex_labels(raw_pairs):
    """Every pair in tot.jsonl must have the expected multiclass_label and negation values."""
    errors = []
    for p in raw_pairs:
        b, cf = p.base, p.counterfactual
        ds = b.descriptors.get("dataset_name", "")
        edge = _edge_type(ds)
        if not edge:
            errors.append(f"Unknown edge type for dataset_name={ds!r} (id={b.id})")
            continue

        exp_b_ml, exp_cf_ml = _EDGE_ML[edge]

        # multiclass_label
        if b.descriptors.get("multiclass_label") != exp_b_ml:
            errors.append(
                f"{b.id}: expected multiclass_label={exp_b_ml}, "
                f"got {b.descriptors.get('multiclass_label')}"
            )
        if cf.descriptors.get("multiclass_label") != exp_cf_ml:
            errors.append(
                f"{cf.id}: expected multiclass_label={exp_cf_ml}, "
                f"got {cf.descriptors.get('multiclass_label')}"
            )

        # negation
        if edge in _NA_EDGES:
            # NA: base is negated, cf is affirmed
            if b.descriptors.get("negation") is not True:
                errors.append(f"{b.id}: NA base should have negation=True")
            if cf.descriptors.get("negation") is not False:
                errors.append(f"{cf.id}: NA cf should have negation=False")
        elif edge.endswith("_aff"):
            if b.descriptors.get("negation") is not False:
                errors.append(f"{b.id}: _aff base should have negation=False")
            if cf.descriptors.get("negation") is not False:
                errors.append(f"{cf.id}: _aff cf should have negation=False")
        elif edge.endswith("_neg"):
            if b.descriptors.get("negation") is not True:
                errors.append(f"{b.id}: _neg base should have negation=True")
            if cf.descriptors.get("negation") is not True:
                errors.append(f"{cf.id}: _neg cf should have negation=True")

    assert not errors, "\n".join(errors[:20])


# ── Test 2: vertex text consistency within each prism ────────────────────────

def test_vertex_text_consistency(raw_pairs):
    """Within each prism group, the shared vertex texts must be identical across edges."""
    # group_id -> edge_type -> {base_text, cf_text}
    by_group: dict[str, dict[str, dict]] = defaultdict(dict)
    for p in raw_pairs:
        gid  = p.base.descriptors.get("group_id", "")
        edge = _edge_type(p.base.descriptors.get("dataset_name", ""))
        if not gid or not edge:
            continue
        by_group[gid][edge] = {"base": p.base.text, "cf": p.counterfactual.text}

    errors = []
    for gid, edges in by_group.items():
        # Affirmed triangle: T/aff shared by tf_aff.base and tn_aff.base
        if "tf_aff" in edges and "tn_aff" in edges:
            if edges["tf_aff"]["base"] != edges["tn_aff"]["base"]:
                errors.append(f"{gid}: T/aff text differs between tf_aff.base and tn_aff.base")
        # F/aff shared by tf_aff.cf and fn_aff.base
        if "tf_aff" in edges and "fn_aff" in edges:
            if edges["tf_aff"]["cf"] != edges["fn_aff"]["base"]:
                errors.append(f"{gid}: F/aff text differs between tf_aff.cf and fn_aff.base")
        # N/aff shared by tn_aff.cf and fn_aff.cf
        if "tn_aff" in edges and "fn_aff" in edges:
            if edges["tn_aff"]["cf"] != edges["fn_aff"]["cf"]:
                errors.append(f"{gid}: N/aff text differs between tn_aff.cf and fn_aff.cf")

        # Negated triangle: T/neg shared by tf_neg.base and tn_neg.base
        if "tf_neg" in edges and "tn_neg" in edges:
            if edges["tf_neg"]["base"] != edges["tn_neg"]["base"]:
                errors.append(f"{gid}: T/neg text differs between tf_neg.base and tn_neg.base")
        # F/neg shared by tf_neg.cf and fn_neg.base
        if "tf_neg" in edges and "fn_neg" in edges:
            if edges["tf_neg"]["cf"] != edges["fn_neg"]["base"]:
                errors.append(f"{gid}: F/neg text differs between tf_neg.cf and fn_neg.base")
        # N/neg shared by tn_neg.cf and fn_neg.cf
        if "tn_neg" in edges and "fn_neg" in edges:
            if edges["tn_neg"]["cf"] != edges["fn_neg"]["cf"]:
                errors.append(f"{gid}: N/neg text differs between tn_neg.cf and fn_neg.cf")

    assert not errors, "\n".join(errors[:20])


# ── Test 3: compound negation filtering ──────────────────────────────────────

def test_compound_negation_filtering(proc_samples):
    """Processed samples: _aff datasets must have negation=False, _neg must have negation=True."""
    errors = []
    for s in proc_samples:
        ds = s.descriptors.get("dataset_name", "")
        neg = s.descriptors.get("negation")
        edge = _edge_type(ds)
        if not edge:
            continue
        if edge in _NA_EDGES:
            continue  # NA edges have mixed negation — skip
        if edge.endswith("_aff") and neg is not False:
            errors.append(f"{s.id}: _aff sample has negation={neg!r}")
        elif edge.endswith("_neg") and neg is not True:
            errors.append(f"{s.id}: _neg sample has negation={neg!r}")

    assert not errors, "\n".join(errors[:20])


# ── Test 4: test-split group completeness ────────────────────────────────────

_AFF_EDGES = {"tf_aff", "tn_aff", "fn_aff"}
_NEG_EDGES = {"tf_neg", "tn_neg", "fn_neg"}


def _check_group_completeness(manifest_path: Path, proc_samples_path: Path,
                              expected_edges: set[str]) -> list[str]:
    if not manifest_path.exists():
        pytest.skip(f"{manifest_path} not found")
    if not proc_samples_path.exists():
        pytest.skip(f"{proc_samples_path} not found")

    from src.data.loader import load_jsonl
    samples = load_jsonl(str(proc_samples_path))
    id_to_sample = {s.id: s for s in samples}

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Deduplicate: only one error per (layer, train_position) combination.
    seen: set[tuple] = set()
    errors = []
    for entry in manifest["entries"]:
        key = (entry.get("layer"), entry.get("train_position"))
        if key in seen:
            continue
        seen.add(key)

        test_ids = set(entry["test_ids"])
        # group_id -> set of edge types present in test set
        groups: dict[str, set[str]] = defaultdict(set)
        for tid in test_ids:
            s = id_to_sample.get(tid)
            if s is None:
                continue
            gid  = s.descriptors.get("group_id", "")
            edge = _edge_type(s.descriptors.get("dataset_name", ""))
            if gid and edge in expected_edges:
                groups[gid].add(edge)

        incomplete = {
            gid: edges
            for gid, edges in groups.items()
            if edges != expected_edges
        }
        if incomplete:
            layer = entry.get("layer")
            pos   = entry.get("train_position")
            n     = len(incomplete)
            example_gid, example_edges = next(iter(incomplete.items()))
            errors.append(
                f"layer={layer} pos={pos}: {n} incomplete prism group(s); "
                f"e.g. {example_gid} has only {sorted(example_edges)} "
                f"(expected {sorted(expected_edges)})"
            )
    return errors


def _tv_runs() -> list[tuple[str, set[str]]]:
    """Every configured ToT true-vs-* compound run, with the edges its groups must carry."""
    import yaml
    params = yaml.safe_load((ROOT / "params.yaml").read_text())
    runs = []
    for run, cfg in sorted((params.get("run_compounds") or {}).items()):
        compound = cfg.get("compound", "")
        if cfg.get("family") != "tot" or "_tv_" not in compound:
            continue
        if compound.endswith("_aff"):
            runs.append((run, _AFF_EDGES))
        elif compound.endswith("_neg"):
            runs.append((run, _NEG_EDGES))
    return runs


@pytest.mark.parametrize("run,edges", _tv_runs(), ids=[r for r, _ in _tv_runs()])
def test_test_split_group_completeness(run, edges):
    """All prism groups in a true-vs-* run's test set must have all 3 edges."""
    manifest = ROOT / "data" / "probes" / run / "manifest.json"
    errors = _check_group_completeness(manifest, PROC_SAMPLES, edges)
    assert not errors, "\n".join(errors)
