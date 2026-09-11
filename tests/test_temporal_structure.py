"""Structural tests for the temporal (days/months) modular-arithmetic graphs.

Self-contained: builds graphs directly from the builder, so no pipeline run or
model is required.

Run from the project root:
    pytest tests/test_temporal_structure.py -v
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.graph import materialize_pairs
from src.data.sources.temporal import build_modular_arithmetic_graphs, build_prediction_graphs

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
OFFSETS = [1, 2, 3, 4, 5, 6]
STEM = "Let's do some day of the week math. {offset_word_cap} {units} from {start} is"


@pytest.fixture(scope="module")
def graphs():
    return build_modular_arithmetic_graphs(
        DAYS, dataset_name="days_of_week", unit="day",
        offsets=OFFSETS, stem_template=STEM, id_prefix="dow",
    )


@pytest.fixture(scope="module")
def pred_graphs():
    return build_prediction_graphs(
        DAYS, dataset_name="days_of_week", unit="day",
        offsets=OFFSETS, stem_template=STEM, id_prefix="dowp",
    )


def test_graph_and_node_counts(graphs):
    n = len(DAYS)
    assert len(graphs) == n * 6  # 7 starts × 6 offsets
    n_circular = n * (n - 1) // 2  # complete graph K_n
    n_truth = n - 1                # correct-incident subset
    for g in graphs:
        assert len(g.nodes) == n
        assert len(g.edges) == n_circular + n_truth
        assert sum(e.type == "circular" for e in g.edges) == n_circular
        assert sum(e.type == "truth" for e in g.edges) == n_truth


def test_modular_arithmetic_is_correct(graphs):
    n = len(DAYS)
    for g in graphs:
        any_node = next(iter(g.nodes.values()))
        start_index = any_node.descriptors["start_index"]
        offset = any_node.descriptors["offset"]
        assert g.descriptors["correct_index"] == (start_index + offset) % n


def test_exactly_one_correct_node_per_graph(graphs):
    for g in graphs:
        correct = [k for k, s in g.nodes.items() if s.descriptors["is_correct"] == 1]
        assert len(correct) == 1


def test_edge_typing_and_truth_orientation(graphs):
    for g in graphs:
        ci = g.descriptors["correct_index"]
        for e in g.edges:
            i = int(e.base_node_id[1:])
            j = int(e.cf_node_id[1:])
            touches_correct = ci in (i, j)
            if e.type == "truth":
                assert touches_correct
                # base is always the correct (label-1) node, cf the incorrect one
                assert g.nodes[e.base_node_id].descriptors["is_correct"] == 1
                assert g.nodes[e.cf_node_id].descriptors["is_correct"] == 0
                assert (e.base_label, e.cf_label) == (1, 0)
                assert e.dataset_name == "days_of_week_truth"
            else:
                # circular edges span the whole complete graph (correct-incident
                # pairs included), so no touches_correct constraint here.
                assert e.type == "circular"
                assert e.dataset_name == "days_of_week_circular"


def test_circular_is_complete_graph_truth_is_correct_incident_subset(graphs):
    n = len(DAYS)
    all_pairs = {frozenset(p) for p in combinations(range(n), 2)}
    for g in graphs:
        ci = g.descriptors["correct_index"]
        circ = {frozenset((int(e.base_node_id[1:]), int(e.cf_node_id[1:])))
                for e in g.edges if e.type == "circular"}
        truth = {frozenset((int(e.base_node_id[1:]), int(e.cf_node_id[1:])))
                 for e in g.edges if e.type == "truth"}
        assert circ == all_pairs                              # circular = K_n
        assert truth == {p for p in all_pairs if ci in p}     # truth = correct-incident
        assert truth <= circ                                  # truth ⊆ circular


def test_node_labels_and_masks(graphs):
    for g in graphs:
        for s in g.nodes.values():
            # multiclass_label is the answer's circular position
            assert s.descriptors["multiclass_label"] == s.descriptors["answer_index"]
            # masks recover the literal start and answer substrings
            assert [s.text[a:b] for a, b in s.masks["start"]] == [s.descriptors["start"]]
            assert [s.text[a:b] for a, b in s.masks["answer"]] == [s.descriptors["answer"]]


def test_circular_center_key_is_per_element_pair(graphs):
    n = len(DAYS)
    for g in graphs:
        circ = [e for e in g.edges if e.type == "circular"]
        # one centering group per unordered element pair, canonical "i_j" (i<j)
        for e in circ:
            i = int(e.base_node_id[1:])
            j = int(e.cf_node_id[1:])
            assert e.descriptors["center_key"] == f"{min(i, j)}_{max(i, j)}"
        keys = {e.descriptors["center_key"] for e in circ}
        assert len(keys) == n * (n - 1) // 2          # C(n,2) distinct pair types
        # truth edges are deliberately left without center_key → one truth
        # direction per dataset_name, not per element pair.
        assert all("center_key" not in e.descriptors
                   for e in g.edges if e.type == "truth")


def test_center_key_survives_materialization(graphs):
    pairs = materialize_pairs(graphs, edge_types=["circular"])
    assert pairs
    for p in pairs:
        # both sides carry the same pair-type centering key
        assert p.base.descriptors["center_key"] == p.counterfactual.descriptors["center_key"]


def test_prediction_center_key_is_per_start_pair(pred_graphs):
    n = len(DAYS)
    for g in pred_graphs:
        keys = {e.descriptors["center_key"] for e in g.edges}
        assert len(keys) == n * (n - 1) // 2
        for e in g.edges:
            i, j = int(e.base_node_id[1:]), int(e.cf_node_id[1:])
            assert e.descriptors["center_key"] == f"{min(i, j)}_{max(i, j)}"


def test_materialized_pairs_have_binary_truth_labels(graphs):
    pairs = materialize_pairs(graphs, edge_types=["truth"])
    assert pairs, "expected truth pairs"
    for p in pairs:
        assert p.base.descriptors["label"] == 1
        assert p.counterfactual.descriptors["label"] == 0


# ── prediction variant (no answer; probe the final "is" token) ──────────────────

def test_prediction_graph_shape(pred_graphs):
    n = len(DAYS)
    assert len(pred_graphs) == len(OFFSETS)        # one complete graph per offset
    for g in pred_graphs:
        assert len(g.nodes) == n                   # nodes vary by start element
        assert len(g.edges) == n * (n - 1) // 2    # complete graph, circular only
        assert all(e.type == "prediction" for e in g.edges)
        assert all(e.dataset_name == "days_of_week_pred" for e in g.edges)


def test_prediction_stem_has_no_answer(pred_graphs):
    for g in pred_graphs:
        for s in g.nodes.values():
            assert s.text.endswith(" is")          # ends at the operator, no answer
            # mask marks the (single) start occurrence
            assert [s.text[a:b] for a, b in s.masks["start"]] == [s.descriptors["start"]]


def test_prediction_label_is_the_predicted_element(pred_graphs):
    n = len(DAYS)
    for g in pred_graphs:
        for s in g.nodes.values():
            exp = (s.descriptors["start_index"] + s.descriptors["offset"]) % n
            assert s.descriptors["multiclass_label"] == exp
            assert s.descriptors["target"] == DAYS[exp]
        # every circle position is predicted exactly once within a graph
        preds = sorted(s.descriptors["multiclass_label"] for s in g.nodes.values())
        assert preds == list(range(n))
