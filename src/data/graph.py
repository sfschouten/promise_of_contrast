from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .loader import Sample

if TYPE_CHECKING:
    from .contrastive import ContrastivePair


@dataclass
class GraphEdge:
    """Directed edge connecting two graph nodes as a contrastive pair."""
    type: str           # semantic edge type, e.g. "tf_aff", "na_t"
    base_node_id: str   # key into Graph.nodes
    cf_node_id: str     # key into Graph.nodes
    base_label: int     # binary label for base side in this pairing
    cf_label: int       # binary label for cf side in this pairing
    dataset_name: str   # e.g. "tot_city_tf_aff"
    descriptors: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "base_node_id": self.base_node_id,
            "cf_node_id": self.cf_node_id,
            "base_label": self.base_label,
            "cf_label": self.cf_label,
            "dataset_name": self.dataset_name,
            "descriptors": self.descriptors,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GraphEdge":
        return cls(
            type=d["type"],
            base_node_id=d["base_node_id"],
            cf_node_id=d["cf_node_id"],
            base_label=d["base_label"],
            cf_label=d["cf_label"],
            dataset_name=d["dataset_name"],
            descriptors=d.get("descriptors", {}),
        )


@dataclass
class Graph:
    """
    A labelled directed graph of Sample nodes connected by typed contrastive edges.

    For the ToT prism: 6 nodes (T+, T-, F+, F-, N+, N-) and 9 typed edges.
    Node keys within the graph are short semantic names (e.g. "t_aff");
    Sample.id is globally unique across all graphs.
    """
    id: str
    nodes: dict[str, Sample]        # local node key → Sample
    edges: list[GraphEdge]
    descriptors: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "nodes": {k: s.to_dict() for k, s in self.nodes.items()},
            "edges": [e.to_dict() for e in self.edges],
            "descriptors": self.descriptors,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Graph":
        return cls(
            id=d["id"],
            nodes={k: Sample.from_dict(sd) for k, sd in d["nodes"].items()},
            edges=[GraphEdge.from_dict(ed) for ed in d["edges"]],
            descriptors=d.get("descriptors", {}),
        )


# ── I/O ───────────────────────────────────────────────────────────────────────

def save_graphs_jsonl(graphs: list[Graph], path: str) -> None:
    with open(path, "w") as f:
        for g in graphs:
            f.write(json.dumps(g.to_dict()) + "\n")


def load_graphs_jsonl(path: str) -> list[Graph]:
    with open(path) as f:
        return [Graph.from_dict(json.loads(line)) for line in f if line.strip()]


def is_graphs_file(path: str) -> bool:
    """Peek at the first non-empty line to determine whether a JSONL file contains Graphs."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                return "nodes" in d and "edges" in d
    return False


# ── Graph utilities ────────────────────────────────────────────────────────────

def graphs_to_nodes(graphs: list[Graph]) -> list[Sample]:
    """Deduplicated unique nodes across all graphs, preserving first-seen order."""
    seen: set[str] = set()
    out: list[Sample] = []
    for g in graphs:
        for sample in g.nodes.values():
            if sample.id not in seen:
                seen.add(sample.id)
                out.append(sample)
    return out


def materialize_pairs(
    graphs: list[Graph],
    edge_types: list[str] | None = None,
) -> list["ContrastivePair"]:
    """
    Flatten graph edges into ContrastivePair objects.

    edge_types filters by GraphEdge.type; None includes all edges.
    Binary labels and dataset_name are stamped onto materialized Sample copies
    from the edge; intrinsic node descriptors (multiclass_label, group_id, etc.)
    are preserved from the node.
    """
    from .contrastive import ContrastivePair
    pairs = []
    for g in graphs:
        for edge in g.edges:
            if edge_types is not None and edge.type not in edge_types:
                continue
            base = deepcopy(g.nodes[edge.base_node_id])
            cf = deepcopy(g.nodes[edge.cf_node_id])
            base.descriptors["label"] = edge.base_label
            base.descriptors["dataset_name"] = edge.dataset_name
            cf.descriptors["label"] = edge.cf_label
            cf.descriptors["dataset_name"] = edge.dataset_name
            for k, v in edge.descriptors.items():
                base.descriptors.setdefault(k, v)
                cf.descriptors.setdefault(k, v)
            pairs.append(ContrastivePair(base=base, counterfactual=cf))
    return pairs


def load_family_as_pairs(
    family_name: str,
    base_dir: str = "data/processed",
) -> list[Sample] | None:
    """
    For graph families: loads templated graphs, materializes edges into
    ContrastivePairs, and returns flattened samples with pair_id/pair_role stamped.
    Returns None for non-graph families (caller should use load_jsonl instead).
    """
    graphs_path = Path(base_dir) / f"graphs_{family_name}.jsonl"
    if not graphs_path.exists() or not is_graphs_file(str(graphs_path)):
        return None
    from .contrastive import pairs_to_samples
    graphs = load_graphs_jsonl(str(graphs_path))
    pairs = materialize_pairs(graphs)
    return pairs_to_samples(pairs)
