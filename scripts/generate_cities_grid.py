"""
Cities 2x2 contrast grid (negation x content) from the Marks & Tegmark city data.

Each city yields one graph with four nodes -- the <pc, pi, nc, ni> tuple:

    pc  positive + correct    "The city of X is in Y."       true
    pi  positive + incorrect  "The city of X is in Z."       false
    nc  negative + correct    "The city of X is not in Y."   false
    ni  negative + incorrect  "The city of X is not in Z."   true

where Y is the city's real country and Z a wrong one.  Truth is the XOR of the
two factors, so all six unordered pairs are meaningful contrasts:

    content   pc-pi, ni-nc    polarity fixed, country varies   truth flips
    negation  pc-nc, ni-pi    country fixed, polarity varies   truth flips
    both      pc-ni, pi-nc    both vary                        truth preserved

The two "both" edges are the point of the design: they hold truth constant while
changing everything else, so a direction that tracks truth should not separate
them, while one tracking surface form should.  They are also why this cannot be
expressed as a flat pairing -- by_descriptor and label_flip pair a true statement
against a false one by construction.

Node text is taken verbatim from the geometry_of_truth CSVs (statement column),
so it matches the `cities` / `neg_cities` samples and reuses their activations.
Structure follows the SNLI meta-picture (src generate_snli.py) and the Farquhar
2x2 grid; node text is self-contained, so templates/statement.j2 passes it through.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.graph import Graph, GraphEdge, save_graphs_jsonl
from src.data.loader import Sample
from src.data.masks import find_char_spans

DATASET = "cities_grid"

# Node key -> (polarity, content correctness, truth value).
NODES = {
    "pc": ("pos", "correct",   1),
    "pi": ("pos", "incorrect", 0),
    "nc": ("neg", "correct",   0),
    "ni": ("neg", "incorrect", 1),
}

# (edge type, base node, cf node) -- base is the true side wherever truth differs.
EDGES = [
    ("content",  "pc", "pi"),
    ("content",  "ni", "nc"),
    ("negation", "pc", "nc"),
    ("negation", "ni", "pi"),
    ("both",     "pc", "ni"),   # both true
    ("both",     "pi", "nc"),   # both false
]


def _read(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _node(gid: str, key: str, row: dict) -> Sample:
    polarity, content, truth = NODES[key]
    text    = row["statement"].strip()
    city    = row["city"].strip()
    country = row["country"].strip()
    return Sample(
        id=f"{DATASET}_{gid}_{key}",
        text=text,
        descriptors={
            "node":            key,
            # The four nodes of one city are a tuple, not four unrelated points.
            # `group_id` is what keeps them together downstream: the scatter report
            # samples whole groups, so a sampled city contributes all six of its
            # edges instead of a lone connector between two of the four points.
            "group_id":        f"{DATASET}_{gid}",
            "polarity":        polarity,
            "content":         content,
            "truth":           truth,
            "city":            city,
            "country":         country,
            "correct_country": row["correct_country"].strip(),
        },
        masks={
            **({"city":    find_char_spans(text, city)}    if city    else {}),
            **({"country": find_char_spans(text, country)} if country else {}),
        },
        metadata={"negation": polarity == "neg"},
    )


def build_graphs(pos_rows: list[dict], neg_rows: list[dict]) -> list[Graph]:
    # Group both CSVs by (city, country); each city contributes a correct and an
    # incorrect country, and the negated file mirrors the same two rows.
    by_city: dict[str, dict[str, dict]] = defaultdict(dict)
    for rows, prefix in ((pos_rows, "p"), (neg_rows, "n")):
        for row in rows:
            city    = row["city"].strip()
            correct = row["country"].strip() == row["correct_country"].strip()
            by_city[city][f"{prefix}{'c' if correct else 'i'}"] = row

    graphs: list[Graph] = []
    for gi, city in enumerate(sorted(by_city)):
        rows = by_city[city]
        if set(rows) != set(NODES):          # skip cities missing any corner
            continue
        gid   = f"{gi:05d}"
        nodes = {k: _node(gid, k, rows[k]) for k in NODES}
        sd    = {"sub_dataset": DATASET, "city": city,
                 "correct_country": nodes["pc"].descriptors["country"],
                 "incorrect_country": nodes["pi"].descriptors["country"]}
        edges: list[GraphEdge] = []
        for kind, base, cf in EDGES:
            desc = {**sd, "contrast": kind, "edge_kind": kind,
                    "base_node": base, "cf_node": cf}
            b_truth = NODES[base][2]
            c_truth = NODES[cf][2]
            # Labels are the nodes' real truth values: equal on the "both" edges,
            # which is the property under test rather than an accident.
            edges.append(GraphEdge(kind, base, cf, b_truth, c_truth,
                                   f"{DATASET}_{kind}", desc))
            # Same six edges pooled into one dataset, for a single combined page.
            edges.append(GraphEdge(f"all_{kind}", base, cf, b_truth, c_truth,
                                   f"{DATASET}_all", desc))
        graphs.append(Graph(id=f"{DATASET}_{gid}", nodes=nodes, edges=edges,
                            descriptors=sd))
    return graphs


def main() -> None:
    params   = yaml.safe_load((ROOT / "params.yaml").read_text())
    data_dir = Path(params["dataset"][DATASET]["data_dir"])
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir

    graphs = build_graphs(_read(data_dir / "cities.csv"),
                          _read(data_dir / "neg_cities.csv"))

    out_path = ROOT / "data" / "raw" / f"{DATASET}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_graphs_jsonl(graphs, str(out_path))
    print(f"Saved {len(graphs)} graphs ({len(graphs) * len(NODES)} nodes, "
          f"{len(graphs) * len(EDGES)} contrast pairs x2 datasets) -> {out_path}")


if __name__ == "__main__":
    main()
