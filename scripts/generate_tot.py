"""Generate Graph objects from the Trilemma of Truth (ToT) dataset.

For each entity (object_1, relation) one Graph is generated whose 6 nodes form
a triangular prism (K₃ □ K₂) and whose 9 directed edges encode contrastive pairs:

  Nodes:  T+, T-, F+, F-, N+, N-
  Edges:
    Affirmed triangle (+): TF+, TN+, FN+
    Negated triangle  (-): TF-, TN-, FN-
    Cross-negation:        NA_T (T-↔T+), NA_F (F-↔F+), NA_N (N-↔N+)

**Vertex classification** is based on entity class, not factual truth:
  T vertex: (obj1, obj2) whose AFFIRMED form is factually true.
            T+ = "X is in Y" (is_true=True, neg=False)
            T- = "X is NOT in Y" (negated, factually false)
  F vertex: whose AFFIRMED form is factually false.
            F+ = "X is in Z" (is_false=True, neg=False)
            F- = "X is NOT in Z" (negated, factually true)

**multiclass_label** on nodes always reflects entity class (T=1, F=0, N=2).
**Binary label** is an edge property, not a node property:
  TF/TN/FN: base=entity-class-winner(1), cf=loser(0)  →  T>F>N
  NA_*:     base=negated(0),             cf=affirmed(1)

Negated forms are synthesised via " is " → " is not " (verified 100% match).
The N vertex is constructed synthetically from neither-class rows.

HuggingFace: carlomarxx/trilemma-of-truth
Sub-datasets: city_locations, med_indications, word_definitions
"""
from __future__ import annotations

import random
import sys
from collections import defaultdict
from pathlib import Path

import yaml
from datasets import concatenate_datasets, load_dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.graph import Graph, GraphEdge, save_graphs_jsonl
from src.data.loader import Sample

SEED = 42

_SUB_SHORT = {
    "city_locations":   "city",
    "med_indications":  "med",
    "word_definitions": "word",
}


def _relation_key(statement: str) -> str:
    s = statement.lower()
    if "a synonym of" in s:
        return "synonym"
    if "a type of" in s:
        return "typeOf"
    return "instanceOf"


def _node(row: dict, vertex: str, graph_id: str, ml: int) -> Sample:
    """Build one graph node; binary label is edge-level and not stored here."""
    return Sample(
        id=f"{graph_id}_{vertex}",
        text=row["statement"],
        descriptors={
            "multiclass_label": ml,
            "negation": bool(row["negation"]),
            "group_id": graph_id,
        },
        metadata={
            "object_1": row["object_1"],
            "object_2": row["object_2"],
            "relation": _relation_key(row["statement"]),
        },
    )


def _negate(statement: str) -> str:
    return statement.replace(" is ", " is not ", 1)


def _collect_synthetic_obj2(neither_rows: list[dict]) -> dict[str, list[str]]:
    """Pool of synthetic object_2 values from affirmed neither rows, keyed by relation."""
    pool: dict[str, list[str]] = defaultdict(list)
    for r in neither_rows:
        if r["object_2"] is None:
            continue
        pool[_relation_key(r["statement"])].append(r["object_2"])
    return pool


def _gen_prisms(
    rows: list[dict],
    short: str,
    rng: random.Random,
    syn_pool: dict[str, list[str]],
    max_entities: int | None,
) -> list[Graph]:
    """Build one Graph per entity; each Graph has 6 nodes and 9 directed edges."""
    # Collect affirmed real rows by (obj1, rel).
    t_aff_by_entity: dict[tuple, list[dict]] = defaultdict(list)
    f_aff_by_entity: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        if r["object_1"] is None or r["object_2"] is None or not r["real_object"]:
            continue
        if r["negation"]:
            continue
        entity_rel = (r["object_1"], _relation_key(r["statement"]))
        if r["is_true"]:
            t_aff_by_entity[entity_rel].append(r)
        elif r["is_false"]:
            f_aff_by_entity[entity_rel].append(r)

    # Build entity list: one (T/aff row, F/aff row, syn_obj2) triple per entity.
    entities: list[tuple] = []
    for entity_rel in sorted(set(t_aff_by_entity) & set(f_aff_by_entity)):
        obj1, rel = entity_rel
        if rel not in syn_pool:
            continue
        syn_list = list(syn_pool[rel])
        rng.shuffle(syn_list)
        for ta, fa, syn_obj2 in zip(
            t_aff_by_entity[entity_rel], f_aff_by_entity[entity_rel], syn_list
        ):
            entities.append((ta, fa, syn_obj2))

    if max_entities is not None and len(entities) > max_entities:
        rng.shuffle(entities)
        entities = entities[:max_entities]

    graphs: list[Graph] = []
    n_dropped_internal_period = 0
    for i, (ta, fa, syn_obj2) in enumerate(entities):
        pid = f"tot_{short}_prism_{i}"

        # Synthesise negated forms.
        tn_row = {**ta, "statement": _negate(ta["statement"]), "negation": True,
                  "is_true": False, "is_false": True, "multiclass_label": 1}
        fn_row = {**fa, "statement": _negate(fa["statement"]), "negation": True,
                  "is_true": True,  "is_false": False, "multiclass_label": 0}

        # Synthetic N rows — same syn_obj2 for both + and -.
        n_aff_row = {
            "statement":  ta["statement"].replace(ta["object_2"], syn_obj2, 1),
            "object_1":   ta["object_1"], "object_2": syn_obj2,
            "negation":   False, "multiclass_label": 2,
            "is_true": False, "is_false": False, "is_neither": True, "real_object": False,
        }
        n_neg_row = {
            "statement":  _negate(ta["statement"]).replace(ta["object_2"], syn_obj2, 1),
            "object_1":   ta["object_1"], "object_2": syn_obj2,
            "negation":   True, "multiclass_label": 2,
            "is_true": False, "is_false": False, "is_neither": True, "real_object": False,
        }

        nodes = {
            "t_aff": _node(ta,        "t_aff", pid, 1),
            "t_neg": _node(tn_row,    "t_neg", pid, 1),
            "f_aff": _node(fa,        "f_aff", pid, 0),
            "f_neg": _node(fn_row,    "f_neg", pid, 0),
            "n_aff": _node(n_aff_row, "n_aff", pid, 2),
            "n_neg": _node(n_neg_row, "n_neg", pid, 2),
        }

        # Drop the whole prism if any statement carries a period before its final one.
        #
        # The period-token analyses read the last token on the assumption that it is the
        # sentence delimiter.  Llama-2-7B places a massive activation on the *first* period
        # of a sequence, so a statement with an earlier period ("... the U.S. Virgin
        # Islands.", or the upstream-malformed "Sydenham is a dr..") leaves its final
        # period without that component -- roughly a hundredth of the usual norm.  After
        # centering those samples sit ~100 sigma from the rest and dominate the
        # cross-covariance: excluding 4 of 1500 prisms moves the leading tuple-contrastive
        # eigenvector by |cos| = 0.38.  The criterion is a property of the input, not of
        # the fitted result, and is applied to both models so they stay comparable.
        if any("." in n.text[:-1] for n in nodes.values()):
            n_dropped_internal_period += 1
            continue

        def ds(k: str) -> str:
            return f"tot_{short}_{k}"

        edges = [
            # Affirmed triangle: base=entity-class-winner(1), cf=loser(0)
            GraphEdge("tf_aff", "t_aff", "f_aff", 1, 0, ds("tf_aff")),
            GraphEdge("tn_aff", "t_aff", "n_aff", 1, 0, ds("tn_aff")),
            GraphEdge("fn_aff", "f_aff", "n_aff", 1, 0, ds("fn_aff")),
            # Negated triangle
            GraphEdge("tf_neg", "t_neg", "f_neg", 1, 0, ds("tf_neg")),
            GraphEdge("tn_neg", "t_neg", "n_neg", 1, 0, ds("tn_neg")),
            GraphEdge("fn_neg", "f_neg", "n_neg", 1, 0, ds("fn_neg")),
            # Cross-negation: base=negated(0), cf=affirmed(1)
            GraphEdge("na_t", "t_neg", "t_aff", 0, 1, ds("na_t")),
            GraphEdge("na_f", "f_neg", "f_aff", 0, 1, ds("na_f")),
            GraphEdge("na_n", "n_neg", "n_aff", 0, 1, ds("na_n")),
        ]

        graphs.append(Graph(
            id=pid,
            nodes=nodes,
            edges=edges,
            descriptors={"sub_dataset": short},
        ))

    if n_dropped_internal_period:
        print(f"  dropped {n_dropped_internal_period} prisms whose statements "
              f"contain a period before the final one")
    return graphs


def _load_sub_dataset(hf_name: str, splits: list[str], source: str) -> list[dict]:
    dsets = [load_dataset(source, name=hf_name, split=s) for s in splits]
    return [dict(r) for r in concatenate_datasets(dsets)]


def _gen_sub_dataset(
    hf_name: str, short: str, splits: list[str], rng: random.Random,
    max_entities: int | None, source: str = "carlomarxx/trilemma-of-truth",
) -> list[Graph]:
    rows = _load_sub_dataset(hf_name, splits, source)

    aff_neithers = [r for r in rows if r["is_neither"] and not r["negation"]]
    syn_pool = _collect_synthetic_obj2(aff_neithers)

    graphs = _gen_prisms(rows, short, rng, syn_pool, max_entities)
    print(f"  {len(graphs)} prisms × 9 edges = {len(graphs) * 9} edge instances")
    return graphs


def main() -> None:
    with open(ROOT / "params.yaml") as f:
        params = yaml.safe_load(f)

    cfg           = params["dataset"]["tot"]
    fam_cfg       = params.get("families", {}).get("tot", {})
    sub_datasets  = cfg.get("sub_datasets", list(_SUB_SHORT.keys()))
    splits        = cfg.get("splits", ["train", "validation", "test"])
    max_entities  = fam_cfg.get("max_pairs_per_type")
    out_path      = ROOT / "data" / "raw" / "tot.jsonl"
    rng           = random.Random(SEED)
    source        = cfg.get("data_dir") or cfg.get("hf_repo", "carlomarxx/trilemma-of-truth")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_graphs: list[Graph] = []
    for hf_name in sub_datasets:
        short = _SUB_SHORT.get(hf_name, hf_name)
        print(f"{hf_name}:", flush=True)
        all_graphs.extend(_gen_sub_dataset(hf_name, short, splits, rng, max_entities, source))

    save_graphs_jsonl(all_graphs, str(out_path))
    print(f"\nSaved {len(all_graphs)} graphs → {out_path}")


if __name__ == "__main__":
    main()
