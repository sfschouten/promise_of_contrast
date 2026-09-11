"""
Constructs ContrastivePair objects for all configured dataset families.

Reads `families` from params.yaml.  For each family:
  - If the raw data file is already in pairs format (e.g. CCS), load it directly.
  - Otherwise load flat samples and apply the family's pairing strategy list.

Sampling is applied here — before template application and activation extraction.
Set `families.<name>.max_pairs` to cap how many pairs enter the rest of the pipeline.
Pairs are shuffled with `preprocessing.seed` before truncation for reproducibility.

Two pairing strategies are supported:
  by_descriptor   Group samples by a descriptor key and pair base_label against
                  cf_label within each group.
  label_flip      Programmatically create a counterfactual by flipping a label.

Each pair has `_family` stamped into sample metadata so preprocess.py can apply
the correct prompt template per family.

Output: data/processed/pairs_<family>.jsonl
"""

import argparse
import random
import sys
from copy import deepcopy
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.contrastive import (
    is_pairs_file,
    label_flip,
    load_pairs_jsonl,
    make_pairs,
    pair_by_descriptor,
    save_pairs_jsonl,
)
from src.data.graph import is_graphs_file, load_graphs_jsonl, save_graphs_jsonl
from src.data.loader import load_jsonl, save_jsonl


def _run_pairing(samples: list, cfg: dict, seed: int = 42) -> list:
    strategy = cfg["strategy"]
    dataset_name = cfg.get("dataset_name")

    if strategy == "by_descriptor":
        pairs = pair_by_descriptor(
            samples,
            key=cfg["key"],
            base_label=cfg.get("base_label", 1),
            cf_label=cfg.get("cf_label", 0),
            label_descriptor=cfg.get("label_descriptor", "label"),
            dataset_name=dataset_name,
        )
        key_repr = cfg["key"] if isinstance(cfg["key"], str) else str(cfg["key"])
        print(f"    by_descriptor key={key_repr!r} dataset={dataset_name!r} → {len(pairs)} pairs")

    elif strategy == "label_flip":
        descriptor = cfg.get("descriptor", "label")
        subset = [s for s in samples if s.descriptors.get("dataset_name") == dataset_name] \
                 if dataset_name else samples
        pairs = make_pairs(subset, lambda s: label_flip(s, descriptor))
        print(f"    label_flip descriptor={descriptor!r} dataset={dataset_name!r} → {len(pairs)} pairs")

    else:
        raise ValueError(
            f"Unknown pairing strategy: {strategy!r}. "
            "Expected 'by_descriptor' or 'label_flip'."
        )

    max_pairs = cfg.get("max_pairs")
    if max_pairs is not None and len(pairs) > max_pairs:
        random.Random(seed).shuffle(pairs)
        pairs = pairs[:max_pairs]
        print(f"      → sampled {max_pairs}")

    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True)
    args = parser.parse_args()

    params = yaml.safe_load(Path("params.yaml").read_text())
    families = params["families"]
    seed = params["preprocessing"]["seed"]

    if args.family not in families:
        raise SystemExit(f"Unknown family {args.family!r}. Known: {list(families)}")

    family_name = args.family
    fcfg = families[family_name]
    path = fcfg["path"]
    pairings_cfgs = fcfg.get("pairings", [])
    max_pairs = fcfg.get("max_pairs")

    print(f"Family: {family_name}  ({path})")

    if not Path(path).exists():
        raise SystemExit(f"Raw data file not found: {path}")

    max_per_type = fcfg.get("max_pairs_per_type")

    out = Path(f"data/processed/pairs_{family_name}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    if is_graphs_file(path):
        graphs = load_graphs_jsonl(path)
        print(f"  Loaded {len(graphs)} graphs (graph file)")
        if max_per_type is not None:
            by_sub: dict[str, list] = {}
            for g in graphs:
                sub = g.descriptors.get("sub_dataset", g.id)
                by_sub.setdefault(sub, []).append(g)
            rng = random.Random(seed)
            graphs = []
            for sub, sub_graphs in sorted(by_sub.items()):
                if len(sub_graphs) > max_per_type:
                    rng.shuffle(sub_graphs)
                    sub_graphs = sub_graphs[:max_per_type]
                print(f"    {sub}: {len(sub_graphs)} graphs")
                graphs.extend(sub_graphs)
        if max_pairs is not None and len(graphs) > max_pairs:
            rng = random.Random(seed)
            rng.shuffle(graphs)
            graphs = graphs[:max_pairs]
            print(f"  Sampled {max_pairs} graphs total (seed={seed})")
        # Cap the number of materialised pairs per edge dataset_name (the edges
        # become ContrastivePairs downstream).  Useful for complete-graph
        # families where one edge type explodes combinatorially, e.g. the
        # incorrect×incorrect "circular" pairs of the temporal datasets.
        edge_caps = fcfg.get("max_pairs_per_dataset") or {}
        if edge_caps:
            edge_locs: dict[str, list[tuple[int, int]]] = {}
            for gi, g in enumerate(graphs):
                for ei, e in enumerate(g.edges):
                    edge_locs.setdefault(e.dataset_name, []).append((gi, ei))
            drop: set[tuple[int, int]] = set()
            rng = random.Random(seed)
            for ds_name, locs in sorted(edge_locs.items()):
                cap = edge_caps.get(ds_name)
                if cap is None or len(locs) <= cap:
                    continue
                # Stratify the cap across pair-types (edge center_key) so the
                # training set is spread evenly over pair-types rather than
                # dropped uniformly.  Round-robin keep one edge from each
                # shuffled pair-type group until the cap is reached: per-type
                # counts stay within 1 of each other and unequal group sizes are
                # tolerated.  Edges without a center_key fall into a single group
                # (→ the previous uniform behaviour).
                groups: dict[str, list[tuple[int, int]]] = {}
                for gi, ei in locs:
                    key = str(graphs[gi].edges[ei].descriptors.get("center_key", ""))
                    groups.setdefault(key, []).append((gi, ei))
                queues = list(groups.values())
                for q in queues:
                    rng.shuffle(q)
                keep: list[tuple[int, int]] = []
                while len(keep) < cap and any(queues):
                    for q in queues:
                        if q and len(keep) < cap:
                            keep.append(q.pop())
                keep_set = set(keep)
                drop.update(loc for loc in locs if loc not in keep_set)
                print(f"  Capped {ds_name}: {len(locs)} → {len(keep)} pairs "
                      f"across {len(groups)} pair-types (seed={seed})")
            if drop:
                for gi, g in enumerate(graphs):
                    g.edges = [e for ei, e in enumerate(g.edges) if (gi, ei) not in drop]
        for g in graphs:
            for sample in g.nodes.values():
                sample.metadata["_family"] = family_name
        save_graphs_jsonl(graphs, str(out))
        print(f"Saved {len(graphs)} graphs → {out}")
        return
    elif is_pairs_file(path):
        pairs = load_pairs_jsonl(path)
        print(f"  Loaded {len(pairs)} pairs (already paired)")
        if max_per_type is not None:
            by_type: dict[str, list] = {}
            for pair in pairs:
                ds = pair.base.descriptors.get("dataset_name", "")
                by_type.setdefault(ds, []).append(pair)
            rng = random.Random(seed)
            pairs = []
            for ds_name, ds_pairs in by_type.items():
                if len(ds_pairs) > max_per_type:
                    rng.shuffle(ds_pairs)
                    ds_pairs = ds_pairs[:max_per_type]
                print(f"    {ds_name}: {len(ds_pairs)} pairs")
                pairs.extend(ds_pairs)
    elif not pairings_cfgs:
        # Flat passthrough: this family declares no contrastive pairings, so the
        # raw samples flow straight through with no pairs constructed (e.g. the
        # multiclass temporal datasets, which are studied for probe geometry
        # rather than truth contrast).  preprocess.py detects the flat format of
        # this file and templates the samples directly.
        samples = load_jsonl(path)
        print(f"  Loaded {len(samples)} samples (no pairings → flat passthrough)")
        if max_pairs is not None and len(samples) > max_pairs:
            rng = random.Random(seed)
            rng.shuffle(samples)
            samples = samples[:max_pairs]
            print(f"  Sampled {max_pairs} samples (seed={seed})")
        for s in samples:
            s.metadata["_family"] = family_name
        save_jsonl(samples, str(out))
        print(f"Saved {len(samples)} flat samples → {out}")
        return
    else:
        samples = load_jsonl(path)
        print(f"  Loaded {len(samples)} samples")
        pairs = []
        for cfg in pairings_cfgs:
            new_pairs = _run_pairing(samples, cfg, seed=seed)
            rename = cfg.get("rename_dataset")
            if rename:
                for pair in new_pairs:
                    # Deepcopy so we don't mutate Sample objects shared with earlier pairings.
                    pair.base = deepcopy(pair.base)
                    pair.counterfactual = deepcopy(pair.counterfactual)
                    pair.base.descriptors["dataset_name"] = rename
                    pair.counterfactual.descriptors["dataset_name"] = rename
            pairs.extend(new_pairs)

    if max_pairs is not None and len(pairs) > max_pairs:
        rng = random.Random(seed)
        rng.shuffle(pairs)
        pairs = pairs[:max_pairs]
        print(f"  Sampled {max_pairs} pairs (seed={seed})")

    for pair in pairs:
        pair.base.metadata["_family"] = family_name
        pair.counterfactual.metadata["_family"] = family_name

    save_pairs_jsonl(pairs, str(out))
    print(f"Saved {len(pairs)} pairs → {out}")


if __name__ == "__main__":
    main()
