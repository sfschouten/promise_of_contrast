"""Which examples each dataset's probes were trained and evaluated on.

The pwl profile splits pairs with `split_strategy: ordered` -- a prefix of the generator's
order -- so the split does not depend on the probe seed or the model, and can be rebuilt
exactly from `data/processed/samples_pwl.jsonl` with the same functions `train_probes`
uses.  Writing it out does two things:

  * makes the pool sizes explicit next to the tables they qualify;
  * publishes the example ids, so a later reproduction can evaluate on exactly these
    examples instead of a same-sized resample.

`check_against_evals` confirms the rebuilt split against the evaluation database, so the
ids written here are the ones the reported numbers were actually computed on.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _common import DATASETS, PROFILE, ROOT
from src.data.loader import load_jsonl
from src.probing.base import _extract_pairs, _split_pairs


def _profile() -> dict:
    params = yaml.safe_load((ROOT / "params.yaml").read_text())
    return {**params.get("probing", {}), **params["probing_profiles"][PROFILE]}


def pool_ids(samples_path: Path | None = None) -> pd.DataFrame:
    """One row per pair: dataset, split, pair id and the ids of both framings."""
    pp = _profile()
    samples = load_jsonl(samples_path or ROOT / "data" / "processed" / f"samples_{PROFILE}.jsonl")
    by_dataset: dict[str, list] = {}
    for s in samples:
        by_dataset.setdefault(s.descriptors.get("dataset_name", ""), []).append(s)

    rows = []
    for ds in [d for d in DATASETS if d in by_dataset] + sorted(set(by_dataset) - set(DATASETS)):
        pairs = _extract_pairs(by_dataset[ds])
        if not pairs:
            continue
        train, test = _split_pairs(pairs, pp.get("train_split", 0.8), pp.get("seed", 42),
                                   pp.get("split_strategy", "random"))
        for split, chunk in (("train", train), ("eval", test)):
            rows += [{"dataset": ds, "split": split,
                      "pair_id": base.descriptors["pair_id"],
                      "base_id": base.id, "counterfactual_id": cf.id}
                     for base, cf in chunk]
    return pd.DataFrame(rows)


def pool_sizes(ids: pd.DataFrame) -> pd.DataFrame:
    """Pairs per dataset and split."""
    counts = ids.groupby(["dataset", "split"]).size().unstack(fill_value=0)
    return pd.DataFrame([{"dataset": ds,
                          "pairs": int(counts.loc[ds].sum()),
                          "train pairs": int(counts.loc[ds, "train"]),
                          "eval pairs": int(counts.loc[ds, "eval"])}
                         for ds in DATASETS if ds in counts.index])


def check_against_evals(sizes: pd.DataFrame, evals: pd.DataFrame) -> list[str]:
    """Datasets whose rebuilt eval pool differs from the n_eval the probes reported.

    `n_eval` counts samples, two per pair.  An empty list means every published id set is
    exactly the one the reported numbers were computed on.
    """
    reported = (evals.groupby("dataset")["n_eval"].max() // 2).to_dict()
    return [f"{r['dataset']}: rebuilt {r['eval pairs']} eval pairs, evaluated on "
            f"{reported[r['dataset']]}"
            for _, r in sizes.iterrows()
            if r["dataset"] in reported and reported[r["dataset"]] != r["eval pairs"]]
