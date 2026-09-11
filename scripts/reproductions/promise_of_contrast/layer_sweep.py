"""Accuracy against layer, to back the choice of layer 16 (block 15) with evidence.

The paper probes a single layer.  This fits the closed-form
methods (tcPCA-1, g-tcPCA-1), the supervised logistic ceiling, and CCS with a few seeds at
every cached layer, for both token positions, with exactly the pwl protocol: the same
ordered 50/50 split, per-branch centring, sign rule and uncalibrated scoring as
`train_probes` + `evaluate_probes`.  Nothing is written to the pwl probe store; the result
is one CSV per model, which `reproduce.py` turns into `fig_layer_sweep.pdf`.

    python scripts/reproductions/promise_of_contrast/layer_sweep.py --model llama2_7b
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _common import PROFILE, ROOT
from src.activations.extract import extract_hidden_states
from src.config import resolve_probing
from src.data.graph import load_family_as_pairs
from src.data.loader import load_jsonl
from src.probing import REGISTRY
from src.probing.base import eval_probe, fit_probe, fit_probe_batched
from src.probing.paper_eval import unsupervised_metrics

N_CCS_SEEDS = 5
LABELS = {"cross_covariance": "tcPCA-1", "generalized_cross_covariance": "GCC-1",
          "logistic": "logistic", "ccs": "CCS"}


def out_path(model: str) -> Path:
    return ROOT / "data" / "layer_sweep" / f"{model}__{PROFILE}.csv"


def _methods(pp: dict) -> list[dict]:
    """The pwl profile's own entries for the four methods, so kwargs cannot drift."""
    picked, seen = [], set()
    for m in pp["methods"]:
        name = m["name"]
        if name not in LABELS or name in seen:
            continue
        if name in ("cross_covariance", "generalized_cross_covariance") and (
                m.get("n_components", 1) != 1 or m.get("beta") is not None):
            continue
        if name == "ccs" and not (m.get("confidence") == "min_sq"
                                  and m.get("consistency") == "default"
                                  and not m.get("weight_norm") and not m.get("svd_reduce")):
            continue
        seen.add(name)
        picked.append(m)
    missing = set(LABELS) - seen
    if missing:
        raise KeyError(f"pwl profile lacks {sorted(missing)}")
    return picked


def _score(probe, info, activations, samples, position, pp) -> float:
    result = eval_probe(probe, info, activations, samples, position)
    if result is None:
        return float("nan")
    extra = unsupervised_metrics(probe, info, activations, samples, position,
                                 centering=pp.get("centering", "midpoint"),
                                 sign_from=pp.get("sign_from", "none"))
    if extra:
        extra.pop("auc_unsupervised", None)
        result = dataclasses.replace(result, **extra)
    acc = getattr(result, "uncalibrated_accuracy", None)
    return float(acc if acc is not None else result.accuracy)


def sweep(model: str) -> pd.DataFrame:
    params = yaml.safe_load((ROOT / "params.yaml").read_text())
    run_key = f"{model}__{PROFILE}"
    run_cfg = params["runs"][run_key]
    pp = resolve_probing(params, run_cfg)
    ap, mp = params["activations"], params["models"][run_cfg["model"]]
    layers = list(ap["layers"])
    positions = list(pp.get("train_positions", ap["token_positions"]))

    nodes = load_jsonl(str(ROOT / "data" / "processed" / f"samples_{run_cfg['family']}.jsonl"))
    activations = extract_hidden_states(
        samples=nodes, model_name=mp["name"],
        cache_dir=ROOT / params["output"]["activations_dir"] / run_key,
        token_positions=ap["token_positions"], mask_positions=ap.get("mask_positions", []),
        layers=layers, batch_size=mp["batch_size"], device=mp["device"],
        dtype=mp.get("dtype", "auto"))
    pairs = load_family_as_pairs(run_cfg["family"])
    samples = pairs if pairs is not None else nodes

    common = dict(descriptor="label", train_fraction=pp.get("train_split", 0.8),
                  seed=pp.get("seed", 42), split_strategy=pp.get("split_strategy", "random"),
                  centering=pp.get("centering", "midpoint"))
    methods = _methods(pp)
    datasets = sorted({s.descriptors.get("dataset_name", "") for s in samples} - {""})

    rows, t0 = [], time.monotonic()
    for ds in datasets:
        ds_samples = [s for s in samples if s.descriptors.get("dataset_name") == ds]
        for pos in positions:
            for m in methods:
                cls = REGISTRY[m["name"]]
                kwargs = {k: v for k, v in m.items()
                          if k not in ("name", "label_descriptor", "seeds")}
                label = LABELS[m["name"]]
                if m["name"] == "ccs":
                    seeds = list(m.get("seeds") or [0])[:N_CCS_SEEDS]
                    fitted = fit_probe_batched(activations=activations, samples=ds_samples,
                                               layers=layers, position=pos, probe_cls=cls,
                                               probe_kwargs=kwargs, seeds=seeds, dataset=ds,
                                               **common)
                    for (layer, s), (probe, info) in fitted.items():
                        rows.append({"dataset": ds, "position": pos, "layer": layer,
                                     "method": label, "seed": s,
                                     "acc": _score(probe, info, activations, ds_samples, pos, pp)})
                else:
                    for layer in layers:
                        probe, info = fit_probe(activations=activations, samples=ds_samples,
                                                layer=layer, position=pos, probe_cls=cls,
                                                probe_kwargs=kwargs, dataset=ds, **common)
                        if probe is None:
                            continue
                        rows.append({"dataset": ds, "position": pos, "layer": layer,
                                     "method": label, "seed": None,
                                     "acc": _score(probe, info, activations, ds_samples, pos, pp)})
            print(f"  {ds:12s} {pos:15s} done  ({time.monotonic() - t0:5.0f}s)", flush=True)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    df = sweep(args.model)
    path = out_path(args.model)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"wrote {len(df)} rows to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
