"""Shared vocabulary for the Probing Without Labels reproduction.

Method names are *derived* from `params.yaml` rather than hard-coded: a probe's `method`
string encodes its non-default options, so writing the strings out by hand here would
silently drift the moment an option changes.  Instead we instantiate each configured
method and ask it for its own name.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.probing import REGISTRY

PROFILE = "pwl"

# Token position → the name the paper uses for it.  The prompt ends "<answer>.", so the
# last token is the period and the one before it is the end of the answer.
POSITION_LABELS = {"second_to_last": "answer", "last": "period"}

# Dataset display order.
DATASETS = ["comparisons", "sp_en_trans", "cities", "amazon", "imdb",
            "ent_bank", "snli", "copa", "rte"]

# The five datasets the loss-term ablation is run on.
ABLATION_DATASETS = ["comparisons", "sp_en_trans", "cities", "amazon", "imdb"]


def _ccs_label(kwargs: dict) -> str:
    """A readable name for a CCS variant, from its loss terms and alterations."""
    conf = kwargs.get("confidence", "burns")
    cons = kwargs.get("consistency", "default")
    if conf != "none" and cons != "none":
        base = "CCS"
    elif cons == "none":
        base = "L_conf"
    else:
        base = "L_cons"
    alterations = ("a1" if kwargs.get("weight_norm") else "",
                   "a2" if kwargs.get("svd_reduce") else "")
    suffix = "+".join(a for a in alterations if a)
    return f"{base}+{suffix}" if suffix else base


def method_catalogue(params: dict | None = None) -> dict[str, dict]:
    """Map method string → {label, kind, kwargs, seeds} for every method in the profile."""
    if params is None:
        params = yaml.safe_load((ROOT / "params.yaml").read_text())
    catalogue: dict[str, dict] = {}
    for m in params["probing_profiles"][PROFILE]["methods"]:
        kwargs = {k: v for k, v in m.items()
                  if k not in ("name", "label_descriptor", "seeds")}
        probe = REGISTRY[m["name"]](**kwargs)
        if m["name"] == "ccs":
            label, kind = _ccs_label(kwargs), "ccs"
        elif m["name"] == "cross_covariance":
            beta = kwargs.get("beta")
            label = f"tcPCA-{kwargs.get('n_components', 1)}"
            if beta is not None:
                label += f" (beta={beta:g})"
            kind = "tcpca"
        elif m["name"] == "generalized_cross_covariance":
            # Its own kind: the generalised eigenvalues are explained-variance ratios,
            # so they are not comparable with the tcPCA spectra on a shared axis.
            label = f"GCC-{kwargs.get('n_components', 1)}"
            kind = "gcc"
        elif m["name"] == "pca_pair":
            k = kwargs.get("component_index", 0)
            label, kind = (f"PC{k + 1}", "crc")
        else:
            label, kind = m["name"], "baseline"
        catalogue[probe.method] = {
            "label": label, "kind": kind, "kwargs": kwargs,
            "seeds": list(m.get("seeds") or [None]),
        }
    return catalogue


def eval_db_path(model: str) -> Path:
    return ROOT / "data" / "eval_results" / f"{model}__{PROFILE}.duckdb"


def probes_dir(model: str, run: str | None = None) -> Path:
    """Probes for `<model>__<run>`; `run` defaults to the paper's own pwl profile.

    The tuple-geometry figures read the `cities_grid` and `days_of_week` runs, so the
    suffix cannot be hard-coded to PROFILE here.
    """
    return ROOT / "data" / "probes" / f"{model}__{run or PROFILE}"


def activations_dir(model: str, run: str | None = None) -> Path:
    return ROOT / "data" / "activations" / f"{model}__{run or PROFILE}"


def output_dir(model: str) -> Path:
    params = yaml.safe_load((ROOT / "params.yaml").read_text())
    d = ROOT / params["output"]["reproductions_dir"] / "promise_of_contrast" / model
    d.mkdir(parents=True, exist_ok=True)
    return d


def model_name(model_alias: str) -> str:
    params = yaml.safe_load((ROOT / "params.yaml").read_text())
    return params["models"][model_alias]["name"]
