from __future__ import annotations

import json
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable

from .loader import Sample

ContrastiveFn = Callable[[Sample], Sample]


@dataclass
class ContrastivePair:
    base: Sample
    counterfactual: Sample

    @property
    def id(self) -> str:
        return self.base.id

    def to_dict(self) -> dict:
        return {
            "base": self.base.to_dict(),
            "counterfactual": self.counterfactual.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ContrastivePair":
        return cls(
            base=Sample.from_dict(d["base"]),
            counterfactual=Sample.from_dict(d["counterfactual"]),
        )


# ── I/O ───────────────────────────────────────────────────────────────────────

def save_pairs_jsonl(pairs: list[ContrastivePair], path: str) -> None:
    with open(path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p.to_dict()) + "\n")


def load_pairs_jsonl(path: str) -> list[ContrastivePair]:
    with open(path) as f:
        return [ContrastivePair.from_dict(json.loads(line)) for line in f if line.strip()]


def is_pairs_file(path: str) -> bool:
    """Peek at the first line to determine whether a JSONL file contains pairs."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                return "base" in json.loads(line)
    return False


# ── Pair ↔ Sample conversion ──────────────────────────────────────────────────

def pairs_to_samples(pairs: list[ContrastivePair]) -> list[Sample]:
    """
    Flatten pairs into a single sample list.  Each sample receives two extra
    descriptors so the pairing can be reconstructed downstream:

      pair_id    — unique per pair (shared by both sides); uses index to handle
                   Cartesian-product pairings where the same base appears many times
      pair_role  — "base" or "counterfactual"
    """
    out = []
    for i, p in enumerate(pairs):
        base = deepcopy(p.base)
        cf   = deepcopy(p.counterfactual)
        pair_id = f"{p.base.id}_{i}"
        base.descriptors["pair_id"]   = pair_id
        base.descriptors["pair_role"] = "base"
        cf.descriptors["pair_id"]     = pair_id
        cf.descriptors["pair_role"]   = "counterfactual"
        out.append(base)
        out.append(cf)
    return out


# ── Pairing utilities ─────────────────────────────────────────────────────────

def pair_by_descriptor(
    samples: list[Sample],
    key: str | list[str],
    base_label: Any = 1,
    cf_label: Any = 0,
    label_descriptor: str = "label",
    dataset_name: str | None = None,
) -> list[ContrastivePair]:
    """
    Group samples by descriptor[key], then within each group pair samples
    whose descriptor[label_descriptor] equals base_label with those whose
    value equals cf_label.

    key may be a single descriptor name or a list of names; when a list is
    given the group key is a tuple of the corresponding values.

    Groups with multiple candidates on either side yield multiple pairs
    (all combinations).  Groups missing either side are silently skipped.

    dataset_name  — if set, only consider samples with that dataset_name value.
    """
    if dataset_name is not None:
        samples = [s for s in samples if s.descriptors.get("dataset_name") == dataset_name]

    keys = [key] if isinstance(key, str) else list(key)

    groups: dict[Any, dict[Any, list[Sample]]] = defaultdict(lambda: defaultdict(list))
    for s in samples:
        key_val = tuple(s.descriptors.get(k) for k in keys)
        if any(v is None for v in key_val):
            continue
        key_val   = key_val[0] if len(key_val) == 1 else key_val
        label_val = s.descriptors.get(label_descriptor)
        groups[key_val][label_val].append(s)

    pairs = []
    for by_label in groups.values():
        for base in by_label.get(base_label, []):
            for cf in by_label.get(cf_label, []):
                pairs.append(ContrastivePair(base=base, counterfactual=cf))
    return pairs


def make_pairs(samples: list[Sample], counterfactual_fn: ContrastiveFn) -> list[ContrastivePair]:
    return [ContrastivePair(base=s, counterfactual=counterfactual_fn(s)) for s in samples]


# ── Counterfactual functions ───────────────────────────────────────────────────

def label_flip(sample: Sample, descriptor: str = "label") -> Sample:
    """Flip a binary descriptor value; text unchanged."""
    cf = deepcopy(sample)
    cf.id = f"{sample.id}_cf"
    cf.descriptors = {**sample.descriptors, descriptor: 1 - sample.descriptors[descriptor]}
    return cf


def text_swap(mapping: dict[str, str]) -> ContrastiveFn:
    """Return a fn that replaces substrings in sample.text per a mapping dict."""
    def fn(sample: Sample) -> Sample:
        cf = deepcopy(sample)
        cf.id = f"{sample.id}_cf"
        for src, tgt in mapping.items():
            cf.text = cf.text.replace(src, tgt)
        return cf
    return fn
