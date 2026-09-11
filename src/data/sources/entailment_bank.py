"""Shared parsing for EntailmentBank proofs and the ARC questions they derive from.

EntailmentBank ships in several shapes — the allenai release nests sentences under
``meta.triples``, community mirrors either expand them into ``sent1``…``sent25`` columns
or keep them in one space-delimited ``context`` string — and a proof references them all
by the same ``sentN`` / ``intN`` keys.  These helpers hide that, so a generator can ask
for "the premises of the last proof step" without caring which shape it was handed.

Used by scripts/generate_entbank.py and scripts/generate_pwl.py.
"""
from __future__ import annotations

import re

from datasets import concatenate_datasets, load_dataset


def _ensure_period(text: str) -> str:
    t = text.rstrip()
    return t if t[-1:] in ".!?" else t + "."


def _last_step_keys(proof: str) -> list[str]:
    """Keys from the immediate-premise side of the '-> hypothesis' step."""
    for part in reversed(proof.split(";")):
        part = part.strip()
        if "-> hypothesis" in part:
            lhs = part.split("-> hypothesis")[0].strip()
            return [k.strip() for k in lhs.split("&") if k.strip()]
    return []


def _all_proof_keys(proof: str) -> set[str]:
    """All sentX / intX tokens referenced anywhere in the proof."""
    return set(re.findall(r"\b(?:sent|int)\d+\b", proof))


def _extract_intermediates(proof: str) -> dict[str, str]:
    """Map intX keys → their conclusion text from the proof string."""
    out: dict[str, str] = {}
    for part in proof.split(";"):
        if "->" not in part:
            continue
        rhs = part.split("->", 1)[1].strip()
        if rhs.startswith("hypothesis"):
            continue
        if ":" in rhs:
            key, text = rhs.split(":", 1)
            out[key.strip()] = text.strip()
    return out


def _sentence(text: str) -> str:
    """Normalise a single proof sentence."""
    s = text.strip()
    if s and not s[-1] in ".!?":
        s += "."
    return s.capitalize()


def _parse_context(row: dict) -> dict[str, str]:
    """Parse 'context' field into a key→text dict.

    ariesutiono/entailment-bank-v3 stores all sentences in a single space-delimited
    string: "sent1: text1 sent2: text2 ...".  We split on whitespace boundaries
    immediately before a sentN:/intN: key.
    """
    context = (row.get("context") or "").strip()
    if not context:
        return {}
    parts = re.split(r'\s+(?=(?:sent|int)\d+\s*:)', context)
    result: dict[str, str] = {}
    for part in parts:
        part = part.strip()
        if ":" in part:
            key, val = part.split(":", 1)
            result[key.strip()] = val.strip()
    return result


def _lookup(key: str, row: dict, intermediates: dict[str, str]) -> str:
    """Look up a sentence key across all three storage formats."""
    if key in intermediates:
        return intermediates[key]
    # ariesutiono datasets-expanded format: sent1-sent25 as top-level fields
    val = (row.get(key) or "").strip()
    if val:
        return val
    # allenai format: sentences nested in meta.triples
    meta = row.get("meta") or {}
    if isinstance(meta, dict):
        val = (meta.get("triples", {}).get(key) or "").strip()
        if val:
            return val
    # ariesutiono raw-JSONL format: sentences in a single 'context' string
    val = _parse_context(row).get(key, "").strip()
    if val:
        return val
    return ""


def _get_supports(row: dict, proof: str) -> list[str]:
    keys = _last_step_keys(proof)
    intermediates = _extract_intermediates(proof)
    sents = [_lookup(k, row, intermediates) for k in keys]
    return [_sentence(s) for s in sents if s]


def _get_distractors(row: dict, proof: str) -> list[str]:
    # allenai format: distractors listed explicitly in meta.distractors
    meta = row.get("meta") or {}
    if isinstance(meta, dict):
        distractor_keys = meta.get("distractors") or []
        if distractor_keys:
            triples = meta.get("triples", {})
            out = []
            for k in distractor_keys[:3]:
                val = (triples.get(k) or row.get(k) or "").strip()
                if val:
                    out.append(_sentence(val))
            return out
    # ariesutiono format: any sentX not referenced anywhere in the proof
    used = _all_proof_keys(proof)
    out = []
    # Try top-level sentX fields (datasets-expanded) then context string (raw JSONL)
    ctx = _parse_context(row)
    for i in range(1, 26):
        key = f"sent{i}"
        val = (row.get(key) or ctx.get(key) or "").strip()
        if val and key not in used:
            out.append(_sentence(val))
    return out


def _load_arc_index(configs: list[str], arc_source: str = "allenai/ai2_arc") -> dict[str, dict]:
    """Map ARC question ID → {correct_label, correct, wrong_choices, choices_str}."""
    index: dict[str, dict] = {}
    for config in configs:
        dsets = [load_dataset(arc_source, config, split=s)
                 for s in ["train", "validation", "test"]]
        for row in concatenate_datasets(dsets):
            choices = row["choices"]
            labels  = choices["label"]
            texts   = choices["text"]
            key     = row["answerKey"]
            correct = next((t for l, t in zip(labels, texts) if l == key), None)
            wrong_choices = [(l, t) for l, t in zip(labels, texts) if l != key]
            choices_str = "  ".join(f"({l}) {t}" for l, t in zip(labels, texts))
            if correct and wrong_choices:
                index[row["id"]] = {
                    "correct_label": key,
                    "correct": correct,
                    "wrong_choices": wrong_choices,
                    "choices_str": choices_str,
                }
    return index

