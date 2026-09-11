"""Nine contrast-pair datasets in the framing-ordered format unsupervised probing needs.

Every example here is a *minimal pair*: one prompt, two completions, exactly one of which
is correct.  The two sides differ only in the final phrase, so the last content token (the
"answer" position) and the trailing period are both meaningful probing sites.

    The city of Tyumen is not in Russia.        <- branch 0
    The city of Tyumen is in Russia.            <- branch 1

Three descriptors carry the labelling, and the distinction between them matters:

    branch   0 or 1 — *which framing* this sample is, fixed by the template.
    label    1 if this sample's framing is the correct one, else 0.
    truth    1 if branch 0 is the correct framing (identical on both sides of a pair).

Ordering by framing rather than by truth is what makes unsupervised accuracy meaningful.
If `base` were always the true side, a probe could score 100% by learning the ordering,
and the below-chance results that characterise these methods could not occur at all.
`src/probing/paper_eval.py` scores `(p_base + 1 - p_cf)/2 > 0.5` against `truth`.

Prompts follow the promptsource convention: a template splits into question and answer at
`|||`, and the rendered text is `question + " " + answer + "."`.

Sources are all local, under `external/` and DVC-tracked, so the pipeline is reproducible
without re-downloading anything.

Run from the project root:
    python scripts/generate_pwl.py [--datasets amazon,imdb]
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.contrastive import ContrastivePair, save_pairs_jsonl
from src.data.sources.entailment_bank import _get_supports, _load_arc_index, _sentence
from src.data.loader import Sample

DATASET = "pwl"


# ── prompt composition ────────────────────────────────────────────────────────

def compose(question: str, answer: str) -> str:
    """The promptsource rendering: question, a space, the answer, a full stop.

    The trailing period is what makes "the last token" and "the answer token" two
    distinct, comparable probing positions.
    """
    return f"{question} {answer}."


# ── per-dataset builders ──────────────────────────────────────────────────────
#
# Each yields dicts of {uid, question, answers, true_branch, meta}, where `answers` is
# (branch-0 completion, branch-1 completion) and `true_branch` says which is correct.

def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _parquet(path: Path, split: str):
    from datasets import load_dataset
    files = sorted(path.glob(f"**/{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no {split} parquet under {path}")
    return load_dataset("parquet", data_files=[str(f) for f in files], split="train")


def build_amazon(cfg: dict, ext: Path):
    """Consider the following example: ''' <review> ''' Between negative and positive,
    the sentiment of this example is ||| negative / positive"""
    choices = ("negative", "positive")
    for i, row in enumerate(_read_jsonl(ext / "ccs/amazon_polarity/train.jsonl")):
        content = row["text"].strip()
        yield {
            "uid": f"amazon_{i}",
            # The curly quotes are deliberate, not a typo.
            "question": f"Consider the following example: ‘‘‘ {content} ’’’ "
                        f"Between {choices[0]} and {choices[1]}, the sentiment of this example is",
            "answers": choices,
            "true_branch": int(row["label"]),
            "meta": {},
        }


def build_imdb(cfg: dict, ext: Path):
    """The following movie review expresses what sentiment? <review> ||| negative / positive"""
    choices = ("negative", "positive")
    for i, row in enumerate(_parquet(ext / "ccs/imdb", "train")):
        yield {
            "uid": f"imdb_{i}",
            "question": f"The following movie review expresses what sentiment? {row['text'].strip()}",
            "answers": choices,
            "true_branch": int(row["label"]),
            "meta": {},
        }


def build_copa(cfg: dict, ext: Path):
    """Consider the following example: '''<premise>''' Choice 1: ... Choice 2: ...
    Q: Which one is more likely to be the <cause|effect>, choice 1 or choice 2?
    ||| choice 1 / choice 2"""
    choices = ("choice 1", "choice 2")
    for split in ("train", "validation"):
        for i, row in enumerate(_parquet(ext / "ccs/super_glue/copa", split)):
            if int(row["label"]) < 0:          # unlabelled test rows
                continue
            yield {
                "uid": f"copa_{split}_{i}",
                "question": f"Consider the following example: ‘‘‘{row['premise']}’’’ "
                            f"Choice 1: {row['choice1']} Choice 2: {row['choice2']} "
                            f"Q: Which one is more likely to be the {row['question']}, "
                            f"choice 1 or choice 2?",
                "answers": choices,
                "true_branch": int(row["label"]),
                "meta": {"question_type": row["question"]},
            }


def build_rte(cfg: dict, ext: Path):
    """Assuming that the following is true: "<premise>"
    Concluding that: "<hypothesis>" is ||| incorrect / correct"""
    choices = ("incorrect", "correct")
    for split in ("train", "validation"):
        for i, row in enumerate(_parquet(ext / "ccs/glue/rte", split)):
            label = int(row["label"])
            if label < 0:
                continue
            # GLUE RTE: 0 = entailment, 1 = not_entailment.  The conclusion is *correct*
            # (branch 1) exactly when the premise entails it.
            yield {
                "uid": f"rte_{split}_{i}",
                "question": f'Assuming that the following is true: "{row["sentence1"].strip()}"\n'
                            f'Concluding that: "{row["sentence2"].strip()}" is',
                "answers": choices,
                "true_branch": 1 - label,
                "meta": {},
            }


def build_cities(cfg: dict, ext: Path):
    """The city of X is ||| (not) in Y — the contrast is negation.

    Each city appears twice, once with its real country and once with a wrong one, so
    truth is the XOR of "right country" and "affirmative".  See src/data/sources/cities.py for
    the GeoNames filter; the wrong-country draw is seeded here.
    """
    from src.data.sources.cities import load_cities

    cities = load_cities(ext / "geometry_of_truth/geonames.csv",
                         min_population=cfg.get("min_population", 500_000))
    rng = random.Random(cfg.get("seed", 0))
    all_countries = sorted({c for _, c in cities})

    for i, (city, true_country) in enumerate(cities):
        false_country = true_country
        while false_country == true_country:
            false_country = rng.choice(all_countries)
        for correct, country in ((True, true_country), (False, false_country)):
            yield {
                "uid": f"cities_{i}_{'t' if correct else 'f'}",
                "question": f"The city of {city} is",
                "answers": (f"not in {country}", f"in {country}"),
                # Affirmative is true exactly when the country is the right one.
                "true_branch": int(correct),
                "meta": {"city": city, "country": country,
                         "correct_country": true_country},
            }


def build_comparisons(cfg: dict, ext: Path):
    """<N1> is ||| smaller / larger than <N2> — the contrast is the relation."""
    rows = _read_csv(ext / "geometry_of_truth/larger_than.csv")
    for i, row in enumerate(rows):
        # The number words live only in the statement; recover them, and capitalise
        # both operands.
        stmt = row["statement"].strip().rstrip(".")
        if " is larger than " not in stmt:
            continue
        n1_text, n2_text = stmt.split(" is larger than ")
        n2_text = n2_text[:1].upper() + n2_text[1:]
        yield {
            "uid": f"comparisons_{i}",
            "question": f"{n1_text} is",
            "answers": (f"smaller than {n2_text}", f"larger than {n2_text}"),
            "true_branch": int(row["label"]),      # 1 when "larger" is true
            "meta": {"n1": row["n1"], "n2": row["n2"]},
        }


def build_sp_en_trans(cfg: dict, ext: Path):
    """The Spanish word 'x' ||| (does not) mean(s) 'y' — the contrast is negation.

    The affirmative and negated CSVs are row-aligned, so the two completions come from
    the same index in each.
    """
    pos = _read_csv(ext / "geometry_of_truth/sp_en_trans.csv")
    neg = _read_csv(ext / "geometry_of_truth/neg_sp_en_trans.csv")
    if len(pos) != len(neg):
        raise ValueError("sp_en_trans and neg_sp_en_trans are not row-aligned")
    for i, (p, n) in enumerate(zip(pos, neg)):
        p_stmt = p["statement"].strip().rstrip(".")
        n_stmt = n["statement"].strip().rstrip(".")
        if " means " not in p_stmt or " does not mean " not in n_stmt:
            continue
        start, pos_end = p_stmt.split(" means ", 1)
        _, neg_end = n_stmt.split(" does not mean ", 1)
        yield {
            "uid": f"sp_en_trans_{i}",
            "question": start,
            "answers": (f"does not mean {neg_end}", f"means {pos_end}"),
            "true_branch": int(p["label"]),        # 1 when the affirmative form is true
            "meta": {},
        }


def build_snli(cfg: dict, ext: Path):
    """A picture framing of NLI: the premise describes a picture, and the hypothesis is
    said to be correct or incorrect about it.  Neutral pairs are dropped, so the contrast
    is entailment vs contradiction."""
    choices = ("incorrect", "correct")
    # HF stanfordnlp/snli: 0 = entailment, 1 = neutral, 2 = contradiction.
    for i, row in enumerate(_parquet(ext / "snli", "train")):
        label = int(row["label"])
        if label not in (0, 2):                    # drop neutral and unlabelled
            continue
        yield {
            "uid": f"snli_{i}",
            "question": "You are looking at a picture.\n"
                        f'Describing it as "{row["premise"].strip()}" is correct.\n'
                        f'Saying (about the picture) that: "{row["hypothesis"].strip()}" is',
            "answers": choices,
            "true_branch": 1 if label == 0 else 0,
            "meta": {},
        }


def build_ent_bank(cfg: dict, ext: Path):
    """An ARC question, the proof premises that support its answer (all stated as
    correct), and a claim that some answer option answers it.

    Each EntailmentBank instance yields two examples — one with the real answer and one
    with a sampled wrong option — so the dataset is balanced by construction.
    """
    choices = ("incorrect", "correct")
    arc = _load_arc_index(cfg.get("arc_configs", ["ARC-Easy", "ARC-Challenge"]),
                          arc_source=str(ext / "arc"))
    rng = random.Random(cfg.get("seed", 0))

    for split in cfg.get("splits", ["train", "dev", "test"]):
        path = ext / "entailment_bank" / f"task2_{split}.jsonl"
        if not path.exists():
            continue
        for i, row in enumerate(_read_jsonl(path)):
            meta = row.get("meta") or {}
            entry = arc.get(row["id"])
            if entry is None:
                continue
            premises = _get_supports(row, row.get("proof", ""))
            if not premises:
                continue
            question_text = (meta.get("question_text") or "").strip()
            if not question_text:
                continue

            wrong_label, wrong_text = rng.choice(entry["wrong_choices"])
            options = [(entry["correct"], 1), (wrong_text.rstrip("."), 0)]
            rng.shuffle(options)
            rendered = [f"({tag}) {text}" for tag, (text, _) in zip("AB", options)]
            question = question_text + " " + " ".join(rendered)

            # All premises are asserted true in this configuration.
            premise_block = "".join(
                f'The statement "{p}" is correct.\n' for p in premises
            )
            for (text, correct), shown in zip(options, rendered):
                yield {
                    "uid": f"ent_bank_{split}_{i}_{'t' if correct else 'f'}",
                    "question": f"You are given the following question:\n> {question}\n"
                                f"{premise_block}\n"
                                f'Answering the question with "{shown}" is',
                    "answers": choices,
                    "true_branch": correct,
                    "meta": {"arc_id": row["id"], "n_premises": len(premises)},
                }


BUILDERS = {
    "amazon":       build_amazon,
    "imdb":         build_imdb,
    "copa":         build_copa,
    "rte":          build_rte,
    "cities":       build_cities,
    "comparisons":  build_comparisons,
    "sp_en_trans":  build_sp_en_trans,
    "snli":         build_snli,
    "ent_bank":     build_ent_bank,
}


# ── assembly ──────────────────────────────────────────────────────────────────

def to_pair(name: str, item: dict) -> ContrastivePair:
    """Turn a builder item into a framing-ordered ContrastivePair."""
    a0, a1 = item["answers"]
    true_branch = int(item["true_branch"])
    truth = int(true_branch == 0)          # is branch 0 the correct framing?

    def _side(branch: int, answer: str) -> Sample:
        return Sample(
            id=f"{item['uid']}_b{branch}",
            text=compose(item["question"], answer),
            descriptors={
                "dataset_name": name,
                "branch": branch,
                "label": int(true_branch == branch),
                "truth": truth,
            },
            metadata={"question": item["question"], "answer": answer,
                      "dataset": name, **item.get("meta", {})},
        )

    return ContrastivePair(base=_side(0, a0), counterfactual=_side(1, a1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default=None,
                        help="comma-separated subset of the configured datasets")
    args = parser.parse_args()

    params = yaml.safe_load((ROOT / "params.yaml").read_text())
    cfg = params["dataset"][DATASET]
    ext = ROOT / cfg.get("external_dir", "external")
    max_samples = cfg.get("max_samples_per_dataset", 1000)
    max_chars = cfg.get("max_chars")
    shuffle_seed = cfg.get("shuffle_seed", 0)

    wanted = args.datasets.split(",") if args.datasets else cfg["datasets"]

    all_pairs: list[ContrastivePair] = []
    for name in wanted:
        items = list(BUILDERS[name](cfg, ext))
        if max_chars:
            # Bound sequence length so a batch fits in memory.  Dropping is safer than
            # truncating: the answer sits at the *end*, so a truncated prompt would move
            # the probing position onto unrelated text.
            kept = [it for it in items
                    if len(compose(it["question"], max(it["answers"], key=len))) <= max_chars]
            dropped = len(items) - len(kept)
            items = kept
        else:
            dropped = 0

        # Shuffle here, not at probe-fit time: with probing.split_strategy "ordered" the
        # train/test split is a prefix of this order, so it is fixed by the data rather
        # than by the probe seed, and stays identical across a seed sweep.
        order = np.random.default_rng(shuffle_seed).permutation(len(items))
        items = [items[i] for i in order[:max_samples]]

        pairs = [to_pair(name, it) for it in items]
        all_pairs.extend(pairs)
        n_true0 = sum(p.base.descriptors["truth"] for p in pairs)
        print(f"  {name:14s} {len(pairs):5d} pairs  "
              f"(branch-0 correct in {n_true0}, {n_true0 / max(len(pairs), 1):.0%})"
              + (f"  [{dropped} dropped as over-long]" if dropped else ""))

    out_path = ROOT / "data" / "raw" / f"{DATASET}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_pairs_jsonl(all_pairs, str(out_path))
    print(f"Saved {len(all_pairs)} pairs across {len(wanted)} datasets → {out_path}")


if __name__ == "__main__":
    main()
