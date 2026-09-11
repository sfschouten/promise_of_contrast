"""Generate contrastive pairs from CCS benchmark datasets (Burns et al. 2023).

Each source sample becomes one ContrastivePair:
  base.descriptors["label"] = 1   (true framing)
  cf.descriptors["label"]   = 0   (false framing)

Both sides share the same source content in sample.text; the framing
statement lives in sample.metadata["claim"].  Apply templates/ccs_claim.j2
(or any template that references sample.metadata["claim"]) to compose them.

Datasets covered (all 11 of the original 10+):
  imdb, amazon_polarity, ag_news, dbpedia_14, rte, qnli, boolq, copa, hellaswag,
  piqa (via nthngdy/piqa mirror), story_cloze (via lecslab/story_cloze mirror)

  Note: ybisk/piqa and LSDSem/story_cloze use loading scripts no longer supported
  by the HF datasets library.  The mirrors above are equivalent parquet conversions.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.contrastive import ContrastivePair, save_pairs_jsonl
from src.data.loader import Sample


# ── helpers ───────────────────────────────────────────────────────────────────

def _ensure_period(text: str) -> str:
    """Append a period if the text doesn't already end with terminal punctuation."""
    t = text.rstrip()
    return t if t[-1:] in ".!?" else t + "."


def _pair(
    idx: int,
    dataset: str,
    text: str,
    true_claim: str,
    false_claim: str,
    metadata_extra: dict | None = None,
) -> ContrastivePair:
    meta_base = {"claim": true_claim, "dataset": dataset, "negation": False, **(metadata_extra or {})}
    meta_cf   = {"claim": false_claim, "dataset": dataset, "negation": False, **(metadata_extra or {})}
    base = Sample(
        id=f"{dataset}_{idx}",
        text=text,
        descriptors={"label": 1, "dataset_name": dataset},
        metadata=meta_base,
    )
    cf = Sample(
        id=f"{dataset}_{idx}_cf",
        text=text,
        descriptors={"label": 0, "dataset_name": dataset},
        metadata=meta_cf,
    )
    return ContrastivePair(base=base, counterfactual=cf)


# ── per-dataset generators ────────────────────────────────────────────────────

# Maps CCS dataset name → subdirectory name within data_dir (or None to use name directly)
_DATA_SUBDIR = {
    "rte":  "glue",
    "qnli": "glue",
    "copa": "super_glue",
}


def _source(name: str, data_dir: str | None) -> str:
    if not data_dir:
        return {
            "imdb":            "stanfordnlp/imdb",
            "amazon_polarity": "SetFit/amazon_polarity",
            "ag_news":         "fancyzhx/ag_news",
            "dbpedia_14":      "fancyzhx/dbpedia_14",
            "rte":             "nyu-mll/glue",
            "qnli":            "nyu-mll/glue",
            "boolq":           "google/boolq",
            "copa":            "super_glue",
            "hellaswag":       "Rowan/hellaswag",
            "piqa":            "nthngdy/piqa",
            "story_cloze":     "lecslab/story_cloze",
        }[name]
    subdir = _DATA_SUBDIR.get(name, name)
    return f"{data_dir}/{subdir}"


def _gen_imdb(max_samples: int | None, data_dir: str | None = None) -> list[ContrastivePair]:
    ds = load_dataset(_source("imdb", data_dir), split="train")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    pairs = []
    for i, row in enumerate(ds):
        true_framing  = "positive" if row["label"] == 1 else "negative"
        false_framing = "negative" if row["label"] == 1 else "positive"
        pairs.append(_pair(
            i, "imdb", row["text"],
            true_claim=f"The sentiment of this text is {true_framing}.",
            false_claim=f"The sentiment of this text is {false_framing}.",
        ))
    return pairs


def _gen_amazon_polarity(max_samples: int | None, data_dir: str | None = None) -> list[ContrastivePair]:
    ds = load_dataset(_source("amazon_polarity", data_dir), split="train")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    pairs = []
    for i, row in enumerate(ds):
        text = f"{row['title']}\n{row['text']}"
        true_framing  = "positive" if row["label"] == 1 else "negative"
        false_framing = "negative" if row["label"] == 1 else "positive"
        pairs.append(_pair(
            i, "amazon_polarity", text,
            true_claim=f"The sentiment of this review is {true_framing}.",
            false_claim=f"The sentiment of this review is {false_framing}.",
        ))
    return pairs


_AG_NEWS_LABELS = ["World", "Sports", "Business", "Sci/Tech"]


def _gen_ag_news(max_samples: int | None, data_dir: str | None = None) -> list[ContrastivePair]:
    ds = load_dataset(_source("ag_news", data_dir), split="train")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    n = len(_AG_NEWS_LABELS)
    pairs = []
    for i, row in enumerate(ds):
        true_label  = row["label"]
        false_label = (true_label + 1) % n
        pairs.append(_pair(
            i, "ag_news", row["text"],
            true_claim=f"This article is about {_AG_NEWS_LABELS[true_label]}.",
            false_claim=f"This article is about {_AG_NEWS_LABELS[false_label]}.",
            metadata_extra={"true_label": true_label},
        ))
    return pairs


_DBPEDIA_LABELS = [
    "Company", "EducationalInstitution", "Artist", "Athlete", "OfficeHolder",
    "MeanOfTransportation", "Building", "NaturalPlace", "Village", "Animal",
    "Plant", "Album", "Film", "WrittenWork",
]


def _gen_dbpedia(max_samples: int | None, data_dir: str | None = None) -> list[ContrastivePair]:
    ds = load_dataset(_source("dbpedia_14", data_dir), split="train")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    n = len(_DBPEDIA_LABELS)
    pairs = []
    for i, row in enumerate(ds):
        text = f"{row['title']}\n{row['content']}"
        true_label  = row["label"]
        false_label = (true_label + 1) % n
        pairs.append(_pair(
            i, "dbpedia_14", text,
            true_claim=f"This article is about a {_DBPEDIA_LABELS[true_label]}.",
            false_claim=f"This article is about a {_DBPEDIA_LABELS[false_label]}.",
            metadata_extra={"true_label": true_label},
        ))
    return pairs


def _gen_rte(max_samples: int | None, data_dir: str | None = None) -> list[ContrastivePair]:
    """RTE: label 0 = entailment (true), 1 = not_entailment (false)."""
    ds = load_dataset(_source("rte", data_dir), name="rte", split="train")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    pairs = []
    for i, row in enumerate(ds):
        text = f"Premise: {row['sentence1']}\nHypothesis: {row['sentence2']}"
        entails = row["label"] == 0
        pairs.append(_pair(
            i, "rte", text,
            true_claim="Does the premise entail the hypothesis? Yes.",
            false_claim="Does the premise entail the hypothesis? No.",
            metadata_extra={"entails": entails},
        ) if entails else _pair(
            i, "rte", text,
            true_claim="Does the premise entail the hypothesis? No.",
            false_claim="Does the premise entail the hypothesis? Yes.",
            metadata_extra={"entails": entails},
        ))
    return pairs


def _gen_qnli(max_samples: int | None, data_dir: str | None = None) -> list[ContrastivePair]:
    """QNLI: label 0 = entailment (sentence answers question), 1 = not_entailment."""
    ds = load_dataset(_source("qnli", data_dir), name="qnli", split="train")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    pairs = []
    for i, row in enumerate(ds):
        text = f"Question: {row['question']}\n{row['sentence']}"
        answers = row["label"] == 0
        pairs.append(_pair(
            i, "qnli", text,
            true_claim="Does the sentence answer the question? Yes.",
            false_claim="Does the sentence answer the question? No.",
            metadata_extra={"answers": answers},
        ) if answers else _pair(
            i, "qnli", text,
            true_claim="Does the sentence answer the question? No.",
            false_claim="Does the sentence answer the question? Yes.",
            metadata_extra={"answers": answers},
        ))
    return pairs


def _gen_boolq(max_samples: int | None, data_dir: str | None = None) -> list[ContrastivePair]:
    ds = load_dataset(_source("boolq", data_dir), split="train")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    pairs = []
    for i, row in enumerate(ds):
        text = f"{row['passage']}\nQuestion: {row['question']}?"
        answer = row["answer"]
        pairs.append(_pair(
            i, "boolq", text,
            true_claim="Answer: Yes." if answer else "Answer: No.",
            false_claim="Answer: No." if answer else "Answer: Yes.",
        ))
    return pairs


_COPA_CONNECTORS = {"cause": "caused by", "effect": "followed by"}


def _gen_copa(max_samples: int | None, data_dir: str | None = None) -> list[ContrastivePair]:
    ds = load_dataset(_source("copa", data_dir), "copa", split="train")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    pairs = []
    for i, row in enumerate(ds):
        connector  = _COPA_CONNECTORS[row["question"]]
        correct    = row["choice1"] if row["label"] == 0 else row["choice2"]
        incorrect  = row["choice2"] if row["label"] == 0 else row["choice1"]
        premise    = row["premise"].rstrip(".")
        pairs.append(_pair(
            i, "copa",
            text=premise,
            true_claim=f"This was {connector}: {correct}",
            false_claim=f"This was {connector}: {incorrect}",
            metadata_extra={"question_type": row["question"]},
        ))
    return pairs


def _gen_hellaswag(max_samples: int | None, data_dir: str | None = None) -> list[ContrastivePair]:
    ds = load_dataset(_source("hellaswag", data_dir), split="train")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    pairs = []
    for i, row in enumerate(ds):
        correct_idx = int(row["label"])
        endings     = row["endings"]
        wrong_idx   = (correct_idx + 1) % len(endings)
        ctx = row["ctx"].strip()
        # ctx + ending form a continuous sentence — compose directly so no
        # template separator appears mid-sentence.
        extra = {"activity_label": row["activity_label"], "claim": ""}
        pairs.append(ContrastivePair(
            base=Sample(
                id=f"hellaswag_{i}",
                text=ctx + " " + endings[correct_idx].strip(),
                descriptors={"label": 1, "dataset_name": "hellaswag"},
                metadata={"dataset": "hellaswag", "negation": False, **extra},
            ),
            counterfactual=Sample(
                id=f"hellaswag_{i}_cf",
                text=ctx + " " + endings[wrong_idx].strip(),
                descriptors={"label": 0, "dataset_name": "hellaswag"},
                metadata={"dataset": "hellaswag", "negation": False, **extra},
            ),
        ))
    return pairs


def _gen_piqa(max_samples: int | None, data_dir: str | None = None) -> list[ContrastivePair]:
    """PIQA via nthngdy/piqa (parquet mirror of ybisk/piqa).
    label 0 = sol1 correct, 1 = sol2 correct.
    """
    ds = load_dataset(_source("piqa", data_dir), split="train")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    pairs = []
    for i, row in enumerate(ds):
        correct   = row["sol1"] if row["label"] == 0 else row["sol2"]
        incorrect = row["sol2"] if row["label"] == 0 else row["sol1"]
        pairs.append(_pair(
            i, "piqa",
            text=f"Goal: {row['goal']}",
            true_claim=f"Solution: {_ensure_period(correct)}",
            false_claim=f"Solution: {_ensure_period(incorrect)}",
        ))
    return pairs


def _gen_story_cloze(max_samples: int | None, data_dir: str | None = None) -> list[ContrastivePair]:
    """Story Cloze via lecslab/story_cloze (parquet mirror, prompt/chosen/rejected format)."""
    ds = load_dataset(_source("story_cloze", data_dir), split="train")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    pairs = []
    for i, row in enumerate(ds):
        pairs.append(_pair(
            i, "story_cloze",
            text=row["prompt"],
            true_claim=row["chosen"],
            false_claim=row["rejected"],
        ))
    return pairs


# ── dispatch ──────────────────────────────────────────────────────────────────

_GENERATORS = {
    "imdb":             _gen_imdb,
    "amazon_polarity":  _gen_amazon_polarity,
    "ag_news":          _gen_ag_news,
    "dbpedia_14":       _gen_dbpedia,
    "rte":              _gen_rte,
    "qnli":             _gen_qnli,
    "boolq":            _gen_boolq,
    "copa":             _gen_copa,
    "hellaswag":        _gen_hellaswag,
    "piqa":             _gen_piqa,
    "story_cloze":      _gen_story_cloze,
}


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    import yaml

    with open(ROOT / "params.yaml") as f:
        params = yaml.safe_load(f)

    cfg         = params["dataset"]["ccs"]
    out_path    = ROOT / "data" / "raw" / "ccs.jsonl"
    datasets    = cfg.get("datasets", list(_GENERATORS))
    max_samples = cfg.get("max_samples_per_dataset", None)
    data_dir    = cfg.get("data_dir") or None

    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_pairs: list[ContrastivePair] = []
    for name in datasets:
        if name not in _GENERATORS:
            print(f"  [warn] unknown dataset {name!r}, skipping", flush=True)
            continue
        print(f"  loading {name}...", end=" ", flush=True)
        pairs = _GENERATORS[name](max_samples, data_dir=data_dir)
        print(f"{len(pairs)} pairs")
        all_pairs.extend(pairs)

    save_pairs_jsonl(all_pairs, str(out_path))
    print(f"\nSaved {len(all_pairs)} pairs → {out_path}")


if __name__ == "__main__":
    main()
