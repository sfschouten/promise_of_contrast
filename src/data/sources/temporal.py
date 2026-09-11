"""
Synthetic modular-arithmetic datasets over cyclic vocabularies — days of the
week and months of the year — following Engels, Liao, Michaud, Gurnee & Tegmark
(2024), "Not all language model features are one-dimensionally linear".

Each (start, offset) arithmetic problem becomes one *fully-connected graph*.
A problem is phrased as a paper-style stem ending right before the answer:

    "Let's do some day of the week math. Two days from Monday is"

and the graph has one node per possible answer — the stem completed with each
element of the vocabulary:

    a0: "... is Monday"     a1: "... is Tuesday"   …   a2 (correct): "... is Wednesday"

Two overlapping edge sets are exposed via the edge ``dataset_name``:

  * **circular** pairs cover *every* unordered pair of nodes — the complete
    graph K_n, including the correct-incident ones.  Their binary labels are
    cosmetic (cross-covariance is base/cf-swap invariant); their purpose is to
    let the cross-covariance probe recover the 2-D circular representation of
    the cyclic variable that the paper studies.  dataset_name "<name>_circular".
  * **truth** pairs are the subset of edges *incident to the correct-answer
    node* (correct vs incorrect completion), oriented base = correct (label 1),
    cf = incorrect (label 0) → suited for truth-value probing.  dataset_name
    "<name>_truth".  (These pairs also appear, unlabelled, among the circular
    edges; the two sets are not disjoint.)

Every node carries `multiclass_label` (= its answer's position on the circle)
and `target` (= its answer name, used as the colour key in the subspace scatter),
so the circular geometry can be probed and visualised directly.

Two graph variants are generated into the same family:

  * **completion** (dataset_names "<name>_truth" / "<name>_circular") — the stem
    is completed with each candidate answer; nodes vary by *answer*.  The
    contrast lives at the answer token, so probe position `last`.
  * **prediction** (dataset_name "<name>_pred") — the stem ENDS at the operator
    ("… is") with no answer; one complete graph per offset whose nodes vary by
    *start*, so each node's final ("is") token predicts a different element.
    The contrast lives at that final token (which is `last`), letting a probe
    read the model's internal answer *before* it is emitted — the paper's setup.
    Circular edges only (no stated answer ⇒ no truth contrast).

The generator is fully synthetic: the only "external" input is the ordered
vocabulary (``external/temporal/days.json`` etc.), which fixes the cyclic order
and acts as the DVC dependency for the generate stage.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from ..graph import Graph, GraphEdge, save_graphs_jsonl
from ..loader import Sample
from ..masks import find_char_spans

# Spelled-out cardinals so prompts read naturally ("two days" not "2 days").
# Indexed by the integer offset; covers every offset used by days (≤6) and
# months (≤11), with headroom.
NUM_WORDS = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
]

# Default stem phrasing (per dataset; overridable via params `stem_template`).
# Available format fields:
#   {start}            start element (e.g. "Monday")
#   {offset}           integer offset
#   {offset_word}      spelled-out offset ("two")
#   {offset_word_cap}  capitalised spelled-out offset ("Two")
#   {unit}             singular unit ("day")
#   {units}            offset-aware unit ("day" for offset 1, else "days")
DEFAULT_STEM_TEMPLATE = "{offset_word_cap} {units} from {start} is"


def _render_stem(stem_template: str, *, start: str, offset: int, unit: str) -> str:
    if not 0 <= offset < len(NUM_WORDS):
        raise ValueError(
            f"offset {offset} out of supported range [0, {len(NUM_WORDS) - 1}] "
            "for spelled-out cardinals"
        )
    offset_word = NUM_WORDS[offset]
    return stem_template.format(
        start=start,
        offset=offset,
        offset_word=offset_word,
        offset_word_cap=offset_word.capitalize(),
        unit=unit,
        units=unit if offset == 1 else f"{unit}s",
    )


def build_modular_arithmetic_graphs(
    vocab: list[str],
    *,
    dataset_name: str,
    unit: str,
    offsets: list[int],
    stem_template: str,
    id_prefix: str,
) -> list[Graph]:
    """
    Build one complete graph per (start element, offset) arithmetic problem.

    Each graph has ``len(vocab)`` nodes — the stem completed with every possible
    answer — and ``C(n, 2)`` undirected edges, typed "truth" (incident to the
    correct node) or "circular" (between two incorrect nodes).
    """
    n = len(vocab)
    graphs: list[Graph] = []
    stem_idx = 0
    for start_index, start in enumerate(vocab):
        for offset in offsets:
            stem = _render_stem(stem_template, start=start, offset=offset, unit=unit)
            correct_index = (start_index + offset) % n
            correct_answer = vocab[correct_index]

            keys = [f"a{ai}" for ai in range(n)]
            nodes: dict[str, Sample] = {}
            for ai, answer in enumerate(vocab):
                text = f"{stem} {answer}"
                # Positional spans: the answer is the appended suffix, and the
                # start element lives in the stem prefix.  (Search would be
                # ambiguous for the node whose answer equals the start, e.g.
                # "… from Monday is Monday".)
                answer_span = (len(stem) + 1, len(text))
                start_spans = find_char_spans(stem, start)
                nodes[keys[ai]] = Sample(
                    id=f"{id_prefix}_{stem_idx:04d}_{ai}",
                    text=text,
                    descriptors={
                        "multiclass_label": ai,        # position on the circle
                        # Ties the n nodes of one graph together so the scatter
                        # report samples whole tuples, not individual edges.
                        "group_id": f"{id_prefix}_{stem_idx:04d}",
                        "answer": answer,
                        "answer_index": ai,
                        "target": answer,              # colour key for subspace scatter
                        "is_correct": int(ai == correct_index),
                        "dataset_name": dataset_name,  # edges override per pairing
                        "start": start,
                        "start_index": start_index,
                        "offset": offset,
                        "correct_answer": correct_answer,
                        "correct_index": correct_index,
                    },
                    masks={
                        "start": start_spans,
                        "answer": [answer_span],
                    },
                    metadata={},
                )

            sd = {"sub_dataset": dataset_name}
            edges: list[GraphEdge] = []
            for i in range(n):
                for j in range(i + 1, n):
                    # Circular pair: EVERY unordered pair, i.e. the complete graph
                    # K_n (including the correct-incident ones).  Labels are
                    # cosmetic (cross-covariance is base/cf-swap invariant).
                    edges.append(GraphEdge(
                        type="circular",
                        base_node_id=keys[i], cf_node_id=keys[j],
                        base_label=1, cf_label=0,
                        dataset_name=f"{dataset_name}_circular",
                        descriptors={**sd, "contrast": "circular",
                                     "base_answer_index": i, "cf_answer_index": j,
                                     # one centering group per unordered element
                                     # pair (i<j): each "pair type" is mean-centred
                                     # independently so cross-cov sees the pairwise
                                     # difference geometry (the circle), not the
                                     # absolute per-element cloud offsets.
                                     "center_key": f"{i}_{j}"},
                    ))
                    # Truth pair: the correct-incident edges are additionally
                    # exposed as a labelled subset, oriented base = correct (1),
                    # cf = incorrect (0), for truth-value probing.
                    if i == correct_index or j == correct_index:
                        base_k, cf_k = (keys[i], keys[j]) if i == correct_index \
                            else (keys[j], keys[i])
                        edges.append(GraphEdge(
                            type="truth",
                            base_node_id=base_k, cf_node_id=cf_k,
                            base_label=1, cf_label=0,
                            dataset_name=f"{dataset_name}_truth",
                            descriptors={**sd, "contrast": "truth"},
                        ))

            graphs.append(Graph(
                id=f"{id_prefix}_{stem_idx:04d}",
                nodes=nodes,
                edges=edges,
                descriptors={**sd, "start": start, "offset": offset,
                             "correct_index": correct_index},
            ))
            stem_idx += 1
    return graphs


def build_prediction_graphs(
    vocab: list[str],
    *,
    dataset_name: str,
    unit: str,
    offsets: list[int],
    stem_template: str,
    id_prefix: str,
) -> list[Graph]:
    """
    Build "prediction" graphs: stems that END at the operator ("… is") with no
    answer appended.  One complete graph per offset; its nodes vary by *start*
    element, so each node's final ("is") token predicts a different element.

    Probe the final token (train position ``last``) to read the model's internal
    answer.  Circular edges only — with no stated answer there is no truth
    contrast.  ``multiclass_label`` / ``target`` is the *predicted* element.
    """
    n = len(vocab)
    graphs: list[Graph] = []
    for gid, offset in enumerate(offsets):
        keys = [f"s{si}" for si in range(n)]
        nodes: dict[str, Sample] = {}
        for start_index, start in enumerate(vocab):
            text = _render_stem(stem_template, start=start, offset=offset, unit=unit)
            predicted_index = (start_index + offset) % n
            predicted = vocab[predicted_index]
            nodes[keys[start_index]] = Sample(
                id=f"{id_prefix}_{gid:04d}_{start_index}",
                text=text,
                descriptors={
                    "multiclass_label": predicted_index,  # the element the model should predict
                    "group_id": f"{id_prefix}_{gid:04d}",  # see build_modular_arithmetic_graphs
                    "target": predicted,                  # colour key = predicted element
                    "predicted": predicted,
                    "predicted_index": predicted_index,
                    "dataset_name": dataset_name,         # edges override per pairing
                    "start": start,
                    "start_index": start_index,
                    "offset": offset,
                },
                masks={"start": find_char_spans(text, start)},
                metadata={},
            )

        sd = {"sub_dataset": dataset_name}
        edges: list[GraphEdge] = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append(GraphEdge(
                    type="prediction",
                    base_node_id=keys[i], cf_node_id=keys[j],
                    base_label=1, cf_label=0,           # cosmetic (paired method)
                    dataset_name=f"{dataset_name}_pred",
                    descriptors={**sd, "contrast": "prediction",
                                 "base_start_index": i, "cf_start_index": j,
                                 # per-start-pair centering group (cf. circular)
                                 "center_key": f"{i}_{j}"},
                ))
        graphs.append(Graph(
            id=f"{id_prefix}_{gid:04d}", nodes=nodes, edges=edges,
            descriptors={**sd, "offset": offset, "variant": "prediction"},
        ))
    return graphs


def generate(dataset_key: str, *, id_prefix: str) -> None:
    """
    Params-driven entry point.  Reads ``dataset.<dataset_key>`` from params.yaml,
    loads the ordered vocabulary from ``<data_dir>/<vocab_file>``, builds the
    graphs and writes ``data/raw/<dataset_key>.jsonl`` (a graphs file).

    Expected config keys under ``dataset.<dataset_key>``:
        data_dir       directory holding the vocab file (DVC dependency)
        vocab_file     JSON list of ordered element names
        unit           singular unit noun ("day" / "month")
        offsets        list of integer offsets to enumerate
        stem_template  optional prompt phrasing (defaults to DEFAULT_STEM_TEMPLATE)
    """
    params = yaml.safe_load(Path("params.yaml").read_text())
    cfg = params["dataset"][dataset_key]

    vocab_path = Path(cfg["data_dir"]) / cfg["vocab_file"]
    vocab = json.loads(vocab_path.read_text())
    if not vocab:
        raise ValueError(f"empty vocabulary in {vocab_path}")

    offsets = cfg.get("offsets") or list(range(1, len(vocab)))
    stem_template = cfg.get("stem_template") or DEFAULT_STEM_TEMPLATE

    completion = build_modular_arithmetic_graphs(
        vocab,
        dataset_name=dataset_key,
        unit=cfg["unit"],
        offsets=offsets,
        stem_template=stem_template,
        id_prefix=id_prefix,
    )
    prediction = build_prediction_graphs(
        vocab,
        dataset_name=dataset_key,
        unit=cfg["unit"],
        offsets=offsets,
        stem_template=stem_template,
        id_prefix=f"{id_prefix}p",
    )
    graphs = completion + prediction

    out = Path(f"data/raw/{dataset_key}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    save_graphs_jsonl(graphs, str(out))

    def _counts(gs):
        e = [edge for g in gs for edge in g.edges]
        by = {t: sum(1 for x in e if x.type == t) for t in {x.type for x in e}}
        return sum(len(g.nodes) for g in gs), by

    c_nodes, c_by = _counts(completion)
    p_nodes, p_by = _counts(prediction)
    print(f"Saved {len(graphs)} graphs → {out}")
    print(f"  completion: {len(completion)} graphs, {c_nodes} nodes, edges {c_by}")
    print(f"  prediction: {len(prediction)} graphs, {p_nodes} nodes, edges {p_by}")
