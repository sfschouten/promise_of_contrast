import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.contrastive import (
    is_pairs_file,
    load_pairs_jsonl,
    pairs_to_samples,
)
from src.data.graph import (
    graphs_to_nodes, is_graphs_file, load_graphs_jsonl, save_graphs_jsonl,
)
from src.data.loader import load_hf_dataset, load_jsonl, save_jsonl
from src.data.masks import find_char_spans
from src.data.templating import PromptTemplater


def _apply_template(s, templater, template_name: str) -> None:
    """Apply template in-place, recomputing mask offsets against the new text."""
    original_text = s.text
    original_masks = s.masks
    mask_substrings = {
        name: [s.text[start:end] for start, end in spans]
        for name, spans in s.masks.items()
    }
    s.text = templater.apply(s, template_name)
    if s.text == original_text:
        # Passthrough template (text unchanged): keep the precomputed masks
        # exactly.  Re-finding substrings would be lossy when a masked word
        # occurs more than once (e.g. the "… from Monday is Monday" node, where
        # the start word equals the answer).
        s.masks = original_masks
        return
    s.masks = {
        name: [span for substr in substrings
               for span in find_char_spans(s.text, substr)]
        for name, substrings in mask_substrings.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True)
    args = parser.parse_args()

    params = yaml.safe_load(Path("params.yaml").read_text())
    pp = params["preprocessing"]
    families_cfg = params.get("families", {})

    input_path = Path(f"data/processed/pairs_{args.family}.jsonl")
    using_graphs = input_path.exists() and is_graphs_file(str(input_path))
    using_pairs = not using_graphs and input_path.exists() and is_pairs_file(str(input_path))

    out_path = Path(f"data/processed/samples_{args.family}.jsonl")
    graphs_out = Path(f"data/processed/graphs_{args.family}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if using_graphs:
        graphs = load_graphs_jsonl(str(input_path))
        print(f"Loaded {len(graphs)} graphs from {input_path}")

        nodes = graphs_to_nodes(graphs)

        fcfg = families_cfg.get(args.family, {})
        template_str = fcfg.get("template") or pp.get("prompt_template", "")
        if template_str:
            template_path = Path(template_str)
            templater = PromptTemplater(template_dir=template_path.parent)
            tname = template_path.name
            for node in nodes:
                _apply_template(node, templater, tname)

        # Update graph node references to the templated copies.
        node_map = {n.id: n for n in nodes}
        for g in graphs:
            for key in list(g.nodes):
                g.nodes[key] = node_map[g.nodes[key].id]

        save_jsonl(nodes, str(out_path))
        save_graphs_jsonl(graphs, str(graphs_out))
        print(f"Saved {len(nodes)} unique nodes → {out_path}")
        print(f"Saved {len(graphs)} templated graphs → {graphs_out}")
        return

    # For non-graph families, write an empty graphs file so DVC outs are satisfied.
    graphs_out.write_text("")

    if using_pairs:
        data = load_pairs_jsonl(str(input_path))
        print(f"Loaded {len(data)} pairs from {input_path}")
    else:
        data = load_jsonl(str(input_path))
        print(f"Loaded {len(data)} samples from {input_path}")

    if using_pairs:
        # Group pairs by the _family stamp set by make_pairs.py.
        family_groups: dict[str, list] = defaultdict(list)
        for pair in data:
            fam = pair.base.metadata.get("_family", "")
            family_groups[fam].append(pair)

        all_samples = []
        t_total = time.time()
        for fam, pairs in family_groups.items():
            t0 = time.time()
            # Per-family template; fall back to the top-level preprocessing key if present.
            fcfg = families_cfg.get(fam, {})
            template_str = fcfg.get("template") or pp.get("prompt_template", "")
            if template_str:
                template_path = Path(template_str)
                templater = PromptTemplater(template_dir=template_path.parent)
                tname = template_path.name
                report_every = max(1, len(pairs) // 10)
                for i, p in enumerate(pairs):
                    _apply_template(p.base, templater, tname)
                    _apply_template(p.counterfactual, templater, tname)
                    if (i + 1) % report_every == 0:
                        pct = 100 * (i + 1) // len(pairs)
                        print(f"    {fam}: {i + 1}/{len(pairs)} pairs templated ({pct}%)",
                              flush=True)

            samples = pairs_to_samples(pairs)
            all_samples.extend(samples)
            print(f"  {fam or '(no family)'}: {len(pairs)} pairs → {len(samples)} samples  "
                  f"({time.time() - t0:.1f}s)")

        save_jsonl(all_samples, str(out_path))
        total_pairs = sum(len(v) for v in family_groups.values())
        print(f"Saved {len(all_samples)} samples ({total_pairs} pairs) → {out_path}  "
              f"(total {time.time() - t_total:.1f}s)")

    else:
        # Flat samples (no pairing stage); apply the family template if set,
        # falling back to the top-level preprocessing template.
        fcfg = families_cfg.get(args.family, {})
        template_str = fcfg.get("template") or pp.get("prompt_template", "")
        if template_str:
            template_path = Path(template_str)
            templater = PromptTemplater(template_dir=template_path.parent)
            tname = template_path.name
            for s in data:
                _apply_template(s, templater, tname)
        save_jsonl(data, str(out_path))
        print(f"Saved {len(data)} samples → {out_path}")


if __name__ == "__main__":
    main()
