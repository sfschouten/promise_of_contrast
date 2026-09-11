"""
Data pairs inspection report.

Two page types:

  Entity pages  (when group_id is present)
    N group_ids are sampled per family.  For each entity, one page shows all
    pair types that exist for that entity across all datasets — so one page
    gives a complete view of e.g. the TF/TN/FN triangle for a single entity.
    The dataset name is shown in each row header.

  Dataset pages  (fallback for pairs without group_id)
    One page per dataset, N randomly sampled pairs shown side by side.

With --tokenize: token boundaries marked with │, spaces as ·, last token ⟪…⟫.

Run from the project root:
    python scripts/report_pairs.py [--n N] [--tokenize] [--output path.pdf]
"""
from __future__ import annotations

import argparse
import random
import sys
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.contrastive import ContrastivePair
from src.data.graph import load_family_as_pairs
from src.data.loader import load_jsonl

WRAP_WIDTH     = 72
SEED           = 42
FONT_SIZE      = 8
LINE_HEIGHT_IN = 0.16
ROW_PAD_IN     = 0.30
FIG_WIDTH      = 12
SUPTITLE_H     = 0.35
BOTTOM_MARGIN  = 0.05


def _wrap(text: str, width: int = WRAP_WIDTH) -> str:
    return "\n".join(
        textwrap.fill(line, width=width) if line else ""
        for line in text.splitlines()
    )


def _format_tokens(tokens: list[str]) -> str:
    lines: list[str] = []
    current = ""
    for i, tok in enumerate(tokens):
        display = tok.replace(" ", "·")
        chunk = f"⟪{display}⟫" if i == len(tokens) - 1 else f"{display}│"
        if len(current) + len(chunk) > WRAP_WIDTH and current:
            lines.append(current)
            current = chunk
        else:
            current += chunk
    if current:
        lines.append(current)
    return "\n".join(lines)


def _body_text(text: str, tokenizer) -> str:
    if tokenizer is None:
        return _wrap(text)
    token_ids = tokenizer.encode(text)
    tokens    = [tokenizer.decode([t]) for t in token_ids]
    return _format_tokens(tokens)


def _row_height(p: ContrastivePair, tokenizer) -> float:
    def n_lines(text: str) -> int:
        return len(_body_text(text, tokenizer).splitlines()) + 1
    return max(n_lines(p.base.text), n_lines(p.counterfactual.text)) * LINE_HEIGHT_IN + ROW_PAD_IN


def _draw_pairs(
    axes,
    pairs: list[ContrastivePair],
    row_heights: list[float],
    tokenizer,
    row_headers: list[str] | None = None,
    group_boundaries: set[int] | None = None,
) -> None:
    """Render pairs into a (n_pairs × 2) axes array."""
    for i, p in enumerate(pairs):
        for j, (sample, role) in enumerate(
            [(p.base, "base"), (p.counterfactual, "counterfactual")]
        ):
            ax = axes[i][j]
            ax.axis("off")

            if group_boundaries and i in group_boundaries:
                ax.plot([0, 1], [1, 1], color="#444444", lw=1.2,
                        transform=ax.transAxes, clip_on=False)

            label_val  = sample.descriptors.get("label", "")
            prefix     = f"{row_headers[i]}  " if row_headers else ""
            if tokenizer is None:
                header = f"{prefix}{role}  (label={label_val})"
            else:
                token_ids = tokenizer.encode(sample.text)
                last_tok  = repr(tokenizer.decode([token_ids[-1]])) if token_ids else "?"
                header = f"{prefix}{role}  (label={label_val})  [{len(token_ids)} tok | last: {last_tok}]"

            ax.text(0.01, 0.98, header,
                    transform=ax.transAxes,
                    fontsize=FONT_SIZE, fontweight="bold", color="#2c7bb6",
                    va="top", ha="left")
            ax.text(0.01, 0.98 - LINE_HEIGHT_IN / row_heights[i],
                    _body_text(sample.text, tokenizer),
                    transform=ax.transAxes,
                    fontsize=FONT_SIZE, va="top", ha="left", family="monospace")

            ax.set_facecolor("#eaf3fb" if j == 0 else "#fef9ec")
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.4)
                spine.set_edgecolor("#cccccc")


def _render_entity_page(
    pdf: PdfPages,
    family: str,
    group_id: str,
    entries: list[tuple[str, ContrastivePair]],  # (dataset_name, pair)
    tokenizer,
) -> None:
    """One page per entity: rows are datasets, columns are base | counterfactual."""
    pairs       = [p for _, p in entries]
    row_headers = [ds for ds, _ in entries]
    row_heights = [_row_height(p, tokenizer) for p in pairs]
    fig_height  = sum(row_heights) + SUPTITLE_H + BOTTOM_MARGIN

    fig, axes = plt.subplots(
        len(pairs), 2,
        figsize=(FIG_WIDTH, fig_height),
        gridspec_kw={"height_ratios": row_heights, "hspace": 0.0, "wspace": 0.02},
        squeeze=False,
    )
    fig.suptitle(f"family={family}  |  entity={group_id}", fontsize=11)
    fig.subplots_adjust(
        left=0.005, right=0.995,
        top=1.0 - SUPTITLE_H / fig_height,
        bottom=BOTTOM_MARGIN / fig_height,
    )
    _draw_pairs(axes, pairs, row_heights, tokenizer, row_headers=row_headers)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _render_dataset_page(
    pdf: PdfPages,
    family: str,
    dataset: str,
    pairs: list[ContrastivePair],
    n_total: int,
    tokenizer,
) -> None:
    """One page per dataset: rows are pairs, columns are base | counterfactual."""
    row_heights = [_row_height(p, tokenizer) for p in pairs]
    fig_height  = sum(row_heights) + SUPTITLE_H + BOTTOM_MARGIN

    fig, axes = plt.subplots(
        len(pairs), 2,
        figsize=(FIG_WIDTH, fig_height),
        gridspec_kw={"height_ratios": row_heights, "hspace": 0.0, "wspace": 0.02},
        squeeze=False,
    )
    fig.suptitle(
        f"family={family}  |  dataset={dataset}  ({n_total} pairs total)",
        fontsize=11,
    )
    fig.subplots_adjust(
        left=0.005, right=0.995,
        top=1.0 - SUPTITLE_H / fig_height,
        bottom=BOTTOM_MARGIN / fig_height,
    )
    _draw_pairs(axes, pairs, row_heights, tokenizer)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n",        type=int, default=5,
                        help="Entities (or pairs for ungrouped datasets) to sample (default: 5)")
    parser.add_argument("--tokenize", action="store_true")
    parser.add_argument("--output",   default=None)
    args = parser.parse_args()

    params  = yaml.safe_load(Path("params.yaml").read_text())
    out_dir = Path(params["output"]["reports_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = (Path(args.output) if args.output else
                out_dir / ("pairs_report_tokens.pdf" if args.tokenize else "pairs_report.pdf"))

    tokenizer = None
    if args.tokenize:
        from transformers import AutoTokenizer
        model_name = next(iter(params["models"].values()))["name"]
        print(f"Loading tokenizer for {model_name} …")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load all processed samples (pair-aware for graph families).
    samples_flat = []
    for fam in params.get("families", {}):
        pair_samples = load_family_as_pairs(fam)
        if pair_samples is not None:
            for s in pair_samples:
                s.metadata["_family"] = fam
            samples_flat.extend(pair_samples)
        else:
            p = Path(f"data/processed/samples_{fam}.jsonl")
            if p.exists():
                loaded = load_jsonl(str(p))
                for s in loaded:
                    s.metadata["_family"] = fam
                samples_flat.extend(loaded)

    # Reconstruct ContrastivePair objects, keyed by (family, dataset_name).
    by_pair: dict[str, dict[str, object]] = defaultdict(dict)
    for s in samples_flat:
        pid  = s.descriptors.get("pair_id")
        role = s.descriptors.get("pair_role")
        if pid is not None and role in ("base", "counterfactual"):
            by_pair[pid][role] = s

    page_pairs: dict[tuple[str, str], list[ContrastivePair]] = defaultdict(list)
    for pid, roles in by_pair.items():
        base = roles.get("base")
        cf   = roles.get("counterfactual")
        if base is None or cf is None:
            continue
        fam = base.metadata.get("_family", "unknown")
        ds  = base.descriptors.get("dataset_name", "unknown")
        page_pairs[(fam, ds)].append(ContrastivePair(base=base, counterfactual=cf))

    # Build group_id → [(family, dataset, pair)] index.
    groups_index: dict[str, list[tuple[str, str, ContrastivePair]]] = defaultdict(list)
    for (fam, ds), pairs in page_pairs.items():
        for p in pairs:
            gid = p.base.descriptors.get("group_id", "")
            if gid:
                groups_index[gid].append((fam, ds, p))

    rng = random.Random(SEED)

    # Per family: sample N group_ids, then render entity pages followed by
    # dataset pages for any datasets that have no group_id.
    families = sorted({fam for fam, _ in page_pairs})
    print(f"Writing pages → {out_path}")
    n_pages = 0

    with PdfPages(out_path) as pdf:
        for family in families:
            family_gids = sorted(
                gid for gid, entries in groups_index.items()
                if any(f == family for f, _, _ in entries)
            )

            if family_gids:
                chosen = rng.sample(family_gids, min(args.n, len(family_gids)))
                for gid in chosen:
                    entries = sorted(
                        [(ds, p) for f, ds, p in groups_index[gid] if f == family],
                        key=lambda x: x[0],
                    )
                    _render_entity_page(pdf, family, gid, entries, tokenizer)
                    n_pages += 1
                    print(f"  [entity] {family} / {gid}  ({len(entries)} datasets)")

            # Ungrouped datasets: one page per dataset, random pair sampling.
            for (fam, ds), all_pairs in sorted(page_pairs.items()):
                if fam != family:
                    continue
                if any(p.base.descriptors.get("group_id") for p in all_pairs):
                    continue  # covered by entity pages above
                sample = rng.sample(all_pairs, min(args.n, len(all_pairs)))
                _render_dataset_page(pdf, family, ds, sample, len(all_pairs), tokenizer)
                n_pages += 1
                print(f"  [dataset] {family} / {ds}  ({len(all_pairs)} pairs, showing {len(sample)})")

        if n_pages == 0:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, "No pairs found.", ha="center", va="center", fontsize=12)
            ax.axis("off")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            print("  (no pairs found — wrote placeholder page)")

    print(f"\nDone. {out_path} ({n_pages} pages)")


if __name__ == "__main__":
    main()
