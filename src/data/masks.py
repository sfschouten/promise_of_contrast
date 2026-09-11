"""
Utilities for character-level and token-level span masks.

Masks are stored on Sample as dict[str, list[tuple[int, int]]], where each
tuple is a (start, end) character span with exclusive end (Python slice convention).

Token-level conversion requires an offset_mapping — the list of (char_start, char_end)
tuples per token produced by a HuggingFace tokenizer called with
return_offsets_mapping=True.  Spans with (0, 0) (e.g. special tokens) are treated
as outside every character span.
"""

from __future__ import annotations


def find_char_spans(text: str, substring: str) -> list[tuple[int, int]]:
    """Return (start, end) for every non-overlapping occurrence of `substring` in `text`."""
    spans: list[tuple[int, int]] = []
    start = 0
    while (idx := text.find(substring, start)) != -1:
        spans.append((idx, idx + len(substring)))
        start = idx + 1
    return spans


def char_spans_to_token_mask(
    spans: list[tuple[int, int]],
    offset_mapping: list[tuple[int, int]],
) -> list[bool]:
    """
    Return a boolean list of length len(offset_mapping).
    A token is True when its character range overlaps with at least one span.
    """
    mask = []
    for tok_start, tok_end in offset_mapping:
        if tok_start == tok_end:  # special / padding token
            mask.append(False)
            continue
        overlaps = any(
            span_start < tok_end and tok_start < span_end
            for span_start, span_end in spans
        )
        mask.append(overlaps)
    return mask


def last_token_in_spans(
    spans: list[tuple[int, int]],
    offset_mapping: list[tuple[int, int]],
) -> int | None:
    """Index of the last token that overlaps with any span, or None."""
    token_mask = char_spans_to_token_mask(spans, offset_mapping)
    indices = [i for i, m in enumerate(token_mask) if m]
    return indices[-1] if indices else None
