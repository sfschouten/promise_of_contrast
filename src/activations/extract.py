"""Batched hidden-state extraction via nnsight.

The residual stream is read from ``model.model.layers[l].output``, which is exactly
transformers' ``outputs.hidden_states[l + 1]`` — i.e. the output of decoder block ``l``.
Layer numbering here is therefore *block* numbering: layer 0 is the first block's output,
and the embedding layer (``hidden_states[0]``) is not addressable through ``layers``.

Two nnsight/transformers details that are easy to get wrong and are asserted below:

* From transformers 5.0, ``LlamaDecoderLayer.forward`` returns a bare tensor rather than
  a ``(hidden_states, ...)`` tuple, so the ``.output[0]`` idiom found in most nnsight
  examples silently slices the *batch* dimension.  We save ``.output`` and unwrap a tuple
  afterwards, which is correct on both transformers 4.x and 5.x.
* ``nnsight.LanguageModel`` sets ``tokenizer.padding_side = "left"`` (it is built for
  generation).  ``_resolve_position`` counts from the left, so we force right padding.
"""

import time
from pathlib import Path

import torch
import transformers
from transformers import AutoConfig

from ..data.loader import Sample
from ..data.masks import last_token_in_spans
from .cache import (
    _META_KEY,
    PosKey,
    SampleCache,
    load_sample_cache,
    update_sample_cache,
)


def _resolve_position(pos_key: PosKey, seq_len: int) -> int:
    if pos_key == "last":
        return seq_len - 1
    elif pos_key == "second_to_last":
        return max(0, seq_len - 2)
    elif pos_key == "first":
        return 0
    else:
        return min(int(pos_key), seq_len - 1)


def extract_hidden_states(
    samples: list[Sample],
    model_name: str,
    cache_dir: Path,
    token_positions: list[PosKey] | str,
    mask_positions: list[str] = (),
    layers: list[int] | str = "all",
    batch_size: int = 32,
    device: str = "cuda",
    dtype: str = "auto",
) -> dict[str, SampleCache]:
    """
    Returns {sample.id: {(layer, pos_key): tensor(hidden_size)}}.

    token_positions  "all" stores every real token; otherwise a list of keys
                     ("last", "first", or integer indices).
    mask_positions   list of mask names (from sample.masks); for each name M the
                     last token overlapping M's spans is stored as pos_key "{M}_last".
                     Samples that lack a given mask are silently skipped for that key.
    layers           block indices; layer l is the output of decoder block l, i.e.
                     transformers' hidden_states[l + 1].
    dtype            torch dtype for the weights, or "auto" to follow the checkpoint.
                     Set it explicitly if you do not want to depend on the transformers
                     default (which changed in v5).
    """
    cache_dir = Path(cache_dir)

    if layers == "all":
        layer_indices = list(range(AutoConfig.from_pretrained(model_name).num_hidden_layers))
    else:
        layer_indices = list(layers)

    mask_positions = list(mask_positions)
    mask_pos_keys = [f"{m}_last" for m in mask_positions]
    needs_offsets = bool(mask_positions)

    # Load all caches up front; use them for both returning data and checking
    # what still needs to be forwarded.
    all_data: dict[str, SampleCache] = {
        s.id: load_sample_cache(cache_dir, model_name, s) for s in samples
    }

    def _needs_fwd(s: Sample) -> bool:
        cache = all_data[s.id]
        # bulk token positions
        if token_positions == "all":
            if _META_KEY not in cache:
                return True
        else:
            if any((l, p) not in cache for l in layer_indices for p in token_positions):
                return True
        # mask-derived positions (only for masks the sample actually has)
        for mask_name, pos_key in zip(mask_positions, mask_pos_keys):
            if mask_name in s.masks and any(
                (l, pos_key) not in cache for l in layer_indices
            ):
                return True
        return False

    to_forward = [s for s in samples if _needs_fwd(s)]
    n_cached = len(samples) - len(to_forward)
    print(f"  {len(samples)} samples total: {n_cached} cached, {len(to_forward)} to compute")

    if to_forward:
        from nnsight import LanguageModel

        prev_verbosity = transformers.logging.get_verbosity()
        transformers.logging.set_verbosity_error()
        model = LanguageModel(model_name, device_map=device, dispatch=True, dtype=dtype)
        transformers.logging.set_verbosity(prev_verbosity)
        tokenizer = model.tokenizer

        # nnsight defaults the tokenizer to left padding; _resolve_position counts from
        # the left, so right padding is required for "last"/"second_to_last" to be real
        # tokens rather than padding.
        tokenizer.padding_side = "right"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        assert tokenizer.padding_side == "right", tokenizer.padding_side

        blocks = model.model.layers

        # transformers <5 returns (hidden_states, ...) from a decoder layer; >=5 returns
        # the tensor itself.  Settle it once with a two-token trace rather than guessing.
        _probe: dict[str, object] = {}
        with torch.no_grad(), model.trace(
            input_ids=torch.zeros(1, 2, dtype=torch.long, device=device)
        ):
            _probe["out"] = blocks[layer_indices[0]].output.save()
        layer_returns_tuple = not torch.is_tensor(_probe["out"])

        def _hidden(layer: int):
            """The (batch, seq, hidden) residual stream leaving decoder block `layer`."""
            out = blocks[layer].output
            return out[0] if layer_returns_tuple else out

        n_batches = (len(to_forward) + batch_size - 1) // batch_size
        t_start = time.monotonic()
        for i in range(0, len(to_forward), batch_size):
            batch = to_forward[i : i + batch_size]
            batch_num = i // batch_size + 1
            if batch_num % 50 == 1:
                elapsed = time.monotonic() - t_start
                if batch_num > 1:
                    eta = elapsed / i * (len(to_forward) - i)
                    eta_str = f"  ETA {int(eta // 60)}m{int(eta % 60):02d}s"
                else:
                    eta_str = ""
                print(f"  batch {batch_num}/{n_batches} ({i}/{len(to_forward)} samples done){eta_str}", flush=True)
            enc = tokenizer(
                [s.text for s in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                return_offsets_mapping=needs_offsets,
            )
            # offset_mapping cannot be moved to GPU; pop it before the device transfer.
            offset_mappings = enc.pop("offset_mapping") if needs_offsets else None
            inputs = enc.to(device)
            seq_lens = [int(n) for n in inputs["attention_mask"].sum(dim=1)]
            rows = torch.arange(len(batch), device=device)

            # Resolve every wanted position to a per-sample token index *before* tracing,
            # then gather inside it.  Saving whole (batch, seq, hidden) tensors for every
            # layer is what runs the GPU out of memory once prompts are long: this keeps
            # one vector per (sample, layer, position) instead of one per token.
            wanted: dict[PosKey, list[int]] = {}
            if token_positions != "all":
                for pos_key in token_positions:
                    wanted[pos_key] = [_resolve_position(pos_key, n) for n in seq_lens]

            per_sample_offsets = []
            for j in range(len(batch)):
                if offset_mappings is None:
                    per_sample_offsets.append(None)
                else:
                    per_sample_offsets.append(
                        [(int(a), int(b)) for a, b in offset_mappings[j][:seq_lens[j]]]
                    )
            for mask_name, pos_key in zip(mask_positions, mask_pos_keys):
                idxs = []
                for j, sample in enumerate(batch):
                    spans = sample.masks.get(mask_name, [])
                    tok = (last_token_in_spans(spans, per_sample_offsets[j])
                           if spans and per_sample_offsets[j] else None)
                    idxs.append(-1 if tok is None else tok)   # -1 marks "not present"
                if any(i >= 0 for i in idxs):
                    wanted[pos_key] = idxs

            traced: dict[tuple[int, PosKey], object] = {}
            # no_grad matters here: without it autograd retains every intermediate
            # activation of all 32 blocks for a backward pass that never comes, which is
            # several GB per batch once prompts are long.
            with torch.no_grad(), model.trace(**inputs):
                for layer in layer_indices:
                    hs = _hidden(layer)
                    if token_positions == "all":
                        traced[(layer, "all")] = hs.save()
                    else:
                        for pos_key, idxs in wanted.items():
                            cols = torch.tensor([max(i, 0) for i in idxs], device=device)
                            traced[(layer, pos_key)] = hs[rows, cols].save()

            gathered = {k: (v if torch.is_tensor(v) else v[0]) for k, v in traced.items()}

            for j, sample in enumerate(batch):
                seq_len = seq_lens[j]
                new_entries: SampleCache = {}

                for layer in layer_indices:
                    if token_positions == "all":
                        hs = gathered[(layer, "all")][j]
                        for pos in range(seq_len):
                            new_entries[(layer, pos)] = hs[pos].detach().cpu()
                        for mask_name, pk in zip(mask_positions, mask_pos_keys):
                            idxs = wanted.get(pk)
                            if idxs is not None and idxs[j] >= 0:
                                new_entries[(layer, pk)] = hs[idxs[j]].detach().cpu()
                    else:
                        for pos_key, idxs in wanted.items():
                            if idxs[j] < 0:            # mask absent for this sample
                                continue
                            new_entries[(layer, pos_key)] = gathered[(layer, pos_key)][j].detach().cpu()

                if token_positions == "all":
                    new_entries[_META_KEY] = torch.tensor(seq_len)

                update_sample_cache(cache_dir, model_name, sample, new_entries)
                all_data[sample.id].update(new_entries)

        del model
        if device != "cpu":
            torch.cuda.empty_cache()

    return all_data
