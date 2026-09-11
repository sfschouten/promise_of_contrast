import hashlib
from pathlib import Path
from typing import Union

import torch

from ..data.loader import Sample

PosKey = Union[str, int]          # "last", "first", or a token index
CacheKey = tuple[int, PosKey]     # (layer_index, position_key)
SampleCache = dict[CacheKey, torch.Tensor]  # -> hidden_state vector (hidden_size,)

_META_KEY = ("_meta", "n_tokens")  # sentinel: all token positions were extracted


def _cache_key(model_name: str, sample: Sample) -> str:
    return hashlib.sha256(f"{model_name}:{sample.text}".encode()).hexdigest()[:16]


def _cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"{key}.pt"


def load_sample_cache(cache_dir: Path, model_name: str, sample: Sample) -> SampleCache:
    path = _cache_path(cache_dir, _cache_key(model_name, sample))
    if not path.exists():
        return {}
    data = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(data, torch.Tensor):
        return {}  # legacy format; will be re-extracted
    # Detach defensively: activations are values, never part of a graph, but a tensor
    # saved from inside a tracing context can carry requires_grad, and every downstream
    # reader calls .numpy() on these.
    return {k: v.detach() for k, v in data.items()}


def update_sample_cache(
    cache_dir: Path,
    model_name: str,
    sample: Sample,
    new_entries: SampleCache,
) -> None:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, _cache_key(model_name, sample))
    existing = load_sample_cache(cache_dir, model_name, sample)
    existing.update(new_entries)
    torch.save(existing, path)


def needs_forward(
    cache_dir: Path,
    model_name: str,
    sample: Sample,
    layer_indices: list[int],
    position_keys: list[PosKey] | str,
) -> bool:
    """Return True if the sample needs a forward pass for the given request."""
    cached = load_sample_cache(cache_dir, model_name, sample)
    if position_keys == "all":
        return _META_KEY not in cached
    return any((l, p) not in cached for l in layer_indices for p in position_keys)
