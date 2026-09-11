import argparse
import sys
import uuid
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.activations.extract import extract_hidden_states
from src.config import resolve_probing
from src.data.graph import load_family_as_pairs
from src.data.loader import load_jsonl
from src.probing import (
    REGISTRY,
    Manifest,
    compute_probe_config_hash,
    fit_probe,
    fit_probe_batched,
    load_probes,
    probe_key,
    save_probes,
)
from src.probing.base import ProbeFitInfo


def _find_run_key(params: dict, model_alias: str, family: str) -> str:
    for rkey, rcfg in params["runs"].items():
        if rcfg["model"] == model_alias and rcfg["family"] == family:
            return rkey
    raise KeyError(f"No run found for model={model_alias!r}, family={family!r}")


def _load_probe_cache(probes_dir: Path) -> tuple[dict, dict]:
    """Load existing probes and their manifest entries for incremental re-training.

    Returns:
        cached_probes: {(dataset, layer, position, method, descriptor): Probe}
        cached_info:   {(dataset, layer, position, method, descriptor): ProbeFitInfo}
    """
    manifest_path = probes_dir / "manifest.json"
    if not manifest_path.exists():
        return {}, {}
    try:
        manifest = Manifest.load(probes_dir)
        probes = load_probes(manifest.params_file)
        cached_info = {
            probe_key(e.dataset, e.layer, e.train_position, e.method, e.descriptor, e.seed): e
            for e in manifest.entries
        }
        return dict(probes), cached_info
    except Exception as exc:
        print(f"  [cache] Could not load existing probes: {exc} — will retrain all")
        return {}, {}


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run")
    group.add_argument("--run-compound")
    parser.add_argument("--token-positions", default=None,
                        help="Comma-separated override for activations.token_positions. "
                             "Probes are fitted at probing.train_positions only, so narrowing "
                             "this to those positions trains from the cache alone instead of "
                             "extracting (and so loading the model for) positions nothing uses.")
    args = parser.parse_args()

    params = yaml.safe_load(Path("params.yaml").read_text())
    run_cfg_for_profile = (params["runs"][args.run] if args.run
                           else params["run_compounds"][args.run_compound])
    ap = params["activations"]
    if args.token_positions:
        ap = {**ap, "token_positions": [p.strip() for p in args.token_positions.split(",")]}
    pp = resolve_probing(params, run_cfg_for_profile)
    op = params["output"]

    act_base = Path(op["activations_dir"])

    def _load_run(run_key: str):
        run_cfg = params["runs"][run_key]
        mp = params["models"][run_cfg["model"]]
        family = run_cfg["family"]
        # Unique nodes for activation extraction (no duplicates).
        nodes = load_jsonl(f"data/processed/samples_{family}.jsonl")
        fam_act = extract_hidden_states(
            samples=nodes,
            model_name=mp["name"],
            cache_dir=act_base / run_key,
            token_positions=ap["token_positions"],
            mask_positions=ap.get("mask_positions", []),
            layers=ap["layers"],
            batch_size=mp["batch_size"],
            device=mp["device"],
            dtype=mp.get("dtype", "auto"),
        )
        # For graph families: materialize pairs (with pair_id/pair_role) for probing.
        pair_samples = load_family_as_pairs(family)
        fam_samples = pair_samples if pair_samples is not None else nodes
        return fam_samples, fam_act

    merge_datasets = False  # for compounds, train on the pooled set as one dataset

    if args.run:
        run_cfg = params["runs"][args.run]
        mp = params["models"][run_cfg["model"]]
        samples, activations = _load_run(args.run)
        output_name = args.run
    else:
        rc_cfg = params["run_compounds"][args.run_compound]
        mp = params["models"][rc_cfg["model"]]
        compound_cfg = params["compounds"][rc_cfg["compound"]]
        samples, activations = [], {}
        for fam in compound_cfg["families"]:
            run_key = _find_run_key(params, rc_cfg["model"], fam)
            fam_samples, fam_act = _load_run(run_key)
            samples.extend(fam_samples)
            activations.update(fam_act)
            print(f"  [{fam}] {len(fam_samples)} samples")
        filter_datasets = compound_cfg.get("datasets")
        if filter_datasets:
            samples = [s for s in samples if s.descriptors.get("dataset_name", "") in filter_datasets]
        output_name = args.run_compound
        merge_datasets = True

    probes_dir = Path(op["probes_dir"]) / output_name
    probes_dir.mkdir(parents=True, exist_ok=True)

    # Load existing probe cache for incremental re-training.
    cached_probes, cached_info = _load_probe_cache(probes_dir)
    if cached_probes:
        print(f"  [cache] {len(cached_probes)} existing probes available for reuse")

    raw = pp.get("label_descriptors", pp.get("label_descriptor", "label"))
    global_descriptors: list[str] = [raw] if isinstance(raw, str) else list(raw)

    probe_configs = []
    for m in pp["methods"]:
        cls = REGISTRY[m["name"]]
        kwargs = {k: v for k, v in m.items()
                  if k not in ("name", "label_descriptor", "seeds")}
        method_name = cls(**kwargs).method   # resolved once; may encode hyperparams (e.g. cross_covariance_8)
        if "label_descriptor" in m:
            ld = m["label_descriptor"]
            method_descriptors = [ld] if isinstance(ld, str) else list(ld)
        else:
            method_descriptors = global_descriptors
        # `seeds` sweeps the probe *initialisation* while the split stays fixed; the
        # default [None] means "one probe, no seed axis" — exactly the old behaviour.
        seeds = list(m.get("seeds") or [None])
        probe_configs.append((cls, kwargs, method_name, method_descriptors, seeds))

    # Methods exposing fit_paired_batched are fitted across all layers in one
    # batched pass; the rest go through the per-layer fit_probe loop.
    normal_configs = [c for c in probe_configs if not hasattr(c[0], "fit_paired_batched")]
    batched_configs = [c for c in probe_configs if hasattr(c[0], "fit_paired_batched")]

    train_positions = pp.get("train_positions", ap["token_positions"])
    if train_positions == "all":
        # Discover layers from activations; positions will be filtered per sample.
        from src.activations.cache import _META_KEY
        all_keys = [k for cache in activations.values() for k in cache if k != _META_KEY]
        train_positions = sorted({p for _, p in all_keys}, key=lambda p: (str(type(p)), p))

    # probing.layers narrows which of the extracted layers are probed; "all" (the
    # default) means every layer that was extracted.  Fitting one layer instead of
    # eleven is the difference between minutes and an hour for a 30-seed sweep.
    probe_layers = pp.get("layers", "all")
    if probe_layers not in (None, "all"):
        layer_list = list(probe_layers)
    elif ap.get("layers") == "all":
        from transformers import AutoConfig
        layer_list = list(range(AutoConfig.from_pretrained(mp["name"]).num_hidden_layers))
    else:
        layer_list = list(ap["layers"])

    seed = pp.get("seed", 42)
    train_fraction = pp.get("train_split", 0.8)
    split_strategy = pp.get("split_strategy", "random")
    centering = pp.get("centering", "midpoint")

    # Map dataset_name → family using the _family metadata stamped by make_pairs.py.
    dataset_to_family: dict[str, str] = {}
    for s in samples:
        ds = s.descriptors.get("dataset_name", "")
        fam = s.metadata.get("_family", "")
        if ds and fam:
            dataset_to_family[ds] = fam

    # Compounds pool everything into one training set; per-family mode splits by dataset.
    if merge_datasets:
        dataset_names = [output_name]
        dataset_map = {output_name: samples}
    else:
        dataset_names = sorted({s.descriptors.get("dataset_name", "") for s in samples})
        dataset_map = {
            dn: ([s for s in samples if s.descriptors.get("dataset_name") == dn] if dn else samples)
            for dn in dataset_names
        }

    # Pre-compute per-(dataset, descriptor) sample fingerprints for cache hashing.
    # Fingerprint = sorted list of (sample_id, label_value) pairs for valid samples.
    def _sample_fps(dataset_samples: list, descriptor: str) -> list[tuple[str, object]]:
        return sorted(
            (s.id, s.descriptors[descriptor])
            for s in dataset_samples
            if descriptor in s.descriptors
        )

    all_probes: dict[tuple, object] = {}
    all_fit_infos: list[ProbeFitInfo] = []
    cache_hits = 0
    newly_trained = 0

    n_layers = len(layer_list)
    for dataset_name in dataset_names:
        dataset_samples = dataset_map[dataset_name]
        print(f"  dataset={dataset_name!r}  ({len(dataset_samples)} samples)")

        for layer_idx, layer in enumerate(layer_list):
            layer_new = 0
            layer_cached = 0
            for position in train_positions:
                for probe_cls, probe_kwargs, method_name, method_descriptors, method_seeds in normal_configs:
                    for descriptor in method_descriptors:
                      for init_seed in method_seeds:
                        cache_key = probe_key(dataset_name, layer, position, method_name,
                                              descriptor, init_seed)
                        config_hash = compute_probe_config_hash(
                            layer=layer, position=position, method_name=method_name,
                            method_kwargs=probe_kwargs, descriptor=descriptor,
                            seed=seed, train_fraction=train_fraction,
                            sample_fingerprints=_sample_fps(dataset_samples, descriptor),
                            init_seed=init_seed, split_strategy=split_strategy,
                            centering=centering,
                        )
                        existing = cached_info.get(cache_key)
                        if existing and existing.config_hash == config_hash and cache_key in cached_probes:
                            all_probes[cache_key] = cached_probes[cache_key]
                            all_fit_infos.append(existing)
                            cache_hits += 1
                            layer_cached += 1
                            continue

                        probe, info = fit_probe(
                            activations, dataset_samples, layer, position,
                            probe_cls, probe_kwargs,
                            descriptor=descriptor,
                            train_fraction=train_fraction,
                            seed=seed,
                            dataset=dataset_name,
                            init_seed=init_seed,
                            split_strategy=split_strategy,
                            centering=centering,
                        )
                        if probe is not None:
                            all_probes[cache_key] = probe
                            info.family = output_name if merge_datasets else dataset_to_family.get(dataset_name, "")
                            info.config_hash = config_hash
                            all_fit_infos.append(info)
                            layer_new += 1
                            newly_trained += 1

            status = ""
            if layer_cached and layer_new:
                status = f"  {layer_cached} cached, {layer_new} new"
            elif layer_cached:
                status = f"  all {layer_cached} cached"
            elif layer_new:
                status = f"  {layer_new} trained"
            print(f"    layer {layer:3d}  ({layer_idx + 1}/{n_layers}){status}")

        # Layer-batched methods (CCS): fit every layer in one pass per
        # (position, descriptor), honouring the incremental cache per layer.
        for probe_cls, probe_kwargs, method_name, method_descriptors, method_seeds in batched_configs:
            for position in train_positions:
                for descriptor in method_descriptors:
                    # A batched pass fits every seed for a layer at once, so a layer is
                    # "to do" as soon as any of its seeds is missing.
                    hashes: dict[tuple, str] = {}
                    todo_layers: list[int] = []
                    for layer in layer_list:
                        layer_needed = False
                        for init_seed in method_seeds:
                            cache_key = probe_key(dataset_name, layer, position,
                                                  method_name, descriptor, init_seed)
                            config_hash = compute_probe_config_hash(
                                layer=layer, position=position, method_name=method_name,
                                method_kwargs=probe_kwargs, descriptor=descriptor,
                                seed=seed, train_fraction=train_fraction,
                                sample_fingerprints=_sample_fps(dataset_samples, descriptor),
                                init_seed=init_seed, split_strategy=split_strategy,
                                centering=centering,
                            )
                            hashes[cache_key] = config_hash
                            existing = cached_info.get(cache_key)
                            if existing and existing.config_hash == config_hash and cache_key in cached_probes:
                                all_probes[cache_key] = cached_probes[cache_key]
                                all_fit_infos.append(existing)
                                cache_hits += 1
                            else:
                                layer_needed = True
                        if layer_needed:
                            todo_layers.append(layer)

                    if not todo_layers:
                        continue

                    results = fit_probe_batched(
                        activations, dataset_samples, todo_layers, position,
                        probe_cls, probe_kwargs, seeds=method_seeds,
                        descriptor=descriptor, train_fraction=train_fraction,
                        seed=seed, dataset=dataset_name,
                        split_strategy=split_strategy, centering=centering,
                    )
                    n_done = 0
                    for (layer, init_seed), (probe, info) in results.items():
                        cache_key = probe_key(dataset_name, layer, position, method_name,
                                              descriptor, init_seed)
                        all_probes[cache_key] = probe
                        info.family = output_name if merge_datasets else dataset_to_family.get(dataset_name, "")
                        info.config_hash = hashes[cache_key]
                        all_fit_infos.append(info)
                        newly_trained += 1
                        n_done += 1
                    if n_done:
                        print(f"  [{method_name}] {dataset_name}: {n_done} probes fit in one "
                              f"batched pass ({len(todo_layers)} layers x {len(method_seeds)} "
                              f"seeds, pos={position}, desc={descriptor})")

    # Remove stale .pt files before writing the new one.
    for old_pt in probes_dir.glob("*.pt"):
        old_pt.unlink()

    run_id = str(uuid.uuid4())[:8]
    params_file = probes_dir / f"{run_id}.pt"
    save_probes(all_probes, params_file)

    manifest = Manifest(
        run_id=run_id,
        model=mp["name"],
        dataset="",
        params_file=str(params_file),
        entries=all_fit_infos,
    )
    manifest.save(probes_dir)

    print(f"\nDone: {cache_hits} loaded from cache, {newly_trained} retrained "
          f"({len(all_fit_infos)} total probes, run {run_id})")
    print(f"Manifest → {probes_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
