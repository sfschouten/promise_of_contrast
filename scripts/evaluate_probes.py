import argparse
import dataclasses
import sys
import time
import uuid
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.activations.extract import extract_hidden_states
from src.config import resolve_probing
from src.data.graph import load_family_as_pairs
from src.data.loader import load_jsonl
from src.probing import (
    Manifest,
    eval_probe,
    load_probes,
    probe_key,
    unsupervised_metrics,
)
from src.probing.base import ProbeEvalResult
from src.results.db import get_connection, init_schema, insert_probe_artifacts, insert_probe_evals, insert_probe_runs


def _find_run_key(params: dict, model_alias: str, family: str) -> str:
    for rkey, rcfg in params["runs"].items():
        if rcfg["model"] == model_alias and rcfg["family"] == family:
            return rkey
    raise KeyError(f"No run found for model={model_alias!r}, family={family!r}")


def _load_run(run_key: str, act_base: Path, params: dict) -> tuple[list, dict]:
    run_cfg = params["runs"][run_key]
    mp = params["models"][run_cfg["model"]]
    ap = params["activations"]
    family = run_cfg["family"]
    samples_path = Path(f"data/processed/samples_{family}.jsonl")
    if not samples_path.exists():
        print(f"  [{run_key}] samples not found, skipping")
        return [], {}
    nodes = load_jsonl(str(samples_path))
    fam_activations = extract_hidden_states(
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
    pair_samples = load_family_as_pairs(family)
    fam_samples = pair_samples if pair_samples is not None else nodes
    return fam_samples, fam_activations


def _run_eval(
    conn,
    label: str,
    samples: list,
    activations: dict,
    probes_dir: Path,
    pp: dict,
    eval_id: str,
) -> int:
    """Evaluate probes in probes_dir against samples/activations; return result count."""
    eval_positions = pp.get("eval_positions", "all")
    paper_scoring = pp.get("paper_scoring", False)
    centering = pp.get("centering", "midpoint")
    sign_from = pp.get("sign_from", "none")
    if eval_positions == "all":
        from src.activations.cache import _META_KEY
        all_keys = [k for cache in activations.values() for k in cache if k != _META_KEY]
        eval_position_list = sorted({p for _, p in all_keys}, key=lambda p: (str(type(p)), p))
    else:
        eval_position_list = list(eval_positions)

    manifest = Manifest.load(probes_dir)
    probes   = load_probes(manifest.params_file)

    all_results: list[ProbeEvalResult] = []
    n_entries = len(manifest.entries)
    n_missing = 0        # manifest entries with no probe in the store
    t_start = time.monotonic()
    for i, entry in enumerate(manifest.entries):
        if i and i % 200 == 0:
            elapsed = time.monotonic() - t_start
            eta = elapsed / i * (n_entries - i)
            print(f"    {i}/{n_entries} probes scored  "
                  f"ETA {int(eta // 60)}m{int(eta % 60):02d}s", flush=True)
        key = probe_key(entry.dataset, entry.layer, entry.train_position,
                        entry.method, entry.descriptor, entry.seed)
        probe = probes.get(key)
        if probe is None:
            # The manifest and the probe store have drifted apart — usually because probes
            # were pruned by hand without re-running train_probes.  Skipping quietly would
            # make a partial evaluation indistinguishable from a complete one, so count it
            # and complain at the end.
            n_missing += 1
            continue
        if not entry.dataset:
            dataset_samples = samples
        else:
            dataset_samples = [s for s in samples if s.descriptors.get("dataset_name", "") == entry.dataset]
            if not dataset_samples:  # compound probe: entry.dataset is the compound name, not a descriptor value
                dataset_samples = samples
        for eval_pos in eval_position_list:
            result = eval_probe(probe, entry, activations, dataset_samples, eval_pos)
            if result is None:
                continue
            if paper_scoring:
                # Same probe, same held-out pairs, scored without the calibration head —
                # merged into the same row so both views live side by side.
                extra = unsupervised_metrics(
                    probe, entry, activations, dataset_samples, eval_pos,
                    centering=centering, sign_from=sign_from,
                )
                if extra:
                    extra.pop("auc_unsupervised", None)
                    result = dataclasses.replace(result, **extra)
            all_results.append(result)

    entries_with_artifacts = []
    for entry in manifest.entries:
        probe = probes.get(probe_key(entry.dataset, entry.layer, entry.train_position,
                                     entry.method, entry.descriptor, entry.seed))
        entries_with_artifacts.append(
            dataclasses.replace(entry, artifacts=probe.artifacts) if probe is not None else entry
        )

    insert_probe_runs(
        conn, entries_with_artifacts,
        run_id=manifest.run_id,
        model=manifest.model,
        params_file=manifest.params_file,
    )
    insert_probe_artifacts(conn, entries_with_artifacts, run_id=manifest.run_id)
    insert_probe_evals(conn, all_results, eval_id=eval_id, run_id=manifest.run_id)

    if n_missing:
        print(f"  WARNING: {n_missing} of {n_entries} manifest entries had no probe in the "
              f"store and were skipped — re-run train_probes to resynchronise.",
              file=sys.stderr)
    print(f"  {label}: {len(manifest.entries)} probes × {len(eval_position_list)} positions "
          f"→ {len(all_results)} results")
    return len(all_results)


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run")
    group.add_argument("--run-compound")
    args = parser.parse_args()

    params = yaml.safe_load(Path("params.yaml").read_text())
    run_cfg_for_profile = (params["runs"][args.run] if args.run
                           else params["run_compounds"][args.run_compound])
    pp = resolve_probing(params, run_cfg_for_profile)
    op = params["output"]

    act_base    = Path(op["activations_dir"])
    probes_base = Path(op["probes_dir"])
    eval_base   = Path(op["eval_results_dir"])
    eval_base.mkdir(parents=True, exist_ok=True)

    eval_id = str(uuid.uuid4())[:8]

    if args.run:
        run_key = args.run
        probes_dir = probes_base / run_key
        if not (probes_dir / "manifest.json").exists():
            print(f"[{run_key}] probes not found, nothing to evaluate")
            return
        out_db = eval_base / f"{run_key}.duckdb"
        # Recreate rather than append: this stage's output is the full evaluation of one
        # run, so re-running it must replace the previous rows, not add to them.
        out_db.unlink(missing_ok=True)
        conn = get_connection(str(out_db))
        init_schema(conn)
        print(f"Evaluating {run_key} …")
        samples, activations = _load_run(run_key, act_base, params)
        if samples:
            _run_eval(conn, run_key, samples, activations, probes_dir, pp, eval_id)
        conn.close()
        print(f"→ {out_db}")

    else:
        rc_key = args.run_compound
        rc_cfg = params["run_compounds"][rc_key]
        compound_cfg = params["compounds"][rc_cfg["compound"]]
        model_alias = rc_cfg["model"]
        probes_dir = probes_base / rc_key
        if not (probes_dir / "manifest.json").exists():
            print(f"[{rc_key}] compound probes not found, nothing to evaluate")
            return
        out_db = eval_base / f"{rc_key}.duckdb"
        out_db.unlink(missing_ok=True)
        conn = get_connection(str(out_db))
        init_schema(conn)
        print(f"Evaluating compound '{rc_key}' …")
        all_samples, all_activations = [], {}
        for fam in compound_cfg["families"]:
            run_key = _find_run_key(params, model_alias, fam)
            fam_samples, fam_act = _load_run(run_key, act_base, params)
            all_samples.extend(fam_samples)
            all_activations.update(fam_act)
        filter_datasets = compound_cfg.get("datasets")
        if filter_datasets:
            all_samples = [s for s in all_samples
                           if s.descriptors.get("dataset_name", "") in filter_datasets]
        if all_samples:
            _run_eval(conn, rc_key, all_samples, all_activations, probes_dir, pp, eval_id)
        conn.close()
        print(f"→ {out_db}")


if __name__ == "__main__":
    main()
