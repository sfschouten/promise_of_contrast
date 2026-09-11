"""
The manifest is a JSON sidecar to the probe .pt file.  It records, for every
fitted probe, the training metadata needed by evaluate_probes.py — in particular
the held-out test_ids so evaluation is always on unseen samples.

Layout inside probes_dir/:
  manifest.json   ← this file
  {run_id}.pt     ← all probe state_dicts, keyed by (layer, train_pos, method, descriptor)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .base import ProbeFitInfo


MANIFEST_FILENAME = "manifest.json"


def probe_key(
    dataset: str,
    layer: int,
    train_position,
    method: str,
    descriptor: str,
    seed: int | None = None,
) -> tuple:
    """The key a probe is stored under, in `.pt` files and in-memory dicts.

    Seedless probes keep the historical 5-tuple, so every probe already on disk stays
    loadable and keeps its identity; a seed sweep appends the initialisation seed as a
    sixth element.  Always build keys through this function rather than by hand — the
    tuple shape is otherwise easy to get inconsistent between writer and reader.
    """
    base = (dataset, layer, train_position, method, descriptor)
    return base if seed is None else base + (seed,)


@dataclass
class Manifest:
    run_id: str
    model: str
    dataset: str
    params_file: str        # path to the .pt file
    entries: list[ProbeFitInfo]

    def save(self, probes_dir: Path) -> None:
        path = Path(probes_dir) / MANIFEST_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._to_dict(), indent=2))

    @staticmethod
    def load(probes_dir: Path) -> Manifest:
        path = Path(probes_dir) / MANIFEST_FILENAME
        return Manifest._from_dict(json.loads(path.read_text()))

    # ── serialisation ─────────────────────────────────────────────────────────

    def _to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "model": self.model,
            "dataset": self.dataset,
            "params_file": self.params_file,
            "entries": [
                {
                    "layer":          e.layer,
                    "train_position": str(e.train_position),
                    "method":         e.method,
                    "descriptor":     e.descriptor,
                    "n_train":        e.n_train,
                    "test_ids":       e.test_ids,
                    "n_pairs":        e.n_pairs,
                    "dataset":        e.dataset,
                    "family":         e.family,
                    "config_hash":    e.config_hash,
                    "seed":           e.seed,
                }
                for e in self.entries
            ],
        }

    @staticmethod
    def _from_dict(d: dict) -> Manifest:
        entries = [
            ProbeFitInfo(
                layer=          e["layer"],
                train_position= e["train_position"],
                method=         e["method"],
                descriptor=     e["descriptor"],
                n_train=        e["n_train"],
                test_ids=       e["test_ids"],
                n_pairs=        e.get("n_pairs"),
                dataset=        e.get("dataset") or d.get("dataset", ""),
                family=         e.get("family", ""),
                config_hash=    e.get("config_hash", ""),
                seed=           e.get("seed"),
            )
            for e in d["entries"]
        ]
        return Manifest(
            run_id=      d["run_id"],
            model=       d["model"],
            dataset=     d["dataset"],
            params_file= d["params_file"],
            entries=     entries,
        )
