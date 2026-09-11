from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from ..activations.cache import PosKey, SampleCache
from ..data.loader import Sample

MIN_SAMPLES = 20


class Probe(ABC):
    method: str

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None: ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of positive class, shape (n_samples,)."""
        ...

    def raw_score(self, X: np.ndarray) -> np.ndarray:
        """Uncalibrated score sigma(X @ theta_hat) along the probe's leading direction.

        No bias, no calibration head and no sign disambiguation — this is the quantity
        unsupervised probing papers threshold at 0.5, and it is what makes a below-chance
        accuracy meaningful rather than a bug.  `predict_proba` remains the calibrated
        number used everywhere else.
        """
        theta = self.subspace[0].detach().double().cpu().numpy()
        theta = theta / (np.linalg.norm(theta) + 1e-12)
        # expit rather than 1/(1+exp(-z)): the latter overflows for large negative z.
        # The saturated value is the same, but the warning is not worth living with.
        return expit(X.astype(np.float64) @ theta)

    def predict_proba_matrix(self, X: np.ndarray) -> np.ndarray:
        """Return full class probability matrix, shape (n_samples, n_classes).
        Binary probes return a 2-column stack; multiclass probes override."""
        p = self.predict_proba(X)
        return np.column_stack([1 - p, p])

    @property
    @abstractmethod
    def subspace(self) -> torch.Tensor:
        """Orthonormal basis of the identified subspace, shape (k, hidden_size)."""
        ...

    @property
    def artifacts(self) -> dict[str, np.ndarray]:
        """Named arrays produced at fit time (eigenvalues, explained variance, …).
        Probes that expose diagnostics override this; default is empty."""
        return {}

    @abstractmethod
    def state_dict(self) -> dict: ...

    @classmethod
    @abstractmethod
    def from_state_dict(cls, d: dict) -> Probe: ...


@dataclass
class ProbeFitInfo:
    """Metadata produced when fitting a probe at a single (layer, position)."""
    layer: int
    train_position: PosKey
    method: str
    descriptor: str
    n_train: int
    test_ids: list[str] = field(default_factory=list)
    n_pairs: int | None = None  # set when a pair-level split was used
    artifacts: dict[str, np.ndarray] = field(default_factory=dict)
    dataset: str = ""            # sub-dataset label (e.g. "cities"); set by caller
    family: str = ""             # source family (e.g. "marks_tegmark"); set by caller
    config_hash: str = ""        # fingerprint used by incremental re-training logic
    seed: int | None = None      # probe-initialisation seed; None for deterministic probes


def compute_probe_config_hash(
    layer: int,
    position: PosKey,
    method_name: str,
    method_kwargs: dict,
    descriptor: str,
    seed: int,
    train_fraction: float,
    sample_fingerprints: list[tuple[str, int]],
    init_seed: int | None = None,
    split_strategy: str = "random",
    centering: str = "midpoint",
) -> str:
    """Deterministic hash of everything that determines a probe's weights.

    Covers the layer, position, method config, descriptor, training hyperparams,
    and the (id, label) fingerprint of the training samples.  Changing any of
    these invalidates the cached probe and triggers re-training.
    """
    payload = {
        "layer": layer,
        "position": str(position),
        "method": method_name,
        "kwargs": sorted(method_kwargs.items()),
        "descriptor": descriptor,
        "seed": seed,
        "train_fraction": train_fraction,
        "samples": sorted(sample_fingerprints),
    }
    # Added only when non-default, so probes fitted before these options existed keep
    # their hash and stay cached.
    if init_seed is not None:
        payload["init_seed"] = init_seed
    if split_strategy != "random":
        payload["split_strategy"] = split_strategy
    if centering != "midpoint":
        payload["centering"] = centering
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


@dataclass
class ProbeEvalResult:
    """Evaluation metrics for a probe measured at a (possibly different) position."""
    layer: int
    train_position: PosKey
    eval_position: PosKey
    method: str
    descriptor: str
    accuracy: float
    auc: float
    n_eval: int
    pair_accuracy: float | None = None  # fraction of pairs where base scores > counterfactual
    dataset: str = ""
    seed: int | None = None
    # Unsupervised scoring (see eval_probe_unsupervised): threshold
    # (p_base + 1 - p_cf)/2 at 0.5 against the sample-level truth.
    uncalibrated_accuracy: float | None = None
    pos_prob_mean: float | None = None
    neg_prob_mean: float | None = None
    sign_flipped: bool | None = None


# ── Pair utilities ────────────────────────────────────────────────────────────

def _extract_pairs(
    samples: list[Sample],
) -> list[tuple[Sample, Sample]] | None:
    """
    Reconstruct (base, counterfactual) pairs from pair_id / pair_role descriptors.
    Only pairs where both roles are present in `samples` are returned.
    Returns None if no pair structure is detected.
    """
    has_pair_fields = [
        s for s in samples
        if "pair_id" in s.descriptors and "pair_role" in s.descriptors
    ]
    if not has_pair_fields:
        return None

    by_pair: dict[str, dict[str, Sample]] = defaultdict(dict)
    for s in has_pair_fields:
        by_pair[s.descriptors["pair_id"]][s.descriptors["pair_role"]] = s

    result = [
        (roles["base"], roles["counterfactual"])
        for roles in by_pair.values()
        if "base" in roles and "counterfactual" in roles
    ]
    return result or None


def _center_group_keys(pairs: list[tuple[Sample, Sample]]) -> list[tuple]:
    """
    Centering-group key per pair: ``(dataset_name, center_key)``.

    Paired fits subtract each group's own mean midpoint so every contrast cloud
    is centred at the origin before the cross-covariance / diff-in-means / CCS
    direction is estimated.  Historically the group was just ``dataset_name`` —
    correct when one dataset_name == one contrast type (ToT, SNLI compounds).

    ``center_key`` lets a *single* dataset_name hold several contrast types that
    must each be centred independently.  The temporal circular set is exactly
    this case: every unordered pair of elements (e.g. {Mon, Wed}) is its own
    "pair type" and gets its own midpoint, so cross-covariance recovers the
    pairwise-difference geometry (the circle) rather than mixing in the absolute
    offsets of the per-element clouds.  Absent ⇒ "" ⇒ grouping collapses to
    dataset_name, i.e. the historical behaviour (unchanged for all other data).
    """
    keys = []
    for b, c in pairs:
        ds = b.descriptors.get("dataset_name", "") or c.descriptors.get("dataset_name", "")
        ck = b.descriptors.get("center_key", "") or c.descriptors.get("center_key", "")
        keys.append((ds, ck))
    return keys


def _grouped_midpoints(
    pairs: list[tuple[Sample, Sample]], Xb: np.ndarray, Xc: np.ndarray
) -> np.ndarray:
    """
    Per-pair midpoint to subtract from both sides, grouped by ``_center_group_keys``.

    Returns an array shaped like ``Xb`` (one midpoint row per pair); subtract it
    from both ``Xb`` and ``Xc`` to centre each group at the origin.
    """
    keys = _center_group_keys(pairs)
    mm = np.zeros_like(Xb)
    for g in dict.fromkeys(keys):  # unique, insertion order
        mask = np.array([k == g for k in keys])
        mm[mask] = (Xb[mask].mean(0) + Xc[mask].mean(0)) / 2
    return mm


def _center_paired(
    pairs: list[tuple[Sample, Sample]],
    Xb: np.ndarray,
    Xc: np.ndarray,
    mode: str = "midpoint",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Centre a paired activation cloud, per ``_center_group_keys`` group.

    ``midpoint``   subtract (mean(Xb) + mean(Xc))/2 from *both* sides — the historical
                   behaviour, and the default everywhere.
    ``per_branch`` subtract each branch's own mean from that branch.  This is what
                   contrast-consistency pipelines normalise with before probing.
    ``none``       leave the data alone.

    The two are not interchangeable: writing d = (mean(Xb) - mean(Xc))/2, the
    symmetrised cross-covariance satisfies

        S_cross(midpoint) = S_cross(per_branch) - 2n . d d^T

    so midpoint centering folds the diff-in-means direction into the recovered contrast
    direction, while per-branch centering removes it.
    """
    if mode == "none":
        return Xb, Xc
    if mode not in ("midpoint", "per_branch"):
        raise ValueError(f"unknown centering mode {mode!r}")

    keys = _center_group_keys(pairs)
    Xb, Xc = Xb.copy(), Xc.copy()
    for g in dict.fromkeys(keys):  # unique, insertion order
        mask = np.array([k == g for k in keys])
        if mode == "midpoint":
            m = (Xb[mask].mean(0) + Xc[mask].mean(0)) / 2
            Xb[mask] -= m
            Xc[mask] -= m
        else:
            Xb[mask] -= Xb[mask].mean(0)
            Xc[mask] -= Xc[mask].mean(0)
    return Xb, Xc


def _split_pairs(
    pairs: list[tuple[Sample, Sample]],
    train_fraction: float,
    seed: int,
    split_strategy: str = "random",
) -> tuple[list, list]:
    """
    Partition pairs into (train, test) at the pair — or, when ``group_id`` is present,
    the group — level, so both sides of a pair and all edges of a group stay together.

    ``random``  shuffle with ``seed`` (the default).
    ``ordered`` keep the incoming order and cut at ``train_fraction``.  Use this when the
                *generator* already shuffled and the split must be reproducible
                independently of the probe seed.
    """
    n_pairs = len(pairs)
    pair_group_ids = [
        pair[0].descriptors.get("group_id") or pair[1].descriptors.get("group_id")
        for pair in pairs
    ]
    rng = np.random.default_rng(seed)

    if any(pair_group_ids):
        by_group: dict[str, list] = defaultdict(list)
        for pair, gid in zip(pairs, pair_group_ids):
            by_group[gid or f"_solo_{id(pair)}"].append(pair)
        group_list = list(by_group.values())
        n_groups = len(group_list)
        perm = np.arange(n_groups) if split_strategy == "ordered" else rng.permutation(n_groups)
        n_train_groups = max(1, int(n_groups * train_fraction))
        train_pairs = [p for i in perm[:n_train_groups] for p in group_list[i]]
        test_pairs = [p for i in perm[n_train_groups:] for p in group_list[i]]
    else:
        perm = np.arange(n_pairs) if split_strategy == "ordered" else rng.permutation(n_pairs)
        n_train_pairs = max(1, int(n_pairs * train_fraction))
        train_pairs = [pairs[i] for i in perm[:n_train_pairs]]
        test_pairs = [pairs[i] for i in perm[n_train_pairs:]]

    return train_pairs, test_pairs


def _pair_accuracy(
    probe: Probe,
    pairs: list[tuple[Sample, Sample]],
    activations: dict[str, SampleCache],
    layer: int,
    eval_position: PosKey,
) -> float | None:
    """Fraction of pairs whose base side outscores its counterfactual.

    Scored in two batched calls rather than one per sample: at sweep scale this is the
    difference between minutes and hours, and the result is identical.
    """
    key = (layer, eval_position)
    usable = [
        (b, c) for b, c in pairs
        if activations.get(b.id, {}).get(key) is not None
        and activations.get(c.id, {}).get(key) is not None
    ]
    if not usable:
        return None
    Xb = np.stack([activations[b.id][key].float().numpy() for b, _ in usable])
    Xc = np.stack([activations[c.id][key].float().numpy() for _, c in usable])
    return float((probe.predict_proba(Xb) > probe.predict_proba(Xc)).mean())


def _eval_metrics(y: np.ndarray, y_prob_matrix: np.ndarray) -> tuple[np.ndarray, float]:
    """Compute (y_pred, auc) from full probability matrix, handling binary and multiclass."""
    n_classes = len(np.unique(y))
    if n_classes > 2:
        return (
            y_prob_matrix.argmax(axis=1),
            float(roc_auc_score(y, y_prob_matrix, multi_class="ovr")),
        )
    y_prob_1d = y_prob_matrix[:, 1]
    return (y_prob_1d >= 0.5).astype(int), float(roc_auc_score(y, y_prob_1d))


# ── Core probe fitting ────────────────────────────────────────────────────────

def fit_probe(
    activations: dict[str, SampleCache],
    samples: list[Sample],
    layer: int,
    position: PosKey,
    probe_cls: type[Probe],
    probe_kwargs: dict | None = None,
    descriptor: str = "label",
    train_fraction: float = 0.8,
    seed: int = 42,
    dataset: str = "",
    init_seed: int | None = None,
    split_strategy: str = "random",
    centering: str = "midpoint",
) -> tuple[Probe, ProbeFitInfo] | tuple[None, None]:
    """
    Fit a probe at (layer, position).  Returns (probe, fit_info) or (None, None)
    when there are too few samples.

    When samples carry pair_id / pair_role descriptors the train/test split is
    performed at the pair level so both sides of each pair land in the same
    partition.  Probes that declare requires_pairs = True receive matched
    (X_base, X_cf) arrays via fit_paired instead of the standard fit(X, y).

    ``seed`` controls the split; ``init_seed``, when given, is passed to the probe as its
    initialisation seed and recorded on the fit info.  Keeping them separate is what lets
    a seed sweep vary the initialisation while holding the split fixed.
    ``split_strategy`` and ``centering`` are documented on _split_pairs / _center_paired.
    """
    valid = [
        s for s in samples
        if (layer, position) in activations.get(s.id, {})
        and descriptor in s.descriptors
    ]

    pairs = _extract_pairs(valid)
    requires_pairs = getattr(probe_cls, "requires_pairs", False)

    if requires_pairs and pairs is None:
        return None, None

    kwargs = dict(probe_kwargs or {})
    if init_seed is not None:
        kwargs["seed"] = init_seed
    probe = probe_cls(**kwargs)

    if pairs is not None:
        n_pairs = len(pairs)
        if n_pairs < max(MIN_SAMPLES // 2, 2):
            return None, None

        # Split at the group level when group_id is present so all edges of a
        # prism group land in the same partition (otherwise a three-edge triangle
        # gets scattered across train and test, producing incomplete connector
        # lines in the subspace scatter).
        train_pairs, test_pairs = _split_pairs(pairs, train_fraction, seed, split_strategy)

        test_ids = [s.id for pair in test_pairs for s in pair]

        if hasattr(probe, "fit_paired"):
            X_base = np.stack([activations[b.id][(layer, position)].float().numpy() for b, _ in train_pairs])
            X_cf   = np.stack([activations[c.id][(layer, position)].float().numpy() for _, c in train_pairs])
            # Per-contrast-type centering: group pairs by (dataset_name, center_key)
            # so every contrast cloud is centred at the origin.  Works for single-family
            # (one group), compound runs (one group per dataset_name) and the temporal
            # circular set (one group per element pair via center_key) without blending.
            X_base, X_cf = _center_paired(train_pairs, X_base, X_cf, centering)
            probe.fit_paired(X_base, X_cf)
        else:
            train_samples = [s for pair in train_pairs for s in pair]
            X = np.stack([activations[s.id][(layer, position)].float().numpy() for s in train_samples])
            y = np.array([s.descriptors[descriptor] for s in train_samples])
            if len(np.unique(y)) < 2:
                return None, None
            probe.fit(X, y)

        info = ProbeFitInfo(
            layer=layer,
            train_position=position,
            method=probe.method,
            descriptor=descriptor,
            n_train=len(train_pairs) * 2,
            test_ids=test_ids,
            n_pairs=n_pairs,
            artifacts=probe.artifacts,
            dataset=dataset,
            seed=init_seed,
        )

    else:
        if len(valid) < MIN_SAMPLES:
            return None, None

        X = np.stack([activations[s.id][(layer, position)].float().numpy() for s in valid])
        y = np.array([s.descriptors[descriptor] for s in valid])

        if len(np.unique(y)) < 2:
            return None, None

        idx = np.arange(len(valid))
        train_idx, test_idx = train_test_split(
            idx, train_size=train_fraction, random_state=seed, stratify=y,
        )
        probe.fit(X[train_idx], y[train_idx])

        info = ProbeFitInfo(
            layer=layer,
            train_position=position,
            method=probe.method,
            descriptor=descriptor,
            n_train=len(train_idx),
            test_ids=[valid[i].id for i in test_idx],
            artifacts=probe.artifacts,
            dataset=dataset,
            seed=init_seed,
        )

    return probe, info


def fit_probe_batched(
    activations: dict[str, SampleCache],
    samples: list[Sample],
    layers: list[int],
    position: PosKey,
    probe_cls: type[Probe],
    probe_kwargs: dict | None = None,
    *,
    seeds: list[int | None] = (None,),
    descriptor: str = "label",
    train_fraction: float = 0.8,
    seed: int = 42,
    dataset: str = "",
    split_strategy: str = "random",
    centering: str = "midpoint",
) -> dict[tuple[int, int | None], tuple[Probe, ProbeFitInfo]]:
    """
    Fit ``probe_cls`` at every (layer, init-seed) in one batched pass, for a single
    (dataset, position).  Returns {(layer, init_seed): (probe, fit_info)}.

    The probe class must expose ``fit_paired_batched(X_base, X_cf, *, seeds, **kwargs)``
    taking (L, n, d) arrays and returning an L x S grid of fitted probes.

    The pair-level train/test split and centering mirror fit_probe exactly; the split is
    layer- and seed-independent, so it is computed once and reused, which is precisely
    what makes a 30-seed sweep comparable across seeds.
    """
    layers = list(layers)
    seed_list = list(seeds)
    if not layers or not seed_list:
        return {}

    # Samples usable at *every* requested layer.  Extraction caches all configured
    # layers uniformly, so this is normally just the full descriptor-labelled set.
    valid = [
        s for s in samples
        if descriptor in s.descriptors
        and all((layer, position) in activations.get(s.id, {}) for layer in layers)
    ]
    pairs = _extract_pairs(valid)
    if pairs is None:
        return {}
    n_pairs = len(pairs)
    if n_pairs < max(MIN_SAMPLES // 2, 2):
        return {}

    train_pairs, test_pairs = _split_pairs(pairs, train_fraction, seed, split_strategy)
    test_ids = [s.id for pair in test_pairs for s in pair]

    # Build per-layer centered (X_base, X_cf), stacked to (L, n_train, d).
    Xb_layers, Xc_layers = [], []
    for layer in layers:
        Xb = np.stack([activations[b.id][(layer, position)].float().numpy() for b, _ in train_pairs])
        Xc = np.stack([activations[c.id][(layer, position)].float().numpy() for _, c in train_pairs])
        Xb, Xc = _center_paired(train_pairs, Xb, Xc, centering)
        Xb_layers.append(Xb)
        Xc_layers.append(Xc)

    kw = dict(probe_kwargs or {})
    sweeping = any(sd is not None for sd in seed_list)
    grid = probe_cls.fit_paired_batched(
        np.stack(Xb_layers), np.stack(Xc_layers),
        seeds=[int(sd) for sd in seed_list] if sweeping else None,
        **kw,
    )

    out: dict[tuple[int, int | None], tuple[Probe, ProbeFitInfo]] = {}
    for layer, row in zip(layers, grid):
        for init_seed, probe in zip(seed_list, row):
            info = ProbeFitInfo(
                layer=layer,
                train_position=position,
                method=probe.method,
                descriptor=descriptor,
                n_train=len(train_pairs) * 2,
                test_ids=test_ids,
                n_pairs=n_pairs,
                artifacts=probe.artifacts,
                dataset=dataset,
                seed=init_seed,
            )
            out[(layer, init_seed)] = (probe, info)
    return out


def fit_probe_ccs_layerbatched(
    activations: dict[str, SampleCache],
    samples: list[Sample],
    layers: list[int],
    position: PosKey,
    probe_kwargs: dict | None = None,
    descriptor: str = "label",
    train_fraction: float = 0.8,
    seed: int = 42,
    dataset: str = "",
) -> dict[int, tuple[Probe, ProbeFitInfo]]:
    """Back-compat wrapper: CCS at every layer, single seed, keyed by layer alone."""
    from .ccs import CCSProbe

    out = fit_probe_batched(
        activations, samples, layers, position, CCSProbe, probe_kwargs,
        descriptor=descriptor, train_fraction=train_fraction, seed=seed, dataset=dataset,
    )
    return {layer: value for (layer, _), value in out.items()}


def eval_probe(
    probe: Probe,
    fit_info: ProbeFitInfo,
    activations: dict[str, SampleCache],
    samples: list[Sample],
    eval_position: PosKey,
) -> ProbeEvalResult | None:
    """
    Evaluate a fitted probe at eval_position on the held-out test samples.
    When pairs are present in the test set, also computes pair_accuracy:
    the fraction of pairs where the base sample scores higher than the
    counterfactual.
    """
    test_id_set = set(fit_info.test_ids)
    valid = [
        s for s in samples
        if s.id in test_id_set
        and (fit_info.layer, eval_position) in activations.get(s.id, {})
        and fit_info.descriptor in s.descriptors
    ]
    if len(valid) < MIN_SAMPLES:
        return None

    X = np.stack([activations[s.id][(fit_info.layer, eval_position)].float().numpy() for s in valid])
    y = np.array([s.descriptors[fit_info.descriptor] for s in valid])

    y_prob = probe.predict_proba(X)
    y_pred, auc = _eval_metrics(y, probe.predict_proba_matrix(X))

    pair_acc = None
    if fit_info.n_pairs is not None:
        test_pairs = _extract_pairs(valid)
        if test_pairs:
            pair_acc = _pair_accuracy(probe, test_pairs, activations,
                                      fit_info.layer, eval_position)

    return ProbeEvalResult(
        layer=fit_info.layer,
        train_position=fit_info.train_position,
        eval_position=eval_position,
        method=fit_info.method,
        descriptor=fit_info.descriptor,
        accuracy=float(accuracy_score(y, y_pred)),
        auc=auc,
        n_eval=len(valid),
        pair_accuracy=pair_acc,
        dataset=fit_info.dataset,
        seed=fit_info.seed,
    )


def eval_probe_ood(
    probe: Probe,
    fit_info: ProbeFitInfo,
    activations: dict[str, SampleCache],
    samples: list[Sample],
    eval_position: PosKey,
    eval_dataset: str = "",
) -> ProbeEvalResult | None:
    """
    Evaluate a fitted probe on all provided samples (no train/test split filter).
    Used for out-of-domain evaluation: probe trained on source, evaluated on target.
    """
    valid = [
        s for s in samples
        if (fit_info.layer, eval_position) in activations.get(s.id, {})
        and fit_info.descriptor in s.descriptors
    ]
    if len(valid) < MIN_SAMPLES:
        return None

    X = np.stack([activations[s.id][(fit_info.layer, eval_position)].float().numpy() for s in valid])
    y = np.array([s.descriptors[fit_info.descriptor] for s in valid])

    y_prob = probe.predict_proba(X)
    y_pred, auc = _eval_metrics(y, probe.predict_proba_matrix(X))

    pair_acc = None
    all_pairs = _extract_pairs(valid)
    if all_pairs:
        pair_acc = _pair_accuracy(probe, all_pairs, activations,
                                  fit_info.layer, eval_position)

    return ProbeEvalResult(
        layer=fit_info.layer,
        train_position=fit_info.train_position,
        eval_position=eval_position,
        method=fit_info.method,
        descriptor=fit_info.descriptor,
        accuracy=float(accuracy_score(y, y_pred)),
        auc=auc,
        n_eval=len(valid),
        pair_accuracy=pair_acc,
        dataset=eval_dataset,
        seed=fit_info.seed,
    )


def eval_probe_transfer(
    probe: Probe,
    fit_info: ProbeFitInfo,
    activations: dict[str, SampleCache],
    samples: list[Sample],
    eval_position: PosKey,
    eval_dataset: str = "",
) -> ProbeEvalResult | None:
    """
    Burns-style no-refit transfer: apply a probe fitted on a *source* dataset to a
    *different* target dataset, with neither the direction nor the calibration head
    refit (cf. eval_probe_ood_refit, which refits a logistic head — not faithful to
    Burns et al. 2023 Fig 2).

    The target is re-centred by its own midpoint before scoring, mirroring the
    per-dataset centering fit_probe applies at train time (subtract the mean of the
    base/cf pair means).  pair_accuracy — the fraction of target pairs where the
    base side scores above the counterfactual — is the primary transfer metric; it
    is invariant to this constant shift, while accuracy/AUC benefit from it.
    """
    valid = [
        s for s in samples
        if (fit_info.layer, eval_position) in activations.get(s.id, {})
        and fit_info.descriptor in s.descriptors
    ]
    if len(valid) < MIN_SAMPLES:
        return None

    def _act(s: Sample) -> np.ndarray:
        return activations[s.id][(fit_info.layer, eval_position)].float().numpy()

    X = np.stack([_act(s) for s in valid]).astype(np.float64)
    y = np.array([s.descriptors[fit_info.descriptor] for s in valid])

    # Re-centre the target by its own midpoint (matches fit_probe's per-contrast
    # centering: per (dataset_name, center_key) group).  Use grouped pair
    # midpoints when pairs are present, else the grand mean.
    pairs = _extract_pairs(valid)
    if pairs is not None:
        Xb = np.stack([_act(b) for b, _ in pairs])
        Xc = np.stack([_act(c) for _, c in pairs])
        mm_pair = _grouped_midpoints(pairs, Xb, Xc)
        mm_by_id: dict[str, np.ndarray] = {}
        for (b, c), m in zip(pairs, mm_pair):
            mm_by_id[b.id] = m
            mm_by_id[c.id] = m
        grand = X.mean(axis=0)
        mm = np.stack([mm_by_id.get(s.id, grand) for s in valid])
    else:
        mm = X.mean(axis=0)

    y_pred, auc = _eval_metrics(y, probe.predict_proba_matrix(X - mm))

    pair_acc = None
    if pairs is not None:
        # Batched, with each pair re-centred by its own midpoint (as above).
        mids = np.stack([mm_by_id[b.id] for b, _ in pairs])
        Pb = probe.predict_proba(np.stack([_act(b) for b, _ in pairs]) - mids)
        Pc = probe.predict_proba(np.stack([_act(c) for _, c in pairs]) - mids)
        pair_acc = float((Pb > Pc).mean())

    return ProbeEvalResult(
        layer=fit_info.layer,
        train_position=fit_info.train_position,
        eval_position=eval_position,
        method=fit_info.method,
        descriptor=fit_info.descriptor,
        accuracy=float(accuracy_score(y, y_pred)),
        auc=auc,
        n_eval=len(valid),
        pair_accuracy=pair_acc,
        dataset=eval_dataset,
        seed=fit_info.seed,
    )


def eval_probe_ood_refit(
    probe: Probe,
    fit_info: ProbeFitInfo,
    activations: dict[str, SampleCache],
    samples: list[Sample],
    eval_position: PosKey,
    eval_dataset: str = "",
    train_fraction: float = 0.8,
    seed: int = 42,
) -> ProbeEvalResult | None:
    """
    OOD evaluation with the logistic head refit on the eval dataset's own train split.

    The geometric subspace (directions) is frozen from `probe` (fitted on the source
    compound).  A fresh LogisticRegression is fitted on the train portion of the eval
    dataset projected onto that subspace; accuracy/AUC are measured on the test portion.

    The split mirrors fit_probe: pair-level when pair structure is present (both roles
    of each pair land in the same partition), stratified by label otherwise.
    """
    valid = [
        s for s in samples
        if (fit_info.layer, eval_position) in activations.get(s.id, {})
        and fit_info.descriptor in s.descriptors
    ]
    if len(valid) < MIN_SAMPLES:
        return None

    W      = probe.subspace.float().numpy().astype(np.float64)  # (k, hidden_size)
    X_all  = np.stack([activations[s.id][(fit_info.layer, eval_position)].float().numpy()
                       for s in valid]).astype(np.float64)
    X_proj = X_all @ W.T                                # (n, k)
    id_to_proj = {s.id: X_proj[i] for i, s in enumerate(valid)}

    pairs = _extract_pairs(valid)

    if pairs is not None:
        n_pairs = len(pairs)
        if n_pairs < max(MIN_SAMPLES // 2, 2):
            return None
        rng    = np.random.default_rng(seed)
        perm   = rng.permutation(n_pairs)
        n_train     = max(1, int(n_pairs * train_fraction))
        train_pairs = [pairs[i] for i in perm[:n_train]]
        test_pairs  = [pairs[i] for i in perm[n_train:]]

        train_samples = [s for pair in train_pairs for s in pair]
        test_samples  = [s for pair in test_pairs  for s in pair]
        X_train = np.stack([id_to_proj[s.id] for s in train_samples])
        y_train = np.array([s.descriptors[fit_info.descriptor] for s in train_samples])
        X_test  = np.stack([id_to_proj[s.id] for s in test_samples])
        y_test  = np.array([s.descriptors[fit_info.descriptor] for s in test_samples])

    else:
        y_all = np.array([s.descriptors[fit_info.descriptor] for s in valid])
        idx   = np.arange(len(valid))
        try:
            train_idx, test_idx = train_test_split(
                idx, train_size=train_fraction, random_state=seed, stratify=y_all,
            )
        except ValueError:
            train_idx, test_idx = train_test_split(
                idx, train_size=train_fraction, random_state=seed,
            )
        train_samples = [valid[i] for i in train_idx]
        test_samples  = [valid[i] for i in test_idx]
        X_train = X_proj[train_idx]
        y_train = y_all[train_idx]
        X_test  = X_proj[test_idx]
        y_test  = y_all[test_idx]

    if len(test_samples) < MIN_SAMPLES:
        return None

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred, auc = _eval_metrics(y_test, clf.predict_proba(X_test))

    pair_acc = None
    if pairs is not None:
        usable = [(b, c) for b, c in test_pairs
                  if id_to_proj.get(b.id) is not None and id_to_proj.get(c.id) is not None]
        if usable:
            Pb = clf.predict_proba(np.stack([id_to_proj[b.id] for b, _ in usable]))[:, 1]
            Pc = clf.predict_proba(np.stack([id_to_proj[c.id] for _, c in usable]))[:, 1]
            pair_acc = float((Pb > Pc).mean())

    return ProbeEvalResult(
        layer=fit_info.layer,
        train_position=fit_info.train_position,
        eval_position=eval_position,
        method=fit_info.method,
        descriptor=fit_info.descriptor,
        accuracy=float(accuracy_score(y_test, y_pred)),
        auc=auc,
        n_eval=len(test_samples),
        pair_accuracy=pair_acc,
        dataset=eval_dataset,
        seed=fit_info.seed,
    )


def save_probes(probes: dict[tuple, Probe], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {key: {"method": p.method, "state_dict": p.state_dict()} for key, p in probes.items()},
        path,
    )


def load_probes(path: Path) -> dict[tuple, Probe]:
    from .ccs import CCSProbe
    from .crosscov import CrossCovarianceProbe, GeneralizedCrossCovarianceProbe
    from .dim import ContrastDiffInMeansProbe, DiffInMeansProbe
    from .linear import LogisticProbe
    from .pca import PairedPCAProbe, PCAProbe

    def _resolve(method: str) -> type[Probe]:
        if method == "logistic":
            return LogisticProbe
        if method == "diff_in_means":
            return DiffInMeansProbe
        if method == "contrast_diff_in_means":
            return ContrastDiffInMeansProbe
        if method == "ccs" or method.startswith("ccs["):
            return CCSProbe
        if method.startswith("pca_pair"):
            return PairedPCAProbe
        if method.startswith("pca"):
            return PCAProbe
        if method.startswith("generalized_cross_covariance"):
            return GeneralizedCrossCovarianceProbe
        if method.startswith("cross_covariance"):
            return CrossCovarianceProbe
        raise ValueError(f"Unknown probe method: {method!r}")

    data = torch.load(path, weights_only=False)
    return {key: _resolve(v["method"]).from_state_dict(v["state_dict"]) for key, v in data.items()}
