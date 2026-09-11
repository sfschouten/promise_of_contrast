"""Unsupervised (uncalibrated) scoring for contrast probes.

`eval_probe` in base.py reports the *calibrated* accuracy: every probe carries a logistic
head fitted on the training split, and accuracy is that head's. That is the right number
for supervised probing, but it is not the number the contrast-probing literature reports.

There, a probe is scored directly from the direction it found:

    p_i   = sigma(x_i . theta_hat)                 no bias, no head
    p_pair = (p_base + 1 - p_cf) / 2               the pair's probability that base is true
    accuracy = mean( (p_pair > 0.5) == truth )

The direction's *sign*, however, is not determined by the objective: both the CCS losses
and the contrastive eigenproblems are invariant under theta -> -theta, so an unsupervised
fit finds the right axis but an arbitrary orientation, and accuracy comes out at either p
or 1-p depending on the seed.  Resolving it is an *evaluation* choice, not a training one,
and `sign_from` names the available conventions:

    "none"           leave the raw orientation.  Makes the ambiguity visible: accuracies
                     split bimodally across seeds, roughly 50/50.
    "train_accuracy" flip if the direction scores below chance on the training split.
    "diff_in_means"  flip if the direction points away from mean(true) - mean(false),
                     computed on the training split.  More stable than the accuracy rule
                     because it compares directions rather than thresholded predictions.
    "branch_means"   flip if the direction points away from mean(branch 0) - mean(branch 1).
                     **Label-free**: it uses only which framing a sample is, which the
                     template fixes.  Deterministic across seeds, but whether the resulting
                     orientation corresponds to "true" is then a property of the dataset
                     rather than something the convention can guarantee.

The first three cost exactly one bit of label information.  That is not a new concession —
accuracy against ground truth is already a supervised measurement.  What stays unsupervised
is the *training*.

Two details matter for reproducing such numbers:

* the evaluation half is centred by its *own* per-branch mean, exactly as the training
  half was (the calibration head would otherwise absorb the offset, which is why
  `eval_probe` can get away with no centering at all);
* `truth` is the sample-level label — which of the two framings is the correct one — and
  is distinct from the branch index. See the note in `scripts/generate_pwl.py`.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from ..activations.cache import PosKey, SampleCache
from ..data.loader import Sample
from .base import (
    MIN_SAMPLES,
    Probe,
    ProbeFitInfo,
    _center_paired,
    _extract_pairs,
)


def _should_flip(probe, train_pairs, activations, key, centering, desc, sign_from,
                 pair_scores) -> bool:
    """Decide whether the probe's orientation should be reversed, on the training split."""
    if sign_from == "train_accuracy":
        p_train = pair_scores(train_pairs)
        y_train = np.array([b.descriptors[desc] for b, _ in train_pairs]).astype(int)
        return bool(((p_train > 0.5).astype(int) == y_train).mean() < 0.5)

    Xb = np.stack([activations[b.id][key].float().numpy() for b, _ in train_pairs])
    Xc = np.stack([activations[c.id][key].float().numpy() for _, c in train_pairs])
    Xb, Xc = _center_paired(train_pairs, Xb, Xc, centering)

    if sign_from == "diff_in_means":
        # mean(correct framing) - mean(incorrect framing).  `truth` says whether branch 0
        # is the correct one, so it selects which side of each pair is which.
        truth = np.array([b.descriptors[desc] for b, _ in train_pairs]).astype(bool)
        true_side = np.where(truth[:, None], Xb, Xc)
        false_side = np.where(truth[:, None], Xc, Xb)
        reference = true_side.mean(0) - false_side.mean(0)
    else:  # "branch_means" — label-free: which framing, not which is true
        reference = Xb.mean(0) - Xc.mean(0)

    theta = probe.subspace.detach().double().cpu().numpy()[0]
    return bool(float(np.dot(theta, reference)) < 0)


def unsupervised_metrics(
    probe: Probe,
    fit_info: ProbeFitInfo,
    activations: dict[str, SampleCache],
    samples: list[Sample],
    eval_position: PosKey,
    centering: str = "per_branch",
    truth_descriptor: str = "truth",
    sign_from: str = "none",
) -> dict[str, float] | None:
    """Score `probe` on its held-out pairs without using its calibration head.

    Returns {"uncalibrated_accuracy", "pos_prob_mean", "neg_prob_mean", "sign_flipped"},
    or None when the probe has no usable pair structure at this position.  Designed to be
    merged into the ProbeEvalResult that `eval_probe` produces, so a single row carries
    both the calibrated and the uncalibrated view.
    """
    test_id_set = set(fit_info.test_ids)
    valid = [
        s for s in samples
        if s.id in test_id_set
        and (fit_info.layer, eval_position) in activations.get(s.id, {})
    ]
    if len(valid) < MIN_SAMPLES:
        return None

    pairs = _extract_pairs(valid)
    if not pairs:
        return None

    key = (fit_info.layer, eval_position)

    def _pair_scores(ps):
        Xb = np.stack([activations[b.id][key].float().numpy() for b, _ in ps])
        Xc = np.stack([activations[c.id][key].float().numpy() for _, c in ps])
        Xb, Xc = _center_paired(ps, Xb, Xc, centering)
        return (probe.raw_score(Xb) + 1.0 - probe.raw_score(Xc)) / 2.0

    p_pair = _pair_scores(pairs)

    # Truth is a property of the *pair*: it says which framing is correct.  Fall back to
    # the fitting descriptor for datasets that predate the distinction.
    desc = truth_descriptor if truth_descriptor in pairs[0][0].descriptors else fit_info.descriptor
    if desc not in pairs[0][0].descriptors:
        return None
    y = np.array([b.descriptors[desc] for b, _ in pairs]).astype(int)
    if len(np.unique(y)) < 2:
        return None

    # Orientation, decided on the training half and then applied unchanged here.
    # Flipping theta maps every pair score p to 1 - p, so the flip is one subtraction.
    flipped = False
    if sign_from != "none":
        if sign_from not in ("train_accuracy", "diff_in_means", "branch_means"):
            raise ValueError(f"unknown sign_from {sign_from!r}")
        held_out = set(fit_info.test_ids)
        train_pairs = _extract_pairs([
            s for s in samples
            if s.id not in held_out
            and (fit_info.layer, eval_position) in activations.get(s.id, {})
        ])
        if train_pairs:
            flipped = _should_flip(probe, train_pairs, activations, key, centering,
                                   desc, sign_from, _pair_scores)
            if flipped:
                p_pair = 1.0 - p_pair

    out = {
        "sign_flipped": flipped,
        # No max(acc, 1-acc): sub-chance scores are a real result for an unsupervised probe.
        "uncalibrated_accuracy": float(((p_pair > 0.5).astype(int) == y).mean()),
        "pos_prob_mean": float(p_pair[y == 1].mean()),
        "neg_prob_mean": float(p_pair[y == 0].mean()),
    }
    try:
        out["auc_unsupervised"] = float(roc_auc_score(y, p_pair))
    except ValueError:
        pass
    return out
