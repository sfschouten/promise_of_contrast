"""Tests for how the orientation of an unsupervised direction is resolved.

Both CCS objectives and the contrastive eigenproblems are invariant under theta -> -theta.
A fit therefore recovers the right *axis* but an arbitrary *orientation*, and accuracy
comes out at either p or 1-p. These tests pin down the two ways of handling that:

    sign_from="none"   leave it, so the ambiguity is visible in the spread across seeds
    sign_from="train_accuracy"  choose the orientation that beats chance on the training split,
                       then apply it unchanged to the held-out split
"""
import numpy as np
import pytest
import torch

from src.activations.cache import SampleCache
from src.data.loader import Sample
from src.probing.base import Probe, ProbeFitInfo
from src.probing.paper_eval import unsupervised_metrics


class _FixedDirection(Probe):
    """A probe that is nothing but a direction, so the sign logic is what is tested."""
    method = "fixed"

    def __init__(self, direction):
        self._d = np.asarray(direction, dtype=float)

    def fit(self, X, y):
        raise NotImplementedError

    def predict_proba(self, X):
        return self.raw_score(X)

    @property
    def subspace(self):
        return torch.from_numpy(self._d).float().unsqueeze(0)

    def state_dict(self):
        return {"d": self._d}

    @classmethod
    def from_state_dict(cls, d):
        return cls(d["d"])


def _dataset(n_pairs=40, d=4, separation=6.0, seed=0):
    """Pairs whose two branches are separated along axis 0, with truth alternating.

    Half the pairs have branch 0 correct, half branch 1, so accuracy is only above chance
    if the direction's orientation matches the labelling.
    """
    rng = np.random.default_rng(seed)
    samples, acts, pairs = [], {}, []
    for i in range(n_pairs):
        truth = i % 2                      # 1 => branch 0 is the correct framing
        offset = separation if truth else -separation
        base_vec = rng.normal(size=d) * 0.1 + np.array([offset] + [0.0] * (d - 1))
        cf_vec = rng.normal(size=d) * 0.1 - np.array([offset] + [0.0] * (d - 1))
        for role, vec, sid in (("base", base_vec, f"b{i}"), ("counterfactual", cf_vec, f"c{i}")):
            s = Sample(id=sid, text="", descriptors={
                "dataset_name": "d", "pair_id": f"p{i}", "pair_role": role,
                "truth": truth, "label": truth,
            })
            samples.append(s)
            acts[sid] = {(0, "last"): torch.tensor(vec, dtype=torch.float32)}
        pairs.append(i)
    return samples, acts


def _info(samples, n_test=20):
    """Hold out the last n_test pairs."""
    test_ids = [s.id for s in samples if int(s.id[1:]) >= (len(samples) // 2 - n_test)]
    return ProbeFitInfo(layer=0, train_position="last", method="fixed",
                        descriptor="label", n_train=0, test_ids=test_ids,
                        n_pairs=len(samples) // 2, dataset="d")


@pytest.mark.parametrize("sign", [+1.0, -1.0])
def test_train_split_sign_resolution_is_orientation_invariant(sign):
    """Flipping the probe must not change the reported accuracy once the sign is
    resolved on the training split — that is the whole point."""
    samples, acts = _dataset()
    info = _info(samples)
    probe = _FixedDirection(sign * np.array([1.0, 0, 0, 0]))
    got = unsupervised_metrics(probe, info, acts, samples, "last",
                               centering="none", sign_from="train_accuracy")
    assert got is not None
    assert got["uncalibrated_accuracy"] == pytest.approx(1.0)


def test_without_resolution_the_two_orientations_are_mirror_images():
    samples, acts = _dataset()
    info = _info(samples)
    a = unsupervised_metrics(_FixedDirection([1.0, 0, 0, 0]), info, acts, samples,
                             "last", centering="none", sign_from="none")
    b = unsupervised_metrics(_FixedDirection([-1.0, 0, 0, 0]), info, acts, samples,
                             "last", centering="none", sign_from="none")
    assert a["uncalibrated_accuracy"] == pytest.approx(1 - b["uncalibrated_accuracy"])


def test_the_flip_is_reported():
    samples, acts = _dataset()
    info = _info(samples)
    kept = unsupervised_metrics(_FixedDirection([1.0, 0, 0, 0]), info, acts, samples,
                                "last", centering="none", sign_from="train_accuracy")
    flipped = unsupervised_metrics(_FixedDirection([-1.0, 0, 0, 0]), info, acts, samples,
                                   "last", centering="none", sign_from="train_accuracy")
    assert kept["sign_flipped"] is False
    assert flipped["sign_flipped"] is True


def test_a_useless_direction_stays_at_chance_rather_than_being_rescued():
    """Resolving the sign must not turn noise into signal: it is one bit, not a fit."""
    samples, acts = _dataset(separation=0.0, seed=3)
    info = _info(samples)
    got = unsupervised_metrics(_FixedDirection([0.0, 1.0, 0, 0]), info, acts, samples,
                               "last", centering="none", sign_from="train_accuracy")
    assert 0.2 < got["uncalibrated_accuracy"] < 0.8


def test_unknown_sign_mode_is_rejected():
    samples, acts = _dataset()
    with pytest.raises(ValueError):
        unsupervised_metrics(_FixedDirection([1.0, 0, 0, 0]), _info(samples), acts,
                             samples, "last", centering="none", sign_from="magic")


@pytest.mark.parametrize("mode", ["train_accuracy", "diff_in_means", "branch_means"])
@pytest.mark.parametrize("sign", [+1.0, -1.0])
def test_every_convention_is_orientation_invariant(mode, sign):
    """Whatever the rule, flipping the probe must not change the reported accuracy."""
    samples, acts = _dataset()
    info = _info(samples)
    a = unsupervised_metrics(_FixedDirection([1.0, 0, 0, 0]), info, acts, samples,
                             "last", centering="none", sign_from=mode)
    b = unsupervised_metrics(_FixedDirection([sign * 1.0, 0, 0, 0]), info, acts, samples,
                             "last", centering="none", sign_from=mode)
    assert a["uncalibrated_accuracy"] == pytest.approx(b["uncalibrated_accuracy"])


def test_branch_means_uses_no_truth_labels():
    """The label-free rule must give the same orientation when the truth labels are
    scrambled — that is what makes it label-free."""
    import dataclasses

    samples, acts = _dataset(seed=17)
    info = _info(samples)
    probe = _FixedDirection([1.0, 0, 0, 0])
    before = unsupervised_metrics(probe, info, acts, samples, "last",
                                  centering="none", sign_from="branch_means")

    scrambled = []
    for s in samples:
        d = dict(s.descriptors)
        d["truth"] = 1 - d["truth"]
        d["label"] = 1 - d["label"]
        scrambled.append(dataclasses.replace(s, descriptors=d))
    after = unsupervised_metrics(probe, info, acts, scrambled, "last",
                                 centering="none", sign_from="branch_means")
    # Accuracy is measured against the (scrambled) labels, so it inverts; the *orientation*
    # decision must not have changed.
    assert before["sign_flipped"] == after["sign_flipped"]
    assert after["uncalibrated_accuracy"] == pytest.approx(
        1 - before["uncalibrated_accuracy"])


def test_diff_in_means_does_use_truth_labels():
    """Contrast with the above: the diff-in-means rule is label-dependent by construction."""
    import dataclasses

    samples, acts = _dataset(seed=19)
    info = _info(samples)
    probe = _FixedDirection([1.0, 0, 0, 0])
    before = unsupervised_metrics(probe, info, acts, samples, "last",
                                  centering="none", sign_from="diff_in_means")
    scrambled = [dataclasses.replace(s, descriptors={**s.descriptors,
                                                     "truth": 1 - s.descriptors["truth"],
                                                     "label": 1 - s.descriptors["label"]})
                 for s in samples]
    after = unsupervised_metrics(probe, info, acts, scrambled, "last",
                                 centering="none", sign_from="diff_in_means")
    assert before["sign_flipped"] != after["sign_flipped"]
