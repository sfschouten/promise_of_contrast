"""Resolution of per-run parameter overrides.

`params.yaml` holds one baseline `probing:` section plus any number of named profiles
under `probing_profiles:`.  A run opts into a profile with `probing: <name>`; runs that
do not are unaffected, so one experiment can change the split, the centering or the
method list without disturbing the others.
"""
from __future__ import annotations


def resolve_probing(params: dict, run_cfg: dict) -> dict:
    """Return the probing config for a run: the baseline, overlaid with its profile."""
    profile = run_cfg.get("probing")
    if not profile:
        return dict(params["probing"])
    profiles = params.get("probing_profiles") or {}
    if profile not in profiles:
        raise KeyError(
            f"run refers to probing profile {profile!r}, which is not in probing_profiles"
        )
    return {**params["probing"], **profiles[profile]}
