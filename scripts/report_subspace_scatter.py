"""
Subspace scatter report.

For each (method, descriptor, train_position) combination, produces one PDF page
per (family, dataset) showing test-split activations projected onto the probe's
leading eigenvectors.  The number of eigenvector pairs per row is determined by a
'bend of the knee' heuristic on the eigenvalue / variance spectrum.

Sampling is stratified by contrast type (the pair of color values in each pair),
so every edge of the label geometry (e.g. T–F, T–N, F–N for trilemma data) is
equally represented.  Pairs are connected by thin dotted lines.

Configuration in reporting.yaml under scatter:
  color_by       field to color points by; checked in descriptors then metadata
                 (default: the probe's label descriptor).
                 Can also be a list of fields — their values are joined with "_"
                 to form a combined color key (e.g. [premise_negation, premise_swap_status]).
  color_labels   optional dict mapping raw values (or combined "_"-joined strings)
                 → display strings (e.g. {0: "F", 1: "T", 2: "N"} or
                 {"False_real": "cc", "True_swapped": "ic_swap"})
  shade_by       descriptor rendered as a light/dark variant of each hue rather
                 than a separate color (e.g. negation), with shade_labels for
                 display names; it also joins color in the sampling stratum key
  max_pairs      pairs (or 2× points) per panel before sampling kicks in
  max_vectors    eigenvectors to show per layer row (hard cap)
  exclude_datasets  (per_source only) list of dataset_names to skip entirely
                 (e.g. drop a family's truth-direction page from a circle plot)
  views_3d       (--3d only) list of [elev, azim] viewing angles, one column each

By default only compound datasets are processed.  Pass --include-families to
also render per-family probes.

Activations are read straight from the cache written by extract_activations, one
sample at a time and only for the points that reach a panel; this report never
loads the model, so a sample missing from the cache is reported rather than computed.

With --3d, each page instead shows one 3-D axes per (layer, viewing angle) over the
first three eigenvectors (EV 0/1/2).  Several viewing angles per layer stand in for
the rotation a static PDF cannot offer; configure them in reporting.yaml under
scatter.views_3d as a list of [elev, azim] pairs.

Run from the project root:
    python scripts/report_subspace_scatter.py [--include-families] [--color-by KEY]
                                              [--max-pairs N] [--max-vectors N]
                                              [--3d | --html] [--components 0,1,3]
                                              [--datasets cities]
                                              [--compounds KEY] [--output-dir path/]
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import hsv_to_rgb, to_hex, to_rgb
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.activations.cache import CacheKey, SampleCache, load_sample_cache
from src.data.graph import load_family_as_pairs
from src.data.loader import load_jsonl
from src.probing import Manifest, load_probes, probe_key
from src.probing.base import Probe


# ── helpers ───────────────────────────────────────────────────────────────────

def _base_method(method: str) -> str:
    return re.sub(r"_\d+$", "", method)


def _n_components_from_method(method: str) -> int:
    m = re.search(r"_(\d+)$", method)
    return int(m.group(1)) if m else 0


def _signal_magnitudes(probe: Probe) -> np.ndarray | None:
    """Descending-order signal magnitudes for the elbow heuristic."""
    arts = probe.artifacts
    base = _base_method(probe.method)
    if base == "pca" and "explained_variance" in arts:
        return np.asarray(arts["explained_variance"], dtype=float)
    if base in ("cross_covariance", "generalized_cross_covariance") and "eigenvalues" in arts:
        eigs   = np.asarray(arts["eigenvalues"], dtype=float)
        n_comp = probe.subspace.shape[0]
        return np.abs(np.sort(eigs)[:n_comp])   # most-negative k; already descending magnitude
    return None


def _knee_n(magnitudes: np.ndarray, max_k: int = 8) -> int:
    """
    Elbow of a descending-magnitude curve; even count in [2, min(len, max_k)].
    Uses the maximum-perpendicular-distance heuristic.
    """
    n    = min(len(magnitudes), max_k)
    vals = magnitudes[:n].astype(float)
    vmin, vmax = vals[-1], vals[0]
    if n <= 2 or (vmax - vmin) < 1e-12:
        return 2
    x    = np.arange(n, dtype=float) / (n - 1)
    y    = (vals - vmin) / (vmax - vmin)
    knee = int(np.argmax(np.abs(x + y - 1))) + 1
    knee = max(2, knee)
    if knee % 2:
        knee_up = min(knee + 1, n)
        knee    = knee_up if knee_up % 2 == 0 else max(2, knee - 1)
    return knee


def _find_run_key(params: dict, model_alias: str, family: str) -> str:
    for rkey, rcfg in params["runs"].items():
        if rcfg["model"] == model_alias and rcfg["family"] == family:
            return rkey
    raise KeyError(f"No run found for model={model_alias!r}, family={family!r}")


def _load_run_samples(run_key: str, params: dict) -> list:
    """Samples for a run — pair-flattened for graph families, flat otherwise."""
    family = params["runs"][run_key]["family"]
    nodes = load_jsonl(f"data/processed/samples_{family}.jsonl")
    pair_samples = load_family_as_pairs(family)
    return pair_samples if pair_samples is not None else nodes


class ActivationStore:
    """
    Reads per-sample activation caches from disk on first use, keeping only the
    (layer, position) keys the probes were actually trained at.

    This report used to call extract_hidden_states, which loads every cached layer
    and position for every sample in the family up front — several GB for the larger
    families (ccs is 20k samples x 22 vectors), all of it held while every page is
    rendered.  It also loaded an 8B model into RAM whenever a sample was missing
    from the cache, which is not something a reporting stage should ever do.

    Only the sampled test-split points are plotted, so entries are pulled in on
    demand: peak residency is the points that actually reach a panel.  A sample
    whose cache lacks the requested key is reported as missing rather than computed.
    """

    def __init__(self, keys: set[CacheKey]):
        self._keys   = keys
        self._source: dict[str, tuple[Path, str]] = {}   # sample id -> (cache_dir, model)
        self._cache:  dict[str, SampleCache] = {}
        self.n_missing = 0

    def register(self, samples: list, cache_dir: Path, model_name: str) -> None:
        for s in samples:
            self._source.setdefault(s.id, (cache_dir, model_name))

    def get(self, sample) -> SampleCache:
        entry = self._cache.get(sample.id)
        if entry is None:
            src = self._source.get(sample.id)
            if src is None:
                entry = {}
            else:
                cache_dir, model_name = src
                cached = load_sample_cache(cache_dir, model_name, sample)
                entry  = {k: v for k, v in cached.items() if k in self._keys}
            if not entry:
                self.n_missing += 1
            self._cache[sample.id] = entry
        return entry


def _get_attr(sample, key: str):
    """Return a display value from descriptors, then metadata; None if absent."""
    if key in sample.descriptors:
        return sample.descriptors[key]
    return sample.metadata.get(key)


# ── page construction ─────────────────────────────────────────────────────────

def _collect_pages(
    manifest: Manifest,
    probes: dict,
    samples: list,
    store: ActivationStore,
    max_vectors: int,
    min_vectors: int = 0,
    max_pairs: int = 200,
    color_by: str = "",
    color_labels: dict | None = None,
    shape_by: str = "",
    shape_labels: dict | None = None,
    shade_by: str = "",
    shade_labels: dict | None = None,
    cyclic_color: bool = False,
    cyclic_order_by: str = "multiclass_label",
) -> list[dict]:
    """
    Returns a list of page descriptors, one per
    (family, dataset, descriptor, train_position, method_base) combination.

    Sampling is stratified: pairs are grouped by the sorted tuple of their
    (base color value, cf color value), and max_pairs // n_strata are sampled
    from each group.  This keeps all contrast types equally visible.
    """
    # For each (dataset, layer, train_position, method_base, descriptor), keep
    # only the entry/probe with the most n_components.
    best: dict[tuple, tuple] = {}
    for entry in manifest.entries:
        base   = _base_method(entry.method)
        n_comp = _n_components_from_method(entry.method)
        key    = (entry.dataset, entry.layer, entry.train_position, base, entry.descriptor)
        probe  = probes.get(
            probe_key(entry.dataset, entry.layer, entry.train_position,
                      entry.method, entry.descriptor, entry.seed)
        )
        if probe is None:
            continue
        # .subspace can raise for a degenerate probe (e.g. a logistic probe fit
        # on a cosmetic label with near-zero weights, as on the circular edges);
        # such probes have no meaningful subspace, so skip them like 1-D ones.
        try:
            if probe.subspace.shape[0] < 2:
                continue
        except ValueError:
            continue
        if _signal_magnitudes(probe) is None:
            continue
        prev = best.get(key)
        if prev is None or n_comp > _n_components_from_method(prev[0].method):
            best[key] = (entry, probe)

    # Paired methods (cross_covariance, generalized_cross_covariance) compute their
    # subspace from X_base / X_cf only — the label descriptor has no effect on the
    # directions.  Collapse all descriptor variants to a single page per method.
    _PAIRED_BASES = {"cross_covariance", "generalized_cross_covariance"}

    # Group by page = (family, dataset, descriptor, train_position, method_base).
    by_page: dict[tuple, dict[int, tuple]] = defaultdict(dict)
    for (dataset, layer, train_position, method_base, descriptor), (entry, probe) in best.items():
        family      = entry.family or manifest.dataset or "unknown"
        page_desc   = "paired" if method_base in _PAIRED_BASES else descriptor
        page_key    = (family, dataset, page_desc, train_position, method_base)
        by_page[page_key][layer] = (entry, probe)

    pages = []
    for page_key, layer_map in sorted(by_page.items()):
        family, dataset, descriptor, train_position, method_base = page_key
        layers = sorted(layer_map.keys())

        # Restrict candidates to this page's dataset.  Graph families can reuse the
        # same node id across configs (e.g. the banana/shed grid shares nodes
        # between imdb_sent/imdb_dist/imdb_all), so the test_id filter below would
        # otherwise pull in pairs from sibling configs that share those nodes.
        # For compounds the page dataset is the compound name rather than a
        # per-sample dataset_name, so fall back to all samples in that case.
        page_samples = [s for s in samples if s.descriptors.get("dataset_name") == dataset]
        if not page_samples:
            page_samples = samples

        # Column count: max elbow across layers, floored by min_vectors, capped at max_vectors // 2.
        n_pairs = max(1, min_vectors // 2)
        for layer in layers:
            _, probe = layer_map[layer]
            mags    = _signal_magnitudes(probe)
            n_dims  = _knee_n(mags, max_k=max_vectors)
            n_pairs = max(n_pairs, n_dims // 2)

        if isinstance(color_by, list):
            effective_color = "+".join(color_by)
            def _color_val(s, _keys=color_by):
                return "_".join(str(_get_attr(s, k)) for k in _keys)
        else:
            # Use the first real descriptor found across layers as fallback color key.
            first_descriptor = next(iter(layer_map.values()))[0].descriptor
            effective_color = color_by if color_by else first_descriptor
            def _color_val(s, _key=effective_color):
                return _get_attr(s, _key)

        rows = []
        for layer in layers:
            entry, probe = layer_map[layer]
            test_id_set  = set(entry.test_ids)

            # Candidate selection uses descriptors only — activations for the
            # points that survive sampling are fetched below, so a page never
            # pulls the whole family's cache into memory.
            valid = [
                s for s in page_samples
                if s.id in test_id_set
                and entry.descriptor in s.descriptors
            ]
            if len(valid) < 4:
                rows.append(None)
                continue

            # Arrays for all valid samples (before sampling).
            pair_ids   = np.array([s.descriptors.get("pair_id",   "")     for s in valid])
            roles_arr  = np.array([s.descriptors.get("pair_role", "none") for s in valid])
            color_arr  = np.array([_color_val(s) for s in valid], dtype=object)
            group_arr  = np.array([s.descriptors.get("group_id",  "")     for s in valid])
            topics_raw = np.array(
                [str(_get_attr(s, shape_by) or "") if shape_by else "" for s in valid],
                dtype=object,
            )
            shade_raw = np.array(
                [str(_get_attr(s, shade_by)) if shade_by else "" for s in valid],
                dtype=object,
            )
            # Points are identified visually by hue *and* shade, so the sampling
            # stratum has to use both — otherwise a shaded factor collapses
            # strata that were previously distinct (e.g. the six grid contrasts).
            strata_arr = (
                np.array([f"{c}|{v}" for c, v in zip(color_arr, shade_raw)], dtype=object)
                if shade_by else color_arr
            )

            # Stratified sampling.
            unique_pids = [p for p in sorted(set(pair_ids)) if p]
            rng = np.random.default_rng(0)

            if unique_pids:
                bc = {pid: cv for pid, role, cv in zip(pair_ids, roles_arr, strata_arr)
                      if role == "base" and pid}
                cc = {pid: cv for pid, role, cv in zip(pair_ids, roles_arr, strata_arr)
                      if role == "counterfactual" and pid}

                # Map pair_id → group_id (empty string = ungrouped).
                pid_to_gid = {pid: gid for pid, gid in zip(pair_ids, group_arr) if pid and gid}

                if pid_to_gid:
                    # Group-level sampling: all pairs within a group are kept or
                    # dropped together.  Ungrouped pairs are treated as singleton
                    # groups.  Stratify groups by their shape signature — the sorted
                    # tuple of (base-color, cf-color) pairs in the group — so that
                    # every geometric structure type is equally represented.
                    gid_to_pids: dict[str, list] = defaultdict(list)
                    for pid in unique_pids:
                        gid = pid_to_gid.get(pid) or f"_solo_{pid}"
                        gid_to_pids[gid].append(pid)

                    buckets: dict[tuple, list] = defaultdict(list)
                    for gid, gpids in gid_to_pids.items():
                        sig = tuple(sorted(
                            (str(bc.get(p, "")), str(cc.get(p, "")))
                            for p in gpids if p in bc and p in cc
                        ))
                        buckets[sig].append(gid)

                    n_b = max(1, len(buckets))
                    per = max(1, max_pairs // n_b)
                    keep_gids: set = set()
                    for glist in buckets.values():
                        n = min(per, len(glist))
                        keep_gids.update(rng.choice(glist, size=n, replace=False))

                    keep_pids: set = set()
                    for gid in keep_gids:
                        keep_pids.update(gid_to_pids[gid])
                else:
                    # No group_id present: existing per-pair stratified sampling.
                    buckets_p: dict[tuple, list] = defaultdict(list)
                    for pid in unique_pids:
                        if pid in bc and pid in cc:
                            strata = tuple(sorted([str(bc[pid]), str(cc[pid])]))
                            buckets_p[strata].append(pid)

                    n_b = max(1, len(buckets_p))
                    per = max(1, max_pairs // n_b)
                    keep_pids = set()
                    for plist in buckets_p.values():
                        n = min(per, len(plist))
                        keep_pids.update(rng.choice(plist, size=n, replace=False))

                mask = np.array([pid in keep_pids for pid in pair_ids])

            else:
                # Unpaired: stratify by color value.
                buckets_idx: dict[str, list] = defaultdict(list)
                for i, cv in enumerate(strata_arr):
                    buckets_idx[str(cv)].append(i)
                n_b  = max(1, len(buckets_idx))
                per  = max(1, (max_pairs * 2) // n_b)
                keep: set = set()
                for idxs in buckets_idx.values():
                    n = min(per, len(idxs))
                    keep.update(int(i) for i in rng.choice(idxs, size=n, replace=False))
                mask = np.zeros(len(valid), dtype=bool)
                for i in keep:
                    mask[i] = True

            valid      = [s for s, m in zip(valid, mask) if m]
            pair_ids   = pair_ids[mask]
            roles_arr  = roles_arr[mask]
            color_arr  = color_arr[mask]
            group_arr  = group_arr[mask]
            topics_raw = topics_raw[mask]
            shade_raw  = shade_raw[mask]
            topics_arr = np.array(
                [str(shape_labels.get(str(tv), tv)) if shape_labels else tv for tv in topics_raw],
                dtype=object,
            )
            shades_arr = np.array(
                [str(shade_labels.get(str(sv), sv)) if shade_labels else sv for sv in shade_raw],
                dtype=object,
            )

            if len(valid) < 2:
                rows.append(None)
                continue

            # Fetch activations for the sampled points only; drop any whose cache
            # lacks this (layer, position) — the store counts them as missing.
            vecs = [store.get(s).get((layer, train_position)) for s in valid]
            keep = [i for i, v in enumerate(vecs) if v is not None]
            if len(keep) < 2:
                rows.append(None)
                continue
            if len(keep) < len(valid):
                vecs       = [vecs[i] for i in keep]
                valid      = [valid[i] for i in keep]
                pair_ids   = pair_ids[keep]
                roles_arr  = roles_arr[keep]
                color_arr  = color_arr[keep]
                topics_arr = topics_arr[keep]
                shades_arr = shades_arr[keep]

            W    = probe.subspace.float().numpy()   # (n_comp, hidden)
            W    = W[:min(W.shape[0], n_pairs * 2)]
            X    = np.stack([v.float().numpy() for v in vecs])
            proj = X @ W.T                         # (n_sampled, n_used)

            # Normalise display values via color_labels mapping.
            display = np.array(
                [str(color_labels.get(cv, cv) if color_labels else cv) for cv in color_arr]
            )

            rows.append({
                "layer":       layer,
                "proj":        proj,
                "color_vals":  display,
                "roles":       roles_arr,
                "pair_ids":    pair_ids,
                "topics":      topics_arr,
                "shades":      shades_arr,
                "eigenvalues": probe.artifacts.get("eigenvalues"),
            })

        # Cyclic coloring: map each display color value to its position on the
        # cycle (e.g. Monday→0 … Sunday→6) so the renderer can assign hues evenly
        # around the wheel, in order.  Derived from cyclic_order_by (default
        # multiclass_label = the element's circular index), which co-occurs with
        # the color value on every node.
        cyclic_order = None
        if cyclic_color:
            cyclic_order = {}
            for s in page_samples:
                idx = _get_attr(s, cyclic_order_by)
                if idx is None:
                    continue
                cv   = _color_val(s)
                disp = str(color_labels.get(cv, cv) if color_labels else cv)
                cyclic_order[disp] = int(idx)

        if any(r is not None for r in rows):
            pages.append({
                "family":         family,
                "dataset":        dataset,
                "descriptor":     descriptor,
                "color_by":       effective_color,
                "shape_by":       shape_by,
                "shade_by":       shade_by,
                "train_position": train_position,
                "method":         method_base,
                "n_pairs":        n_pairs,
                "rows":           rows,
                "cyclic_order":   cyclic_order,
            })

    return pages


# ── rendering ─────────────────────────────────────────────────────────────────

# Qualitative palette — first two match the existing binary blue/red scheme.
# Pairs (i, i+4) share a hue family to encode non-swapped vs swapped variants.
_QUAL_COLORS = [
    "#4477AA",  # 0: blue   (cc)
    "#EE6677",  # 1: red    (ci)
    "#228833",  # 2: green  (ic)
    "#CCBB44",  # 3: yellow (ii)
    "#66CCEE",  # 4: cyan   (cc_swap — blue family)
    "#AA3377",  # 5: purple (ci_swap — red family)
    "#BBBBBB",  # 6: grey   (ic_swap)
    "#EE8833",  # 7: orange (ii_swap — yellow family)
    # 8+: extra hues so color keys with many categories (e.g. the 12 months of
    # months_of_year) stay distinct.  Appended, so indices 0–7 are unchanged and
    # existing reports keep their colors.
    "#882255",  # 8:  wine
    "#44AA99",  # 9:  teal
    "#999933",  # 10: olive
    "#332288",  # 11: indigo
    "#AA4499",  # 12: magenta
    "#117733",  # 13: dark green
]

_TOPIC_MARKERS = ["o", "s", "^", "D", "v", "p", "*", "h", "P", "X"]


def _parse_component_sets(spec: str) -> list[tuple[int, int, int]]:
    """
    "0,1,2" or "0,1,2;0,1,3" -> [(0,1,2)] / [(0,1,2), (0,1,3)].

    Each triple is rendered as its own figure (HTML) or page (PDF).  Selecting the
    triple by toggling traces instead would need two independent controls over the
    same visibility array as the layer slider, which plotly cannot keep in sync.
    """
    sets: list[tuple[int, int, int]] = []
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        idx = tuple(int(v) for v in chunk.split(","))
        if len(idx) != 3:
            raise ValueError(f"--components expects triples, got {chunk!r}")
        if len(set(idx)) != 3 or min(idx) < 0:
            raise ValueError(f"--components triple must be three distinct non-negative indices: {chunk!r}")
        sets.append(idx)  # type: ignore[arg-type]
    return sets or [(0, 1, 2)]


def _separating_plane(P: np.ndarray, y: np.ndarray):
    """
    Logistic boundary for a binary split, fitted *in the plotted 3-D space* rather
    than projected into it.

    The probes are fitted on full activations, and only ~70-85% of a truth
    probe's direction survives projection into three components, so a projected
    boundary would show points on the wrong side of its own plane.  Fitting on
    the plotted coordinates keeps the plane exactly consistent with the dots.

    Returns (corner coordinates (4, 3), accuracy), or None for a degenerate split.
    """
    from sklearn.linear_model import LogisticRegression

    if y.min() == y.max():
        return None

    clf = LogisticRegression(max_iter=1000).fit(P, y)
    w, b = clf.coef_[0], float(clf.intercept_[0])
    nrm = np.linalg.norm(w)
    if nrm < 1e-12:
        return None
    n = w / nrm

    # Anchor the quad at the point of the plane nearest the cloud's centre, and
    # span it with the data's own extent along two in-plane axes.
    c  = P.mean(0)
    u0 = c - ((w @ c + b) / nrm**2) * w
    seed = np.eye(3)[int(np.argmin(np.abs(n)))]
    e1 = seed - (seed @ n) * n
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    d  = P - u0
    s1 = 1.15 * max(np.abs(d @ e1).max(), 1e-6)
    s2 = 1.15 * max(np.abs(d @ e2).max(), 1e-6)
    corners = np.array([u0 - s1 * e1 - s2 * e2, u0 + s1 * e1 - s2 * e2,
                        u0 + s1 * e1 + s2 * e2, u0 - s1 * e1 + s2 * e2])
    return corners, float(clf.score(P, y))


def _factor_planes(P: np.ndarray, labels: np.ndarray, fname: str) -> list[tuple]:
    """
    Separating planes for one factor: a single plane when it is binary, and a
    one-vs-rest plane per class when it has three or four.  A three-class factor
    such as the trilemma's T/F/N has no single optimal plane, but each class
    against the other two is well defined.

    Returns [(display name, short key, corners, accuracy), ...].
    """
    vals = sorted({str(v) for v in labels})
    if not 2 <= len(vals) <= 4:
        return []

    out = []
    if len(vals) == 2:
        y = np.array([str(v) == vals[1] for v in labels], dtype=int)
        got = _separating_plane(P, y)
        if got is not None:
            out.append((f"{fname} plane ({got[1]:.0%})", fname, got[0], got[1]))
    else:
        for cls in vals:
            y = np.array([str(v) == cls for v in labels], dtype=int)
            got = _separating_plane(P, y)
            if got is not None:
                out.append((f"{fname}={cls} vs rest ({got[1]:.0%})",
                            f"{fname}={cls}", got[0], got[1]))
    return out


def _shade_hex(color: str, amount: float) -> str:
    """Blend a hex color toward black (amount < 0) or white (amount > 0)."""
    if not amount:
        return color
    r, g, b = to_rgb(color)
    target = 1.0 if amount > 0 else 0.0
    f = abs(amount)
    return to_hex(tuple(c + (target - c) * f for c in (r, g, b)))


def _shade_amounts(n: int) -> list[float]:
    """Blend factors for n shade levels; a single level leaves the color untouched."""
    if n <= 1:
        return [0.0]
    if n == 2:
        return [-0.25, 0.45]          # darker / lighter around the base hue
    return list(np.linspace(-0.30, 0.50, n))

# Rasterizing a panel makes matplotlib keep a figure-sized RGBA buffer for every
# page until the PDF is closed — ~650 MB per page at dpi 150 on an 11-layer report,
# which is what made this script exhaust memory on multi-page runs.  At the point
# counts stratified sampling produces, the raster is also *bigger* on disk than the
# vector marks it replaces, so it is only worth it for unusually dense panels.
_RASTER_MIN_POINTS = 5000

# Pair-connector opacity levels offered by the interactive page's slider.
# The base grey is dark (40) rather than mid (120) because the lines composite
# against a white pane: at base 120 a 22% line lands on #e1e1e1, which is the
# same lightness as the default gridlines (#DFE8F3), so the low half of the
# slider was invisible.  Grid lines are lightened below for the same reason.
_PAIR_LINE_ALPHAS = [0.1, 0.18, 0.28, 0.4, 0.6, 0.8, 1.0]
_PAIR_LINE_DEFAULT = 0.18
_PAIR_LINE_ALPHAS_RGBA = {a: f"rgba(40,40,40,{a})" for a in _PAIR_LINE_ALPHAS}
_GRID_COLOR = "#F2F2F2"
# Separating-plane fills, deliberately outside the categorical point palette.
_PLANE_COLORS = ["#2B7A78", "#B07D2B", "#6A5ACD", "#A63A50", "#4F7942"]

# Default 3-D viewing angles (elev, azim).  Three angles per layer give enough
# parallax to read depth from a static page.
_VIEWS_3D = [(22, -60), (22, 30), (68, -60)]


def _style_maps(page: dict, rows: list[dict]) -> dict:
    """Color / marker maps and legend ordering, shared by the 2-D and 3-D renderers."""
    cyclic_order = page.get("cyclic_order")
    color_set = {cv for r in rows for cv in r["color_vals"]}
    if cyclic_order:
        # Order legend by position on the cycle (Jan…Dec), not lexically.
        all_color_vals = sorted(color_set, key=lambda v: (cyclic_order.get(v, 1e9), v))
    else:
        all_color_vals = sorted(color_set, key=lambda v: (len(v), v))
    all_topics = sorted({str(t) for r in rows for t in r["topics"]})
    has_topics = not (len(all_topics) == 1 and all_topics[0] == "")
    all_shades = sorted({str(v) for r in rows for v in r.get("shades", [""])}) or [""]
    has_shades = not (len(all_shades) == 1 and all_shades[0] == "")

    if cyclic_order:
        # Cyclic hue ramp: hue = position / period, fixed saturation & value, so
        # the ordered elements wrap around the color wheel.  If the recovered
        # geometry is a circle, the colors run smoothly around the ring.
        period = max(cyclic_order.values()) + 1 if cyclic_order else 1
        _S, _V = 0.65, 0.95
        color_map = {
            cv: to_hex(hsv_to_rgb([(cyclic_order[cv] % period) / period, _S, _V]))
            if cv in cyclic_order else "#999999"
            for cv in all_color_vals
        }
    else:
        color_map = {cv: _QUAL_COLORS[i % len(_QUAL_COLORS)]
                     for i, cv in enumerate(all_color_vals)}
    # Shade splits each hue into light/dark variants for a second binary factor
    # (e.g. negated vs affirmative), so hue stays free for the primary contrast.
    amounts = _shade_amounts(len(all_shades))
    shaded_map = {(cv, sv): _shade_hex(color_map[cv], amounts[si])
                  for cv in all_color_vals
                  for si, sv in enumerate(all_shades)}
    topic_marker_map = {t: _TOPIC_MARKERS[i % len(_TOPIC_MARKERS)]
                        for i, t in enumerate(all_topics)}
    return {
        "color_vals": all_color_vals,
        "color_map":  color_map,
        "shades":     all_shades,
        "shaded_map": shaded_map,
        "has_shades": has_shades,
        "topics":     all_topics,
        "marker_map": topic_marker_map,
        "has_topics": has_topics,
    }


def _legend_handles(page: dict, style: dict) -> list[Line2D]:
    shade_by = page.get("shade_by", "")
    handles = []
    for cv in style["color_vals"]:
        for sv in style["shades"]:
            label = f"{page['color_by']}={cv}"
            if style["has_shades"]:
                label += f", {shade_by}={sv}"
            handles.append(
                Line2D([0], [0], marker="o", linestyle="", color="w",
                       markerfacecolor=style["shaded_map"][(cv, sv)], markersize=7,
                       label=label)
            )
    if style["has_topics"]:
        shape_by = page.get("shape_by", "")
        handles.append(Line2D([0], [0], linestyle="", color="none", label=""))
        for topic in style["topics"]:
            handles.append(
                Line2D([0], [0], marker=style["marker_map"][topic],
                       linestyle="", color="k", markersize=7,
                       label=f"{shape_by}={topic}")
            )
    return handles


def _ev_labeler(row: dict):
    """Returns f(i) -> 'EV i [eigenvalue]' for this row's probe."""
    eigs = row.get("eigenvalues")

    def label(i: int) -> str:
        if eigs is not None and i < len(eigs):
            return f"EV {i} [{eigs[i]:.3g}]"
        return f"EV {i}"

    return label


def _page_title(page: dict) -> str:
    fam, ds  = page["family"], page["dataset"]
    title_ds = ds if ds == fam else f"{fam} / {ds}"
    source   = page.get("source", "")
    prefix   = f"{source}  |  " if source and source != title_ds else ""
    return (
        f"{prefix}{title_ds}  |  {page['method']}  |  probe={page['descriptor']}  |  "
        f"color={page['color_by']}  |  train={page['train_position']}"
    )


def _pair_segments(pair_ids: np.ndarray, roles: np.ndarray) -> list[tuple[int, int]]:
    """Index pairs (base, counterfactual) to connect with a line."""
    if not (pair_ids != "").any():
        return []
    base_pos = {pid: k for k, (pid, r) in enumerate(zip(pair_ids, roles))
                if r == "base" and pid}
    cf_pos   = {pid: k for k, (pid, r) in enumerate(zip(pair_ids, roles))
                if r == "counterfactual" and pid}
    return [(bi, cf_pos[pid]) for pid, bi in base_pos.items() if pid in cf_pos]


def _plot_page(pdf: PdfPages, page: dict, dpi: int = 150) -> None:
    rows = [r for r in page["rows"] if r is not None]
    if not rows:
        return

    n_layers = len(rows)
    n_pairs  = page["n_pairs"]
    fig, axes = plt.subplots(
        n_layers, n_pairs,
        figsize=(max(4.0, 3.5 * n_pairs + 1.0), max(3.0, 3.0 * n_layers)),
        squeeze=False,
    )

    style            = _style_maps(page, rows)
    all_color_vals   = style["color_vals"]
    all_topics       = style["topics"]
    topic_marker_map = style["marker_map"]

    for ri, row in enumerate(rows):
        proj      = row["proj"]
        color_arr = row["color_vals"]
        roles     = row["roles"]
        pair_ids  = row["pair_ids"]
        topics    = row["topics"]
        shades    = row["shades"]
        layer     = row["layer"]

        for pi in range(n_pairs):
            ax = axes[ri, pi]
            ix = pi * 2
            iy = pi * 2 + 1

            if ix >= proj.shape[1] or iy >= proj.shape[1]:
                ax.axis("off")
                continue

            xv = proj[:, ix]
            yv = proj[:, iy]
            raster = len(xv) > _RASTER_MIN_POINTS

            # Pair connector lines (drawn below markers).
            for bi, ci in _pair_segments(pair_ids, roles):
                ax.plot([xv[bi], xv[ci]], [yv[bi], yv[ci]],
                        color="gray", lw=0.5, ls=":", alpha=0.35, zorder=1)

            # Scatter markers — shape encodes topic/dataset.
            for cv in all_color_vals:
                for sv in style["shades"]:
                    for topic in all_topics:
                        mask = (color_arr == cv) & (shades == sv) & (topics == topic)
                        if not mask.any():
                            continue
                        ax.scatter(
                            xv[mask], yv[mask],
                            c=style["shaded_map"][(cv, sv)],
                            marker=topic_marker_map[topic],
                            s=12, alpha=0.6, linewidths=0,
                            rasterized=raster, zorder=2,
                        )

            _ev_label = _ev_labeler(row)
            ylabel = f"Layer {layer}\n{_ev_label(iy)}" if pi == 0 else _ev_label(iy)
            ax.set_ylabel(ylabel, fontsize=7)
            ax.set_xlabel(_ev_label(ix), fontsize=7)
            ax.tick_params(labelsize=6)
            if ri == 0:
                ax.set_title(f"EV {ix} vs {iy}", fontsize=8)

    # Figure-level legend.
    fig.legend(handles=_legend_handles(page, style), fontsize=7, loc="center left",
               bbox_to_anchor=(1.01, 0.5), borderaxespad=0)
    fig.suptitle(_page_title(page), fontsize=9)

    plt.tight_layout(rect=[0, 0, 0.85, 0.93])
    pdf.savefig(fig, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def _plot_page_3d(pdf: PdfPages, page: dict, dpi: int = 150,
                  views: list[tuple[float, float]] | None = None,
                  comps: tuple[int, int, int] = (0, 1, 2)) -> None:
    """
    One 3-D axes per (layer, viewing angle), showing the first three eigenvectors
    (EV 0/1/2) of the page's probe.  Layers are rows, viewing angles are columns —
    the extra angles substitute for the rotation a static page cannot offer.
    """
    need = max(comps) + 1
    rows = [r for r in page["rows"] if r is not None and r["proj"].shape[1] >= need]
    if not rows:
        # No layer row has enough eigenvectors for the requested triple.
        print(f"  skipping 3-D page (needs EV {max(comps)}): {_page_title(page)}")
        return

    views    = views or _VIEWS_3D
    n_layers = len(rows)
    n_views  = len(views)
    fig, axes = plt.subplots(
        n_layers, n_views,
        figsize=(max(4.0, 4.0 * n_views + 1.0), max(3.8, 3.8 * n_layers)),
        subplot_kw={"projection": "3d"},
        squeeze=False,
    )

    style            = _style_maps(page, rows)
    all_color_vals   = style["color_vals"]
    all_topics       = style["topics"]
    topic_marker_map = style["marker_map"]

    for ri, row in enumerate(rows):
        proj      = row["proj"]
        color_arr = row["color_vals"]
        topics    = row["topics"]
        shades    = row["shades"]
        segments  = _pair_segments(row["pair_ids"], row["roles"])
        xv, yv, zv = proj[:, comps[0]], proj[:, comps[1]], proj[:, comps[2]]
        raster     = len(xv) > _RASTER_MIN_POINTS
        _ev_label  = _ev_labeler(row)

        for vi, (elev, azim) in enumerate(views):
            ax = axes[ri, vi]
            ax.view_init(elev=elev, azim=azim)

            # Pair connector lines (drawn below markers).
            for bi, ci in segments:
                ax.plot([xv[bi], xv[ci]], [yv[bi], yv[ci]], [zv[bi], zv[ci]],
                        color="gray", lw=0.5, ls=":", alpha=0.35, zorder=1)

            # Scatter markers — shape encodes topic/dataset.
            for cv in all_color_vals:
                for sv in style["shades"]:
                    for topic in all_topics:
                        mask = (color_arr == cv) & (shades == sv) & (topics == topic)
                        if not mask.any():
                            continue
                        ax.scatter(
                            xv[mask], yv[mask], zv[mask],
                            c=style["shaded_map"][(cv, sv)],
                            marker=topic_marker_map[topic],
                            s=10, alpha=0.6, linewidths=0, depthshade=False,
                            rasterized=raster, zorder=2,
                        )

            ax.set_xlabel(_ev_label(comps[0]), fontsize=6, labelpad=1)
            ax.set_ylabel(_ev_label(comps[1]), fontsize=6, labelpad=1)
            ax.set_zlabel(_ev_label(comps[2]), fontsize=6, labelpad=-1)
            ax.tick_params(labelsize=5, pad=-3)
            ax.locator_params(nbins=4)          # sparse ticks: the panels are small
            ax.set_box_aspect((1, 1, 1), zoom=1.1)
            if ri == 0:
                ax.set_title(f"elev {elev}° / azim {azim}°", fontsize=8)
            if vi == 0:
                ax.text2D(-0.10, 0.5, f"Layer {row['layer']}",
                          transform=ax.transAxes, rotation=90,
                          va="center", ha="center", fontsize=8)

    fig.legend(handles=_legend_handles(page, style), fontsize=7, loc="center left",
               bbox_to_anchor=(1.01, 0.5), borderaxespad=0)
    ev_note = f"  |  EV {comps[0]}/{comps[1]}/{comps[2]}"
    fig.suptitle(_page_title(page) + ev_note, fontsize=9)

    plt.tight_layout(rect=[0, 0, 0.85, 0.93])
    pdf.savefig(fig, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def _html_figure(go, page: dict, comps: tuple[int, int, int], eye: dict):
    """
    One interactive figure: a page rendered in one component triple, layers on a
    slider.  Returns (figure, plane accuracies) or None when no layer of this page
    has enough eigenvectors for the triple.
    """
    need = max(comps) + 1
    rows = [r_ for r_ in page["rows"] if r_ is not None and r_["proj"].shape[1] >= need]
    if not rows:
        return None
    style = _style_maps(page, rows)
    fig  = go.Figure()
    # (layer index, kind) per trace, so the sliders can address them by role
    meta: list[tuple[int, str]] = []
    plane_acc: dict[str, list[float]] = {}

    for ri, row in enumerate(rows):
        vis  = ri == 0
        proj = row["proj"]
        x, y, z = proj[:, comps[0]], proj[:, comps[1]], proj[:, comps[2]]
        cat_of = np.array([f"{c}|{v}" for c, v in
                           zip(row["color_vals"], row["shades"])], dtype=object)

        segs = _pair_segments(row["pair_ids"], row["roles"])
        if segs:
            sx: list = []
            sy: list = []
            sz: list = []
            for bi, ci in segs:
                sx += [float(x[bi]), float(x[ci]), None]
                sy += [float(y[bi]), float(y[ci]), None]
                sz += [float(z[bi]), float(z[ci]), None]
            fig.add_trace(go.Scatter3d(
                x=sx, y=sy, z=sz, mode="lines", name="contrast pairs",
                line=dict(color=_PAIR_LINE_ALPHAS_RGBA[_PAIR_LINE_DEFAULT], width=2),
                hoverinfo="skip", showlegend=False, visible=vis,
            ))
            meta.append((ri, "line"))

        centroids: dict[str, np.ndarray] = {}
        cen_color: dict[str, str] = {}
        cen_label: dict[str, str] = {}
        for cv in style["color_vals"]:
            for sv in style["shades"]:
                mask = (row["color_vals"] == cv) & (row["shades"] == sv)
                if not mask.any():
                    continue
                label = f"{page['color_by']}={cv}"
                if style["has_shades"]:
                    label += f", {page.get('shade_by', '')}={sv}"
                color = style["shaded_map"][(cv, sv)]
                fig.add_trace(go.Scatter3d(
                    x=x[mask], y=y[mask], z=z[mask], mode="markers", name=label,
                    marker=dict(size=2.6, color=color, opacity=0.8),
                    legendgroup=f"{label}@{ri}", showlegend=vis, visible=vis,
                    text=[str(t) for t in row["pair_ids"][mask]],
                    hovertemplate="%{text}<extra>" + label + "</extra>",
                ))
                meta.append((ri, "marker"))
                key = f"{cv}|{sv}"
                centroids[key] = np.array([x[mask].mean(), y[mask].mean(), z[mask].mean()])
                cen_color[key] = color
                cen_label[key] = label

        # Category centroids and the links between them.  Which centroids are
        # joined is taken from the data rather than assumed: every category
        # pair that some contrast pair actually spans gets a link, so the
        # skeleton mirrors the contrast structure (six edges for a 2x2 grid).
        if centroids:
            keys = list(centroids)
            fig.add_trace(go.Scatter3d(
                x=[float(centroids[k][0]) for k in keys],
                y=[float(centroids[k][1]) for k in keys],
                z=[float(centroids[k][2]) for k in keys],
                mode="markers", name="centroids", legendgroup=f"centroids@{ri}",
                marker=dict(size=9, symbol="diamond",
                            color=[cen_color[k] for k in keys],
                            line=dict(color="#222222", width=1)),
                text=[cen_label[k] for k in keys],
                hovertemplate="%{text}<extra>centroid</extra>",
                visible="legendonly", showlegend=vis,
            ))
            meta.append((ri, "centroid"))

            links = sorted({tuple(sorted((cat_of[bi], cat_of[ci])))
                            for bi, ci in segs
                            if cat_of[bi] != cat_of[ci]})
            if links:
                lx: list = []
                ly: list = []
                lz: list = []
                for a, b in links:
                    if a not in centroids or b not in centroids:
                        continue
                    lx += [float(centroids[a][0]), float(centroids[b][0]), None]
                    ly += [float(centroids[a][1]), float(centroids[b][1]), None]
                    lz += [float(centroids[a][2]), float(centroids[b][2]), None]
                fig.add_trace(go.Scatter3d(
                    x=lx, y=ly, z=lz, mode="lines", name="centroid links",
                    legendgroup=f"centroids@{ri}",
                    line=dict(color="rgba(40,40,40,0.85)", width=4),
                    hoverinfo="skip", visible="legendonly", showlegend=vis,
                ))
                meta.append((ri, "centroid"))

        # Optimal separating planes, fitted on these very coordinates so each
        # boundary matches the dots exactly.
        P = np.column_stack([x, y, z])
        factors = [(page["color_by"], row["color_vals"])]
        if style["has_shades"]:
            factors.append((page.get("shade_by", "shade"), row["shades"]))
        pi_ = 0
        for fname, farr in factors:
            for disp, key, corners, acc in _factor_planes(P, farr, fname):
                fig.add_trace(go.Mesh3d(
                    x=corners[:, 0], y=corners[:, 1], z=corners[:, 2],
                    i=[0, 0], j=[1, 2], k=[2, 3],
                    color=_PLANE_COLORS[pi_ % len(_PLANE_COLORS)],
                    opacity=0.25, flatshading=True, hoverinfo="skip",
                    name=disp,
                    legendgroup=f"plane{pi_}@{ri}", showlegend=vis,
                    visible="legendonly",
                ))
                meta.append((ri, "plane"))
                plane_acc.setdefault(key, []).append(acc)
                pi_ += 1

    # ── layer slider ─────────────────────────────────────────────────────
    steps = []
    for ri, row in enumerate(rows):
        # Centroids stay opt-in on every layer: "legendonly" keeps them in the
        # legend, one click away, without cluttering the default view.
        visible    = [("legendonly" if kind in ("centroid", "plane") else True)
                      if li == ri else False
                      for li, kind in meta]
        showlegend = [li == ri and kind != "line" for li, kind in meta]
        lab = _ev_labeler(row)
        steps.append(dict(
            method="update", label=str(row["layer"]),
            args=[{"visible": visible, "showlegend": showlegend},
                  {"scene.xaxis.title.text": lab(comps[0]),
                   "scene.yaxis.title.text": lab(comps[1]),
                   "scene.zaxis.title.text": lab(comps[2])}],
        ))

    # ── pair-line opacity slider ─────────────────────────────────────────
    line_idx = [k for k, (_, kind) in enumerate(meta) if kind == "line"]
    alpha_steps = [
        dict(method="restyle", label=f"{int(a * 100)}%",
             args=[{"line.color": _PAIR_LINE_ALPHAS_RGBA[a]}, line_idx])
        for a in _PAIR_LINE_ALPHAS
    ]

    lab0 = _ev_labeler(rows[0])
    sliders = [dict(active=0, currentvalue=dict(prefix="Layer "),
                    pad=dict(t=45), y=0, x=0, len=1.0, steps=steps)]
    if line_idx:
        sliders.append(dict(
            active=_PAIR_LINE_ALPHAS.index(_PAIR_LINE_DEFAULT),
            currentvalue=dict(prefix="Pair lines "),
            pad=dict(t=45), y=-0.17, x=0, len=0.55, steps=alpha_steps))

    fig.update_layout(
        template="plotly_white",
        height=760,
        margin=dict(l=0, r=0, t=10, b=110 if line_idx else 60),
        legend=dict(itemsizing="constant", y=0.95),
        scene=dict(
            aspectmode="cube",
            camera=dict(eye=eye),
            xaxis=dict(title=lab0(comps[0]), gridcolor=_GRID_COLOR, zerolinecolor=_GRID_COLOR),
            yaxis=dict(title=lab0(comps[1]), gridcolor=_GRID_COLOR, zerolinecolor=_GRID_COLOR),
            zaxis=dict(title=lab0(comps[2]), gridcolor=_GRID_COLOR, zerolinecolor=_GRID_COLOR),
        ),
        sliders=sliders,
    )
    return fig, plane_acc


def _write_html(pages: list[dict], out_path: Path,
                views: list[tuple[float, float]] | None = None,
                comp_sets: list[tuple[int, int, int]] | None = None) -> int:
    """
    Self-contained interactive page: one figure per (page, component triple), with
    plotly.js inlined so it opens from disk with no server.

    Each triple gets its own figure rather than a selector on a shared one -- the
    layer slider already owns trace visibility, and a second control writing the
    same array cannot stay in sync with it.
    """
    import plotly.graph_objects as go      # only this path needs plotly

    comp_sets = comp_sets or [(0, 1, 2)]
    elev, azim = (views or _VIEWS_3D)[0]
    # Spherical eye position for the initial camera, matching the PDF's first view.
    r = 1.9
    eye = dict(
        x=r * np.cos(np.radians(elev)) * np.cos(np.radians(azim)),
        y=r * np.cos(np.radians(elev)) * np.sin(np.radians(azim)),
        z=r * np.sin(np.radians(elev)),
    )

    blocks: list[str] = []
    n_written = 0
    for page in pages:
        for comps in comp_sets:
            built = _html_figure(go, page, comps, eye)
            if built is None:
                continue
            fig, plane_acc = built
            body = fig.to_html(full_html=False,
                               include_plotlyjs=True if n_written == 0 else False,
                               config={"displaylogo": False})
            heading = f"{_page_title(page)} &nbsp;|&nbsp; EV {comps[0]}/{comps[1]}/{comps[2]}"
            blocks.append(f"<section><h2>{heading}</h2>{body}</section>")
            if plane_acc:
                summary = "  ".join(
                    f"{k}: {min(v):.0%}-{max(v):.0%}" for k, v in plane_acc.items())
                print(f"    EV {comps[0]}/{comps[1]}/{comps[2]}  in-slice separation — {summary}")
            n_written += 1

    out_path.write_text(
        "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
        f"<title>{out_path.stem}</title>"
        "<style>body{font-family:system-ui,sans-serif;margin:0 auto;padding:1.5rem;"
        "max-width:1100px;color:#222}h1{font-size:1.2rem}h2{font-size:.85rem;"
        "font-weight:600;color:#555;border-top:1px solid #ddd;padding-top:1rem;"
        "margin-top:2rem;word-break:break-word}</style></head><body>"
        f"<h1>{out_path.stem}</h1>" + "".join(blocks) + "</body></html>"
    )
    return n_written


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-families", action="store_true",
                        help="Also render per-family probes (default: compounds only)")
    parser.add_argument("--color-by", default=None,
                        help="Attribute to color points by; checked in descriptors then metadata "
                             "(default: from reporting.yaml scatter.color_by, "
                             "or the probe's label descriptor)")
    parser.add_argument("--max-pairs", type=int, default=None,
                        help="Max pairs per layer panel before stratified sampling kicks in "
                             "(default: from reporting.yaml or 200)")
    parser.add_argument("--max-vectors", type=int, default=None,
                        help="Upper bound on eigenvectors per layer row (default: from reporting.yaml or 8)")
    parser.add_argument("--components", default="0,1,2",
                        help="Eigenvector triple(s) to plot, e.g. '0,1,2' or "
                             "'0,1,2;0,1,3;0,2,3;1,2,3'.  Each triple becomes its own "
                             "3-D page (PDF) or figure (HTML).  3-D/HTML modes only.")
    parser.add_argument("--html", action="store_true",
                        help="Write a self-contained interactive 3-D page (plotly.js inlined, "
                             "opens from disk with no server) instead of a PDF")
    parser.add_argument("--3d", dest="three_d", action="store_true",
                        help="Render one 3-D axes per (layer, viewing angle) over the first "
                             "three eigenvectors instead of 2-D eigenvector-pair panels")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--compounds", default=None,
                        help="Comma-separated run_compound keys to restrict to "
                             "(e.g. llama3_8b__mt_cities); other compounds are skipped")
    parser.add_argument("--datasets", default=None,
                        help="Comma-separated dataset names to keep (e.g. cities); "
                             "default: every dataset the probes cover")
    parser.add_argument("--runs", default=None,
                        help="Comma-separated run keys to restrict to (e.g. llama3_8b__farquhar). "
                             "When set, only these per-family runs are rendered and compounds are skipped.")
    parser.add_argument("--output-name", default=None,
                        help="Output PDF basename, without extension "
                             "(default: subspace_scatter, or subspace_scatter_3d with --3d)")
    args = parser.parse_args()
    output_name = args.output_name or (
        "subspace_scatter_3d" if (args.three_d or args.html) else "subspace_scatter")
    only_runs = [r.strip() for r in args.runs.split(",")] if args.runs else None
    only_datasets = {d.strip() for d in args.datasets.split(",")} if args.datasets else None
    only_compounds = {c.strip() for c in args.compounds.split(",")} if args.compounds else None
    comp_sets = _parse_component_sets(args.components)
    # _collect_pages keeps an even number of eigenvectors; make sure the highest
    # requested index is among them.
    need_vectors = 2 * ((max(max(c) for c in comp_sets) + 2) // 2)

    params   = yaml.safe_load(Path("params.yaml").read_text())
    op       = params["output"]
    sc       = yaml.safe_load(Path("reporting.yaml").read_text()).get("scatter", {})

    color_by    = args.color_by    if args.color_by    is not None else sc.get("color_by", "")
    max_pairs   = args.max_pairs   if args.max_pairs   is not None else sc.get("max_pairs", 200)
    max_vectors = args.max_vectors if args.max_vectors is not None else sc.get("max_vectors", 8)
    dpi         = sc.get("dpi", 150)
    views_3d    = [tuple(v) for v in sc.get("views_3d", [])] or _VIEWS_3D

    act_base = Path(op["activations_dir"])
    prb_base = Path(op["probes_dir"])
    out_dir  = Path(args.output_dir) if args.output_dir else Path(op["reports_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{output_name}.{'html' if args.html else 'pdf'}"

    all_pages: list[dict] = []

    def _process(name: str, sources: list[tuple[list, Path, str]], probes_dir: Path,
                 lookup: str | None = None) -> None:
        """sources: (samples, activation cache dir, model name) per contributing run."""
        if not (probes_dir / "manifest.json").exists():
            print(f"  [{name}] skipping (no manifest)")
            return
        # Per-source settings: try the short compound/family name first, then the full run key.
        per = sc.get("per_source", {})
        src = per.get(lookup or name, per.get(name, {}))
        eff_color_by = src.get("color_by", color_by)
        eff_shape_by = src.get("shape_by", sc.get("shape_by", ""))
        raw_labels   = src.get("color_labels", sc.get("color_labels", {}))
        if raw_labels:
            if isinstance(eff_color_by, list):
                # Combined string keys — keep as strings.
                eff_labels = {str(k): v for k, v in raw_labels.items()}
            else:
                # Single numeric field — convert keys to int for backward compat.
                try:
                    eff_labels = {int(k): v for k, v in raw_labels.items()}
                except (ValueError, TypeError):
                    eff_labels = {str(k): v for k, v in raw_labels.items()}
        else:
            eff_labels = None
        raw_shape_labels = src.get("shape_labels", sc.get("shape_labels", {}))
        eff_shape_labels = {str(k): str(v) for k, v in raw_shape_labels.items()} if raw_shape_labels else None
        eff_shade_by     = src.get("shade_by", sc.get("shade_by", ""))
        raw_shade_labels = src.get("shade_labels", sc.get("shade_labels", {}))
        eff_shade_labels = {str(k): str(v) for k, v in raw_shade_labels.items()} if raw_shade_labels else None
        eff_cyclic       = src.get("cyclic_color", sc.get("cyclic_color", False))
        eff_cyclic_order = src.get("cyclic_order_by", sc.get("cyclic_order_by", "multiclass_label"))
        min_v    = src.get("min_vectors", sc.get("min_vectors", 0))
        if args.three_d or args.html:
            min_v = max(min_v, need_vectors)
        manifest = Manifest.load(probes_dir)
        probes   = load_probes(manifest.params_file)

        # Only the (layer, position) pairs some probe was fitted at are ever
        # plotted; everything else in the cache stays on disk.
        store = ActivationStore({(e.layer, e.train_position) for e in manifest.entries})
        samples = []
        for run_samples, cache_dir, model_name in sources:
            store.register(run_samples, cache_dir, model_name)
            samples.extend(run_samples)

        # A probe trained before the data was regenerated holds test ids that no
        # longer exist, which silently yields empty pages.  Say so instead.
        sample_ids = {s.id for s in samples}
        stale = sum(1 for e in manifest.entries if not sample_ids.intersection(e.test_ids))
        if stale:
            print(f"  [{name}] {stale}/{len(manifest.entries)} probe entries reference sample ids "
                  f"that are no longer in the processed data (retrain probes for this source)")

        pages    = _collect_pages(
            manifest, probes, samples, store,
            max_vectors=max_vectors,
            min_vectors=min_v,
            max_pairs=max_pairs,
            color_by=eff_color_by,
            color_labels=eff_labels,
            shape_by=eff_shape_by,
            shape_labels=eff_shape_labels,
            shade_by=eff_shade_by,
            shade_labels=eff_shade_labels,
            cyclic_color=eff_cyclic,
            cyclic_order_by=eff_cyclic_order,
        )
        if only_datasets is not None:
            kept = [p for p in pages if p["dataset"] in only_datasets]
            if len(kept) != len(pages):
                print(f"  [{name}] --datasets {sorted(only_datasets)}: "
                      f"kept {len(kept)} of {len(pages)} page(s)")
            pages = kept
        excl = set(src.get("exclude_datasets", []) or [])
        if excl:
            kept = [p for p in pages if p["dataset"] not in excl]
            dropped = len(pages) - len(kept)
            if dropped:
                print(f"  [{name}] excluding {sorted(excl)}: dropped {dropped} page(s)")
            pages = kept
        for p in pages:
            p["source"] = name
        all_pages.extend(pages)
        missing = f", {store.n_missing} sample(s) not in the activation cache" if store.n_missing else ""
        print(f"  [{name}] {len(pages)} page(s){missing}")

    if args.include_families or only_runs:
        for run_key, run_cfg in params.get("runs", {}).items():
            if only_runs is not None and run_key not in only_runs:
                continue
            samples_path = Path(f"data/processed/samples_{run_cfg['family']}.jsonl")
            if not samples_path.exists():
                print(f"  [{run_key}] skipping (no samples file)")
                continue
            print(f"Processing run {run_key} …")
            model_name = params["models"][run_cfg["model"]]["name"]
            sources = [(_load_run_samples(run_key, params),
                        act_base / run_key, model_name)]
            _process(run_key, sources, prb_base / run_key,
                 lookup=run_cfg["family"])

    compound_runs = {} if only_runs else params.get("run_compounds", {})
    if only_compounds is not None:
        compound_runs = {k: v for k, v in compound_runs.items() if k in only_compounds}
    for rc_key, rc_cfg in compound_runs.items():
        compound_cfg = params["compounds"][rc_cfg["compound"]]
        model_alias = rc_cfg["model"]
        print(f"Processing compound run {rc_key} …")
        model_name = params["models"][model_alias]["name"]
        filter_ds  = compound_cfg.get("datasets")
        sources: list[tuple[list, Path, str]] = []
        for fam in compound_cfg["families"]:
            run_key = _find_run_key(params, model_alias, fam)
            if not Path(f"data/processed/samples_{fam}.jsonl").exists():
                continue
            fam_samples = _load_run_samples(run_key, params)
            if filter_ds:
                fam_samples = [s for s in fam_samples
                               if s.descriptors.get("dataset_name", "") in filter_ds]
            sources.append((fam_samples, act_base / run_key, model_name))
        _process(rc_key, sources, prb_base / rc_key,
                 lookup=rc_cfg["compound"])

    print(f"\nWriting {len(all_pages)} page(s) to {out_path} …")
    if args.html:
        n = _write_html(all_pages, out_path, views=views_3d, comp_sets=comp_sets)
        size_mb = out_path.stat().st_size / 1e6
        print(f"  {n} interactive page(s), {size_mb:.1f} MB self-contained")
    else:
        with PdfPages(out_path) as pdf:
            for page in all_pages:
                if args.three_d:
                    for comps in comp_sets:
                        _plot_page_3d(pdf, page, dpi=dpi, views=views_3d, comps=comps)
                else:
                    _plot_page(pdf, page, dpi=dpi)

    print("Done.")


if __name__ == "__main__":
    main()
