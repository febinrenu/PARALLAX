"""Cluster bootstrap confidence intervals, one implementation for every metric we report.

Resampling unit is the *group* (lesion, patient, duplicate cluster), not the image, so images that
share a group are never treated as independent. Optional class stratification resamples groups
inside each stratum, which keeps every class present in every replicate (needed for balanced
accuracy and per-class recall). Intervals are 95% percentile by default; `method="bca"` applies the
bias-corrected and accelerated adjustment (jackknife over groups), used when the number of groups is
small, e.g. 11 test patients.
"""

from __future__ import annotations

from typing import Callable, Mapping

import numpy as np
from scipy.stats import norm

B_DEFAULT = 2000
SEED = 20261006


def _index(groups: np.ndarray, strata: np.ndarray | None):
    uniq, inv = np.unique(groups, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    bounds = np.cumsum(np.bincount(inv))[:-1]
    members = np.split(order, bounds)  # row indices per group
    if strata is None:
        gstrat = np.zeros(len(uniq), int)
    else:
        s = np.asarray(strata)
        first = np.array([s[m[0]] if len(set(s[m])) == 1 else _majority(s[m]) for m in members], dtype=object)
        gstrat = np.unique(first, return_inverse=True)[1]
    return members, gstrat


def _majority(v: np.ndarray):
    vals, cnt = np.unique(v, return_counts=True)
    return vals[np.argmax(cnt)]


def bootstrap_values(
    arrays: Mapping[str, np.ndarray],
    stat_fn: Callable[..., float],
    groups: np.ndarray | None = None,
    strata: np.ndarray | None = None,
    B: int = B_DEFAULT,
    seed: int = SEED,
) -> tuple[float, np.ndarray, list]:
    """Return (point estimate, B replicate values, group index lists). stat_fn(**arrays) -> float."""
    n = len(next(iter(arrays.values())))
    g = np.arange(n) if groups is None else np.asarray(groups)
    members, gstrat = _index(g, strata)
    if len(members) < 2:
        raise ValueError("need at least 2 groups to bootstrap")
    rng = np.random.RandomState(seed)
    arr = {k: np.asarray(v) for k, v in arrays.items()}
    point = float(stat_fn(**arr))
    by_stratum = [np.nonzero(gstrat == s)[0] for s in np.unique(gstrat)]
    vals = np.empty(B)
    for b in range(B):
        picked = np.concatenate([rng.choice(ids, size=len(ids), replace=True) for ids in by_stratum])
        idx = np.concatenate([members[i] for i in picked])
        vals[b] = stat_fn(**{k: v[idx] for k, v in arr.items()})
    return point, vals, members


def _bca_bounds(point: float, vals: np.ndarray, jack: np.ndarray, alpha: float) -> tuple[float, float]:
    frac = np.clip(np.mean(vals < point) + 0.5 * np.mean(vals == point), 1e-6, 1 - 1e-6)
    z0 = norm.ppf(frac)
    d = jack.mean() - jack
    denom = 6.0 * (np.sum(d**2) ** 1.5)
    a = 0.0 if denom == 0 else np.sum(d**3) / denom
    out = []
    for q in (alpha / 2, 1 - alpha / 2):
        z = norm.ppf(q)
        adj = norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
        out.append(float(np.quantile(vals, np.clip(adj, 0, 1))))
    return out[0], out[1]


def ci(
    arrays: Mapping[str, np.ndarray],
    stat_fn: Callable[..., float],
    groups: np.ndarray | None = None,
    strata: np.ndarray | None = None,
    B: int = B_DEFAULT,
    seed: int = SEED,
    alpha: float = 0.05,
    method: str = "percentile",
) -> dict:
    point, vals, members = bootstrap_values(arrays, stat_fn, groups, strata, B, seed)
    vals_f = vals[np.isfinite(vals)]
    if method == "bca":
        n = len(next(iter(arrays.values())))
        jack = []
        for i in range(len(members)):
            keep = np.concatenate([m for j, m in enumerate(members) if j != i])
            jack.append(stat_fn(**{k: np.asarray(v)[keep] for k, v in arrays.items()}))
        lo, hi = _bca_bounds(point, vals_f, np.array(jack, float), alpha)
    elif method == "percentile":
        lo, hi = (float(x) for x in np.quantile(vals_f, [alpha / 2, 1 - alpha / 2]))
    else:
        raise ValueError(method)
    return {"point": point, "lo": lo, "hi": hi, "level": 1 - alpha, "B": int(B), "method": method, "n_groups": len(members), "n": int(len(next(iter(arrays.values())))),
            "stratified": strata is not None, "seed": seed, "n_valid": int(len(vals_f))}


def diff_ci(values_a: np.ndarray, values_b: np.ndarray, point_a: float, point_b: float, alpha: float = 0.05) -> dict:
    """CI for a - b where a and b come from independent test sets (e.g. leaky vs leakage-free accuracy)."""
    n = min(len(values_a), len(values_b))
    d = values_a[:n] - values_b[:n]
    lo, hi = np.quantile(d[np.isfinite(d)], [alpha / 2, 1 - alpha / 2])
    return {"point": point_a - point_b, "lo": float(lo), "hi": float(hi), "level": 1 - alpha, "method": "independent percentile difference"}


def fmt(c: dict, digits: int = 3, pct: bool = False) -> str:
    k = 100 if pct else 1
    return f"{c['point'] * k:.{digits}f} [{c['lo'] * k:.{digits}f}, {c['hi'] * k:.{digits}f}]"
