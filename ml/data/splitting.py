"""Group-aware, class-stratified split assignment.

Whole groups (a lesion, a patient, a duplicate cluster) go to exactly one split. Inside each stratum
the groups are shuffled with a fixed seed, then handed to whichever split is furthest below its target
share of images, so image-level proportions come out close to the requested fractions.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

SEED = 20261006


def group_split(df: pd.DataFrame, group_col: str, strat_col: str, fractions: dict[str, float], seed: int = SEED) -> pd.Series:
    if abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise ValueError("fractions must sum to 1")
    rng = np.random.RandomState(seed)
    names = list(fractions)
    out = pd.Series(index=df.index, dtype=object)
    # one stratum label per group: the most common label among its images (ties -> alphabetical)
    g = df.groupby(group_col, sort=True)
    strat_of = g[strat_col].agg(lambda s: sorted(s.value_counts().sort_index().index, key=lambda v: (-(s == v).sum(), str(v)))[0])
    size_of = g.size()
    split_of: dict = {}
    for stratum in sorted(strat_of.unique(), key=str):
        gids = sorted(strat_of.index[strat_of == stratum], key=str)
        rng.shuffle(gids)
        total = int(size_of[gids].sum())
        target = {n: fractions[n] * total for n in names}
        have = {n: 0 for n in names}
        for gid in gids:
            n = max(names, key=lambda k: (target[k] - have[k], -names.index(k)))
            split_of[gid] = n
            have[n] += int(size_of[gid])
    out[:] = df[group_col].map(split_of)
    return out


def assert_no_straddle(df: pd.DataFrame, group_col: str, split_col: str = "split") -> None:
    bad = df.groupby(group_col)[split_col].nunique()
    bad = bad[bad > 1]
    if len(bad):
        raise AssertionError(f"{len(bad)} groups appear in more than one split, e.g. {list(bad.index[:3])}")


def split_hash(df: pd.DataFrame, cols: tuple[str, ...] = ("image_id", "split")) -> str:
    """Stable fingerprint of an assignment, recorded in the model registry."""
    rows = df.sort_values("image_id")[list(cols)].astype(str).agg("|".join, axis=1)
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()
