from __future__ import annotations

import numpy as np
import pandas as pd

from ml.data import splitting as sp

FR = {"train": 0.7, "val": 0.1, "cal": 0.1, "test": 0.1}


def _toy(n_groups=600, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        label = rng.choice(["a", "b", "c"], p=[0.7, 0.25, 0.05])
        for k in range(int(rng.integers(1, 5))):
            rows.append((f"g{g}", f"g{g}_{k}", label))
    return pd.DataFrame(rows, columns=["group", "image_id", "label"])


def test_groups_never_straddle_and_all_rows_assigned():
    df = _toy()
    df["split"] = sp.group_split(df, "group", "label", FR)
    assert df.split.notna().all()
    sp.assert_no_straddle(df, "group")


def test_proportions_close_overall_and_per_class():
    df = _toy(2000)
    df["split"] = sp.group_split(df, "group", "label", FR)
    share = df.split.value_counts(normalize=True)
    for k, v in FR.items():
        assert abs(share[k] - v) < 0.02, (k, share[k])
    for label, sub in df.groupby("label"):
        s = sub.split.value_counts(normalize=True)
        assert abs(s["train"] - 0.7) < 0.06, (label, s)
        assert (sub.split == "test").sum() > 0


def test_deterministic_and_seed_sensitive():
    df = _toy()
    a = sp.group_split(df, "group", "label", FR, seed=1)
    b = sp.group_split(df, "group", "label", FR, seed=1)
    c = sp.group_split(df, "group", "label", FR, seed=2)
    assert a.equals(b) and not a.equals(c)


def test_split_hash_changes_with_assignment():
    df = _toy(50)
    df["split"] = sp.group_split(df, "group", "label", FR)
    h1 = sp.split_hash(df)
    df2 = df.copy()
    df2.loc[df2.index[0], "split"] = "test" if df2.split.iloc[0] != "test" else "train"
    assert h1 != sp.split_hash(df2)
    assert h1 == sp.split_hash(df.sample(frac=1, random_state=0))
