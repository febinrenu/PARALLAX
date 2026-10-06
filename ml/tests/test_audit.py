from __future__ import annotations

import numpy as np
import pandas as pd

from ml.data import audit as au


def _frame(rows):
    return pd.DataFrame(rows, columns=["image_id", "label", "orig_split"])


H = {k: np.uint64(v) for k, v in {"a": 0xAAAAAAAAAAAAAAAA, "b": 0x0F0F0F0F0F0F0F0F, "c": 0x123456789ABCDEF0, "d": 0xFEDCBA9876543210}.items()}


def test_cross_split_duplicate_is_counted_and_one_copy_kept():
    df = _frame([("i1", "x", "train"), ("i2", "x", "test"), ("i3", "y", "train")])
    h = np.array([H["a"], H["a"] ^ np.uint64(1), H["b"]], dtype=np.uint64)
    out, rep = au.audit(df, h)
    assert rep["pairs_across_original_splits"] == 1
    assert rep["images_with_duplicate_in_another_original_split"] == {"test": 1, "train": 1}
    assert out.dropped_reason.tolist() == ["", "duplicate_of:i1", ""]


def test_label_conflict_drops_every_copy():
    df = _frame([("i1", "x", "train"), ("i2", "y", "train"), ("i3", "x", "train")])
    h = np.array([H["a"], H["a"], H["c"]], dtype=np.uint64)
    out, rep = au.audit(df, h)
    assert out.dropped_reason.tolist() == ["label_conflict", "label_conflict", ""]
    assert rep["label_conflict_clusters"] == 1


def test_immutable_test_images_are_never_dropped_but_their_copies_are():
    df = _frame([("t1", "x", "test"), ("tr1", "x", "train"), ("tr2", "x", "train")])
    h = np.array([H["a"], H["a"], H["a"]], dtype=np.uint64)
    out, rep = au.audit(df, h, immutable=pd.Series([True, False, False]))
    assert out.dropped_reason.tolist() == ["", "duplicate_of_immutable:t1", "duplicate_of_immutable:t1"]
    assert rep["removed_by_reason"]["duplicate_of_immutable_test_image"] == 2


def test_different_labels_at_looser_distance_are_two_views_not_noise():
    df = _frame([("i1", "x", "train"), ("i2", "y", "train"), ("i3", "x", "train")])
    h = np.array([H["a"], H["a"] ^ np.uint64(0b111), H["a"]], dtype=np.uint64)  # i2 is 3 bits away from i1/i3
    out, rep = au.audit(df, h)
    assert out.dropped_reason.tolist() == ["", "", "duplicate_of:i1"]
    assert rep["label_conflict_clusters"] == 0


def test_immutable_stays_even_when_labels_conflict():
    df = _frame([("t1", "x", "test"), ("tr1", "y", "train")])
    h = np.array([H["a"], H["a"]], dtype=np.uint64)
    out, _ = au.audit(df, h, immutable=pd.Series([True, False]))
    assert out.dropped_reason.tolist() == ["", "label_conflict"]


def test_loose_clusters_group_nearby_but_not_duplicate_images():
    df = _frame([("a", "x", "train"), ("b", "x", "train"), ("c", "x", "train")])
    near = H["a"] ^ np.uint64(0b111111111)  # 9 bits away: not a duplicate, inside a loose radius of 10
    h = np.array([H["a"], near, H["d"]], dtype=np.uint64)
    out, rep = au.audit(df, h, loose_dist=10)
    assert out.dropped_reason.eq("").all()
    assert out.loose_cluster[0] == out.loose_cluster[1] != out.loose_cluster[2]
    assert rep["loose_grouping"]["clusters"] == 2


def test_merge_groups_joins_native_groups_through_duplicates():
    native = pd.Series(["L1", "L1", "L2", "L3"])
    dup = pd.Series([0, 1, 1, 3])  # image 1 and 2 are duplicates, so L1 and L2 must merge
    g = au.merge_groups(native, dup)
    assert g[0] == g[1] == g[2] != g[3]


def test_stratified_subset_is_deterministic_and_keeps_rare_classes():
    rows = [(f"i{k:04d}", "common" if k < 1800 else "rare") for k in range(1900)]
    df = pd.DataFrame(rows, columns=["image_id", "label"])
    a = au.stratified_subset(df, "label", 1000, seed=1)
    b = au.stratified_subset(df, "label", 1000, seed=1)
    assert a.equals(b) and len(a) <= 1000
    assert (a.label == "rare").sum() >= 25
    assert au.stratified_subset(df.head(50), "label", 1000, seed=1).shape[0] == 50
