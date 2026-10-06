"""Leakage audit core: near-duplicate clusters, removal rules and report numbers.

Used by make_splits.py for every dataset. Rules (see progress.md Decisions):
  * duplicate = pHash Hamming <= 4
  * immutable images (official test sets) are never dropped; copies of them elsewhere are
  * near-identical images (Hamming <= 1) with different labels are label noise: both are dropped
  * anything within a duplicate cluster that contains an immutable image is dropped
  * otherwise one representative (lowest image_id) is kept per (cluster, label); a cluster holding two labels keeps both
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.data import phash as ph


def merge_groups(native: pd.Series, dup_cluster: pd.Series) -> pd.Series:
    """Group id = union of the native group (lesion, patient) and the duplicate cluster, as one string per row."""
    n = len(native)
    key_native = pd.factorize(native.astype(str))[0]
    key_dup = pd.factorize(dup_cluster.astype(str))[0] + key_native.max() + 1
    # nodes are native ids and dup ids; each row links its two nodes
    lab = ph.components(int(key_dup.max()) + 1, [(int(a), int(b)) for a, b in zip(key_native, key_dup)])
    return pd.Series([f"g{lab[a]}" for a in key_native], index=native.index)


def audit(
    df: pd.DataFrame,
    hashes: np.ndarray,
    *,
    label_col: str = "label",
    orig_split_col: str = "orig_split",
    immutable: pd.Series | None = None,
    max_dist: int = ph.DUP_MAX_DIST,
    noise_dist: int = 1,
    loose_dist: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    df = df.reset_index(drop=True).copy()
    n = len(df)
    imm = np.zeros(n, bool) if immutable is None else immutable.reset_index(drop=True).to_numpy(bool)
    pairs = ph.near_pairs(hashes, max_dist)
    cluster = ph.components(n, [(i, j) for i, j, _ in pairs])
    df["dup_cluster"] = cluster
    if loose_dist is not None:
        df["loose_cluster"] = ph.components(n, [(i, j) for i, j, _ in ph.near_pairs(hashes, loose_dist)])

    orig = df[orig_split_col].astype(str).to_numpy() if orig_split_col in df else np.array(["none"] * n)
    ids = df["image_id"].astype(str).to_numpy()
    labels = df[label_col].astype(str).to_numpy()
    dropped = np.array([""] * n, dtype=object)
    # label noise: (near-)identical images that carry different labels. Looser pairs with different labels are
    # usually another view or scan of the same case, so they are kept and only grouped together.
    noise = np.zeros(n, bool)
    for i, j, d in pairs:
        if d <= noise_dist and labels[i] != labels[j]:
            noise[i] = noise[j] = True
    conflict_clusters = len({int(cluster[i]) for i in np.nonzero(noise)[0]})
    for cl, idx in pd.Series(np.arange(n)).groupby(cluster):
        idx = idx.to_numpy()
        if len(idx) < 2:
            continue
        keep_imm = [i for i in idx if imm[i]]
        if keep_imm:  # anything near an immutable test image leaves the other sets, whatever its label
            for i in idx:
                if not imm[i]:
                    dropped[i] = f"duplicate_of_immutable:{ids[keep_imm[0]]}" if labels[i] == labels[keep_imm[0]] and not noise[i] else "label_conflict"
            continue
        for lab in set(labels[idx]):  # one representative per (cluster, label)
            members = [i for i in idx if labels[i] == lab]
            rep = min(members, key=lambda i: ids[i])
            for i in members:
                if i != rep:
                    dropped[i] = f"duplicate_of:{ids[rep]}"
    for i in np.nonzero(noise)[0]:
        if not imm[i]:
            dropped[i] = "label_conflict"
    df["dropped_reason"] = dropped

    dist_hist = {int(d): 0 for d in range(max_dist + 1)}
    within = cross = 0
    touched_by_split: dict[str, set[int]] = {}
    for i, j, d in pairs:
        dist_hist[d] += 1
        if orig[i] != "none" and orig[j] != "none" and orig[i] != orig[j]:
            cross += 1
            touched_by_split.setdefault(orig[i], set()).add(i)
            touched_by_split.setdefault(orig[j], set()).add(j)
        else:
            within += 1
    sizes = pd.Series(cluster).value_counts()
    report = {
        "n_images": int(n),
        "max_hamming": max_dist,
        "duplicate_pairs": len(pairs),
        "pairs_by_distance": dist_hist,
        "pairs_within_original_split": within,
        "pairs_across_original_splits": cross,
        "images_with_duplicate_in_another_original_split": {k: len(v) for k, v in sorted(touched_by_split.items())},
        "duplicate_clusters": int((sizes > 1).sum()),
        "largest_cluster": int(sizes.max()) if len(sizes) else 0,
        "label_conflict_clusters": int(conflict_clusters),
        "label_noise_max_hamming": noise_dist,
        "images_removed": int((dropped != "").sum()),
        "removed_by_reason": {
            "label_conflict": int((dropped == "label_conflict").sum()),
            "duplicate": int(pd.Series(dropped).str.startswith("duplicate_of:").sum()),
            "duplicate_of_immutable_test_image": int(pd.Series(dropped).str.startswith("duplicate_of_immutable").sum()),
        },
    }
    if loose_dist is not None:
        lsz = df["loose_cluster"].value_counts()
        report["loose_grouping"] = {"max_hamming": loose_dist, "clusters": int(len(lsz)), "largest": int(lsz.max()), "singletons": int((lsz == 1).sum())}
    return df, report


def stratified_subset(df: pd.DataFrame, label_col: str, cap: int, seed: int, min_per_class: int = 25) -> pd.DataFrame:
    """Deterministic label-stratified subset of at most `cap` rows; the whole frame when it is smaller."""
    if len(df) <= cap:
        return df
    rng = np.random.RandomState(seed)
    counts = df[label_col].value_counts()
    quota = {k: min(int(v), max(min_per_class, int(round(cap * v / len(df))))) for k, v in counts.items()}
    # trim the largest classes until the quotas fit the cap
    while sum(quota.values()) > cap:
        big = max(quota, key=lambda k: quota[k])
        quota[big] -= 1
    parts = []
    for k, q in quota.items():
        sub = df[df[label_col] == k].sort_values("image_id")
        parts.append(sub.iloc[np.sort(rng.choice(len(sub), size=q, replace=False))])
    return pd.concat(parts).sort_values("image_id")
