"""Build leakage-audited, group-aware splits for every dataset (P2.2).

    python ml/data/make_splits.py fracatlas ham10000 brain_mri lgg_seg rsna     # any subset
    python ml/data/make_splits.py all                                           # every dataset that is downloaded

Per dataset this writes ml/data/splits/<name>.csv, updates reports/leakage.json and
ml/data/splits/split_hashes.json, and (re)writes ml/data/eval_index.json for P3's batch jobs.

Split column values: train / val / cal / test, plus official_test (ISIC, never modified) and dropped
(removed by the audit; reason in dropped_reason). Seeded with SEED = 20261006.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):  # allow `python ml/data/make_splits.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

from ml.data import audit as au
from ml.data import phash as ph
from ml.data import splitting as sp
from ml.data.common import CACHE_DIR, REPO_ROOT, REPORTS_DIR, SEED, SPLITS_DIR, data_root

LOW_INFO_STD = 2.0  # grey-level std (0..255) below which an image is "flat" and is excluded from duplicate matching
BATCH_CAP = 1000  # MedGemma batch subset size per dataset
COLUMNS = ["dataset", "source_dir", "relpath", "image_id", "label", "group", "split", "orig_split", "dup_cluster", "loose_cluster", "dropped_reason", "eval_batch", "phash"]


# ----------------------------------------------------------------------------- hashing with a cache


def hash_dataset(name: str, source_dirs: pd.Series, relpaths: pd.Series, root: Path, workers: int = 4) -> np.ndarray:
    """pHash per row, cached by (source_dir/relpath). Flat images get a unique pseudo-hash so they never match each other."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_p = CACHE_DIR / f"{name}_phash.npz"
    keys = (source_dirs + "/" + relpaths).to_numpy()
    have: dict[str, tuple[int, float, bool]] = {}
    if cache_p.is_file():
        z = np.load(cache_p, allow_pickle=False)
        have = {k: (int(h), float(s), bool(t)) for k, h, s, t in zip(z["keys"], z["hashes"], z["stds"], z["trunc"])}
    todo = [i for i, k in enumerate(keys) if k not in have]
    if todo:
        print(f"[{name}] hashing {len(todo)} images", flush=True)
        paths = [root / source_dirs.iloc[i] / relpaths.iloc[i] for i in todo]
        h, s, t = ph.hash_paths_stats(paths, workers)
        for i, hv, sv, tv in zip(todo, h, s, t):
            have[keys[i]] = (int(hv), float(sv), bool(tv))
        ks = np.array(list(have))
        np.savez(cache_p, keys=ks, hashes=np.array([have[k][0] for k in ks], dtype=np.uint64), stds=np.array([have[k][1] for k in ks], dtype=np.float32),
                 trunc=np.array([have[k][2] for k in ks], dtype=bool))
    hashes = np.array([have[k][0] for k in keys], dtype=np.uint64)
    stds = np.array([have[k][1] for k in keys], dtype=np.float32)
    hash_dataset.truncated = np.array([have[k][2] for k in keys], dtype=bool)  # type: ignore[attr-defined]
    flat = stds < LOW_INFO_STD
    if flat.any():
        rng = np.random.RandomState(SEED)
        hashes[flat] = rng.randint(0, 2**63 - 1, size=int(flat.sum()), dtype=np.int64).astype(np.uint64) << np.uint64(1)
    hash_dataset.last_flat = int(flat.sum())  # type: ignore[attr-defined]
    return hashes


# ----------------------------------------------------------------------------- finishing


def _finish(name: str, df: pd.DataFrame, hashes: np.ndarray, fractions: dict[str, float], strat_col: str, extra_report: dict, *, group_col: str | None = "native_group",
            immutable: pd.Series | None = None, loose_dist: int | None = None, use_loose_as_group: bool = False, keep_cols: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    df, rep = au.audit(df, hashes, immutable=immutable, loose_dist=loose_dist)
    rep["flat_images_excluded_from_matching"] = int(getattr(hash_dataset, "last_flat", 0))
    trunc = getattr(hash_dataset, "truncated", None)
    if trunc is not None and len(trunc) == len(df):
        df["truncated_file"] = trunc
        rep["truncated_source_files"] = int(trunc.sum())
    if "loose_cluster" not in df:
        df["loose_cluster"] = -1
    df["phash"] = [f"{int(h):016x}" for h in hashes]
    if "exclude_reason" in df:  # reasons the builder already knows (overlap with a training set, a class we do not model)
        for reason in sorted(set(df.exclude_reason) - {""}):
            newly = (df.dropped_reason == "") & (df.exclude_reason == reason)
            df.loc[newly, "dropped_reason"] = reason
            rep["removed_by_reason"][reason] = int(newly.sum())
        rep["images_removed"] = int((df.dropped_reason != "").sum())
    kept = df.dropped_reason == ""
    fixed = df["split"].notna() if "split" in df else pd.Series(False, index=df.index)  # rows pre-assigned (official tests)
    movable = kept & ~fixed
    if use_loose_as_group:
        df["group"] = "c" + df["loose_cluster"].astype(str)
    else:
        df["group"] = au.merge_groups(df[group_col], df["dup_cluster"])
    if "split" not in df:
        df["split"] = pd.Series([None] * len(df), index=df.index, dtype=object)
    df.loc[movable, "split"] = sp.group_split(df[movable], "group", strat_col, fractions)
    df.loc[~kept, "split"] = "dropped"
    sp.assert_no_straddle(df[df.split.isin(fractions)], "group")
    counts = df.split.value_counts().to_dict()
    rep.update(extra_report)
    rep["split_counts"] = {k: int(v) for k, v in sorted(counts.items())}
    rep["split_class_counts"] = {s: {str(k): int(v) for k, v in sub["label"].value_counts().items()} for s, sub in df.groupby("split")}
    rep["split_hash"] = sp.split_hash(df)
    rep["seed"] = SEED
    return df, rep


def _batch_subset(df: pd.DataFrame, test_splits: list[str]) -> pd.DataFrame:
    df["eval_batch"] = False
    t = df[df.split.isin(test_splits)]
    sub = au.stratified_subset(t, "label", BATCH_CAP, SEED)
    df.loc[sub.index, "eval_batch"] = True
    return df


def _write(name: str, df: pd.DataFrame, rep: dict, test_splits: list[str]) -> None:
    df = _batch_subset(df.reset_index(drop=True), test_splits)
    cols = COLUMNS + [c for c in df.columns if c not in COLUMNS and c not in ("native_group", "immutable")]
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    df[[c for c in cols if c in df.columns]].to_csv(SPLITS_DIR / f"{name}.csv", index=False)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    lp = REPORTS_DIR / "leakage.json"
    leak = json.loads(lp.read_text()) if lp.is_file() else {"datasets": {}}
    leak["generated_utc"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    leak["method"] = {"hash": "64-bit DCT pHash", "duplicate_max_hamming": ph.DUP_MAX_DIST, "seed": SEED}
    leak["datasets"][name] = rep
    lp.write_text(json.dumps(leak, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    hp = SPLITS_DIR / "split_hashes.json"
    hashes = json.loads(hp.read_text()) if hp.is_file() else {}
    hashes[name] = {"split_hash": rep["split_hash"], "rows": int(len(df)), "counts": rep["split_counts"]}
    hp.write_text(json.dumps(hashes, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    write_eval_index()
    print(f"[{name}] wrote {SPLITS_DIR / (name + '.csv')}: {rep['split_counts']}; removed {rep['images_removed']}; cross-split dup pairs {rep['pairs_across_original_splits']}", flush=True)


# ----------------------------------------------------------------------------- dataset builders


def _find(root: Path, name: str, must_have: str | None = None) -> Path:
    cands = sorted((p for p in root.rglob(name) if (not must_have or (p / must_have).exists() or must_have == "*")), key=lambda p: len(p.parts))
    if not cands:
        raise FileNotFoundError(f"{name} not found under {root}")
    return cands[0]


def build_fracatlas(root: Path) -> tuple[pd.DataFrame, np.ndarray, dict]:
    base = _find(root / "fracatlas", "FracAtlas")
    meta = pd.read_csv(base / "dataset.csv")
    folder = {p.name: sub for sub in ("Fractured", "Non_fractured") for p in (base / "images" / sub).iterdir()}
    orig = {}
    for fname, tag in (("train", "train"), ("valid", "val"), ("test", "test")):
        for i in pd.read_csv(base / "Utilities" / "Fracture Split" / f"{fname}.csv").image_id:
            orig[i] = tag
    flags = meta[["hand", "leg", "hip", "shoulder"]].sum(axis=1)
    part = np.where((flags >= 2) | (meta["mixed"] == 1), "mixed", meta[["hand", "leg", "hip", "shoulder"]].idxmax(axis=1))
    rows = pd.DataFrame({
        "dataset": "fracatlas", "source_dir": "fracatlas/" + base.name,
        "relpath": [f"images/{folder[i]}/{i}" for i in meta.image_id], "image_id": meta.image_id.str.replace(".jpg", "", regex=False),
        "label": np.where(meta.fractured == 1, "fracture", "no_fracture"), "native_group": meta.image_id,
        "orig_split": meta.image_id.map(orig).fillna("none"), "body_part": part, "view": np.select([meta.frontal == 1, meta.lateral == 1, meta.oblique == 1], ["frontal", "lateral", "oblique"], "unknown"),
        "hardware": meta.hardware, "fracture_count": meta.fracture_count,
        "yolo_label": [f"Annotations/YOLO/{i[:-4]}.txt" for i in meta.image_id],
    })
    rows["strat"] = rows.body_part + "|" + rows.label
    mism = int(((meta.fractured == 1) != meta.image_id.map(folder).eq("Fractured")).sum())
    extra = {"label_folder_mismatches": mism, "note": "no patient id exists; different views of one patient cannot be linked", "images_listed": int(len(meta)), "fractured": int((meta.fractured == 1).sum())}
    hashes = hash_dataset("fracatlas", rows.source_dir, rows.relpath, root)
    return rows, hashes, extra


def _find_in(root: Path, pattern: str) -> Path:
    hits = sorted(root.rglob(pattern), key=lambda p: len(p.parts))
    if not hits:
        raise FileNotFoundError(f"{pattern} not found under {root}")
    return hits[0]


# HAM10000 'localization' -> the ISIC general site vocabulary used by the official test metadata
HAM_SITE_TO_GENERAL = {
    "abdomen": "Trunk", "back": "Trunk", "chest": "Trunk", "trunk": "Trunk",
    "lower extremity": "Lower extremity", "foot": "Lower extremity",
    "upper extremity": "Upper extremity", "hand": "Upper extremity",
    "scalp": "Head and neck", "face": "Head and neck", "neck": "Head and neck", "ear": "Head and neck",
    "genital": "Anogenital region", "acral": "Palms and soles",
}


def build_ham10000(root: Path) -> tuple[pd.DataFrame, np.ndarray, dict]:
    classes = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
    tr_dir, te_dir = root / "isic2018_t3_train", root / "isic2018_t3_test"
    gt_tr = pd.read_csv(_find_in(tr_dir, "ISIC2018_Task3_Training_GroundTruth.csv"))
    gt_te = pd.read_csv(_find_in(te_dir, "ISIC2018_Task3_Test_GroundTruth.csv"))
    lab_tr, lab_te = gt_tr[classes].idxmax(axis=1).str.lower(), gt_te[classes].idxmax(axis=1).str.lower()
    img_tr = _find_in(tr_dir, "ISIC2018_Task3_Training_Input").relative_to(tr_dir).as_posix()
    img_te = _find_in(te_dir, "ISIC2018_Task3_Test_Input").relative_to(te_dir).as_posix()
    meta = pd.read_csv(root / "ham10000_meta" / "HAM10000_metadata.csv").drop_duplicates("image_id").set_index("image_id")
    groups = pd.read_csv(_find_in(tr_dir, "ISIC2018_Task3_Training_LesionGroupings.csv"))
    gcol = [c for c in groups.columns if "lesion" in c.lower()][0]
    icol = [c for c in groups.columns if "image" in c.lower()][0]
    lesion = groups.set_index(icol)[gcol]
    tr = pd.DataFrame({"dataset": "ham10000", "source_dir": "isic2018_t3_train", "relpath": [f"{img_tr}/{i}.jpg" for i in gt_tr.image], "image_id": gt_tr.image,
                       "label": lab_tr, "native_group": gt_tr.image.map(lesion).fillna(gt_tr.image.map(meta.lesion_id)).fillna(gt_tr.image), "orig_split": "train"})
    tr["age"] = tr.image_id.map(meta.age)
    tr["sex"] = tr.image_id.map(meta.sex)
    tr["site"] = tr.image_id.map(meta.localization)
    tr["site_general"] = tr.site.map(HAM_SITE_TO_GENERAL)
    tr["dx_type"] = tr.image_id.map(meta.dx_type)
    te = pd.DataFrame({"dataset": "ham10000", "source_dir": "isic2018_t3_test", "relpath": [f"{img_te}/{i}.jpg" for i in gt_te.image], "image_id": gt_te.image,
                       "label": lab_te, "orig_split": "official_test", "split": "official_test"})
    mp = root / "isic2018_t3_test_meta" / "isic2018_t3_test_meta.json"
    have_meta = mp.is_file()
    if have_meta:
        m = pd.DataFrame(json.loads(mp.read_text())).drop_duplicates("image_id").set_index("image_id")
        te["age"], te["sex"], te["site"] = te.image_id.map(m.get("age")), te.image_id.map(m.get("sex")), te.image_id.map(m.get("site"))
        te["site_general"] = te.site
        te["native_group"] = te.image_id.map(m.get("lesion_id")).fillna(te.image_id)
    else:
        te["native_group"] = te.image_id
    df = pd.concat([tr, te], ignore_index=True)
    imm = df.orig_split.eq("official_test")
    lbl_mismatch = int((tr.label != tr.image_id.map(meta.dx)).sum())
    df["immutable"] = imm
    extra = {"label_mismatches_vs_metadata_dx": lbl_mismatch, "official_test_metadata_from_isic_api": bool(have_meta), "n_lesions_train": int(tr.native_group.nunique()),
             "note": "official ISIC 2018 Task 3 test rows are immutable; copies of them in the training set are removed"}
    hashes = hash_dataset("ham10000", df.source_dir, df.relpath, root)
    return df, hashes, extra


def build_brain_mri(root: Path) -> tuple[pd.DataFrame, np.ndarray, dict]:
    base = root / "brain_mri"
    rows = []
    for part in ("Training", "Testing"):
        d = _find_in(base, part)
        for cdir in sorted(p for p in d.iterdir() if p.is_dir()):
            for f in sorted(cdir.iterdir()):
                if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    rows.append(("brain_mri", "brain_mri", f.relative_to(base).as_posix(), f"{part[:2]}_{cdir.name}_{f.stem}", cdir.name, part.lower()))
    df = pd.DataFrame(rows, columns=["dataset", "source_dir", "relpath", "image_id", "label", "orig_split"])
    df["native_group"] = df.image_id
    hashes = hash_dataset("brain_mri", df.source_dir, df.relpath, root)
    return df, hashes, {"note": "merged from Br35H, SARTAJ and Figshare; no patient ids, so grouping is by pHash similarity (approximation)"}


def build_lgg_seg(root: Path) -> tuple[pd.DataFrame, np.ndarray, dict]:
    from PIL import Image

    base = root / "lgg_seg"
    k3m = _find_in(base, "kaggle_3m")
    rows = []
    for pdir in sorted(p for p in k3m.iterdir() if p.is_dir()):
        for f in sorted(pdir.glob("*.tif")):
            if f.stem.endswith("_mask"):
                continue
            mask = np.asarray(Image.open(pdir / f"{f.stem}_mask.tif")) > 0
            rows.append(("lgg_seg", "lgg_seg", f.relative_to(base).as_posix(), f.stem, "tumor" if mask.any() else "empty", pdir.name, int(mask.sum())))
    df = pd.DataFrame(rows, columns=["dataset", "source_dir", "relpath", "image_id", "label", "native_group", "mask_pixels"])
    df["orig_split"] = "none"
    df["mask_relpath"] = df.relpath.str.replace(".tif", "_mask.tif", regex=False)
    hashes = hash_dataset("lgg_seg", df.source_dir, df.relpath, root)
    return df, hashes, {"n_patients": int(df.native_group.nunique()), "note": "slices of one patient are never separated; splits are by TCGA patient"}


def build_rsna(root: Path) -> tuple[pd.DataFrame, np.ndarray, dict]:
    base = root / "rsna_pneumonia"
    lab = _find_in(base, "stage_2_train_labels.csv")
    cls = pd.read_csv(_find_in(base, "stage_2_detailed_class_info.csv")).drop_duplicates("patientId").set_index("patientId")["class"]
    imgdir = _find_in(base, "stage_2_train_images")
    pids = sorted(pd.read_csv(lab).patientId.unique())
    df = pd.DataFrame({"dataset": "rsna", "source_dir": "rsna_pneumonia", "relpath": [f"{imgdir.relative_to(base).as_posix()}/{p}.dcm" for p in pids], "image_id": pids,
                       "label": [cls[p].replace(" / ", "_").replace(" ", "_") for p in pids], "native_group": pids, "orig_split": "none"})
    hashes = hash_dataset("rsna", df.source_dir, df.relpath, root)
    return df, hashes, {"note": "TorchXRayVision 'all' weights were trained on RSNA; use chex / mimic_ch weights for external numbers"}


def build_bdneuro(root: Path) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """BDNeuro-MRI is an external test set: nothing is trained on it. Images that duplicate a brain_mri image are excluded."""
    base = root / "bdneuro"
    top = _find_in(base, "Brain_Tumor_MRI_Dataset_Final")
    rows = []
    for part in ("train", "val", "test"):
        if not (top / part).is_dir():
            continue
        for cdir in sorted(p for p in (top / part).iterdir() if p.is_dir()):
            for f in sorted(cdir.iterdir()):
                if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    rows.append(("bdneuro", "bdneuro", f.relative_to(base).as_posix(), f.stem, cdir.name.replace("_", ""), part))
    df = pd.DataFrame(rows, columns=["dataset", "source_dir", "relpath", "image_id", "label", "orig_split"])
    df["native_group"] = df.image_id
    hashes = hash_dataset("bdneuro", df.source_dir, df.relpath, root)
    ours = pd.read_csv(SPLITS_DIR / "brain_mri.csv", keep_default_na=False)
    oh = np.array([int(x, 16) for x in ours.phash], dtype=np.uint64)
    hit = {i for i, _, _ in ph.near_pairs(hashes, ph.DUP_MAX_DIST, other=oh)}
    df["exclude_reason"] = ["duplicate_of_brain_mri" if i in hit else "" for i in range(len(df))]
    extra = {"overlap_with_brain_mri_images": len(hit), "note": "external test only; images that duplicate a brain_mri image are excluded; labels follow the folder names (no_tumor -> notumor)"}
    return df, hashes, extra


def build_milk10k(root: Path) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """MILK10k dermoscopic images as an external skin test. Classes we do not model, and any image that duplicates a HAM10000 or ISIC 2018 image, are excluded."""
    base = root / "milk10k"
    gt = pd.read_csv(base / "training_gt.csv")
    inp = pd.read_csv(base / "training_input.csv")
    derm = inp[inp.image_type == "dermoscopic"].merge(gt, on="lesion_id")
    cols = [c for c in gt.columns if c != "lesion_id"]
    top = derm[cols].idxmax(axis=1)
    ours = {"AKIEC": "akiec", "BCC": "bcc", "BKL": "bkl", "DF": "df", "MEL": "mel", "NV": "nv", "VASC": "vasc"}
    site = {"trunk": "Trunk", "head_neck_face": "Head and neck", "lower_extremity": "Lower extremity", "upper_extremity": "Upper extremity", "hand": "Upper extremity", "foot": "Lower extremity", "genital": "Anogenital region"}
    tone = derm.skin_tone_class.map(lambda t: "0-1 (darkest)" if t <= 1 else "5 (lightest)" if t >= 5 else str(int(t)))
    df = pd.DataFrame({"dataset": "milk10k", "source_dir": "milk10k", "relpath": "images/" + derm.isic_id + ".jpg", "image_id": derm.isic_id.to_numpy(), "label": top.map(lambda c: ours.get(c, c.lower())).to_numpy(),
                       "native_group": derm.lesion_id.to_numpy(), "orig_split": "none", "age": derm.age_approx.to_numpy(), "sex": derm.sex.to_numpy(), "site": derm.site.to_numpy(), "site_general": derm.site.map(site).to_numpy(),
                       "skin_tone": tone.to_numpy(), "skin_tone_class": derm.skin_tone_class.to_numpy()})
    df["exclude_reason"] = ["" if c in ours else f"unmapped_class:{c.lower()}" for c in top]
    hashes = hash_dataset("milk10k", df.source_dir, df.relpath, root)
    ham = pd.read_csv(SPLITS_DIR / "ham10000.csv", keep_default_na=False)
    oh = np.array([int(x, 16) for x in ham.phash], dtype=np.uint64)
    hit = {i for i, _, _ in ph.near_pairs(hashes, ph.DUP_MAX_DIST, other=oh)}
    df.loc[[i in hit and df.exclude_reason.iloc[i] == "" for i in range(len(df))], "exclude_reason"] = "duplicate_of_ham10000"
    extra = {"overlap_with_ham10000_images": len(hit), "classes_not_modelled": {c: int((top == c).sum()) for c in cols if c not in ours},
             "note": "external test only; skin-tone grade 0 (very dark) to 5 (very light), grades 0 and 1 are merged because only 6 lesions have grade 0"}
    return df, hashes, extra


# name -> (builder, fractions, strat column, test splits, extra finish kwargs)
SPECS = {
    "fracatlas": (build_fracatlas, {"train": 0.65, "val": 0.10, "cal": 0.10, "test": 0.15}, "strat", ["test"], {}),
    "ham10000": (build_ham10000, {"train": 0.70, "val": 0.10, "cal": 0.10, "test": 0.10}, "label", ["official_test"], {}),
    "brain_mri": (build_brain_mri, {"train": 0.70, "val": 0.10, "cal": 0.10, "test": 0.10}, "label", ["test"], {"loose_dist": 6, "use_loose_as_group": True}),
    "lgg_seg": (build_lgg_seg, {"train": 0.80, "val": 0.10, "test": 0.10}, "label", ["test"], {}),
    "bdneuro": (build_bdneuro, {"external_test": 1.0}, "label", ["external_test"], {}),
    "milk10k": (build_milk10k, {"external_test": 1.0}, "label", ["external_test"], {}),
    "rsna": (build_rsna, {"cal": 0.30, "test": 0.70}, "label", ["test"], {}),
}
SOURCE_DIRS = {"fracatlas": "fracatlas", "ham10000": "isic2018_t3_train", "brain_mri": "brain_mri", "lgg_seg": "lgg_seg", "rsna": "rsna_pneumonia", "bdneuro": "bdneuro", "milk10k": "milk10k"}


def run(name: str, root: Path) -> None:
    builder, fractions, strat, test_splits, kw = SPECS[name]
    df, hashes, extra = builder(root)
    imm = df["immutable"] if "immutable" in df else None
    if name == "lgg_seg":
        df, rep = _finish_lgg(df, hashes, extra)
    else:
        df, rep = _finish(name, df, hashes, fractions, strat, extra, immutable=imm, **kw)
    _write(name, df, rep, test_splits)


def _finish_lgg(df: pd.DataFrame, hashes: np.ndarray, extra: dict) -> tuple[pd.DataFrame, dict]:
    """Patient-level 10-fold CV (stratified by tumour burden quartile). `split` = fold 0: test fold 0, val fold 1, train the rest = 88/11/11 patients."""
    df, rep = au.audit(df.assign(orig_split="none"), hashes)
    df["loose_cluster"] = -1
    df["phash"] = [f"{int(h):016x}" for h in hashes]
    # Slices are never removed. Two patients are merged into one group only when a pair of *tumour-bearing* slices from
    # different patients is a near-duplicate; dark, near-empty top and bottom slices look alike across patients and say nothing.
    pid = pd.factorize(df["native_group"])[0]
    tumour = (df["mask_pixels"] > 0).to_numpy()
    pairs = ph.near_pairs(hashes, ph.DUP_MAX_DIST)
    cross = [(i, j) for i, j, _ in pairs if pid[i] != pid[j]]
    both = [(i, j) for i, j in cross if tumour[i] and tumour[j]]
    lab = ph.components(int(pid.max()) + 1, [(int(pid[i]), int(pid[j])) for i, j in both])
    df["group"] = ["p" + str(lab[p]) for p in pid]
    rep["cross_patient_duplicate_pairs"] = len(cross)
    rep["cross_patient_pairs_both_tumour"] = len(both)
    rep["patients_merged_into_groups"] = int(len(set(pid)) - len(set(lab)))
    for k in ("removed_by_reason", "label_conflict_clusters", "label_noise_max_hamming"):
        rep.pop(k, None)
    pat = df.groupby("group").agg(burden=("mask_pixels", "sum"), n=("image_id", "size")).sort_index()
    pat["quartile"] = pd.qcut(pat.burden.rank(method="first"), 4, labels=False)
    rng = np.random.RandomState(SEED)
    fold = pd.Series(-1, index=pat.index)
    for q in range(4):
        ids = list(pat.index[pat.quartile == q])
        rng.shuffle(ids)
        for k, g in enumerate(ids):
            fold[g] = (k + q * 3) % 10  # offset per quartile keeps the fold sizes level
    df["fold"] = df.group.map(fold)
    df["split"] = np.select([df.fold == 0, df.fold == 1], ["test", "val"], "train")
    sp.assert_no_straddle(df, "group")
    rep.update(extra)
    rep["images_removed"] = 0  # slices are not removed from a segmentation set; duplicates only merge patient groups
    rep["split_counts"] = {k: int(v) for k, v in df.split.value_counts().sort_index().items()}
    rep["patients_per_split"] = {k: int(v) for k, v in df.groupby("split").group.nunique().items()}
    rep["split_hash"] = sp.split_hash(df)
    rep["seed"] = SEED
    return df, rep


def write_eval_index() -> None:
    idx: dict = {"generated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "batch_cap": BATCH_CAP, "datasets": {}}
    for name, (_, _, _, test_splits, _) in SPECS.items():
        p = SPLITS_DIR / f"{name}.csv"
        if not p.is_file():
            continue
        df = pd.read_csv(p, keep_default_na=False, na_values=[""])
        t = df[df.split.isin(test_splits)]
        b = df[df.eval_batch]
        idx["datasets"][name] = {
            "split_file": f"ml/data/splits/{name}.csv", "test_splits": test_splits, "n_test": int(len(t)), "n_batch": int(len(b)),
            "batch_filter": "eval_batch == True", "test_label_counts": {str(k): int(v) for k, v in t.label.value_counts().items()},
            "batch_label_counts": {str(k): int(v) for k, v in b.label.value_counts().items()},
            "train_split_for_retrieval": "train", "source_dirs": sorted(df.source_dir.unique().tolist()),
        }
    (REPO_ROOT / "ml" / "data" / "eval_index.json").write_text(json.dumps(idx, indent=1) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="+", help=f"{', '.join(SPECS)} or all")
    ap.add_argument("--root", type=Path, default=None)
    a = ap.parse_args(argv)
    root = a.root or data_root()
    names = list(SPECS) if a.names == ["all"] else a.names
    rc = 0
    for n in names:
        if n not in SPECS:
            print(f"unknown dataset {n}", file=sys.stderr)
            rc = 1
            continue
        if not (root / SOURCE_DIRS[n] / "manifest.json").is_file():
            print(f"[{n}] skipped: not downloaded (python ml/data/download.py {SOURCE_DIRS[n]})", file=sys.stderr)
            rc = 1 if a.names != ["all"] else rc
            continue
        run(n, root)
    return rc


if __name__ == "__main__":
    sys.exit(main())
