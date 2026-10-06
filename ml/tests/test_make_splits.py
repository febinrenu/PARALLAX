"""Builders run end to end on tiny synthetic datasets laid out like the real ones."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from ml.data import common, make_splits as ms
from ml.data import splitting as sp
from ml.tests.test_phash import _jpeg, _scene

P1_CONFTEST = Path(__file__).resolve().parents[2] / "backend" / "tests" / "conftest.py"


@pytest.fixture()
def env(tmp_path, monkeypatch):
    root = tmp_path / "raw"
    (tmp_path / "ml" / "data").mkdir(parents=True)
    for mod in (ms, common):
        monkeypatch.setattr(mod, "SPLITS_DIR", tmp_path / "splits", raising=False)
        monkeypatch.setattr(mod, "CACHE_DIR", tmp_path / "cache", raising=False)
        monkeypatch.setattr(mod, "REPORTS_DIR", tmp_path / "reports", raising=False)
    monkeypatch.setattr(ms, "REPO_ROOT", tmp_path)
    return root, tmp_path


def _save(a: np.ndarray, path: Path, fmt="JPEG"):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(a).save(path, fmt)


def _manifest(root: Path, name: str):
    (root / name).mkdir(parents=True, exist_ok=True)
    (root / name / "manifest.json").write_text("{}")


def test_brain_mri_cross_split_duplicate_is_found_and_groups_do_not_straddle(env):
    root, tmp = env
    _manifest(root, "brain_mri")
    k = 0
    for part in ("Training", "Testing"):
        for cls in ("glioma", "notumor"):
            for i in range(30 if part == "Training" else 12):
                _save(_scene(1000 * (cls == "glioma") + 100 * (part == "Testing") + i), root / "brain_mri" / part / cls / f"{k}.jpg")
                k += 1
    # plant a leak: a Testing image that is a recompressed copy of a Training image
    src = root / "brain_mri" / "Training" / "glioma" / "0.jpg"
    _save(_jpeg(np.asarray(Image.open(src)), 55), root / "brain_mri" / "Testing" / "glioma" / "leak.jpg")
    ms.run("brain_mri", root)
    df = common.load_split("brain_mri", tmp / "splits")
    rep = json.loads((tmp / "reports" / "leakage.json").read_text())["datasets"]["brain_mri"]
    assert rep["pairs_across_original_splits"] >= 1
    assert rep["images_removed"] >= 1
    assert set(df.split) <= {"train", "val", "cal", "test", "dropped"}
    sp.assert_no_straddle(df[df.split != "dropped"], "group")
    pair = df[df.relpath.str.endswith(("Training/glioma/0.jpg", "Testing/glioma/leak.jpg"))]
    kept = pair[pair.split != "dropped"]
    assert len(kept) == 1, "exactly one of the two copies survives"
    ei = json.loads((tmp / "ml" / "data" / "eval_index.json").read_text())
    assert ei["datasets"]["brain_mri"]["n_test"] == (df.split == "test").sum()


def test_lgg_splits_by_patient_with_ten_folds(env):
    root, tmp = env
    _manifest(root, "lgg_seg")
    rng = np.random.default_rng(0)
    for p in range(30):
        for s in range(6):
            img = _scene(p * 10 + s)
            rgb = np.stack([img, img, img], -1)
            mask = np.zeros((256, 256), np.uint8)
            if s % 2 == 0:
                mask[100 : 100 + int(rng.integers(5, 40)), 100:140] = 255
            d = root / "lgg_seg" / "kaggle_3m" / f"TCGA_XX_{p:04d}"
            _save(rgb, d / f"TCGA_XX_{p:04d}_{s}.tif", "TIFF")
            _save(mask, d / f"TCGA_XX_{p:04d}_{s}_mask.tif", "TIFF")
    ms.run("lgg_seg", root)
    df = common.load_split("lgg_seg", tmp / "splits")
    assert len(df) == 180
    sp.assert_no_straddle(df, "native_group" if "native_group" in df else "group")
    per = df.groupby("split").group.nunique()
    assert per["test"] == 3 and per["val"] == 3 and per["train"] == 24
    assert sorted(df.fold.unique()) == list(range(10))


def test_ham10000_official_test_untouched_and_train_copies_of_it_removed(env):
    root, tmp = env
    for n in ("isic2018_t3_train", "isic2018_t3_test", "ham10000_meta"):
        _manifest(root, n)
    classes = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
    tr_dir = root / "isic2018_t3_train"
    te_dir = root / "isic2018_t3_test"
    rows, lesions, meta = [], [], []
    for i in range(80):
        name = f"ISIC_{i:07d}"
        _save(_scene(i), tr_dir / "ISIC2018_Task3_Training_Input" / f"{name}.jpg")
        c = classes[i % 3]
        rows.append([name] + [1.0 if c == k else 0.0 for k in classes])
        lesions.append((name, f"HAM_{i // 2:04d}"))  # two images per lesion
        meta.append((f"HAM_{i // 2:04d}", name, c.lower(), "histo", 50.0, "male", "back", "x"))
    pd.DataFrame(rows, columns=["image"] + classes).to_csv(tr_dir / "ISIC2018_Task3_Training_GroundTruth.csv", index=False)
    pd.DataFrame(lesions, columns=["image", "lesion_id"]).to_csv(tr_dir / "ISIC2018_Task3_Training_LesionGroupings.csv", index=False)
    pd.DataFrame(meta, columns=["lesion_id", "image_id", "dx", "dx_type", "age", "sex", "localization", "dataset"]).to_csv(root / "ham10000_meta" / "HAM10000_metadata.csv", index=False)
    trows = []
    for i in range(20):
        name = f"ISIC_{9000 + i:07d}"
        # test image 0 is a recompressed copy of training image 5
        img = _jpeg(np.asarray(Image.open(tr_dir / "ISIC2018_Task3_Training_Input" / "ISIC_0000005.jpg")), 60) if i == 0 else _scene(5000 + i)
        _save(img, te_dir / "ISIC2018_Task3_Test_Input" / f"{name}.jpg")
        trows.append([name] + [1.0 if classes[(5 if i == 0 else i) % 3] == k else 0.0 for k in classes])
    pd.DataFrame(trows, columns=["image"] + classes).to_csv(te_dir / "ISIC2018_Task3_Test_GroundTruth.csv", index=False)
    ms.run("ham10000", root)
    df = common.load_split("ham10000", tmp / "splits")
    test = df[df.split == "official_test"]
    assert len(test) == 20 and (test.dropped_reason.isna()).all(), "official test rows are never dropped"
    leaked = df[df.image_id == "ISIC_0000005"].iloc[0]
    assert leaked.split == "dropped" and str(leaked.dropped_reason).startswith("duplicate_of_immutable")
    trn = df[df.split.isin(["train", "val", "cal", "test"])]
    sp.assert_no_straddle(trn, "group")
    # both images of a lesion always share a split
    assert trn.groupby("group").split.nunique().max() == 1
    assert df[df.orig_split == "train"].site_general.dropna().eq("Trunk").all()


def test_rsna_splits_by_patient_and_reads_dicom(env):
    root, tmp = env
    spec = importlib.util.spec_from_file_location("p1_conftest", P1_CONFTEST)
    p1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p1)
    _manifest(root, "rsna_pneumonia")
    base = root / "rsna_pneumonia"
    (base / "stage_2_train_images").mkdir(parents=True)
    classes = ["Normal", "No Lung Opacity / Not Normal", "Lung Opacity"]
    lab, cls = [], []
    for i in range(24):
        pid = f"pid{i:03d}"
        (base / "stage_2_train_images" / f"{pid}.dcm").write_bytes(p1.make_dicom(_scene(i, 128).astype(np.float32) / 255.0, photometric="MONOCHROME1" if i % 5 == 0 else "MONOCHROME2"))
        lab.append((pid, np.nan, np.nan, np.nan, np.nan, int(i % 3 == 2)))
        cls.append((pid, classes[i % 3]))
    pd.DataFrame(lab, columns=["patientId", "x", "y", "width", "height", "Target"]).to_csv(base / "stage_2_train_labels.csv", index=False)
    pd.DataFrame(cls, columns=["patientId", "class"]).to_csv(base / "stage_2_detailed_class_info.csv", index=False)
    ms.run("rsna", root)
    df = common.load_split("rsna", tmp / "splits")
    assert set(df.split) <= {"cal", "test", "dropped"} and (df.split == "cal").sum() > 0 and (df.split == "test").sum() > 0
    assert set(df.label) == {"Normal", "No_Lung_Opacity_Not_Normal", "Lung_Opacity"}
    sp.assert_no_straddle(df[df.split != "dropped"], "group")


def test_flat_images_do_not_merge_unrelated_groups(env):
    root, tmp = env
    _manifest(root, "brain_mri")
    for part in ("Training", "Testing"):
        for i in range(10):
            _save(np.zeros((64, 64), np.uint8), root / "brain_mri" / part / "notumor" / f"black{i}.jpg")
        for i in range(10):
            _save(_scene(i + (50 if part == "Testing" else 0)), root / "brain_mri" / part / "glioma" / f"{i}.jpg")
    ms.run("brain_mri", root)
    rep = json.loads((tmp / "reports" / "leakage.json").read_text())["datasets"]["brain_mri"]
    assert rep["flat_images_excluded_from_matching"] == 20
    assert rep["images_removed"] == 0, "blank images are not duplicates of one another"
