"""The notebook sources build into valid notebooks and the classifier / segmenter ones run end to end on synthetic data."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from ml.data import make_splits as ms
from ml.tests.test_make_splits import _manifest, _save, env  # noqa: F401  (fixture)
from ml.tests.test_phash import _scene
from ml.train import build_notebooks as bn

REPO = Path(__file__).resolve().parents[2]


def test_every_source_builds_into_a_valid_notebook_that_starts_with_the_header_cell(tmp_path):
    paths = bn.build("tester", nb_dir=tmp_path / "nb", kg_dir=tmp_path / "kg")
    assert {p.stem for p in paths} == set(bn.JOBS)
    for p in paths:
        nb = json.loads(p.read_text())
        assert nb["nbformat"] == 4 and nb["cells"][0]["cell_type"] == "markdown"
        code = [c for c in nb["cells"] if c["cell_type"] == "code"]
        first = "".join(code[0]["source"])
        assert "print_header" in first and '"clone"' in first and "rev-parse" in Path(REPO / "ml/train/common.py").read_text()
        for c in code:
            compile("".join(c["source"]), p.name, "exec")
        meta = json.loads((tmp_path / "kg" / p.stem / "kernel-metadata.json").read_text())
        assert meta["id"] == f"tester/{bn.JOBS[p.stem][0]}" and meta["enable_gpu"] == "true" and meta["enable_internet"] == "true"


def test_attached_datasets_are_declared_in_kernel_metadata(tmp_path):
    bn.build("u", nb_dir=tmp_path / "nb", kg_dir=tmp_path / "kg")
    brain = json.loads((tmp_path / "kg" / "brain_cls" / "kernel-metadata.json").read_text())
    seg = json.loads((tmp_path / "kg" / "brain_seg" / "kernel-metadata.json").read_text())
    assert brain["dataset_sources"] == ["masoudnickparvar/brain-tumor-mri-dataset"]
    assert seg["dataset_sources"] == ["mateuszbuda/lgg-mri-segmentation"]


def _run(script: str, root: Path, tmp: Path) -> Path:
    out = tmp / "out"
    envv = {**os.environ, "SMOKE": "1", "PARALLAX_DATA_ROOT": str(root), "PARALLAX_SPLITS_DIR": str(tmp / "splits"), "PARALLAX_OUT": str(out), "TMP": str(tmp), "TEMP": str(tmp), "PYTHONIOENCODING": "utf-8"}
    r = subprocess.run([sys.executable, str(REPO / "ml" / "train" / "src" / script)], cwd=REPO, env=envv, capture_output=True, text=True, timeout=1500)
    assert r.returncode == 0, r.stdout[-3000:] + "\n" + r.stderr[-3000:]
    return out


def test_brain_classifier_notebook_runs_on_a_synthetic_dataset(env):  # noqa: F811
    root, tmp = env
    _manifest(root, "brain_mri")
    classes = ["glioma", "meningioma", "notumor", "pituitary"]
    k = 0
    for part, n in (("Training", 24), ("Testing", 10)):
        for ci, cls in enumerate(classes):
            for i in range(n):
                _save(_scene(10_000 * ci + 100 * (part == "Testing") + i + 1), root / "brain_mri" / part / cls / f"{k}.jpg")
                k += 1
    ms.run("brain_mri", root)
    out = _run("brain_cls.py", root, tmp)
    m = json.loads((out / "metrics.json").read_text())
    for key in ("A_original_testing", "A_testing_without_duplicates", "leakage_free_test", "inflation", "headline"):
        assert key in m, key
    assert m["leakage_free_test"]["accuracy"]["lo"] <= m["leakage_free_test"]["accuracy"]["hi"]
    meta = json.loads((out / "model_meta.json").read_text())
    assert meta["classes"] == classes and meta["preproc_spec"] == "brain_effnet" and meta["weights_sha256"]
    assert (out / "predictions" / "test.npz").exists() and (out / "weights.pt").exists()


def test_brain_segmenter_notebook_runs_on_a_synthetic_dataset(env):  # noqa: F811
    root, tmp = env
    _manifest(root, "lgg_seg")
    rng = np.random.default_rng(0)
    for p in range(30):
        for s in range(8):
            img = _scene(p * 20 + s + 1)
            mask = np.zeros((256, 256), np.uint8)
            if s % 2 == 0:
                y, x = rng.integers(40, 180, 2)
                mask[y : y + 30, x : x + 30] = 255
            rgb = np.stack([img, img, img], -1)
            rgb[mask > 0] = 255
            d = root / "lgg_seg" / "kaggle_3m" / f"TCGA_XX_{p:04d}"
            _save(rgb, d / f"TCGA_XX_{p:04d}_{s}.tif", "TIFF")
            _save(mask, d / f"TCGA_XX_{p:04d}_{s}_mask.tif", "TIFF")
    ms.run("lgg_seg", root)
    out = _run("brain_seg.py", root, tmp)
    m = json.loads((out / "metrics.json").read_text())
    assert m["headline"]["n_patients"] == 3 and m["headline"]["mean_patient_dice"]["method"] == "bca"
    assert 0 <= m["fold_0"]["slice_detection_auroc"] <= 1
    meta = json.loads((out / "model_meta.json").read_text())
    assert meta["task"] == "segmentation" and meta["preproc_spec"]["name"] == "brain_seg" and meta["input_size"] == 256
    z = np.load(out / "predictions" / "test.npz")
    assert z["probs_u8"].shape[1:] == (256, 256) and len(z["patients"]) == len(z["probs_u8"])
