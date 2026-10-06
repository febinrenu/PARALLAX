from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from ml.train import seg


def _square(n=64, y=10, x=10, s=20):
    m = np.zeros((n, n), bool)
    m[y : y + s, x : x + s] = True
    return m


def test_hd95_of_shifted_square_equals_the_shift_and_empty_is_nan():
    a, b = _square(), _square(x=15)
    assert seg.hd95_2d(a, a) == 0.0
    assert seg.hd95_2d(a, b) == pytest.approx(5.0, abs=1.0)
    assert np.isnan(seg.hd95_2d(a, np.zeros_like(a)))


def test_patient_dice_pools_slices_and_ignores_empty_slice_inflation():
    n = 6
    gt = np.zeros((n, 64, 64), np.uint8)
    gt[0] = _square()
    gt[1] = _square(y=30)
    probs = np.zeros((n, 64, 64), np.float32)
    probs[0] = _square()  # perfect on slice 0
    # slice 1 missed, slices 2..5 are empty on empty
    pats = np.array(["A", "A", "A", "B", "B", "B"])
    gt[3] = _square()
    probs[3] = _square()
    ev = seg.evaluate_patients(probs, gt, pats)
    by = {r["patient"]: r for r in ev["patients"]}
    assert by["A"]["dice"] == pytest.approx(2 * 400 / (400 + 800))  # one of two tumour slices found
    assert by["B"]["dice"] == 1.0 and by["B"]["tumour_slices"] == 1
    assert ev["slice_auroc"] > 0.7


def test_channel_replicate_copies_one_channel_for_selected_samples():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(40, 3, 8, 8)
    out = seg.channel_replicate(x, g, p=1.0)
    assert torch.equal(out[:, 0], out[:, 1]) and torch.equal(out[:, 1], out[:, 2])
    none = seg.channel_replicate(x, g, p=0.0)
    assert torch.equal(none, x)


def test_load_slices_reads_tif_pairs(tmp_path):
    rng = np.random.default_rng(0)
    ips, mps = [], []
    for i in range(3):
        a = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
        m = np.zeros((256, 256), np.uint8)
        m[100:140, 100:140] = 255 * (i % 2)
        Image.fromarray(a).save(tmp_path / f"s{i}.tif")
        Image.fromarray(m).save(tmp_path / f"s{i}_mask.tif")
        ips.append(tmp_path / f"s{i}.tif")
        mps.append(tmp_path / f"s{i}_mask.tif")
    X, M = seg.load_slices(ips, mps)
    assert X.shape == (3, 3, 256, 256) and X.dtype == torch.float16 and M.shape == (3, 1, 256, 256)
    assert M[1].sum() == 0 or M[0].sum() == 0 and set(M.unique().tolist()) <= {0, 1}


def test_unet_learns_a_synthetic_blob_task():
    rng = np.random.RandomState(0)
    n, s = 48, 64

    def make(n):
        X = rng.normal(0, 0.2, (n, 3, s, s)).astype(np.float32)
        M = np.zeros((n, 1, s, s), np.uint8)
        for i in range(n):
            if rng.rand() < 0.7:
                y, x = rng.randint(8, s - 24, 2)
                M[i, 0, y : y + 16, x : x + 16] = 1
                X[i] += 2.0 * M[i]
        return torch.from_numpy(X).half(), torch.from_numpy(M)

    Xtr, Mtr = make(48)
    Xva, Mva = make(16)
    res = seg.train_unet(Xtr, Mtr, Xva, Mva, encoder="resnet18", epochs=6, lr=3e-3, batch_size=8, pretrained=False, chan_aug_p=0.0, log=lambda s: None)
    assert res["history"][-1]["val_dice_pooled"] > 0.6
