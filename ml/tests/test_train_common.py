from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from ml.eval import metrics as mt
from ml.train import common as tc


def _toy(n=96, c=3, size=24, seed=0):
    rng = np.random.RandomState(seed)
    y = rng.randint(0, c, n)
    x = rng.normal(0, 0.3, (n, 3, size, size)).astype(np.float32)
    for k in range(c):  # class k lights up a different horizontal band, so flips cannot destroy the signal
        x[y == k, :, k * 6 : k * 6 + 6, :] += 1.5
    return torch.from_numpy(x).half(), y


def test_augment_keeps_shape_and_mask_alignment():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(4, 3, 32, 32)
    mask = (x[:, :1] > 0).float()
    xa, ma = tc.augment(x, g, rot_deg=15, flip_h=True, flip_v=True, mask=mask)
    assert xa.shape == x.shape and ma.shape == mask.shape
    assert set(ma.unique().tolist()) <= {0.0, 1.0}
    # identity parameters leave the image essentially unchanged
    g2 = torch.Generator().manual_seed(0)
    same = tc.augment(x, g2, rot_deg=0, scale=(1, 1), shift=0, flip_h=False, brightness=0, contrast=0)
    assert torch.allclose(same, x, atol=1e-4)


def test_augment_flip_actually_mirrors_some_samples():
    g = torch.Generator().manual_seed(1)
    x = torch.zeros(64, 1, 8, 8)
    x[:, :, :, :2] = 1.0  # bright left edge
    out = tc.augment(x, g, rot_deg=0, scale=(1, 1), shift=0, flip_h=True, brightness=0, contrast=0)
    left_bright = (out[:, 0, :, 0] > 0.5).any(1)
    assert 10 < int(left_bright.sum()) < 54


def test_epoch_order_balances_classes_by_inverse_sqrt_frequency():
    y = np.array([0] * 900 + [1] * 100)
    rng = np.random.RandomState(0)
    order = tc.epoch_order(1000, y, rng, 0.5)
    frac_minor = np.mean(y[order] == 1)
    assert 0.2 < frac_minor < 0.35  # sqrt balancing: 0.1 -> ~0.25, not 0.5
    assert len(tc.epoch_order(10, y[:10], rng, None)) == 10


def test_train_classifier_learns_toy_task_and_returns_best_state():
    Xtr, ytr = _toy(240, seed=1)
    Xva, yva = _toy(60, seed=2)
    res = tc.train_classifier(Xtr, ytr, Xva, yva, arch="resnet18", n_classes=3, epochs=5, lr=3e-3, batch_size=32, pretrained=False, aug=dict(rot_deg=5, flip_h=False, brightness=0.05, contrast=0.05), log=lambda s: None)
    assert res["history"][-1]["val_bal_acc"] > 0.85
    assert 1 <= res["best_epoch"] <= 5
    lg = tc.predict_logits(res["model"], Xva)
    assert mt.balanced_accuracy(yva, lg.argmax(1), 3) > 0.85


def test_bootstrap_report_block_has_cis_for_every_metric():
    rng = np.random.RandomState(0)
    y = rng.randint(0, 3, 300)
    logits = rng.normal(0, 1, (300, 3)) + 2 * np.eye(3)[y]
    p = mt.softmax(logits)
    rep = tc.bootstrap_report(y, p, np.arange(300), 3, ["a", "b", "c"], B=100, positive="b")
    for k in ("accuracy", "balanced_accuracy", "macro_f1", "auroc_macro_ovr", "nll", "ece_15", "auroc_b_vs_rest"):
        assert rep[k]["lo"] <= rep[k]["point"] <= rep[k]["hi"], k
    assert set(rep["per_class_recall"]) == {"a", "b", "c"} and np.array(rep["confusion"]).sum() == 300


def test_write_run_records_hashes_and_predictions(tmp_path):
    sd = {"w": torch.ones(2)}
    out = tc.write_run(tmp_path, model_id="toy", task="cls", arch="resnet18", classes=["a", "b"], preproc_spec="skin_cls", state_dict=sd, weights_name="weights.pt",
                       split_name="toy", split_hash="abc", training_data=["toy"], metrics={"x": np.float32(1.5)}, predictions={"test": {"ids": np.array(["i1"]), "logits": np.zeros((1, 2))}})
    meta = json.loads((out / "model_meta.json").read_text())
    assert meta["weights_sha256"] == tc.sha256_file(out / "weights.pt") and meta["split_hash"] == "abc"
    assert json.loads((out / "metrics.json").read_text()) == {"x": 1.5}
    assert np.load(out / "predictions" / "test.npz")["ids"][0] == "i1"


def test_link_attached_finds_dataset_by_marker(tmp_path):
    inp = tmp_path / "input" / "brain-tumor-mri-dataset"
    (inp / "Training" / "glioma").mkdir(parents=True)
    (inp / "Training" / "glioma" / "a.jpg").write_bytes(b"x")
    dst = tc.link_attached(tmp_path / "raw", "brain_mri", "Training", search=tmp_path / "input")
    assert (dst / "Training" / "glioma" / "a.jpg").exists()
    assert tc.link_attached(tmp_path / "raw", "brain_mri", "Training", search=tmp_path / "input") == dst
    with pytest.raises(FileNotFoundError):
        tc.link_attached(tmp_path / "raw", "other", "Nope", search=tmp_path / "input")
