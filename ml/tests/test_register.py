from __future__ import annotations

import json

import pytest

from ml.artifacts import register as rg


def _fake_run(d, sha_ok=True):
    d.mkdir(parents=True)
    (d / "weights.pt").write_bytes(b"abc")
    meta = {"model_id": "skin_cls@v1", "task": "classification", "arch": "convnext_tiny", "classes": ["a", "b"], "preproc_spec": "skin_cls", "weights_file": "weights.pt",
            "weights_sha256": rg.sha256_file(d / "weights.pt") if sha_ok else "0" * 64, "training_data": ["ham10000"], "split_name": "ham10000", "split_hash": "h", "git_commit": "c0ffee", "seed": 1}
    (d / "model_meta.json").write_text(json.dumps(meta))
    (d / "metrics.json").write_text(json.dumps({"official_test": {"balanced_accuracy": {"point": 0.8, "lo": 0.75, "hi": 0.85}}}))


def test_register_writes_entry_with_hash_and_headline(tmp_path):
    _fake_run(tmp_path / "skin_cls")
    reg = tmp_path / "registry.json"
    e = rg.register(tmp_path / "skin_cls", registry=reg)
    data = json.loads(reg.read_text())
    assert data["models"]["skin_cls@v1"]["weights_sha256"] == e["weights_sha256"]
    assert data["models"]["skin_cls@v1"]["headline"]["official_test.balanced_accuracy"]["point"] == 0.8
    assert data["models"]["skin_cls@v1"]["classes"] == ["a", "b"] and data["models"]["skin_cls@v1"]["temperature"] is None
    assert "CC BY-NC" in e["license"]


def test_register_is_idempotent_and_keeps_other_models(tmp_path):
    reg = tmp_path / "registry.json"
    reg.write_text(json.dumps({"schema": 1, "models": {"other": {"x": 1}}}))
    _fake_run(tmp_path / "skin_cls")
    rg.register(tmp_path / "skin_cls", registry=reg)
    rg.register(tmp_path / "skin_cls", registry=reg)
    assert set(json.loads(reg.read_text())["models"]) == {"other", "skin_cls@v1"}


def test_corrupt_download_is_rejected(tmp_path):
    _fake_run(tmp_path / "skin_cls", sha_ok=False)
    with pytest.raises(SystemExit):
        rg.register(tmp_path / "skin_cls", registry=tmp_path / "r.json")
