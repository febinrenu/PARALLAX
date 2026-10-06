"""Logic tests for the probe trainer and probe inference, on synthetic embeddings (not model output)."""

import numpy as np
import pytest

pytest.importorskip("sklearn")

from medproof.intake.router_config import BODY_PARTS, CLASSES  # noqa: E402
from medproof.intake.router_train import (  # noqa: E402
    Probe,
    fit_temperature,
    group_split,
    train_probe,
    wilson_interval,
)
from tests.intake.router_helpers import DIM, clustered  # noqa: E402


@pytest.fixture(scope="module")
def data():
    return clustered()


def test_split_has_no_group_overlap_and_covers_every_class(data):
    X, y, g = data
    tr, va, te = group_split(y, g, seed=1)
    for a, b in ((tr, va), (tr, te), (va, te)):
        assert not set(g[a]) & set(g[b]), "a group appears in two splits"
    assert len(tr) + len(va) + len(te) == len(y)
    for idx in (tr, va, te):
        assert set(y[idx]) == set(CLASSES)


def test_split_is_deterministic_per_seed(data):
    X, y, g = data
    a, b = group_split(y, g, seed=3), group_split(y, g, seed=3)
    assert all(np.array_equal(p, q) for p, q in zip(a, b))
    assert not all(np.array_equal(p, q) for p, q in zip(a, group_split(y, g, seed=4)))


def test_trained_probe_separates_held_out_groups(data):
    X, y, g = data
    probe, report = train_probe(X, y, g, embedder_id="test-double-embedder", seed=0)
    assert report["test"]["accuracy"] >= 0.95
    assert report["classes"] == list(CLASSES)
    p = probe.predict_proba(X[:10])
    assert p.shape == (10, len(CLASSES)) and np.allclose(p.sum(axis=1), 1.0, atol=1e-5)
    assert [CLASSES[i] for i in p.argmax(1)] == list(y[:10])


def test_report_has_per_class_recall_confusion_matrix_and_intervals(data):
    X, y, g = data
    _, report = train_probe(X, y, g, embedder_id="e", seed=0)
    t = report["test"]
    assert set(t["per_class_recall"]) == set(CLASSES)
    cm = np.asarray(t["confusion_matrix"])
    assert cm.shape == (5, 5) and cm.sum() == t["n"]
    lo, hi = t["accuracy_ci95"]
    assert lo <= t["accuracy"] <= hi


def test_temperature_does_not_worsen_validation_likelihood(data):
    X, y, g = data
    probe, report = train_probe(X, y, g, embedder_id="e", seed=0)
    assert report["temperature"] > 0
    assert report["val_nll_after"] <= report["val_nll_before"] + 1e-9


def test_fit_temperature_recovers_overconfidence():
    rng = np.random.default_rng(0)
    n, k = 400, 3
    y = rng.integers(0, k, n)
    logits = rng.normal(0, 1, (n, k))
    logits[np.arange(n), y] += 1.0  # modest signal
    T = fit_temperature(logits * 5.0, y)  # logits inflated 5x: should be cooled
    assert T > 2.0


def test_save_load_round_trip_is_exact(data, tmp_path):
    X, y, g = data
    probe, _ = train_probe(X, y, g, embedder_id="emb-v1", seed=0)
    path = tmp_path / "probe.npz"
    probe.save(path)
    again = Probe.load(path)
    np.testing.assert_array_equal(probe.predict_proba(X), again.predict_proba(X))
    assert again.embedder_id == "emb-v1" and again.classes == list(CLASSES) and again.dim == DIM


def test_probe_file_is_not_a_pickle(data, tmp_path):
    X, y, g = data
    probe, _ = train_probe(X, y, g, embedder_id="e", seed=0)
    probe.save(tmp_path / "p.npz")
    with np.load(tmp_path / "p.npz", allow_pickle=False) as f:  # raises if any object array is inside
        assert "coef" in f.files


def test_wrong_dimension_is_rejected(data):
    X, y, g = data
    probe, _ = train_probe(X, y, g, embedder_id="e", seed=0)
    with pytest.raises(ValueError):
        probe.predict_proba(np.zeros((2, DIM + 1), np.float32))


def test_body_part_head_trains_on_bone_only_and_predicts_parts(tmp_path):
    rng = np.random.default_rng(1)
    X, y = [], []
    for k, part in enumerate(BODY_PARTS):
        c = np.zeros(DIM, np.float32)
        c[8 + k] = 1.0
        for _ in range(30):
            v = c + rng.normal(0, 0.1, DIM)
            X.append(v / np.linalg.norm(v))
            y.append(part)
    Xc, yc, gc = clustered()
    probe, _ = train_probe(Xc, yc, gc, embedder_id="e", seed=0, body_part=(np.stack(X), np.asarray(y)))
    parts = probe.predict_body_part(np.stack(X[:4]))
    assert parts.shape == (4, len(BODY_PARTS))
    assert [BODY_PARTS[i] for i in parts.argmax(1)] == list(y[:4])
    probe.save(tmp_path / "bp.npz")
    again = Probe.load(tmp_path / "bp.npz")
    np.testing.assert_array_equal(again.predict_body_part(np.stack(X[:4])), parts)


def test_probe_without_body_part_head_returns_none(data):
    X, y, g = data
    probe, _ = train_probe(X, y, g, embedder_id="e", seed=0)
    assert probe.predict_body_part(X[:2]) is None


def test_unknown_labels_are_rejected(data):
    X, y, g = data
    y = y.copy()
    y[0] = "mri_of_a_cat"
    with pytest.raises(ValueError):
        train_probe(X, y, g, embedder_id="e", seed=0)


def test_wilson_interval_known_values():
    lo, hi = wilson_interval(95, 100)
    assert lo == pytest.approx(0.8882, abs=1e-3) and hi == pytest.approx(0.9784, abs=1e-3)
    assert wilson_interval(0, 0) == (0.0, 1.0)
    lo, hi = wilson_interval(100, 100)
    assert hi == pytest.approx(1.0) and lo > 0.95


def test_collect_folder_layout_groups_and_body_parts(tmp_path):
    from medproof.intake.router_train import collect_folder

    def touch(rel):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")

    touch("cxr/a.png")
    touch("cxr/b.png")
    touch("brain_mri/patient1/s1.png")
    touch("brain_mri/patient1/s2.png")
    touch("bone_xray/hand/h1.jpg")
    touch("bone_xray/hip/p1.jpg")
    touch("bone_xray/misc.jpg")
    touch("skin_dermoscopy/notes.txt")  # not an image
    touch("not_a_class/z.png")  # unknown class folder is ignored
    rows = {p.name: (cls, group, part) for p, cls, group, part in collect_folder(tmp_path)}
    assert set(rows) == {"a.png", "b.png", "s1.png", "s2.png", "h1.jpg", "p1.jpg", "misc.jpg"}
    assert rows["a.png"][1] != rows["b.png"][1]  # flat files are their own groups
    assert rows["h1.jpg"][1] != rows["p1.jpg"][1]  # a body-part folder is a label, not a group
    assert rows["s1.png"][1] == rows["s2.png"][1]  # same patient folder is one group
    assert rows["h1.jpg"][2] == "hand" and rows["p1.jpg"][2] == "hip" and rows["misc.jpg"][2] is None
    assert rows["h1.jpg"][0] == "bone_xray"
