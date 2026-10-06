"""OOD detector logic tests on synthetic embeddings (not model output)."""

import numpy as np
import pytest

pytest.importorskip("sklearn")

from medproof.intake.ood import OODModel, auroc, energy_score, fit_ood  # noqa: E402

DIM = 64


def _cluster(centre_axis, n, rng, noise=0.25):
    c = np.zeros(DIM, np.float32)
    c[centre_axis] = 1.0
    x = c + rng.normal(0, noise, (n, DIM)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


@pytest.fixture(scope="module")
def world():
    rng = np.random.default_rng(0)
    data = {"cxr": _cluster(0, 120, rng), "brain_mri": _cluster(1, 120, rng), "skin_dermoscopy": _cluster(2, 120, rng)}
    groups = {m: np.array([f"{m}-{i % 12}" for i in range(len(x))]) for m, x in data.items()}
    return data, groups, rng


def _fit(world, seed=0):
    data, groups, _ = world
    X = np.vstack(list(data.values()))
    y = np.concatenate([[m] * len(x) for m, x in data.items()])
    g = np.concatenate(list(groups.values()))
    return fit_ood(X, y, g, embedder_id="test-double-embedder", seed=seed)


def test_in_distribution_scores_stay_under_the_threshold_about_95_percent_of_the_time(world):
    model, report = _fit(world)
    for m, r in report["per_modality"].items():
        assert 0.93 <= r["oof_below_threshold"] <= 1.0
        assert r["n_fit"] > 0 and r["n_groups"] >= 3 and r["threshold"] > 0


def test_held_out_in_distribution_images_are_rarely_flagged(world):
    """The test split is untouched by the fit, so its flag rate is an honest false-alarm estimate."""
    from medproof.intake.ood import evaluate

    data, groups, _ = world
    X = np.vstack(list(data.values()))
    y = np.concatenate([[m] * len(x) for m, x in data.items()])
    g = np.concatenate(list(groups.values()))
    model, _ = fit_ood(X, y, g, embedder_id="e", seed=0)
    ev = evaluate(model, X, y, g, seed=0)
    for m, r in ev.items():
        assert r["in_flagged"] <= 0.25 and r["auroc"] >= 0.98


def test_other_modalities_are_flagged_and_separated(world):
    data, _, rng = world
    model, report = _fit(world)
    rng2 = np.random.default_rng(5)
    held_in = _cluster(0, 60, rng2)
    other = np.vstack([_cluster(1, 40, rng2), _cluster(2, 40, rng2), _cluster(30, 40, rng2)])
    d_in, d_out = model.distance(held_in, "cxr"), model.distance(other, "cxr")
    assert auroc(d_in, d_out) > 0.99
    flagged = (d_out > model.threshold("cxr")).mean()
    assert flagged > 0.95
    assert (d_in > model.threshold("cxr")).mean() < 0.2


def test_works_with_fewer_samples_than_dimensions(world):
    rng = np.random.default_rng(1)
    X = _cluster(0, 30, rng)  # 30 samples, 64 dims: the raw covariance is singular
    y = np.array(["cxr"] * 30)
    g = np.array([f"g{i % 6}" for i in range(30)])
    model, _ = fit_ood(X, y, g, embedder_id="e", seed=0)
    d = model.distance(_cluster(0, 5, rng), "cxr")
    assert np.isfinite(d).all() and (d >= 0).all()


def test_score_is_distance_over_threshold_and_decision_matches(world):
    model, _ = _fit(world)
    far = _cluster(40, 3, np.random.default_rng(2))
    res = model.score(far[0], "cxr")
    assert res["is_ood"] is True and res["score"] > 1.0
    assert res["score"] == pytest.approx(res["distance"] / res["threshold"])
    near = _cluster(0, 3, np.random.default_rng(3))
    assert model.score(near[0], "cxr")["score"] < res["score"]


def test_unknown_or_unsupported_modality_has_no_score(world):
    model, _ = _fit(world)
    assert model.score(np.zeros(DIM, np.float32), "other") is None
    assert model.score(np.zeros(DIM, np.float32), "bone_xray") is None  # not fitted here


def test_save_load_round_trip_and_no_pickle(world, tmp_path):
    model, _ = _fit(world)
    model.save(tmp_path / "ood.npz")
    with np.load(tmp_path / "ood.npz", allow_pickle=False) as f:
        assert any(k.startswith("mu_") for k in f.files)
    again = OODModel.load(tmp_path / "ood.npz")
    x = _cluster(0, 4, np.random.default_rng(9))
    np.testing.assert_allclose(model.distance(x, "cxr"), again.distance(x, "cxr"), rtol=1e-5)
    assert again.embedder_id == "test-double-embedder" and set(again.modalities) == set(model.modalities)


def test_wrong_dimension_is_rejected(world):
    model, _ = _fit(world)
    with pytest.raises(ValueError):
        model.distance(np.zeros((2, DIM + 1), np.float32), "cxr")


def test_deterministic_for_a_seed(world):
    a, _ = _fit(world, seed=1)
    b, _ = _fit(world, seed=1)
    assert a.threshold("cxr") == b.threshold("cxr")


def test_energy_score_math():
    logits = np.array([[2.0, 0.0, 0.0], [10.0, 0.0, 0.0]], np.float32)
    e = energy_score(logits)
    assert e[1] < e[0]  # a confident prediction has lower (more in-distribution) energy
    assert e[0] == pytest.approx(-np.log(np.exp(2) + 2))
    assert energy_score(logits, temperature=2.0)[0] == pytest.approx(-2 * np.log(np.exp(1) + 2))


def test_auroc_known_values():
    assert auroc(np.array([0.1, 0.2]), np.array([0.9, 0.8])) == 1.0
    assert auroc(np.array([0.9, 0.8]), np.array([0.1, 0.2])) == 0.0
    assert auroc(np.array([0.5, 0.5]), np.array([0.5, 0.5])) == pytest.approx(0.5)


def test_score_folder_returns_one_row_per_image_for_fitted_classes(tmp_path, world):
    from medproof.intake.decode import load_image  # noqa: F401
    from medproof.intake.ood import score_folder
    from medproof.intake.router_config import RouterConfig
    from tests.conftest import make_phantom, png_bytes, to_u8
    from tests.intake.router_helpers import TestDoubleEmbedder

    emb = TestDoubleEmbedder()
    rng = np.random.default_rng(0)
    for cls, n in (("cxr", 12), ("brain_mri", 12), ("other", 3)):
        d = tmp_path / "data" / cls
        d.mkdir(parents=True)
        for i in range(n):
            arr = make_phantom(64, seed=i) if cls == "cxr" else rng.random((64, 64)).astype(np.float32)
            (d / f"{i}.png").write_bytes(png_bytes(to_u8(arr)))
    from medproof.intake.decode import load_image as li

    X = np.vstack([emb.embed_images([li(p).analysis]) for p in sorted((tmp_path / "data" / "cxr").glob("*.png"))])
    Xb = np.vstack([emb.embed_images([li(p).analysis]) for p in sorted((tmp_path / "data" / "brain_mri").glob("*.png"))])
    y = np.array(["cxr"] * len(X) + ["brain_mri"] * len(Xb))
    g = np.array([f"g{i}" for i in range(len(y))])
    model, _ = fit_ood(np.vstack([X, Xb]), y, g, embedder_id=emb.model_id, seed=0)
    rows = score_folder(emb, tmp_path / "data", model, RouterConfig(cache_dir=str(tmp_path / "cache")))
    assert len(rows) == 24 and {r["class"] for r in rows} == {"cxr", "brain_mri"}  # "other" has no model, so it is skipped
    for r in rows:
        assert r["score"] == pytest.approx(r["distance"] / r["threshold"]) and isinstance(r["is_ood"], bool)
