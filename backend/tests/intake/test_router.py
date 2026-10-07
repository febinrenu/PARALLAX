"""Router logic tests using a test double embedder (see router_helpers). Not a model-accuracy claim."""

import numpy as np
import pytest

pytest.importorskip("sklearn")

from medproof.core.schemas import StageResult  # noqa: E402
from medproof.intake.decode import load_image  # noqa: E402
from medproof.intake.embedder import EmbeddingCache  # noqa: E402
from medproof.intake.router import Router, RouterOutput, run  # noqa: E402
from medproof.intake.router_config import BODY_PARTS, CLASSES, RouterConfig  # noqa: E402
from medproof.intake.router_train import train_probe  # noqa: E402
from tests.conftest import make_phantom, png_bytes, to_u8  # noqa: E402
from tests.intake.router_helpers import TestDoubleEmbedder, clustered, prototype_vectors  # noqa: E402

CFG = RouterConfig()


def _img(arr):
    return load_image(png_bytes(to_u8(arr)))


def _families(n=24, seed=0):
    """Image families that differ in simple statistics, so the double embedder can tell them apart."""
    rng = np.random.default_rng(seed)
    fam = {}
    fam["cxr"] = [make_phantom(128, seed=s) for s in range(n)]
    fam["skin_dermoscopy"] = [np.clip(rng.normal(0.6, 0.02, (128, 128, 3)) + np.array([0.2, -0.1, -0.2]), 0, 1).astype(np.float32) for _ in range(n)]
    fam["other"] = [rng.random((128, 128)).astype(np.float32) for _ in range(n)]
    return fam


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    emb = TestDoubleEmbedder()
    fam = _families()
    X, y, g = [], [], []
    for cls, imgs in fam.items():
        for i, im in enumerate(imgs):
            X.append(emb.embed_images([im])[0])
            y.append(cls)
            g.append(f"{cls}-{i % 6}")
    # the remaining classes get synthetic vectors so the probe has all five (logic test)
    Xc, yc, gc = clustered()
    keep = np.isin(yc, ["brain_mri", "bone_xray"])
    X = np.vstack([np.stack(X), Xc[keep]])
    y = np.concatenate([np.asarray(y), yc[keep]])
    g = np.concatenate([np.asarray(g), gc[keep]])
    probe, _ = train_probe(X, y, g, embedder_id=emb.model_id, seed=0)
    path = tmp_path_factory.mktemp("probe") / "probe.npz"
    probe.save(path)
    return path


def _router(trained, **cfg):
    emb = TestDoubleEmbedder(prototype_vectors(CFG))
    return Router(emb, config=RouterConfig(probe_path=str(trained), **cfg)), emb


def test_probe_route_returns_modality_confidence_and_all_probabilities(trained):
    r, _ = _router(trained)
    out = r.predict(_img(make_phantom(128, seed=99)))
    assert isinstance(out, RouterOutput)
    assert out.modality == "cxr" and out.method == "probe"
    assert set(out.probs) == set(CLASSES) and abs(sum(out.probs.values()) - 1.0) < 1e-4
    assert out.confidence == pytest.approx(max(out.probs.values()))
    assert out.body_part is None and out.flags == []


def test_confident_probe_does_not_call_the_text_tower(trained):
    r, emb = _router(trained)
    r.predict(_img(make_phantom(128, seed=98)))
    assert emb.text_calls == 0


def test_low_probe_confidence_falls_back_to_zero_shot(trained):
    r, emb = _router(trained, min_confidence=1.01)  # nothing can be confident enough
    out = r.predict(_img(make_phantom(128, seed=97)))
    assert out.method == "zero_shot" and emb.text_calls > 0
    assert set(out.probs) == set(CLASSES)


def test_missing_probe_uses_zero_shot_and_says_so(tmp_path):
    emb = TestDoubleEmbedder(prototype_vectors(CFG))
    r = Router(emb, config=RouterConfig(probe_path=str(tmp_path / "absent.npz")))
    out = r.predict(_img(make_phantom(128)))
    assert out.method == "zero_shot" and any("probe" in w for w in out.warnings)


def test_zero_shot_math_picks_the_class_whose_text_vector_is_closest():
    """With text vectors set to class axes, an image embedded onto axis k must score class k highest."""

    class AxisEmbedder(TestDoubleEmbedder):
        def __init__(self, axis, vecs):
            super().__init__(vecs)
            self.axis = axis

        def embed_images(self, images):
            v = np.zeros((len(images), 16), np.float32)
            v[:, self.axis] = 1.0
            v[:, 15] = 0.2
            return v / np.linalg.norm(v, axis=1, keepdims=True)

    for k, cls in enumerate(CLASSES):
        r = Router(AxisEmbedder(k, prototype_vectors(CFG)), config=RouterConfig(probe_path="missing.npz"))
        out = r.predict(_img(make_phantom(64)))
        assert out.modality == cls and out.method == "zero_shot"
        assert out.confidence > 0.6


def test_below_zero_shot_floor_is_other_with_uncertain_flag():
    class FlatEmbedder(TestDoubleEmbedder):
        def embed_images(self, images):
            v = np.ones((len(images), 16), np.float32)  # equally close to every class axis
            return v / np.linalg.norm(v, axis=1, keepdims=True)

    r = Router(FlatEmbedder(prototype_vectors(CFG)), config=RouterConfig(probe_path="missing.npz", zero_shot_floor=0.5))
    out = r.predict(_img(make_phantom(64)))
    assert out.modality == "other" and "router_uncertain" in out.flags
    assert out.confidence < 0.5


def test_body_part_only_for_bone(tmp_path):
    rng = np.random.default_rng(2)
    Xc, yc, gc = clustered()
    bx, by = [], []
    for k, part in enumerate(BODY_PARTS):
        c = np.zeros(16, np.float32)
        c[8 + k] = 1.0
        for _ in range(20):
            v = c + rng.normal(0, 0.05, 16)
            bx.append(v / np.linalg.norm(v))
            by.append(part)
    probe, _ = train_probe(Xc, yc, gc, embedder_id="test-double-embedder", seed=0, body_part=(np.stack(bx), np.asarray(by)))
    path = tmp_path / "p.npz"
    probe.save(path)

    class Fixed(TestDoubleEmbedder):
        def __init__(self, vec):
            super().__init__()
            self.vec = vec / np.linalg.norm(vec)

        def embed_images(self, images):
            return np.tile(self.vec, (len(images), 1)).astype(np.float32)

    bone = np.zeros(16, np.float32)
    bone[3] = 1.0
    bone[8 + 2] = 1.0  # bone axis plus the "hip" body-part axis
    out = Router(Fixed(bone), config=RouterConfig(probe_path=str(path), min_confidence=0.0)).predict(_img(make_phantom(64)))
    assert out.modality == "bone_xray" and out.body_part == BODY_PARTS[2]
    cxr = np.zeros(16, np.float32)
    cxr[0] = 1.0
    out = Router(Fixed(cxr), config=RouterConfig(probe_path=str(path), min_confidence=0.0)).predict(_img(make_phantom(64)))
    assert out.modality == "cxr" and out.body_part is None


def test_probe_trained_with_another_embedder_is_refused(trained):
    class Other(TestDoubleEmbedder):
        model_id = "some-other-embedder"

    r = Router(Other(prototype_vectors(CFG)), config=RouterConfig(probe_path=str(trained)))
    out = r.predict(_img(make_phantom(128)))
    assert out.method == "zero_shot" and any("embedder" in w for w in out.warnings)


def test_embedding_cache_avoids_recomputing_and_is_keyed_by_model(trained, tmp_path):
    cache = EmbeddingCache(tmp_path / "cache")
    emb = TestDoubleEmbedder(prototype_vectors(CFG))
    r = Router(emb, config=RouterConfig(probe_path=str(trained)), cache=cache)
    img = _img(make_phantom(128, seed=5))
    a = r.predict(img)
    b = r.predict(img)
    assert emb.image_calls == 1 and a.probs == b.probs
    other = TestDoubleEmbedder(prototype_vectors(CFG))
    other.model_id = "different-model"
    Router(other, config=RouterConfig(probe_path=str(trained)), cache=cache).predict(img)
    assert other.image_calls == 1  # a different model id must not reuse the entry


def test_cache_roundtrip_and_miss(tmp_path):
    c = EmbeddingCache(tmp_path)
    v = np.arange(4, dtype=np.float32)
    assert c.get("m", "abc") is None
    c.put("m", "abc", v)
    np.testing.assert_array_equal(c.get("m", "abc"), v)
    assert c.get("m2", "abc") is None


def test_run_returns_stage_result_and_never_raises(trained):
    r, _ = _router(trained)
    res = run(type("Ctx", (), {"decoded": _img(make_phantom(128, seed=3))})(), router=r)
    assert isinstance(res, StageResult) and res.stage == "router" and res.ok
    assert res.payload["modality"] == "cxr" and res.payload["method"] == "probe"
    assert set(res.payload["probs"]) == set(CLASSES)

    class Boom:
        def predict(self, img):
            raise RuntimeError("model gone")

    bad = run(type("Ctx", (), {"decoded": _img(make_phantom(64))})(), router=Boom())
    assert bad.ok is False and bad.warnings
    assert run(type("Ctx", (), {})(), router=r).ok is False
    assert run(type("Ctx", (), {"decoded": _img(make_phantom(64))})(), router=None).ok is False


def test_prompts_and_thresholds_come_from_config():
    cfg = RouterConfig()
    assert set(cfg.prompts) == set(CLASSES) and all(len(v) >= 2 for v in cfg.prompts.values())
    assert 0.5 < cfg.min_confidence < 1.0 and 0.0 < cfg.zero_shot_floor < 1.0
    assert cfg.model_id == "google/medsiglip-448"


def test_model_id_path_and_probe_can_be_overridden_by_environment(monkeypatch):
    monkeypatch.setenv("MEDPROOF_ROUTER_MODEL_ID", "org/other-model")
    monkeypatch.setenv("MEDPROOF_ROUTER_MODEL", "D:/weights/local-copy")
    monkeypatch.setenv("MEDPROOF_ROUTER_PROBE", "somewhere/probe.npz")
    cfg = RouterConfig()
    assert cfg.model_id == "org/other-model" and cfg.model_path == "D:/weights/local-copy" and cfg.probe_path == "somewhere/probe.npz"
    assert RouterConfig().model_path == "D:/weights/local-copy"


def test_cache_paths_stay_short_for_long_model_ids(tmp_path):
    c = EmbeddingCache(tmp_path)
    long_id = "C:/Users/someone/AppData/Local/Temp/" + "very-long-folder-name/" * 8 + "medsiglip"
    c.put(long_id, "ab" * 32, np.arange(3, dtype=np.float32))
    np.testing.assert_array_equal(c.get(long_id, "ab" * 32), np.arange(3, dtype=np.float32))
    assert len(str(next(tmp_path.rglob("*.npy")).relative_to(tmp_path))) < 130


def test_ood_score_is_reported_and_flags_unusual_images():
    from medproof.intake.ood import fit_ood

    rng = np.random.default_rng(0)

    def around(axis, n, noise=0.05):
        c = np.zeros(16, np.float32)
        c[axis] = 1.0
        x = c + rng.normal(0, noise, (n, 16)).astype(np.float32)
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    X = around(0, 90)
    ood = fit_ood(X, np.array(["cxr"] * 90), np.array([f"g{i % 9}" for i in range(90)]), embedder_id="test-double-embedder", seed=0)[0]

    class Fixed(TestDoubleEmbedder):
        def __init__(self, vec):
            super().__init__(prototype_vectors(CFG))
            self.vec = vec / np.linalg.norm(vec)

        def embed_images(self, images):
            return np.tile(self.vec, (len(images), 1)).astype(np.float32)

    cfg = RouterConfig(probe_path="missing.npz", ood_path="missing.npz")
    typical = Router(Fixed(around(0, 1)[0]), config=cfg, ood=ood).predict(_img(make_phantom(64)))
    assert typical.modality == "cxr" and typical.ood is not None and typical.ood["is_ood"] is False and "ood" not in typical.flags
    odd_vec = around(0, 1)[0] + 0.6 * np.eye(16, dtype=np.float32)[9]  # still closest to cxr, but off its usual region
    odd = Router(Fixed(odd_vec), config=cfg, ood=ood).predict(_img(make_phantom(64)))
    assert odd.modality == "cxr" and odd.ood["is_ood"] is True and "ood" in odd.flags
    assert odd.ood["score"] > typical.ood["score"] > 0
    d = odd.to_dict()
    assert d["ood"]["is_ood"] is True and "ood" in d["flags"]


def test_no_ood_model_means_no_ood_field():
    out = Router(TestDoubleEmbedder(prototype_vectors(CFG)), config=RouterConfig(probe_path="missing.npz", ood_path="missing.npz")).predict(_img(make_phantom(64)))
    assert out.ood is None and out.to_dict()["ood"] is None
