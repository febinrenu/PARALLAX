"""Tests against the real MedSigLIP model. Skipped with an explicit reason when it is not accessible."""

import os

import numpy as np
import pytest

pytest.importorskip("transformers")

from medproof.intake.embedder import load_default  # noqa: E402
from medproof.intake.router import Router  # noqa: E402
from medproof.intake.router_config import CLASSES, RouterConfig  # noqa: E402
from tests.conftest import make_phantom  # noqa: E402

_loaded = None


def _locally_available(cfg: RouterConfig) -> bool:
    """Real-model tests run only when the weights are already on disk, never starting a 3.5 GB download."""
    if cfg.model_path:
        return os.path.isdir(cfg.model_path)
    try:
        from huggingface_hub import try_to_load_from_cache

        return isinstance(try_to_load_from_cache(cfg.model_id, "model.safetensors"), str)
    except Exception:
        return False


def _embedder():
    global _loaded
    if _loaded is None:
        cfg = RouterConfig()
        _loaded = load_default(cfg) if _locally_available(cfg) else (None, "weights not on disk (set MEDPROOF_ROUTER_MODEL to a local copy)")
    return _loaded


def _need_model():
    emb, reason = _embedder()
    if emb is None:
        pytest.skip(f"MedSigLIP not accessible: {reason}")
    return emb


def test_embedding_shape_norm_and_determinism():
    emb = _need_model()
    imgs = [make_phantom(256, seed=1), make_phantom(256, seed=2)]
    a = emb.embed_images(imgs)
    b = emb.embed_images(imgs)
    assert a.shape == (2, emb.dim) and a.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-4)
    np.testing.assert_allclose(a, b, atol=1e-5)
    assert not np.allclose(a[0], a[1])


def test_text_embeddings_share_the_image_space():
    emb = _need_model()
    t = emb.embed_texts(["a chest X-ray", "a photograph of a cat"])
    assert t.shape == (2, emb.dim) and np.allclose(np.linalg.norm(t, axis=1), 1.0, atol=1e-4)
    assert emb.logit_scale > 1.0


def test_zero_shot_on_a_phantom_returns_a_valid_distribution():
    emb = _need_model()
    r = Router(emb, config=RouterConfig(probe_path="does-not-exist.npz"))
    from medproof.intake.decode import load_image
    from tests.conftest import png_bytes, to_u8

    out = r.predict(load_image(png_bytes(to_u8(make_phantom(256)))))
    assert out.method in {"zero_shot"} and set(out.probs) == set(CLASSES)
    assert abs(sum(out.probs.values()) - 1.0) < 1e-4


@pytest.mark.skipif(not os.environ.get("MEDPROOF_ROUTER_DATA"), reason="MEDPROOF_ROUTER_DATA (staged labelled images) not set")
def test_probe_accuracy_on_staged_held_out_images():
    from medproof.intake.router_train import evaluate_folder

    emb = _need_model()
    report = evaluate_folder(emb, os.environ["MEDPROOF_ROUTER_DATA"])
    assert report["n"] >= 30 and report["accuracy"] >= 0.97, report


# --- API-compatibility check with a public random-weight SigLIP (NOT MedSigLIP, no accuracy meaning) ---
TINY = "hf-internal-testing/tiny-random-SiglipModel"


@pytest.fixture(scope="module")
def tiny_embedder():
    import dataclasses

    emb, reason = load_default(dataclasses.replace(RouterConfig(), model_id=TINY, model_path=""))
    if emb is None:
        pytest.skip(f"tiny SigLIP test model not available: {reason}")
    return emb


def test_embedder_code_path_works_with_the_installed_transformers_api(tiny_embedder):
    """Checks shapes, normalisation, determinism, text path and logit scale of OUR wrapper against the
    installed transformers version, using random weights. It proves the code runs, nothing about routing quality."""
    e = tiny_embedder
    imgs = [make_phantom(96, seed=1), np.stack([make_phantom(96, seed=2)] * 3, axis=2)]  # gray and colour
    a, b = e.embed_images(imgs), e.embed_images(imgs)
    assert a.shape == (2, e.dim) and a.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-4)
    np.testing.assert_allclose(a, b, atol=1e-6)
    t = e.embed_texts(["a chest X-ray", "a photograph of a cat"])
    assert t.shape == (2, e.dim) and np.allclose(np.linalg.norm(t, axis=1), 1.0, atol=1e-4)
    assert e.logit_scale > 0 and e._spec.size == 30  # size read from the model config


def test_router_runs_end_to_end_with_the_tiny_model(tiny_embedder, tmp_path):
    from medproof.intake.decode import load_image
    from tests.conftest import png_bytes, to_u8

    r = Router(tiny_embedder, config=RouterConfig(probe_path=str(tmp_path / "none.npz")))
    out = r.predict(load_image(png_bytes(to_u8(make_phantom(96)))))
    assert out.method == "zero_shot" and set(out.probs) == set(CLASSES)
    assert abs(sum(out.probs.values()) - 1.0) < 1e-4
