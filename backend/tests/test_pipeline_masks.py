"""Mask models and the OOD score in the orchestrator: the brain U-Net and MedSAM reach the reader stages, load once,
fail soft, and the router's OOD score lands on the study."""

from __future__ import annotations

import functools
from types import SimpleNamespace

import pytest

from medproof import pipeline
from medproof.core.config import PipelineConfig
from medproof.core.schemas import StageResult
from medproof.pipeline import PIPELINE, StageSpec, run_study
from tests.conftest import png_bytes, to_u8
from tests.readers.test_cxr import _image

pytest.importorskip("torch")


def _cfg(tmp_path) -> PipelineConfig:
    return PipelineConfig(cache_dir=tmp_path / "cache", artifact_root=tmp_path / "artifacts", models_root=tmp_path / "models")


def _ctx(modality, tmp_path):
    return SimpleNamespace(modality_hint=modality, config=_cfg(tmp_path), router_flags=[], reader=None, reader_output=None, findings=None, stage_results=[])


@pytest.fixture(autouse=True)
def _fresh_mask_cache():
    pipeline._MASK_MODELS.clear()
    yield
    pipeline._MASK_MODELS.clear()


@pytest.fixture
def captured(monkeypatch):
    seen = {}

    def fake_classifier_run(ctx, reader=None, unet=None, medsam=None):
        seen["classifier"] = {"unet": unet, "medsam": medsam}
        return StageResult(stage="reader", ok=True, ms=0, payload={})

    def fake_bone_run(ctx, reader=None, unet=None, medsam=None):
        seen["bone"] = {"unet": unet, "medsam": medsam}
        return StageResult(stage="reader", ok=True, ms=0, payload={})

    monkeypatch.setattr("medproof.readers.classifier.run", fake_classifier_run)
    monkeypatch.setattr("medproof.readers.bone.run", fake_bone_run)
    return seen


def _loaders(monkeypatch, unet="UNET", medsam="SAM", log=None):
    def load(kind, config):
        if log is not None:
            log.append(kind)
        value = {"unet": unet, "medsam": medsam}[kind]
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(pipeline, "_build_mask_model", load)


def test_brain_gets_the_unet_skin_and_bone_get_medsam(monkeypatch, captured, tmp_path):
    _loaders(monkeypatch)
    pipeline._run_classifier(_ctx("brain_mri", tmp_path), reader=object())
    assert captured["classifier"] == {"unet": "UNET", "medsam": None}
    pipeline._run_classifier(_ctx("skin_dermoscopy", tmp_path), reader=object())
    assert captured["classifier"] == {"unet": None, "medsam": "SAM"}
    pipeline._run_bone(_ctx("bone_xray", tmp_path), reader=object())
    assert captured["bone"] == {"unet": None, "medsam": "SAM"}


def test_mask_models_load_once_per_process(monkeypatch, captured, tmp_path):
    log = []
    _loaders(monkeypatch, log=log)
    for _ in range(3):
        pipeline._run_classifier(_ctx("skin_dermoscopy", tmp_path), reader=object())
    assert log == ["medsam"]


def test_a_mask_model_that_cannot_load_does_not_stop_the_reader(monkeypatch, captured, tmp_path):
    _loaders(monkeypatch, unet=OSError("no weights"), medsam=RuntimeError("hub unreachable"))
    assert pipeline._run_classifier(_ctx("brain_mri", tmp_path), reader=object()).ok
    assert captured["classifier"] == {"unet": None, "medsam": None}
    assert pipeline._run_bone(_ctx("bone_xray", tmp_path), reader=object()).ok


def test_a_failed_load_is_not_retried_on_every_study(monkeypatch, captured, tmp_path):
    log = []
    _loaders(monkeypatch, medsam=RuntimeError("hub unreachable"), log=log)
    for _ in range(3):
        pipeline._run_bone(_ctx("bone_xray", tmp_path), reader=object())
    assert log == ["medsam"]  # a MedSAM load that has to time out against the hub would stall every study


def test_masks_can_be_switched_off(monkeypatch, captured, tmp_path):
    monkeypatch.setenv("MEDPROOF_MASKS", "0")
    log = []
    _loaders(monkeypatch, log=log)
    pipeline._run_classifier(_ctx("brain_mri", tmp_path), reader=object())
    assert log == [] and captured["classifier"] == {"unet": None, "medsam": None}


def test_the_default_builders_point_at_the_model_folders(tmp_path):
    cfg = _cfg(tmp_path)
    with pytest.raises(Exception, match="brain_seg"):
        pipeline._build_mask_model("unet", cfg)


# -- OOD score ---------------------------------------------------------------------------------------
def _study_with_router_payload(tmp_path, payload):
    def router(ctx):
        return StageResult(stage="router", ok=True, ms=0, payload=payload)

    stages = [PIPELINE[0], StageSpec(name="router", run=router, cacheable=False)]
    study, _ = run_study(png_bytes(to_u8(_image())), modality_hint="cxr", stages=stages, config=_cfg(tmp_path))
    return study


def test_the_routers_ood_score_lands_on_the_study(tmp_path):
    study = _study_with_router_payload(tmp_path, {"ood": {"distance": 10.0, "threshold": 100.0, "score": 0.42, "is_ood": False}})
    assert study.ood_score == pytest.approx(0.42)


def test_an_ood_flag_without_a_score_counts_as_out_of_distribution(tmp_path):
    assert _study_with_router_payload(tmp_path, {"ood": True}).ood_score == 1.0


def test_no_router_ood_means_zero(tmp_path):
    assert _study_with_router_payload(tmp_path, {"skipped": "modality given at upload"}).ood_score == 0.0
    assert _study_with_router_payload(tmp_path, {"ood": None}).ood_score == 0.0
    assert _study_with_router_payload(tmp_path, {"ood": {"score": "bad"}}).ood_score == 0.0


def test_the_functools_partial_stage_signature_still_works(tmp_path):
    spec = StageSpec(name="reader", run=functools.partial(pipeline._reader_step, load=lambda m, c: None))
    assert spec.name == "reader"


# -- the router's `ood` is a dict: only is_ood=True may flag the study ------------------------------------
def _route(tmp_path, ood):
    from tests.test_pipeline_wiring import StubRouter

    router = StubRouter("cxr", confidence=0.95)
    original = router.predict

    def predict(img):
        out = original(img)
        d = out.to_dict()
        out.to_dict = lambda: {**d, "flags": [], "ood": ood}
        return out

    router.predict = predict
    ctx = SimpleNamespace(modality_hint=None, raw_bytes=None, decoded=None, router_flags=[], body_part=None)
    from medproof.intake.decode import load_image

    ctx.decoded = load_image(png_bytes(to_u8(_image())))
    pipeline._router_step(ctx, router=router)
    return ctx


def test_an_in_distribution_ood_dict_does_not_flag_the_study(tmp_path):
    ctx = _route(tmp_path, {"distance": 10.0, "threshold": 100.0, "score": 0.1, "is_ood": False})
    assert "ood" not in ctx.router_flags and ctx.modality_hint == "cxr"


def test_an_out_of_distribution_dict_flags_the_study(tmp_path):
    ctx = _route(tmp_path, {"distance": 500.0, "threshold": 100.0, "score": 5.0, "is_ood": True})
    assert "ood" in ctx.router_flags
