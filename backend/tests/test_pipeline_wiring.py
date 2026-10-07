"""P1's stages wired into the orchestrator: router, modality readers, faithfulness and stability
(progress.md, P1's "P4 wiring recipe"). Real models are replaced by the stub chest reader and a stub
router, so these run without weights."""

from __future__ import annotations

import functools
from types import SimpleNamespace

import pytest

from medproof import pipeline
from medproof.core.cache import StageCache
from medproof.core.config import PipelineConfig
from medproof.pipeline import PIPELINE, StageSpec, run_study
from medproof.readers.classifier import ModelBundleError
from medproof.verify.faithfulness import FaithfulnessConfig
from tests.conftest import png_bytes, to_u8
from tests.readers.test_cxr import _image, make_reader

torch = pytest.importorskip("torch")

FAST_FAITH = FaithfulnessConfig(n_random=9)  # the smallest placement count that can still reach alpha


def _bytes() -> bytes:
    return png_bytes(to_u8(_image()))


def _config(tmp_path, **kw) -> PipelineConfig:
    return PipelineConfig(cache_dir=tmp_path / "cache", artifact_root=tmp_path / "artifacts", **kw)


class StubRouter:
    def __init__(self, modality="cxr", confidence=0.9, flags=(), body_part=None):
        self.out = SimpleNamespace(modality=modality, confidence=confidence, flags=list(flags), body_part=body_part)

    def predict(self, img):
        o = self.out
        return SimpleNamespace(
            warnings=[],
            to_dict=lambda: {"modality": o.modality, "body_part": o.body_part, "confidence": o.confidence,
                             "probs": {}, "method": "probe", "flags": o.flags, "ood": "ood" in o.flags},
        )


def _stub_loader(reader):
    def load(modality, config):
        if modality == "cxr":
            return reader
        raise ModelBundleError(f"ml/artifacts/{modality}: weights file 'weights.pt' not found; pull the training output first")

    return load


def _stages(*, router=None, reader=None, top_k_cfg=FAST_FAITH) -> list[StageSpec]:
    reader = reader or make_reader()
    return [
        PIPELINE[0],
        StageSpec(name="router", run=functools.partial(pipeline._router_step, router=router or StubRouter()), cacheable=False),
        StageSpec(name="reader", run=functools.partial(pipeline._reader_step, load=_stub_loader(reader)), cacheable=False),
        StageSpec(name="faithfulness", run=functools.partial(pipeline._faithfulness_step, cfg=top_k_cfg), timeout_s=120.0),
        StageSpec(name="stability", run=pipeline._stability_step, timeout_s=60.0),
    ]


# -- registry ---------------------------------------------------------------------------------------
def test_p1_stages_run_in_recipe_order_and_the_reader_is_not_cached():
    names = [s.name for s in PIPELINE]
    assert names[:5] == ["intake", "router", "reader", "faithfulness", "stability"]
    by = {s.name: s for s in PIPELINE}
    assert not by["router"].cacheable and not by["reader"].cacheable  # both leave state on ctx for later stages
    assert by["faithfulness"].cacheable and by["stability"].cacheable  # expensive and deterministic


# -- router -----------------------------------------------------------------------------------------
def test_a_confident_route_sets_the_modality_and_its_flags_reach_every_finding(tmp_path):
    router = StubRouter("cxr", confidence=0.91, flags=["ood"])
    study, results = run_study(_bytes(), modality_hint=None, stages=_stages(router=router), config=_config(tmp_path))
    assert study.modality == "cxr"
    assert next(r for r in results if r.stage == "reader").ok
    assert study.findings and all("ood" in f.flags for f in study.findings)
    assert all(f.status != "verified" for f in study.findings)  # the status rule downgrades ood


def test_a_low_confidence_route_is_not_trusted(tmp_path):
    router = StubRouter("cxr", confidence=0.40, flags=["router_uncertain"])
    study, results = run_study(_bytes(), modality_hint=None, stages=_stages(router=router), config=_config(tmp_path))
    assert study.modality == "other"
    reader = next(r for r in results if r.stage == "reader")
    assert not reader.ok and "modality" in reader.warnings[0]
    assert study.findings == []


def test_the_router_is_skipped_when_the_uploader_gave_the_modality(tmp_path):
    router = StubRouter("brain_mri", confidence=0.99)
    study, results = run_study(_bytes(), modality_hint="cxr", stages=_stages(router=router), config=_config(tmp_path))
    routed = next(r for r in results if r.stage == "router")
    assert routed.ok and routed.payload.get("skipped")
    assert study.modality == "cxr"


# -- readers ----------------------------------------------------------------------------------------
@pytest.mark.parametrize("modality", ["brain_mri", "skin_dermoscopy", "bone_xray"])
def test_missing_weights_degrade_the_reader_with_the_reason(tmp_path, modality):
    study, results = run_study(_bytes(), modality_hint=modality, stages=_stages(), config=_config(tmp_path))
    reader = next(r for r in results if r.stage == "reader")
    assert not reader.ok and "weights" in reader.warnings[0]
    assert study.findings == []
    verify = [r for r in results if r.stage in ("faithfulness", "stability")]
    assert all(not r.ok for r in verify)  # skipped with a reason, never crashed


def test_default_loader_points_each_modality_at_its_model_bundle(tmp_path):
    root = tmp_path / "models"
    for modality, bundle in pipeline.READER_BUNDLES.items():
        with pytest.raises(ModelBundleError, match=str(root / bundle).replace("\\", "\\\\")):
            pipeline._load_reader(modality, PipelineConfig(models_root=root))


# -- verification -----------------------------------------------------------------------------------
def test_faithfulness_and_stability_land_on_the_same_finding(tmp_path):
    study, results = run_study(_bytes(), modality_hint="cxr", stages=_stages(), config=_config(tmp_path, faithfulness_top_k=1))
    faith = next(r for r in results if r.stage == "faithfulness")
    stab = next(r for r in results if r.stage == "stability")
    assert faith.ok and stab.ok, (faith.warnings, stab.warnings)
    assert len(faith.payload["results"]) == 1  # only the most probable finding is tested

    top = max(study.findings, key=lambda f: f.prob_raw)
    assert top.image_evidence[0].faithfulness_drop is not None  # set by faithfulness
    assert top.stability is not None  # set by stability, which ran after it and kept it
    assert all(f.stability is not None for f in study.findings)
    others = [f for f in study.findings if f.finding_id != top.finding_id]
    assert all(f.image_evidence[0].faithful is None for f in others)  # untested, so never "valid evidence"


def test_a_repeat_upload_reruns_the_reader_and_reuses_the_verification(tmp_path):
    cfg = _config(tmp_path, faithfulness_top_k=1)
    cache = StageCache(cfg.cache_dir)
    try:
        first, _ = run_study(_bytes(), modality_hint="cxr", stages=_stages(), config=cfg, cache=cache)
        calls = []
        stages = _stages()
        stages[3] = StageSpec(name="faithfulness", run=lambda ctx: calls.append(1), timeout_s=5.0)  # must not run
        second, results = run_study(_bytes(), modality_hint="cxr", stages=stages, config=cfg, cache=cache)
    finally:
        cache.close()
    assert calls == []
    assert [f.model_dump() for f in second.findings] == [f.model_dump() for f in first.findings]
    assert all(f.stability is not None for f in second.findings)


def test_the_cache_key_includes_the_modality(tmp_path):
    cfg = _config(tmp_path, faithfulness_top_k=1)
    cache = StageCache(cfg.cache_dir)
    try:
        run_study(_bytes(), modality_hint="cxr", stages=_stages(), config=cfg, cache=cache)
        _, results = run_study(_bytes(), modality_hint="brain_mri", stages=_stages(), config=cfg, cache=cache)
    finally:
        cache.close()
    faith = next(r for r in results if r.stage == "faithfulness")
    assert not faith.ok  # no brain reader here, so no cached chest verification may answer for it
