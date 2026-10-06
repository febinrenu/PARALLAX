"""Tests for the pipeline orchestrator (plan.md P4.3): ordering, timeouts, caching, and that a
failed stage degrades the study instead of crashing it.
"""

from __future__ import annotations

import time

import pytest
from medproof.core.cache import StageCache
from medproof.core.config import PipelineConfig
from medproof.core.context import StudyContext
from medproof.core.schemas import StageResult
from medproof.core.status_rule import compute_status
from medproof.pipeline import PIPELINE, StageSpec, run_study
from medproof.readers import cxr as cxr_reader

from tests.conftest import png_bytes, to_u8
from tests.readers.test_cxr import _image, make_reader

torch = pytest.importorskip("torch")  # make_reader needs torch, same as its own test file


def _cxr_bytes() -> bytes:
    return png_bytes(to_u8(_image()))


def _config(tmp_path) -> PipelineConfig:
    return PipelineConfig(cache_dir=tmp_path / "cache", artifact_root=tmp_path / "artifacts")


def _cxr_stages() -> list[StageSpec]:
    """The real intake stage plus a reader stage wired to the stub reader from test_cxr.py —
    exercises the real pipeline mechanics without needing torchxrayvision weights."""
    reader = make_reader()
    return [
        PIPELINE[0],  # intake, unchanged
        StageSpec(name="reader", run=lambda ctx: cxr_reader.run(ctx, reader=reader)),
    ]


def test_happy_path_assembles_findings_with_real_status(tmp_path):
    study, stage_results = run_study(
        _cxr_bytes(), modality_hint="cxr", stages=_cxr_stages(), config=_config(tmp_path)
    )
    assert [sr.stage for sr in stage_results] == ["intake", "reader"]
    assert all(sr.ok for sr in stage_results)
    assert study.modality == "cxr" and study.findings
    for f in study.findings:
        assert f.status == compute_status(f)
    assert len(study.ledger_head) == 64  # a hex sha256


def test_ledger_head_is_deterministic_for_the_same_input(tmp_path):
    cfg = _config(tmp_path)
    stages = _cxr_stages()
    raw = _cxr_bytes()
    study1, _ = run_study(raw, modality_hint="cxr", stages=stages, config=cfg, cache=StageCache(cfg.cache_dir))
    study2, _ = run_study(raw, modality_hint="cxr", stages=stages, config=cfg, cache=StageCache(cfg.cache_dir))
    assert study1.ledger_head == study2.ledger_head


def test_no_modality_hint_completes_with_warning(tmp_path):
    study, stage_results = run_study(_cxr_bytes(), modality_hint=None, config=_config(tmp_path))
    assert study.modality == "other" and study.findings == []
    reader_result = stage_results[1]
    assert reader_result.ok is False and reader_result.warnings


def test_unsupported_modality_completes_with_warning(tmp_path):
    study, stage_results = run_study(_cxr_bytes(), modality_hint="brain_mri", config=_config(tmp_path))
    assert study.findings == []
    reader_result = stage_results[1]
    assert reader_result.ok is False
    assert "brain_mri" in reader_result.warnings[0]


def test_corrupt_bytes_still_returns_a_study_result(tmp_path):
    study, stage_results = run_study(b"not an image", modality_hint="cxr", config=_config(tmp_path))
    assert len(stage_results) == 1  # intake is required: the pipeline stops there
    assert stage_results[0].ok is False
    assert study.findings == [] and study.input_sha256  # still a valid StudyResult, not an exception


def test_slow_stage_times_out_instead_of_hanging(tmp_path):
    def slow(ctx: StudyContext) -> StageResult:
        time.sleep(1.0)
        return StageResult(stage="slow", ok=True, ms=1000, payload={}, warnings=[])

    stages = [PIPELINE[0], StageSpec(name="slow", run=slow, timeout_s=0.05)]
    t0 = time.perf_counter()
    study, stage_results = run_study(_cxr_bytes(), modality_hint="cxr", stages=stages, config=_config(tmp_path))
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.9, "run_study should not block for the slow stage's full duration"
    assert stage_results[1].ok is False and "timed out" in stage_results[1].warnings[0]


def test_raising_stage_degrades_instead_of_crashing(tmp_path):
    def boom(ctx: StudyContext) -> StageResult:
        raise RuntimeError("boom")

    stages = [PIPELINE[0], StageSpec(name="boom", run=boom)]
    study, stage_results = run_study(_cxr_bytes(), modality_hint="cxr", stages=stages, config=_config(tmp_path))
    assert stage_results[1].ok is False and "boom" in stage_results[1].warnings[0]
    assert study.findings == []  # degraded gracefully, no exception propagated


def test_cache_serves_the_second_call_without_rerunning_the_stage(tmp_path):
    calls = {"n": 0}

    def counted(ctx: StudyContext) -> StageResult:
        calls["n"] += 1
        return StageResult(stage="counted", ok=True, ms=1, payload={"n": calls["n"]}, warnings=[])

    stages = [PIPELINE[0], StageSpec(name="counted", run=counted)]
    cfg = _config(tmp_path)
    cache = StageCache(cfg.cache_dir)
    raw = _cxr_bytes()

    run_study(raw, modality_hint="cxr", stages=stages, config=cfg, cache=cache)
    run_study(raw, modality_hint="cxr", stages=stages, config=cfg, cache=cache)
    assert calls["n"] == 1
