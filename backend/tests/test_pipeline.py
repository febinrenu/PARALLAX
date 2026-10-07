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


def test_a_modality_without_weights_completes_with_warning(tmp_path):
    # An empty models folder, so the result doesn't depend on whose laptop has the brain weights.
    config = PipelineConfig(cache_dir=tmp_path / "cache", artifact_root=tmp_path / "artifacts", models_root=tmp_path / "no_models")
    study, stage_results = run_study(_cxr_bytes(), modality_hint="brain_mri", config=config)
    assert study.findings == []
    reader_result = next(sr for sr in stage_results if sr.stage == "reader")
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


def test_repeat_run_with_a_shared_cache_still_feeds_the_reader(tmp_path):
    """Intake mutates ctx (it attaches the decoded image), so it must never be served from
    cache; otherwise the second run's reader gets no image."""
    cfg = _config(tmp_path)
    cache = StageCache(cfg.cache_dir)
    raw = _cxr_bytes()
    stages = _cxr_stages()
    first, _ = run_study(raw, modality_hint="cxr", stages=stages, config=cfg, cache=cache)
    second, results = run_study(raw, modality_hint="cxr", stages=stages, config=cfg, cache=cache)
    assert all(r.ok for r in results), [r.warnings for r in results]
    assert [f.label for f in second.findings] == [f.label for f in first.findings]


def test_a_failed_stage_is_not_cached(tmp_path):
    calls = {"n": 0}

    def flaky(ctx: StudyContext) -> StageResult:
        calls["n"] += 1
        ok = calls["n"] > 1  # fails the first time only
        return StageResult(stage="flaky", ok=ok, ms=1, payload={}, warnings=[] if ok else ["transient"])

    stages = [PIPELINE[0], StageSpec(name="flaky", run=flaky)]
    cfg = _config(tmp_path)
    cache = StageCache(cfg.cache_dir)
    raw = _cxr_bytes()
    _, first = run_study(raw, modality_hint="cxr", stages=stages, config=cfg, cache=cache)
    _, second = run_study(raw, modality_hint="cxr", stages=stages, config=cfg, cache=cache)
    assert first[1].ok is False and second[1].ok is True
    assert calls["n"] == 2


# -- P3-1: later stages see earlier results; findings merge by id; claims reach the study ---------
def _finding_update(finding: dict, **changes) -> dict:
    return {**finding, **changes}


def _reader_then(*later: StageSpec) -> list[StageSpec]:
    return [*_cxr_stages(), *later]


def test_later_stages_see_every_earlier_result_on_the_context(tmp_path):
    seen: list[list[str]] = []

    def probe(ctx: StudyContext) -> StageResult:
        seen.append([sr.stage for sr in ctx.stage_results])
        return StageResult(stage="probe", ok=True, ms=0, payload={})

    run_study(_cxr_bytes(), modality_hint="cxr", stages=_reader_then(StageSpec(name="probe", run=probe)), config=_config(tmp_path))
    assert seen == [["intake", "reader"]]


def test_a_stage_that_updates_findings_replaces_them_instead_of_duplicating(tmp_path):
    def second_read(ctx: StudyContext) -> StageResult:
        reader = next(sr for sr in ctx.stage_results if sr.stage == "reader")
        updated = [_finding_update(f, flags=[*f["flags"], "second_read_seen"]) for f in reader.payload["findings"]]
        return StageResult(stage="second_read", ok=True, ms=0, payload={"findings": updated})

    def context(ctx: StudyContext) -> StageResult:
        latest = next(sr for sr in reversed(ctx.stage_results) if sr.payload.get("findings"))
        updated = [_finding_update(f, flags=[*f["flags"], "context_seen"]) for f in latest.payload["findings"]]
        return StageResult(stage="context", ok=True, ms=0, payload={"findings": updated})

    stages = _reader_then(StageSpec(name="second_read", run=second_read), StageSpec(name="context", run=context))
    study, results = run_study(_cxr_bytes(), modality_hint="cxr", stages=stages, config=_config(tmp_path))

    reader_ids = [f["finding_id"] for f in results[1].payload["findings"]]
    assert reader_ids, "the stub reader should produce at least one finding"
    assert [f.finding_id for f in study.findings] == reader_ids  # one per id, reader order kept
    for f in study.findings:
        assert {"second_read_seen", "context_seen"} <= set(f.flags)  # both updates on the same finding
        assert f.status == compute_status(f)


def test_claims_from_any_stage_reach_the_study(tmp_path):
    claim = {"claim_id": "c1", "template": "Doctor, consider {f1.label}.", "evidence_ids": ["e1"], "rendered": "Doctor, consider x."}

    def report(ctx: StudyContext) -> StageResult:
        return StageResult(stage="report", ok=True, ms=0, payload={"claims": [claim, {"claim_id": "broken"}]})

    study, _ = run_study(_cxr_bytes(), modality_hint="cxr", stages=_reader_then(StageSpec(name="report", run=report)), config=_config(tmp_path))
    assert [c.claim_id for c in study.claims] == ["c1"]  # the malformed one is dropped, not fatal


def test_merge_matches_the_reasoning_module_helper(tmp_path):
    reasoning = pytest.importorskip("medproof.reasoning_stages")
    from medproof.pipeline import _collect_claims, _collect_findings

    def update(ctx: StudyContext) -> StageResult:
        latest = ctx.stage_results[-1].payload["findings"]
        return StageResult(stage="update", ok=True, ms=0, payload={"findings": [_finding_update(f, flags=["x"]) for f in latest[:1]]})

    _, results = run_study(_cxr_bytes(), modality_hint="cxr", stages=_reader_then(StageSpec(name="update", run=update)), config=_config(tmp_path))
    ours = _collect_findings(results)
    theirs = reasoning.merge_findings(results)
    for f in theirs:
        f.status = compute_status(f)
    assert [f.model_dump() for f in ours] == [f.model_dump() for f in theirs]
    assert _collect_claims(results) == reasoning.collect_claims(results)


def test_reasoning_stages_are_registered_after_verification_and_never_cached():
    pytest.importorskip("medproof.reasoning_stages")
    names = [s.name for s in PIPELINE]
    start = names.index("calibration") + 1  # after verification and calibration, which they build on
    assert names[start:start + 4] == ["second_read", "context", "report", "precedents"]
    assert not any(s.cacheable for s in PIPELINE[start:start + 4])
