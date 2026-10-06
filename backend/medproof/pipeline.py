"""The pipeline orchestrator (plan.md section 7.4, P4.3): runs stages in order with per-stage
timeouts, caching, and graceful degradation — a failed stage yields a warning, never a crash.

Timeout caveat, stated plainly because it matters: stages run on a worker thread and are
bounded with `future.result(timeout=...)`. A thread cannot be forcibly killed in Python, so a
timeout here means "stop waiting for this stage", not "kill it" — a stage blocked in CPU-bound
native code (e.g. a torch forward pass) keeps running in the background and its result is
discarded when it eventually finishes. This is the right tool for stages that raise, hang on
I/O (an HTTP call to MedGemma or Groq), or simply run long — not a hard kill switch for a
genuinely hung CPU-bound stage. `signal.alarm` isn't available on Windows (this team develops
on Windows laptops and Kaggle/Colab Linux alike); multiprocessing was rejected as the pickling
and model-reload cost per call is worse than the problem it solves.

Current stage registry is deliberately short: only `intake` and the CXR `reader` stage exist
anywhere in the repo today (no router, no verify stages, no brain/skin/bone readers, no
generalist/MedGemma stage, no calibration, no report stage, no ledger). New stages append to
`PIPELINE` as P1/P2/P3 land them; nothing else here needs to change for that.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from pathlib import Path

from medproof.core.cache import StageCache, cache_key
from medproof.core.config import PipelineConfig
from medproof.core.context import StudyContext
from medproof.core.schemas import Finding, Modality, StageResult, StudyResult
from medproof.core.status_rule import compute_status
from medproof.intake.decode import sha256_hex

DISCLAIMER = (
    "Decision support only. Not a diagnosis. A qualified clinician must confirm every finding "
    "before acting on it."
)

_GENESIS_HASH = "0" * 64


@dataclass(frozen=True)
class StageSpec:
    name: str
    run: Callable[[StudyContext], StageResult]
    timeout_s: float | None = None
    cache_version: str = "v1"
    required: bool = False  # if it fails, the pipeline stops (nothing downstream can proceed)


def _intake_step(ctx: StudyContext) -> StageResult:
    """Calls `run_bytes` directly (not the generic `intake.run(ctx)` wrapper), because that
    wrapper discards the `DecodedImage` it decodes. `run_bytes` returns both specifically so a
    caller can hand the image to the next stage — exactly what's needed here."""
    from medproof.intake.run import run_bytes

    raw = ctx.raw_bytes
    if raw is None and ctx.path:
        raw = Path(ctx.path).read_bytes()
    if raw is None:
        return StageResult(stage="intake", ok=False, ms=0, payload={"error": "no input"}, warnings=["no input"])
    result, decoded = run_bytes(raw, ctx.modality_hint)
    ctx.decoded = decoded
    return result


def _reader_step(ctx: StudyContext) -> StageResult:
    modality = ctx.modality_hint
    if modality == "cxr":
        from medproof.readers import cxr  # torch-free at import time; failures degrade inside run()

        return cxr.run(ctx)
    if modality is None:
        msg = "no modality hint and no router available yet (P1.4): reader stage skipped"
    else:
        msg = f"no reader available for modality {modality!r} yet"
    return StageResult(stage="reader", ok=False, ms=0, payload={"error": msg}, warnings=[msg])


PIPELINE: list[StageSpec] = [
    StageSpec(name="intake", run=_intake_step, timeout_s=5.0, required=True),
    StageSpec(name="reader", run=_reader_step, timeout_s=20.0),
]


def _run_stage(spec: StageSpec, ctx: StudyContext, cache: StageCache | None, config: PipelineConfig) -> StageResult:
    key = None
    if cache is not None and ctx.input_sha256:
        key = cache_key(ctx.input_sha256, spec.name, spec.cache_version)
        cached = cache.get(key)
        if cached is not None:
            return cached

    timeout = spec.timeout_s if spec.timeout_s is not None else config.default_timeout_s
    t0 = time.perf_counter()
    # Not a `with` block: ThreadPoolExecutor.__exit__ calls shutdown(wait=True), which would
    # block on the abandoned thread until it finishes — defeating the timeout entirely. Shut
    # down without waiting instead, so a hung stage's thread is left to finish (or not) on its
    # own in the background, exactly as documented at the top of this module.
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(spec.run, ctx)
    try:
        result = future.result(timeout=timeout)
    except FutureTimeoutError:
        ms = int((time.perf_counter() - t0) * 1000)
        msg = f"{spec.name} timed out after {timeout}s"
        result = StageResult(stage=spec.name, ok=False, ms=ms, payload={"error": "timeout"}, warnings=[msg])
    except Exception as exc:  # a stage that raises must not crash the study
        ms = int((time.perf_counter() - t0) * 1000)
        msg = f"{spec.name} failed ({type(exc).__name__}: {exc})"
        result = StageResult(stage=spec.name, ok=False, ms=ms, payload={"error": msg}, warnings=[msg])
    finally:
        pool.shutdown(wait=False)

    if cache is not None and key is not None:
        cache.set(key, result)
    return result


def _collect_findings(stage_results: list[StageResult]) -> list[Finding]:
    """Union of every `payload["findings"]` across all stages, status recomputed by the real
    rule. `evidence.to_findings` hardcodes every finding to "uncertain" on the way out of a
    reader — this is the first place that rule actually gets applied. Merging findings that a
    future verify stage updates in place (e.g. faithfulness setting `faithful=True`) is work
    for whichever stage introduces that; nothing does yet, so there's nothing to merge today."""
    findings: list[Finding] = []
    for sr in stage_results:
        for raw in sr.payload.get("findings", []) or []:
            finding = Finding.model_validate(raw)
            finding.status = compute_status(finding)
            findings.append(finding)
    return findings


def _fold_ledger_head(stage_results: list[StageResult]) -> str:
    """A real, order-and-content-dependent hash so `StudyResult.ledger_head` isn't a fake
    placeholder string — but it is provisional, not the ledger. No persistence, no append-only
    JSONL, no `/ledger/verify`: that's P4.5. The fold formula (prev + name + payload hash)
    mirrors what the real ledger will likely do per entry, so P4.5 generalizes this rather than
    replacing it outright."""
    head = _GENESIS_HASH
    for sr in stage_results:
        payload_hash = hashlib.sha256(json.dumps(sr.payload, sort_keys=True, default=str).encode()).hexdigest()
        head = hashlib.sha256(f"{head}:{sr.stage}:{payload_hash}".encode()).hexdigest()
    return head


def run_study(
    raw_bytes: bytes,
    *,
    study_id: str | None = None,
    modality_hint: Modality | None = None,
    stages: list[StageSpec] | None = None,
    cache: StageCache | None = None,
    config: PipelineConfig | None = None,
) -> tuple[StudyResult, list[StageResult]]:
    """Run every stage in order. Never raises: any failure degrades to a `StageResult(ok=False)`
    and the study still completes. Returns `(StudyResult, [StageResult, ...])` — the API (P4.4)
    streams the latter over SSE as each stage finishes."""
    config = config or PipelineConfig()
    if cache is None:
        cache = StageCache(config.cache_dir)
    stages = stages if stages is not None else PIPELINE

    ctx = StudyContext(
        study_id=study_id or str(uuid.uuid4()),
        raw_bytes=raw_bytes,
        modality_hint=modality_hint,
        input_sha256=sha256_hex(raw_bytes),
        artifact_dir=config.artifact_root / (study_id or "pending"),
    )

    stage_results: list[StageResult] = []
    for spec in stages:
        result = _run_stage(spec, ctx, cache, config)
        stage_results.append(result)
        if spec.required and not result.ok:
            break

    quality = next((sr.payload.get("quality", {}) for sr in stage_results if sr.stage == "intake"), {})
    if not stage_results or not stage_results[0].ok:
        quality = {**quality, "pipeline_error": stage_results[0].warnings if stage_results else ["no stages ran"]}

    study = StudyResult(
        study_id=ctx.study_id,
        input_sha256=ctx.input_sha256,
        modality=modality_hint or "other",
        ood_score=0.0,  # placeholder: P1.5 OOD score isn't built yet
        quality=quality,
        findings=_collect_findings(stage_results),
        claims=[],  # placeholder: P3.9 slot-filled report isn't built yet
        ledger_head=_fold_ledger_head(stage_results),
        disclaimer=DISCLAIMER,
    )
    return study, stage_results
