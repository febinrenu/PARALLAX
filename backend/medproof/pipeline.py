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

Stage registry: `intake`, `router` (only when the uploader gave no modality), the modality's
`reader` (chest, brain, skin or bone), `faithfulness` and `stability` (P1), `calibration` (P2),
then P3's
`second_read`, `context`, `report` and `precedents` (`medproof.reasoning_stages`). New stages
append to `PIPELINE`. Each stage sees every
earlier result on `ctx.stage_results`; a stage that updates findings returns the full list under
`payload["findings"]`, and the study keeps one finding per id with the latest stage winning.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from medproof.core.cache import StageCache, cache_key
from medproof.core.config import PipelineConfig
from medproof.core.context import StudyContext
from medproof.core.schemas import Claim, Finding, Modality, StageResult, StudyResult
from medproof.core.status_rule import compute_status
from medproof.intake.decode import sha256_hex

DISCLAIMER = (
    "Decision support only. Not a diagnosis. A qualified clinician must confirm every finding "
    "before acting on it."
)

_GENESIS_HASH = "0" * 64

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class StageSpec:
    name: str
    run: Callable[[StudyContext], StageResult]
    timeout_s: float | None = None
    cache_version: str = "v1"
    required: bool = False  # if it fails, the pipeline stops (nothing downstream can proceed)
    # A stage that hands state to later stages through ctx (intake sets ctx.decoded) must run
    # every time: a cache hit would skip that side effect and starve the stages after it.
    cacheable: bool = True


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


def _failed(stage: str, msg: str, t0: float | None = None) -> StageResult:
    ms = int((time.perf_counter() - t0) * 1000) if t0 is not None else 0
    return StageResult(stage=stage, ok=False, ms=ms, payload={"error": msg}, warnings=[msg])


def _router_step(ctx: StudyContext, router: Any = None) -> StageResult:
    """P1.4. Runs only when the uploader gave no modality. The routed modality is trusted at or above
    `RouterConfig().trust_confidence`; below it the study stays unrouted and the reader is skipped.
    The router's flags (`ood`, `router_uncertain`) are kept for the reader step to copy onto every
    finding, since the status rule downgrades `ood`."""
    if ctx.modality_hint is not None:
        return StageResult(stage="router", ok=True, ms=0, payload={"skipped": "modality given at upload", "modality": ctx.modality_hint})
    from medproof.intake import router as router_mod
    from medproof.intake.router_config import RouterConfig

    result = router_mod.run(ctx, router=router)
    if not result.ok:
        return result
    payload = result.payload
    ctx.router_flags = [f for f in payload.get("flags", []) or [] if isinstance(f, str)]
    ood = payload.get("ood")  # a dict {distance, threshold, score, is_ood}; only is_ood=True flags the study
    if (ood is True or (isinstance(ood, dict) and ood.get("is_ood") is True)) and "ood" not in ctx.router_flags:
        ctx.router_flags.append("ood")
    confidence = float(payload.get("confidence") or 0.0)
    trusted = confidence >= RouterConfig().trust_confidence and payload.get("modality") in READER_STEPS
    warnings = list(result.warnings)
    if trusted:
        ctx.modality_hint = payload["modality"]
        ctx.body_part = payload.get("body_part")
    else:
        warnings.append(f"router not confident enough ({confidence:.2f}); choose the modality at upload")
    return result.model_copy(update={"payload": {**payload, "trusted": trusted}, "warnings": warnings})


# P2's model bundle for each non-chest reader, under `PipelineConfig.models_root`.
READER_BUNDLES: dict[str, str] = {"brain_mri": "brain_cls", "skin_dermoscopy": "skin_cls", "bone_xray": "bone_det"}
_READERS: dict[str, Any] = {}
_READERS_LOCK = threading.Lock()


def _load_reader(modality: str, config: PipelineConfig) -> Any:
    """The shared reader for a modality, loaded once per process. Raises (e.g. `ModelBundleError`
    with the missing-weights reason) when it can't load; failures aren't remembered, so weights
    copied in later are picked up without a restart."""
    if modality == "cxr":
        from medproof.readers.cxr import get_reader

        return get_reader()
    key = f"{modality}:{config.models_root}"
    with _READERS_LOCK:
        if key not in _READERS:
            bundle = config.models_root / READER_BUNDLES[modality]
            if modality == "bone_xray":
                from medproof.readers.bone import BoneReader

                _READERS[key] = BoneReader.from_dir(bundle)
            else:
                from medproof.readers.classifier import ImageClassifierReader

                _READERS[key] = ImageClassifierReader.from_dir(bundle, modality=modality)
        return _READERS[key]


def _run_cxr(ctx: StudyContext, reader: Any) -> StageResult:
    from medproof.readers import cxr

    return cxr.run(ctx, reader=reader)


# Mask models (P1-A): the brain U-Net and MedSAM. Loaded once per process; a model that cannot load is
# remembered as missing for `_MASK_RETRY_S`, so a MedSAM load that has to give up on an unreachable hub does
# not stall every study. Without a mask model the reader still returns its heatmap or box evidence.
_MASK_MODELS: dict[str, tuple[Any, float]] = {}
_MASK_LOCK = threading.Lock()
_MASK_RETRY_S = 300.0


def _build_mask_model(kind: str, config: PipelineConfig) -> Any:
    if kind == "unet":
        from medproof.readers.segmenter import UNetSegmenter

        return UNetSegmenter.from_dir(config.models_root / "brain_seg")
    from medproof.segment.medsam import MedSAM

    return MedSAM.load(cache_dir=config.cache_dir / "medsam")  # MEDPROOF_MEDSAM_MODEL names a local folder


def _mask_model(kind: str, config: PipelineConfig) -> Any:
    if os.environ.get("MEDPROOF_MASKS", "1") == "0":
        return None
    key = f"{kind}:{config.models_root}:{config.cache_dir}"
    with _MASK_LOCK:
        hit = _MASK_MODELS.get(key)
        if hit is not None and (hit[0] is not None or time.monotonic() - hit[1] < _MASK_RETRY_S):
            return hit[0]
        try:
            model = _build_mask_model(kind, config)
        except Exception as exc:  # missing weights, no segmentation-models-pytorch, hub unreachable
            log.warning("%s mask model unavailable (%s: %s)", kind, type(exc).__name__, exc)
            model = None
        _MASK_MODELS[key] = (model, time.monotonic())
        return model


def _run_classifier(ctx: StudyContext, reader: Any) -> StageResult:
    from medproof.readers import classifier

    config = ctx.config or PipelineConfig()
    unet = _mask_model("unet", config) if ctx.modality_hint == "brain_mri" else None
    medsam = _mask_model("medsam", config) if ctx.modality_hint == "skin_dermoscopy" else None
    return classifier.run(ctx, reader=reader, unet=unet, medsam=medsam)


def _run_bone(ctx: StudyContext, reader: Any) -> StageResult:
    from medproof.readers import bone

    return bone.run(ctx, reader=reader, medsam=_mask_model("medsam", ctx.config or PipelineConfig()))


READER_STEPS: dict[str, Callable[[StudyContext, Any], StageResult]] = {
    "cxr": _run_cxr,
    "brain_mri": _run_classifier,
    "skin_dermoscopy": _run_classifier,
    "bone_xray": _run_bone,
}


def _reader_step(ctx: StudyContext, load: Callable[[str, PipelineConfig], Any] | None = None) -> StageResult:
    """Dispatches on the (given or routed) modality. Every reader leaves `ctx.reader`,
    `ctx.reader_output` and `ctx.findings` for the verify stages; router flags go onto each finding."""
    t0 = time.perf_counter()
    modality = ctx.modality_hint
    if modality is None:
        return _failed("reader", "no modality: none given at upload and the router could not decide")
    step = READER_STEPS.get(modality)
    if step is None:
        return _failed("reader", f"no reader available for modality {modality!r}")
    try:
        reader = (load or _load_reader)(modality, ctx.config or PipelineConfig())
    except Exception as exc:  # missing or corrupt weights, missing torch/timm/ultralytics
        return _failed("reader", f"{modality} reader unavailable: {exc}", t0)
    result = step(ctx, reader)
    if ctx.router_flags and result.payload.get("findings"):
        result = _tag_findings(result, ctx, ctx.router_flags)
    return result


def _tag_findings(result: StageResult, ctx: StudyContext, flags: list[str]) -> StageResult:
    def add(existing: list[str]) -> list[str]:
        return [*existing, *(f for f in flags if f not in existing)]

    findings = [{**f, "flags": add(list(f.get("flags", [])))} for f in result.payload["findings"]]
    if ctx.findings:
        ctx.findings = [f.model_copy(update={"flags": add(list(f.flags))}) for f in ctx.findings]
    return result.model_copy(update={"payload": {**result.payload, "findings": findings}})


def _verify_view(ctx: StudyContext, top_k: int | None = None) -> SimpleNamespace | None:
    """What P1's verify stages read, built from the latest merged findings rather than the reader's
    snapshot: otherwise stability would start from pre-faithfulness findings and, as the later
    stage, overwrite the faithfulness result when findings merge."""
    if ctx.reader_output is None or ctx.decoded is None:
        return None
    out = ctx.reader_output
    if top_k is not None and len(out.findings) > top_k:
        out = replace(out, findings=sorted(out.findings, key=lambda rf: rf.prob_raw, reverse=True)[:top_k])
    return SimpleNamespace(decoded=ctx.decoded, reader_output=out, findings=_merged_findings(ctx.stage_results))


def _faithfulness_step(ctx: StudyContext, cfg: Any = None) -> StageResult:
    """P1.9 deletion test on the `faithfulness_top_k` most probable findings."""
    from medproof.verify import faithfulness

    view = _verify_view(ctx, (ctx.config or PipelineConfig()).faithfulness_top_k)
    if view is None or ctx.reader is None:
        return _failed("faithfulness", "faithfulness skipped: no reader output for this study")
    return faithfulness.run(view, reader=ctx.reader, cfg=cfg)


def _stability_step(ctx: StudyContext, cfg: Any = None) -> StageResult:
    """P1.11: eight perturbations, flip rate per finding (eight forward passes in total)."""
    from medproof.verify import stability

    view = _verify_view(ctx)
    if view is None or ctx.reader is None:
        return _failed("stability", "stability skipped: no reader output for this study")
    return stability.run(view, reader=ctx.reader, cfg=cfg)


# P2's serving-time calibrators, under `PipelineConfig.models_root`. Chest has none on purpose: the
# product reads the all-data weights, which saw RSNA, and P2 fitted the chest calibrator for the
# external `chex` weights, so applying it would claim a calibration that was never measured.
CALIBRATION_FILES: dict[str, str] = {
    "brain_mri": "brain_cls/calibration.json",
    "skin_dermoscopy": "skin_cls/calibration.json",
    "bone_xray": "bone_det/calibration.json",
}
_CALIBRATORS: dict[str, Any] = {}


def _load_calibrator(modality: str, config: PipelineConfig) -> Any:
    path = config.models_root / CALIBRATION_FILES[modality]
    key = str(path)
    if key not in _CALIBRATORS:
        if modality == "bone_xray":
            from medproof.calibrate.binary import BinaryCalibrator

            _CALIBRATORS[key] = BinaryCalibrator.load(path)
        else:
            from medproof.calibrate.calibrator import Calibrator

            _CALIBRATORS[key] = Calibrator.load(path)
    return _CALIBRATORS[key]


def _calibrated(f: Finding, prob: float, tier: str, conformal_set: list[str] | None, coverage: float | None) -> Finding:
    update: dict[str, Any] = {"prob_calibrated": float(prob), "tier": tier, "flags": [x for x in f.flags if x != "uncalibrated"]}
    if conformal_set is not None:
        update["conformal_set"] = list(conformal_set)
    if coverage is not None:
        update["coverage_target"] = float(coverage)
    return f.model_copy(update=update)


def _calibration_step(ctx: StudyContext) -> StageResult:
    """P2.7 to P2.9 at serving time: calibrated probability, conformal set and tier per finding.
    Multi-class readers (brain, skin) use temperature scaling and an APS conformal set; the bone
    detector uses Platt scaling with an abstention band. Findings stay as they are when no
    calibrator applies, and the stage says why."""
    t0 = time.perf_counter()
    modality = ctx.modality_hint
    findings = _merged_findings(ctx.stage_results)
    if not findings:
        return StageResult(stage="calibration", ok=True, ms=0, payload={"calibrated": 0})
    if modality == "cxr":
        return _failed("calibration", "chest findings stay uncalibrated: the product's chest weights saw RSNA, and the fitted calibrator belongs to the external chex weights")
    if modality not in CALIBRATION_FILES:
        return _failed("calibration", f"no calibrator for modality {modality!r}")
    try:
        cal = _load_calibrator(modality, ctx.config or PipelineConfig())
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return _failed("calibration", f"calibration file unavailable for {modality} ({type(exc).__name__})", t0)

    if modality == "bone_xray":
        import numpy as np

        from medproof.readers.cxr_config import CxrConfig  # the readers' own tier thresholds

        updated = []
        for f in findings:
            p = float(cal.prob(np.array([f.prob_raw]))[0])
            abstain = cal.abstain_low <= p <= cal.abstain_high
            updated.append(_calibrated(f, p, "abstain" if abstain else CxrConfig().tier(p), None, 1 - cal.alpha))
        payload = {"method": "platt", "calibrated": len(updated)}
    else:
        out = ctx.reader_output
        if out is None or getattr(out, "logits", None) is None:
            return _failed("calibration", "calibration skipped: no reader logits for this study", t0)
        if list(out.labels) != list(cal.classes):
            return _failed("calibration", "calibration refused: the reader's class order does not match the calibrator's", t0)
        from medproof.calibrate import abstain

        res = cal.calibrate(out.logits)
        probs, cset = res.probs[0], res.conformal_sets[0]
        top = int(probs.argmax())
        updated = []
        for f in findings:
            i = cal.classes.index(f.label) if f.label in cal.classes else None
            if i is None:
                updated.append(f)
                continue
            tier = res.tiers[0] if i == top else abstain.tier(float(probs[i]), len(cset), cal.tier_rule)
            updated.append(_calibrated(f, float(probs[i]), tier, cset, 1 - cal.alpha))
        payload = {"method": cal.method, "temperature": cal.temperature, "conformal_set": cset, "calibrated": len(updated)}
    return StageResult(stage="calibration", ok=True, ms=int((time.perf_counter() - t0) * 1000),
                       payload={**payload, "findings": [f.model_dump() for f in updated]})


def _reasoning_specs() -> list[StageSpec]:
    """P3's stages (second read, context, report, precedents). They set their own timeouts and are
    never cached: the notes change their output and the cache key is the image hash alone. Their
    module pulls in optional dependencies (`backend[services]`); without them the API still runs
    image-only and says why in the log, instead of failing to start."""
    try:
        from medproof.reasoning_stages import stage_specs
    except ImportError as exc:
        log.warning("reasoning stages unavailable (%s); install backend[services] to enable them", exc)
        return []
    return stage_specs()


PIPELINE: list[StageSpec] = [
    StageSpec(name="intake", run=_intake_step, timeout_s=5.0, required=True, cacheable=False),
    # The router sets the modality and the reader leaves the live model on ctx for the verify
    # stages; a cache hit would skip both side effects, so neither is cached. The expensive,
    # deterministic part (faithfulness, stability) is cached instead, keyed by image and modality.
    StageSpec(name="router", run=_router_step, timeout_s=30.0, cacheable=False),
    # 45 s: the first chest read in a process loads the DenseNet and anatomy models (~17 s cold).
    StageSpec(name="reader", run=_reader_step, timeout_s=45.0, cacheable=False),
    StageSpec(name="faithfulness", run=_faithfulness_step, timeout_s=120.0, cache_version="v1"),
    StageSpec(name="stability", run=_stability_step, timeout_s=60.0, cache_version="v1"),
    # Cheap and reads the merged findings, so it always runs (not cached).
    StageSpec(name="calibration", run=_calibration_step, timeout_s=10.0, cacheable=False),
]
PIPELINE.extend(_reasoning_specs())


def _inputs_digest(stage_results: list[StageResult]) -> str:
    material = json.dumps([sr.payload.get("findings") for sr in stage_results], sort_keys=True, default=str)
    return hashlib.sha256(material.encode()).hexdigest()[:16]


def _run_stage(spec: StageSpec, ctx: StudyContext, cache: StageCache | None, config: PipelineConfig) -> StageResult:
    key = None
    if cache is not None and ctx.input_sha256 and spec.cacheable:
        # The key covers the modality (the same bytes read as another modality is another result) and the
        # findings this stage starts from: cached verify results carry full finding copies, so a reader that
        # now adds evidence (a mask, a new model) must not be overruled by a copy made before it did.
        key = cache_key(ctx.input_sha256, spec.name, f"{spec.cache_version}:{ctx.modality_hint}:{_inputs_digest(ctx.stage_results)}")
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

    # Only successes are cached: a timeout or a missing input is transient and must not stick.
    if cache is not None and key is not None and result.ok:
        cache.set(key, result)
    return result


def _collect_findings(stage_results: list[StageResult]) -> list[Finding]:
    """One finding per id across all stages, in first-seen order; a later stage's version replaces
    an earlier one (verify, second-read and context stages return the full updated list). Same
    semantics as `reasoning_stages.merge_findings`, pinned by a parity test, but defined here so
    the orchestrator never depends on that module's optional imports. Status is then recomputed
    by the real rule: readers hand findings out as "uncertain", and this is where it's decided."""
    findings = _merged_findings(stage_results)
    for finding in findings:
        finding.status = compute_status(finding)
    return findings


def _merged_findings(stage_results: list[StageResult]) -> list[Finding]:
    by_id: dict[str, Finding] = {}
    for sr in stage_results:
        for raw in sr.payload.get("findings", []) or []:
            try:
                finding = raw if isinstance(raw, Finding) else Finding.model_validate(raw)
            except ValueError:
                continue  # one malformed entry must not lose the rest of the study
            by_id[finding.finding_id] = finding
    return list(by_id.values())


def _collect_claims(stage_results: list[StageResult]) -> list[Claim]:
    """Every `payload["claims"]` in stage order (the report stage is the only producer today)."""
    claims: list[Claim] = []
    for sr in stage_results:
        for raw in sr.payload.get("claims", []) or []:
            try:
                claims.append(raw if isinstance(raw, Claim) else Claim.model_validate(raw))
            except ValueError:
                continue
    return claims


def _ood_score(stage_results: list[StageResult]) -> float:
    """The router's out-of-distribution score (distance over the modality's threshold; 1 or more is flagged),
    0.0 when the router did not run. A bare `ood: true` without a score counts as 1.0."""
    for sr in stage_results:
        if sr.stage != "router":
            continue
        ood = sr.payload.get("ood")
        if isinstance(ood, dict):
            try:
                return float(ood["score"])
            except (KeyError, TypeError, ValueError):
                return 0.0
        return 1.0 if ood is True else 0.0
    return 0.0


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
    notes: str | None = None,
    stages: list[StageSpec] | None = None,
    cache: StageCache | None = None,
    config: PipelineConfig | None = None,
    on_stage_result: Callable[[StageResult], None] | None = None,
) -> tuple[StudyResult, list[StageResult]]:
    """Run every stage in order. Never raises: any failure degrades to a `StageResult(ok=False)`
    and the study still completes. Returns `(StudyResult, [StageResult, ...])`.

    `on_stage_result`, if given, is called synchronously right after every stage (cache hit or
    miss, so a cached run still produces a full event sequence) — the API (P4.4) uses this to
    push each result onto an SSE queue and to append it to the evidence ledger (P4.5). Nothing
    here imports `core.ledger`: that wiring belongs to the caller, so this function stays usable
    (and testable) with no ledger at all."""
    config = config or PipelineConfig()
    owns_cache = cache is None
    if cache is None:
        cache = StageCache(config.cache_dir)
    try:
        return _run(raw_bytes, study_id, modality_hint, notes, stages, cache, config, on_stage_result)
    finally:
        if owns_cache:
            cache.close()  # otherwise every API upload leaks an open SQLite handle


def _run(
    raw_bytes: bytes,
    study_id: str | None,
    modality_hint: Modality | None,
    notes: str | None,
    stages: list[StageSpec] | None,
    cache: StageCache,
    config: PipelineConfig,
    on_stage_result: Callable[[StageResult], None] | None,
) -> tuple[StudyResult, list[StageResult]]:
    stages = stages if stages is not None else PIPELINE

    input_sha256 = sha256_hex(raw_bytes)
    ctx = StudyContext(
        study_id=study_id or str(uuid.uuid4()),
        raw_bytes=raw_bytes,
        modality_hint=modality_hint,
        notes=notes,
        input_sha256=input_sha256,
        # Content-addressed, like the stage cache: a cached stage result refers to artifacts
        # written by an earlier study of the same image, so both must live under the input hash.
        artifact_dir=config.artifact_root / input_sha256,
        config=config,
    )

    stage_results = ctx.stage_results  # later stages read earlier results from the context
    for spec in stages:
        result = _run_stage(spec, ctx, cache, config)
        stage_results.append(result)
        if on_stage_result is not None:
            on_stage_result(result)
        if spec.required and not result.ok:
            break

    quality = next((sr.payload.get("quality", {}) for sr in stage_results if sr.stage == "intake"), {})
    if not stage_results or not stage_results[0].ok:
        quality = {**quality, "pipeline_error": stage_results[0].warnings if stage_results else ["no stages ran"]}

    study = StudyResult(
        study_id=ctx.study_id,
        input_sha256=ctx.input_sha256,
        modality=ctx.modality_hint or "other",  # the router may have set it
        ood_score=_ood_score(stage_results),
        quality=quality,
        findings=_collect_findings(stage_results),
        claims=_collect_claims(stage_results),
        ledger_head=_fold_ledger_head(stage_results),
        disclaimer=DISCLAIMER,
    )
    return study, stage_results
