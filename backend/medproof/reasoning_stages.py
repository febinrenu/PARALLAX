"""Pipeline-facing adapters for the clinical-reasoning stages (P3).

Each step is `run(ctx) -> StageResult` and never raises. A step reads the findings that earlier stages
produced from `ctx.stage_results` (the orchestrator appends every StageResult there), and returns the
updated findings under `payload["findings"]`; the report step returns `payload["claims"]` instead and
never re-emits findings. Two pure helpers, `merge_findings` and `collect_claims`, let the orchestrator fold
those payloads into the StudyResult: one finding per id (the latest stage wins), and the claims list.

All four are `cacheable=False`: their output depends on the notes and on live services, while the stage
cache is keyed by the image hash alone. The services that do the slow work (MedGemma reads, the language
model pool, the precedent embedder) keep their own caches.

External services are reached through `Services`, so tests and the orchestrator can replace them. Defaults
read the environment: GROQ_KEY_* for the language models, MEDGEMMA_URL / MEDGEMMA_SEED_DIR for the second
reader, MEDPROOF_RETRIEVAL_DIR for the precedent indices. Missing configuration degrades with a warning.
"""

from __future__ import annotations

import functools
import os
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from medproof.context.contradictions import Demographics
from medproof.core.config import PipelineConfig
from medproof.core.schemas import Claim, Finding, Precedent, StageResult
from medproof.intake.decode import DecodedImage
from medproof.reasoning import run_context, run_report
from medproof.verify.concordance import apply as apply_concordance

NOTE_ID = "n1"  # the single uploaded note; P4's fixtures and notes panel use the same id
REPO_ROOT = Path(__file__).resolve().parents[2]
PRECEDENT_DATASET = {"skin_dermoscopy": "ham10000", "bone_xray": "fracatlas", "brain_mri": "brain_mri"}
PRECEDENT_EMBEDDER = "medsiglip-448"
PRECEDENT_K = 5


# ---------------------------------------------------------------- helpers the orchestrator can reuse

def merge_findings(stage_results: Iterable[StageResult]) -> list[Finding]:
    """One finding per id across all stages, in first-seen order; a later stage's version replaces an earlier one."""
    by_id: dict[str, Finding] = {}
    for sr in stage_results:
        for raw in sr.payload.get("findings", []) or []:
            try:
                f = raw if isinstance(raw, Finding) else Finding.model_validate(raw)
            except ValueError:
                continue  # a malformed entry must not lose the rest
            by_id[f.finding_id] = f
    return list(by_id.values())


def collect_claims(stage_results: Iterable[StageResult]) -> list[Claim]:
    claims: list[Claim] = []
    for sr in stage_results:
        for raw in sr.payload.get("claims", []) or []:
            try:
                claims.append(raw if isinstance(raw, Claim) else Claim.model_validate(raw))
            except ValueError:
                continue
    return claims


def prior_results(ctx: Any) -> list[StageResult]:
    return list(getattr(ctx, "stage_results", None) or [])


def _reader_findings(ctx: Any) -> list[Finding]:
    """The contract findings the reader stage leaves on the context (a snapshot taken before verification)."""
    out: list[Finding] = []
    raw = getattr(ctx, "findings", None)
    for item in raw if isinstance(raw, list) else []:
        try:
            out.append(item if isinstance(item, Finding) else Finding.model_validate(item))
        except ValueError:
            continue
    return out


def prior_findings(ctx: Any) -> list[Finding]:
    """Findings from earlier stages. `ctx.stage_results` is preferred because it includes what later stages
    changed (faithfulness, stability); without it, the reader's own findings on `ctx.findings` are used."""
    return merge_findings(prior_results(ctx)) or _reader_findings(ctx)


def _notes(ctx: Any) -> dict[str, str]:
    text = getattr(ctx, "notes", None)
    return {NOTE_ID: text} if isinstance(text, str) and text.strip() else {}


def _flagged_spans(ctx: Any) -> dict[str, list[tuple[int, int]]]:
    flagged: dict[str, list[tuple[int, int]]] = {}
    for sr in prior_results(ctx):
        for note_id, spans in (sr.payload.get("flagged_spans") or {}).items():
            flagged[note_id] = [(int(a), int(b)) for a, b in spans]
    return flagged


def _modality(ctx: Any, findings: list[Finding]) -> str:
    hint = getattr(ctx, "modality_hint", None)
    return str(hint) if hint else (findings[0].modality if findings else "other")


def _dump(findings: list[Finding]) -> list[dict[str, Any]]:
    return [f.model_dump() for f in findings]


def _done(stage: str, t0: float, ok: bool, payload: dict[str, Any], warnings: list[str] | None = None) -> StageResult:
    return StageResult(stage=stage, ok=ok, ms=int((time.perf_counter() - t0) * 1000), payload=payload, warnings=warnings or [])


def _crashed(stage: str, t0: float, exc: Exception, findings: list[Finding]) -> StageResult:
    msg = f"{stage} stage failed ({type(exc).__name__}); continuing without it"
    return _done(stage, t0, False, {"error": type(exc).__name__, "findings": _dump(findings)}, [msg])


# ---------------------------------------------------------------- default services

@functools.lru_cache(maxsize=1)
def _default_llm_pool() -> Any:
    from medproof.llm.config import LLMConfig
    from medproof.llm.groq_pool import GroqPool

    cfg = LLMConfig(cache_dir=PipelineConfig().cache_dir / "llm")
    if not any(os.environ.get(var) for var in cfg.key_env.values()):
        return None
    return GroqPool(cfg)


def _default_generalist() -> Any:
    from medproof.readers.generalist import GeneralistReader

    seed = os.environ.get("MEDGEMMA_SEED_DIR") or (str(REPO_ROOT / "demo" / "medgemma_reads") if (REPO_ROOT / "demo" / "medgemma_reads").is_dir() else None)
    return GeneralistReader(cache_dir=PipelineConfig().cache_dir / "medgemma", seed_dir=seed)


@functools.lru_cache(maxsize=1)
def _embedder() -> Any:
    from medproof.retrieval.embedder import MedSigLipEmbedder

    return MedSigLipEmbedder()


@functools.lru_cache(maxsize=4)
def _index(dataset: str, directory: str) -> Any:
    from medproof.retrieval.index import PrecedentIndex

    return PrecedentIndex.load(Path(directory) / f"{dataset}_{PRECEDENT_EMBEDDER}", expect_embedder=PRECEDENT_EMBEDDER)


def _default_precedents(modality: str, image: DecodedImage) -> tuple[list[Precedent], str]:
    """Nearest confirmed training cases for the image. Returns (precedents, "") or ([], why it is unavailable)."""
    dataset = PRECEDENT_DATASET.get(modality)
    if dataset is None:
        return [], f"no precedent index for {modality}"
    directory = os.environ.get("MEDPROOF_RETRIEVAL_DIR") or str(REPO_ROOT / "ml" / "artifacts" / "retrieval")
    try:
        index = _index(dataset, directory)
    except (OSError, ValueError) as exc:
        return [], f"precedent index for {dataset} unavailable ({type(exc).__name__})"
    from PIL import Image

    from medproof.retrieval.index import to_precedents

    pil = Image.fromarray(image.display).convert("RGB")
    vec = _embedder().embed([pil])[0]
    hits = index.search(vec, PRECEDENT_K)
    return to_precedents(hits, dataset, lambda h: f"ml/artifacts/retrieval/thumbs/{dataset}/{h.id}.jpg"), ""


@dataclass
class Services:
    llm_pool: Callable[[], Any] = _default_llm_pool
    generalist: Callable[[], Any] = _default_generalist
    precedents: Callable[[str, DecodedImage], tuple[list[Precedent], str]] = field(default=_default_precedents)


# ---------------------------------------------------------------- the steps

def second_read_step(ctx: Any, services: Services | None = None) -> StageResult:
    services = services or Services()
    t0 = time.perf_counter()
    findings = prior_findings(ctx)
    image = getattr(ctx, "decoded", None)
    if not findings:
        return _done("second_read", t0, True, {"findings": []}, ["no findings to compare with a second read"])
    if image is None:
        return _done("second_read", t0, False, {"findings": _dump(findings)}, ["second read skipped: no decoded image"])
    reader = None
    try:
        reader = services.generalist()
        if reader is None:
            return _done("second_read", t0, False, {"findings": _dump(findings)}, ["second read unavailable: no reader configured"])
        read = reader.read(image, _modality(ctx, findings))
        updated, st = apply_concordance(findings, read)
        payload = {**st.payload, "findings": _dump(updated), "source": read.source}
        return _done("second_read", t0, st.ok, payload, list(st.warnings))
    except Exception as exc:  # noqa: BLE001 - a stage must never take the study down
        return _crashed("second_read", t0, exc, findings)
    finally:
        if reader is not None and hasattr(reader, "close"):
            try:
                reader.close()
            except Exception:  # noqa: BLE001, S110 - closing is best effort
                pass


def context_step(ctx: Any, services: Services | None = None) -> StageResult:
    services = services or Services()
    t0 = time.perf_counter()
    findings = prior_findings(ctx)
    try:
        out = run_context(findings, _notes(ctx), Demographics(), services.llm_pool())
    except Exception as exc:  # noqa: BLE001
        return _crashed("context", t0, exc, findings)
    payload = {**out.stage.payload, "findings": _dump(out.findings), "missing_context": out.missing_context,
               "flagged_spans": {k: [list(s) for s in v] for k, v in out.flagged_spans.items()}}
    return _done("context", t0, out.stage.ok, payload, list(out.stage.warnings))


def report_step(ctx: Any, services: Services | None = None) -> StageResult:
    services = services or Services()
    t0 = time.perf_counter()
    findings = prior_findings(ctx)
    try:
        pool = services.llm_pool()
        claims, stages = run_report(findings, _notes(ctx), _flagged_spans(ctx), pool, pool)
    except Exception as exc:  # noqa: BLE001
        msg = f"report stage failed ({type(exc).__name__}); no report"
        return _done("report", t0, False, {"error": type(exc).__name__, "claims": []}, [msg])
    warnings = [w for st in stages for w in st.warnings]
    ok = all(st.ok for st in stages)
    payload: dict[str, Any] = {"claims": [c.model_dump() for c in claims], "stages": {st.stage: st.payload for st in stages}}
    return _done("report", t0, ok, payload, warnings)


def precedents_step(ctx: Any, services: Services | None = None) -> StageResult:
    services = services or Services()
    t0 = time.perf_counter()
    findings = prior_findings(ctx)
    image = getattr(ctx, "decoded", None)
    if not findings or image is None:
        return _done("precedents", t0, bool(not findings), {"findings": _dump(findings)}, [] if not findings else ["precedents skipped: no decoded image"])
    try:
        modality = _modality(ctx, findings)
        found, why = services.precedents(modality, image)
    except Exception as exc:  # noqa: BLE001
        return _crashed("precedents", t0, exc, findings)
    if not found:
        return _done("precedents", t0, False, {"findings": _dump(findings)}, [why or "no precedents found"])
    updated = [f.model_copy(update={"precedents": list(found)}) if f.modality == modality else f for f in findings]
    return _done("precedents", t0, True, {"findings": _dump(updated), "precedents": len(found)})


# ---------------------------------------------------------------- registry

def stage_specs(services: Services | None = None) -> list[Any]:
    """StageSpecs for P4's PIPELINE, in order: `PIPELINE.extend(stage_specs())`.

    Imported lazily so this module has no import-time dependency on the orchestrator."""
    from medproof.pipeline import StageSpec

    services = services or Services()
    steps: list[tuple[str, Callable[..., StageResult], float]] = [
        ("second_read", second_read_step, 130.0),  # a live MedGemma read is ~6-11 s on a laptop GPU; the client allows 120 s
        ("context", context_step, 25.0),
        ("report", report_step, 45.0),
        ("precedents", precedents_step, 30.0),
    ]
    return [
        StageSpec(name=name, run=functools.partial(fn, services=services), timeout_s=timeout, cache_version="v1", cacheable=False)
        for name, fn, timeout in steps
    ]
