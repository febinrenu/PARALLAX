"""One call that runs the clinical-reasoning stages in order and never crashes the study.

    concordance -> context (guard, extraction, contradictions) -> report (drafts + firewall)
    -> entailment -> FHIR

It takes plain arguments instead of a study context so it can be wired into the pipeline as one
stage, or called stage by stage. Any stage that fails is recorded as a failed StageResult and the
rest carry on with what they have: no second read, no note evidence, a template-only report,
unchecked claims. The inputs are never modified.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from medproof.context.contradictions import Demographics, check, facts_with_cues
from medproof.context.extract import ExtractedFact, extract
from medproof.context.injection_guard import InjectionGuard
from medproof.core.schemas import Claim, Finding, StageResult
from medproof.readers.generalist import GeneralistRead
from medproof.report.drafts import draft_claims
from medproof.report.entailment import judge
from medproof.report.fhir import build_bundle
from medproof.report.study_view import StudyView
from medproof.verify.concordance import apply as apply_concordance


@dataclass
class ReasoningResult:
    findings: list[Finding]
    claims: list[Claim]
    fhir: dict[str, Any]
    stages: list[StageResult] = field(default_factory=list)
    missing_context: list[str] = field(default_factory=list)
    flagged_spans: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _failed(stage: str, t0: float, exc: Exception) -> StageResult:
    msg = f"{stage} stage failed ({type(exc).__name__}); continuing without it"
    return StageResult(stage=stage, ok=False, ms=int((time.perf_counter() - t0) * 1000), payload={"error": type(exc).__name__}, warnings=[msg])


@dataclass
class ContextOutput:
    findings: list[Finding]
    missing_context: list[str]
    flagged_spans: dict[str, list[tuple[int, int]]]
    stage: StageResult


def run_context(
    findings: list[Finding],
    notes: dict[str, str],
    demographics: Demographics,
    text_pool: Any,
    *,
    evidence_start: int = 1,
) -> ContextOutput:
    """Guard each note, extract its facts, add the closed-class cues, and check them against the findings."""
    t0 = time.perf_counter()
    facts: list[ExtractedFact] = []
    stage_warnings: list[str] = []
    flagged: dict[str, list[tuple[int, int]]] = {}
    ok = True
    guard = InjectionGuard(text_pool)
    for note_id, text in notes.items():
        g = guard.check(text)
        stage_warnings.extend(g.warnings)
        if g.flagged:
            flagged[note_id] = g.spans()
        got = extract(text, note_id, text_pool, g) if text_pool is not None else None
        if got is None or not got.ok:
            ok = False
            stage_warnings.extend(got.warnings if got else ["fact extraction unavailable: no language model configured"])
        facts.extend(facts_with_cues(got.facts if got is not None else [], text, note_id))
    res = check(findings, facts, demographics, evidence_start=evidence_start)
    st = res.stage.model_copy(update={"ok": ok, "warnings": [*stage_warnings, *res.stage.warnings],
                                      "ms": int((time.perf_counter() - t0) * 1000)})
    st.payload["flagged_notes"] = len(flagged)
    return ContextOutput(res.findings, res.missing_context, flagged, st)


def run_report(
    findings: list[Finding],
    notes: dict[str, str],
    flagged: dict[str, list[tuple[int, int]]],
    report_pool: Any,
    judge_pool: Any,
) -> tuple[list[Claim], list[StageResult]]:
    """Draft claims (the firewall runs inside), then have them checked by the entailment judge.

    A failed draft gives no claims; a failed judge keeps the claims, unchecked."""
    stages: list[StageResult] = []
    view = StudyView.build(findings, notes, flagged)
    t0 = time.perf_counter()
    try:
        claims, st = draft_claims(view, report_pool)
        stages.append(st)
    except Exception as exc:  # noqa: BLE001 - a stage must never take the study down
        return [], [_failed("report", t0, exc)]
    t0 = time.perf_counter()
    try:
        claims, st = judge(claims, view, judge_pool)
        stages.append(st)
    except Exception as exc:  # noqa: BLE001
        stages.append(_failed("entailment", t0, exc))
    return claims, stages


def run_reasoning(
    findings: list[Finding],
    *,
    notes: dict[str, str],
    demographics: Demographics,
    generalist_read: GeneralistRead | None,
    text_pool: Any,
    report_pool: Any,
    judge_pool: Any,
    study_id: str,
    issued: datetime,
    ledger_head: str,
    include_quotes: bool = False,
    evidence_start: int = 1,
) -> ReasoningResult:
    stages: list[StageResult] = []
    current = list(findings)
    claims: list[Claim] = []
    missing: list[str] = []
    flagged: dict[str, list[tuple[int, int]]] = {}

    def guarded(name: str, fn: Callable[[], None]) -> None:
        t0 = time.perf_counter()
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - a stage must never take the study down
            stages.append(_failed(name, t0, exc))

    def concordance() -> None:
        nonlocal current
        assert generalist_read is not None
        current, st = apply_concordance(current, generalist_read)
        stages.append(st)

    if generalist_read is not None:
        guarded("concordance", concordance)

    def context() -> None:
        nonlocal current, missing, flagged
        out = run_context(current, notes, demographics, text_pool, evidence_start=evidence_start)
        current, missing, flagged = out.findings, out.missing_context, out.flagged_spans
        stages.append(out.stage)

    guarded("context", context)

    def report() -> None:
        nonlocal claims
        claims, sts = run_report(current, notes, flagged, report_pool, judge_pool)
        stages.extend(sts)

    guarded("report", report)

    fhir: dict[str, Any] = {}

    def export() -> None:
        nonlocal fhir
        t0 = time.perf_counter()
        fhir = build_bundle(current, claims, study_id=study_id, issued=issued, ledger_head=ledger_head, include_quotes=include_quotes)
        stages.append(StageResult(stage="fhir", ok=True, ms=int((time.perf_counter() - t0) * 1000),
                                  payload={"entries": len(fhir.get("entry", []))}))

    guarded("fhir", export)
    warnings: list[str] = []
    for st in stages:
        warnings.extend(st.warnings)
    return ReasoningResult(current, claims, fhir, stages, missing, flagged, warnings)
