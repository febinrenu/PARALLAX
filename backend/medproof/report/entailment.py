"""Entailment judge (D12): a second wall behind the firewall.

The firewall checks structure; this asks a second model whether each rendered sentence is fully
supported by the evidence it cites. A sentence that is not supported keeps its text (so the audit
view can show it struck through) but gets `entailed=False` and a `blocked_reason`, and the report
shows only claims with no `blocked_reason`. If the judge is unavailable, claims stay unchecked
(`entailed=None`) and the stage says so; an unchecked claim is never presented as checked.
"""

from __future__ import annotations

import json
import time
from typing import Any, Protocol

from pydantic import BaseModel

from medproof.core.schemas import Claim, StageResult
from medproof.llm.errors import LLMError
from medproof.report import lexicon
from medproof.report.render import display_label
from medproof.report.slots import SLOT_RE
from medproof.report.study_view import StudyView

BATCH = 6


class ClaimVerdict(BaseModel):
    claim_id: str
    entailed: bool
    reason: str = ""


class Verdicts(BaseModel):
    verdicts: list[ClaimVerdict] = []


class _Pool(Protocol):
    def chat(self, service: str, prompt_name: str, user: str, schema: type[Verdicts], **kw: object) -> Any: ...


def _second_reader(f: Any) -> str:
    if f.second_read is None:
        return "unavailable"
    return {True: "agrees", False: "disagrees", None: "inconclusive"}[f.second_read.agrees]


def notes_context(view: StudyView) -> dict[str, int]:
    """How much the notes say, so a sentence like "the notes do not mention history" can be checked."""
    counts = {"symptoms": 0, "history_or_device_or_medication": 0, "laterality": 0, "total_facts": 0}
    for t in view.text_evidence.values():
        counts["total_facts"] += 1
        if t.fact_type in ("symptom", "negation"):
            counts["symptoms"] += 1
        elif t.fact_type in ("history", "device", "medication"):
            counts["history_or_device_or_medication"] += 1
        elif t.fact_type == "laterality":
            counts["laterality"] += 1
    return counts


def evidence_for(claim: Claim, view: StudyView) -> dict[str, Any]:
    """What the judge may rely on: the findings the sentence names and the evidence it cites."""
    fids: list[str] = []
    for m in SLOT_RE.finditer(claim.template):
        ref = m.group(1)
        owner = ref if ref.startswith("f") else view.owner.get(ref)
        if owner and owner in view.findings and owner not in fids:
            fids.append(owner)
    for eid in claim.evidence_ids:
        owner = view.owner.get(eid)
        if owner and owner not in fids:
            fids.append(owner)
    findings = []
    for fid in fids:
        f = view.findings[fid]
        region = next((e.region_name for e in f.image_evidence if e.region_name), None)
        findings.append(
            {
                "id": f.finding_id,
                "label": display_label(f.modality, f.label),
                "tier": f.tier,
                "status": f.status,
                "region": region,
                "flags": f.flags,
                "flag_phrases": [lexicon.FLAG_PHRASE[x] for x in f.flags if x in lexicon.FLAG_PHRASE],
                "conformal_set": [x.replace("_", " ") for x in f.conformal_set],
                "second_reader": _second_reader(f),
            }
        )
    return {
        "findings": findings,
        "notes_context": notes_context(view),
        "image_evidence": [
            {"id": e.evidence_id, "region": e.region_name, "method": e.method, "faithful": e.faithful}
            for eid in claim.evidence_ids
            if (e := view.image_evidence.get(eid))
        ],
        "note_evidence": [
            {"id": t.evidence_id, "quote": t.quote, "polarity": t.polarity}
            for eid in claim.evidence_ids
            if (t := view.text_evidence.get(eid))
        ],
    }


def judge(claims: list[Claim], view: StudyView, pool: _Pool | None) -> tuple[list[Claim], StageResult]:
    t0 = time.perf_counter()
    todo = [c for c in claims if c.rendered and c.blocked_reason is None]
    verdicts: dict[str, ClaimVerdict] = {}
    warnings: list[str] = []
    ok = True
    if todo and pool is None:
        ok = False
        warnings.append("claims not independently checked: no judge model configured")
    elif todo:
        assert pool is not None
        for i in range(0, len(todo), BATCH):
            chunk = todo[i : i + BATCH]
            payload = json.dumps(
                {"claims": [{"claim_id": c.claim_id, "sentence": c.rendered, "evidence": evidence_for(c, view)} for c in chunk]},
                sort_keys=True,
            )
            try:
                reply = pool.chat("judge", "judge", payload, Verdicts).data
            except LLMError as exc:
                ok = False
                warnings.append(f"claims not independently checked: judge unavailable ({exc})")
                continue
            ids = {c.claim_id for c in chunk}
            for v in reply.verdicts if reply else []:
                if v.claim_id in ids:
                    verdicts[v.claim_id] = v
    out: list[Claim] = []
    entailed = rejected = 0
    for c in claims:
        v = verdicts.get(c.claim_id)
        if v is None:
            out.append(c)
        elif v.entailed:
            entailed += 1
            out.append(c.model_copy(update={"entailed": True}))
        else:
            rejected += 1
            reason = (v.reason or "not supported by the cited evidence").strip()[:160]
            out.append(c.model_copy(update={"entailed": False, "blocked_reason": f"E1_not_entailed: {reason}"}))
    unchecked = len(todo) - entailed - rejected
    if todo and unchecked and ok:
        warnings.append(f"{unchecked} claim(s) received no verdict and stay unchecked")
    stage = StageResult(
        stage="entailment", ok=ok, ms=int((time.perf_counter() - t0) * 1000),
        payload={"entailed": entailed, "not_entailed": rejected, "unchecked": unchecked}, warnings=warnings,
    )
    return out, stage
