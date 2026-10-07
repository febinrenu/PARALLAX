"""Turn findings into report claims. The language model only chooses a frame and a note reference.

Every label, number, region and quote in the final sentence comes from a slot that code fills, and
every claim goes through the firewall afterwards, so a bad choice by the model is blocked rather
than shown. With no model (offline, rate limited, key missing) a deterministic choice of frames
produces the same kind of claims, so the report never disappears.
"""

from __future__ import annotations

import json
import time
from typing import Literal, Protocol

from pydantic import BaseModel

from medproof.core.schemas import Claim, StageResult
from medproof.llm.errors import LLMError
from medproof.report.firewall import is_supported, run
from medproof.report.study_view import StudyView

Frame = Literal["consider", "supported_by_note", "discordant_review", "abstain_note", "context_missing"]
MAX_DRAFTS = 12

FRAMES: dict[str, str] = {
    "consider": "Doctor, consider {F.label} in the {F.region} ({F.tier}).",
    "supported_by_note": "Doctor, consider {F.label} ({F.tier}); the note states {N.quote}.",
    "discordant_review": "Doctor, the two readers disagree on {F.label}; please review the image directly.",
    "abstain_note": "Doctor, there is no confident reading for {F.label}; clinical correlation is needed.",
    "context_missing": "Doctor, the notes do not mention history relevant to {F.label}; please add context.",
}
_PRIORITY = {"verified": 0, "uncertain": 1, "discordant": 2}


class ClaimDraft(BaseModel):
    frame: Frame
    finding: str
    note_ref: str | None = None


class ClaimDrafts(BaseModel):
    drafts: list[ClaimDraft] = []


class _Pool(Protocol):
    def chat(self, service: str, prompt_name: str, user: str, schema: type[ClaimDrafts], **kw: object) -> object: ...


def _first_evidence(view: StudyView, fid: str) -> list[str]:
    f = view.findings.get(fid)
    if f is None:
        return []
    return [e.evidence_id for e in f.image_evidence[:1]] or [t.evidence_id for t in f.text_evidence[:1]]


def to_claim(draft: ClaimDraft, n: int, view: StudyView) -> Claim:
    """Expand a draft into a slot template. Unknown ids are kept so the firewall can name the reason."""
    template = FRAMES[draft.frame].replace("{F.", "{" + draft.finding + ".")
    ids = _first_evidence(view, draft.finding)
    if draft.frame == "supported_by_note":
        ref = draft.note_ref or "te_0"
        template = template.replace("{N.", "{" + ref + ".")
        ids = [*ids, ref]
    return Claim(claim_id=f"c{n}", template=template, evidence_ids=ids or ["ie_missing"])


def template_drafts(view: StudyView) -> ClaimDrafts:
    """The offline choice: one claim per supported, non-rejected finding, most reliable first."""
    chosen: list[ClaimDraft] = []
    order = sorted(
        (f for f in view.findings.values() if f.status != "rejected" and is_supported(f)),
        key=lambda f: (_PRIORITY.get(f.status, 3), f.finding_id),
    )
    for f in order:
        support = next((t.evidence_id for t in f.text_evidence if t.polarity == "supports"), None)
        if f.status == "discordant":
            chosen.append(ClaimDraft(frame="discordant_review", finding=f.finding_id))
        elif f.tier == "abstain":
            chosen.append(ClaimDraft(frame="abstain_note", finding=f.finding_id))
        elif support:
            chosen.append(ClaimDraft(frame="supported_by_note", finding=f.finding_id, note_ref=support))
        else:
            chosen.append(ClaimDraft(frame="consider", finding=f.finding_id))
    return ClaimDrafts(drafts=chosen)


def build_input(view: StudyView) -> str:
    """What the model sees: ids, tiers, statuses and note-evidence polarity. No labels, quotes or regions."""
    items = []
    for f in view.findings.values():
        if f.status == "rejected":
            continue
        items.append(
            {
                "id": f.finding_id,
                "tier": f.tier,
                "status": f.status,
                "flags": f.flags,
                "second_read_agrees": f.second_read.agrees if f.second_read else None,
                "supported": is_supported(f),
                "notes": [{"id": t.evidence_id, "polarity": t.polarity} for t in f.text_evidence],
            }
        )
    return json.dumps({"findings": items}, sort_keys=True)


def draft_claims(view: StudyView, pool: _Pool | None) -> tuple[list[Claim], StageResult]:
    t0 = time.perf_counter()
    warnings: list[str] = []
    if not view.findings:
        return [], StageResult(stage="report", ok=True, ms=0, payload={"rendered": 0, "blocked": 0, "reasons": {}})
    drafts = template_drafts(view)
    if pool is None:
        warnings.append("template-only report: no language model configured")
    else:
        try:
            res = pool.chat("report", "report", build_input(view), ClaimDrafts, fallback=lambda: template_drafts(view))
            drafts = res.data  # type: ignore[attr-defined]
            if getattr(res, "source", "live") == "fallback":
                warnings.append(f"template-only report: {getattr(res, 'warning', 'language model unavailable')}")
        except LLMError as exc:
            warnings.append(f"template-only report: {exc}")
    claims = [to_claim(d, i, view) for i, d in enumerate(drafts.drafts[:MAX_DRAFTS], 1)]
    out, stage = run(claims, view)
    stage.warnings.extend(warnings)
    stage.ms = int((time.perf_counter() - t0) * 1000)
    return out, stage
