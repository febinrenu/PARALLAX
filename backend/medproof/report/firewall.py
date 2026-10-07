"""Hallucination firewall: deterministic rules every claim must pass before a doctor sees it.

Each rule has a stable code that the audit view shows. Rules R1 and R2 stop the check (nothing
can be resolved past a bad parse); R3 to R11 all run so the audit lists every reason.
R12 (duplicates, size cap) is applied across claims in `run`.
"""

from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import dataclass, field

from medproof.core.schemas import Claim, Finding, StageResult
from medproof.report import lexicon
from medproof.report.render import render
from medproof.report.slots import (
    Parsed,
    SlotError,
    TemplateSyntaxError,
    check_fields,
    parse_template,
)
from medproof.report.study_view import StudyView

MAX_CLAIMS = 6
STATUS_PRIORITY = {"verified": 0, "uncertain": 1, "discordant": 2, "rejected": 3}


@dataclass
class Verdict:
    reasons: list[str] = field(default_factory=list)
    rendered: str | None = None

    @property
    def blocked_reason(self) -> str | None:
        return self.reasons[0] if self.reasons else None


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().lower()


def is_supported(f: Finding) -> bool:
    """Same evidence requirement as the status rule: faithful image evidence or supporting text."""
    return any(e.faithful is True for e in f.image_evidence) or any(
        t.polarity == "supports" for t in f.text_evidence
    )


def _resolve(parsed: Parsed, view: StudyView) -> None:
    for s in parsed.slots:
        table = view.findings if s.kind == "f" else view.text_evidence if s.kind == "te" else view.image_evidence
        if s.ref not in table:
            raise SlotError(f"unknown ref '{s.ref}'")


def _confidence_rules(parsed: Parsed, text: str, view: StudyView) -> bool:
    """True when the wording is inconsistent with the findings' tiers and statuses."""
    if any(p in text for p in lexicon.CONFIDENCE_LITERALS):
        return True
    for s in {s.ref for s in parsed.slots if s.kind == "f"}:
        f = view.findings[s]
        cautious = f.tier in ("low", "abstain") or f.status in ("uncertain", "discordant")
        if cautious and lexicon.STRONG_WORDS.search(text):
            return True
        if f.tier == "high" and f.status == "verified" and lexicon.HEDGE_WORDS.search(text):
            return True
        if f.tier == "abstain" and lexicon.SUGGEST_WORDS.search(text):
            return True
        if f.status == "discordant" and not lexicon.REVIEW_WORDS.search(text):
            return True
    return False


def check(claim: Claim, view: StudyView) -> Verdict:
    v = Verdict()
    try:
        parsed = parse_template(claim.template)
    except TemplateSyntaxError:
        v.reasons.append("R1_slot_syntax")
        return v
    try:
        check_fields(parsed)
        _resolve(parsed, view)
    except SlotError:
        v.reasons.append("R2_unknown_slot")
        return v

    ids = list(claim.evidence_ids)
    if any(i not in view.owner for i in ids):
        v.reasons.append("R3_evidence_exists")

    cited_owners = {view.owner[i] for i in ids if i in view.owner}
    finding_refs = {s.ref for s in parsed.slots if s.kind == "f"}
    r4 = any(view.findings[o].status == "rejected" for o in cited_owners)
    for ref in finding_refs:
        r4 = r4 or view.findings[ref].status == "rejected" or ref not in cited_owners
    for s in parsed.slots:
        if s.kind in ("te", "ie") and (s.ref not in ids or view.owner.get(s.ref) not in finding_refs):
            r4 = True
    if r4:
        v.reasons.append("R4_evidence_owner")

    if any(not is_supported(view.findings[ref]) for ref in finding_refs):
        v.reasons.append("R5_support")

    cited_text = [view.text_evidence[i] for i in ids if i in view.text_evidence]
    slot_text = [view.text_evidence[s.ref] for s in parsed.slots if s.kind == "te"]
    for te in {t.evidence_id: t for t in [*cited_text, *slot_text]}.values():
        note = view.notes.get(te.note_id)
        if note is None or note[te.span[0] : te.span[1]] != te.quote:
            v.reasons.append("R6_quote_integrity")
            break

    text = _norm(parsed.text)
    if lexicon.VOCAB_RE.search(text):
        v.reasons.append("R7_vocab_outside_slot")
    if _confidence_rules(parsed, text, view):
        v.reasons.append("R8_confidence_lexicon")
    if any(p in text for p in lexicon.BANNED_PHRASES):
        v.reasons.append("R9_banned_phrases")
    if not claim.template.startswith("Doctor,"):
        v.reasons.append("R10_framing")
    for te in slot_text:
        if not view.is_flagged(te) and lexicon.INSTRUCTION_RE.search(te.quote):
            v.reasons.append("R11_injection_echo")
            break

    if not v.reasons:
        v.rendered = render(parsed, view)
    return v


def _priority(claim: Claim, view: StudyView) -> int:
    try:
        refs = {s.ref for s in parse_template(claim.template).slots if s.kind == "f"}
        return max((STATUS_PRIORITY[view.findings[r].status] for r in refs), default=3)
    except (TemplateSyntaxError, KeyError):
        return 3


def run(claims: list[Claim], view: StudyView) -> tuple[list[Claim], StageResult]:
    """Check every claim, fill `rendered` or `blocked_reason`, never raise."""
    t0 = time.perf_counter()
    verdicts: list[Verdict] = []
    for c in claims:
        try:
            verdicts.append(check(c, view))
        except Exception:  # noqa: BLE001 - one bad claim must not take the report stage down
            verdicts.append(Verdict(reasons=["R0_internal_error"]))

    seen: set[str] = set()
    for v in verdicts:
        if v.rendered is not None:
            if v.rendered in seen:
                v.reasons.append("R12_duplicate")
                v.rendered = None
            else:
                seen.add(v.rendered)

    passing = [i for i, v in enumerate(verdicts) if v.rendered is not None]
    if len(passing) > MAX_CLAIMS:
        ranked = sorted(passing, key=lambda i: (_priority(claims[i], view), i))
        for i in ranked[MAX_CLAIMS:]:
            verdicts[i].reasons.append("R12_over_cap")
            verdicts[i].rendered = None

    out: list[Claim] = []
    reason_counts: dict[str, int] = {}
    warnings: list[str] = []
    for c, v in zip(claims, verdicts, strict=True):
        out.append(c.model_copy(update={"rendered": v.rendered, "blocked_reason": v.blocked_reason}))
        if v.blocked_reason:
            reason_counts[v.blocked_reason] = reason_counts.get(v.blocked_reason, 0) + 1
            warnings.append(f"claim {getattr(c, 'claim_id', '?')} blocked: {v.blocked_reason}")
    payload = {
        "rendered": sum(1 for v in verdicts if v.rendered is not None),
        "blocked": sum(1 for v in verdicts if v.blocked_reason),
        "reasons": reason_counts,
    }
    return out, StageResult(
        stage="report", ok=True, ms=int((time.perf_counter() - t0) * 1000), payload=payload, warnings=warnings
    )
