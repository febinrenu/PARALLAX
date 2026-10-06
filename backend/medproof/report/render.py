"""Deterministic renderer: fills slots from the study. The LLM never supplies a value."""

from __future__ import annotations

import re

from medproof.core.schemas import Finding
from medproof.report import lexicon
from medproof.report.slots import Parsed, Slot, SlotError
from medproof.report.study_view import StudyView

UNSPECIFIED_REGION = "unspecified region"


def _region(f: Finding) -> str:
    for ev in f.image_evidence:
        if ev.region_name:
            return ev.region_name
    return UNSPECIFIED_REGION


def _second_read(f: Finding) -> str:
    sr = f.second_read
    if sr is None:
        return "second read unavailable"
    if sr.agrees is True:
        return "second reader agrees"
    if sr.agrees is False:
        return "second reader disagrees"
    return "second read inconclusive"


def _flags(f: Finding) -> str:
    phrases = [lexicon.FLAG_PHRASE[x] for x in f.flags if x in lexicon.FLAG_PHRASE]
    return ", ".join(phrases) if phrases else "no additional flags"


def _finding_field(f: Finding, name: str) -> str:
    if name == "label":
        return f.label.replace("_", " ")
    if name == "region":
        return _region(f)
    if name == "tier":
        return lexicon.TIER_PHRASE[f.tier]
    if name == "status":
        return lexicon.STATUS_PHRASE[f.status]
    if name == "conformal_set":
        return ", ".join(x.replace("_", " ") for x in f.conformal_set) or "no set recorded"
    if name == "second_read":
        return _second_read(f)
    if name == "flag_phrase":
        return _flags(f)
    raise SlotError(f"unknown finding field '{name}'")


def slot_value(slot: Slot, view: StudyView) -> str:
    if slot.kind == "f":
        f = view.findings.get(slot.ref)
        if f is None:
            raise SlotError(f"unknown finding '{slot.ref}'")
        return _finding_field(f, slot.field)
    if slot.kind == "te":
        te = view.text_evidence.get(slot.ref)
        if te is None:
            raise SlotError(f"unknown text evidence '{slot.ref}'")
        if view.is_flagged(te):
            return lexicon.WITHHELD
        return '"' + re.sub(r"\s+", " ", te.quote).strip() + '"'
    ie = view.image_evidence.get(slot.ref)
    if ie is None:
        raise SlotError(f"unknown image evidence '{slot.ref}'")
    return ie.region_name or UNSPECIFIED_REGION


def render(parsed: Parsed, view: StudyView) -> str:
    out: list[str] = []
    for part in parsed.parts:
        out.append(part if isinstance(part, str) else slot_value(part, view))
    return "".join(out)
