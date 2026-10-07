"""Clinical consistency checks between extracted note facts and image findings (D7).

Deterministic rules only, so every flag can be explained from the note text and the image region:
  laterality   note side vs image side, in patient-side convention (the reader already names regions that way);
  history      e.g. "right pneumonectomy" with a chest finding in the right lung;
  demographic  pregnancy or menopause that cannot fit the age or sex, or an age/sex that disagrees with the record;
  symptom      "asymptomatic" beside a large, confident finding.
It also attaches the note facts that bear on each finding as TextEvidence (supports, contradicts or
neutral), which is what lets the status rule count a note as support, and lists missing context.
"""

from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal

from medproof.context.extract import ExtractedFact
from medproof.core.schemas import Finding, StageResult, TextEvidence

Polarity = Literal["supports", "contradicts", "neutral"]


@dataclass(frozen=True)
class Demographics:
    age: int | None = None
    sex: Literal["M", "F"] | None = None


@dataclass
class ContextResult:
    findings: list[Finding]
    missing_context: list[str]
    stage: StageResult = field(default=None)  # type: ignore[assignment]


_RIGHT = re.compile(r"\b(?:right|rt|r)\b", re.IGNORECASE)
_LEFT = re.compile(r"\b(?:left|lt|l)\b", re.IGNORECASE)
_BOTH = re.compile(r"\b(?:bilateral|both)\b", re.IGNORECASE)
_PNEUMONECTOMY = re.compile(r"\b(left|right|rt|lt)\s+(?:total\s+)?pneumonectomy\b", re.IGNORECASE)
_PREGNANCY = re.compile(r"pregnan|gravida|antenatal|\bweeks pregnant\b", re.IGNORECASE)
_MENOPAUSE = re.compile(r"menopaus", re.IGNORECASE)
_AGE_SEX = re.compile(r"(\d{1,3})\s*(?:y/?o|yo|years?|yrs?|ans|años)?\s*([MFmf])?")
_LARGE = {"Effusion", "Pneumothorax", "Consolidation", "Pneumonia", "Edema", "Mass"}

_SUPPORTING: dict[str, tuple[str, ...]] = {
    **dict.fromkeys(("Consolidation", "Pneumonia", "Infiltration", "Lung Opacity"), ("fever", "cough", "sputum", "dyspnea", "chills")),
    "Effusion": ("dyspnea", "pleuritic", "chest pain", "orthopnea"),
    "Pneumothorax": ("chest pain", "dyspnea"),
    **dict.fromkeys(("Nodule", "Mass", "Lung Lesion"), ("weight loss", "hemoptysis", "cough")),
    **dict.fromkeys(("Edema", "Cardiomegaly"), ("orthopnea", "dyspnea", "swelling")),
    "Atelectasis": ("cough", "dyspnea", "post-operative"),
    "fracture": ("pain", "swelling", "deformity", "bruising", "fall", "weight"),
    **dict.fromkeys(("mel", "nv", "bkl", "bcc", "akiec", "df", "vasc"), ("itching", "bleeding", "growth", "colour change", "changing", "tenderness")),
    **dict.fromkeys(("glioma", "meningioma", "pituitary"), ("headache", "seizure", "weakness", "vision", "vomiting")),
}


# Closed-class cues read straight from the note text. The model tends to skip statements that look
# odd (pregnancy in a man, "asymptomatic"), and those are exactly the ones the rules need, so the
# phrases are also found deterministically, with exact spans and no model in the loop.
_NEG_BEFORE = re.compile(r"\b(?:not|no|never|denies|denied|without)\W+(?:\w+\W+){0,2}$", re.IGNORECASE)
_CUES: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    ("demographic", "pregnant", re.compile(
        r"\b(?:(?:\d+\s*(?:weeks?|wks?)\s*(?:pregnant|gestation))|pregnant|primigravida|multigravida|gravid(?:a|ity)|G\d+\s*P\d+)\b", re.IGNORECASE)),
    ("demographic", "post-menopausal", re.compile(r"\b(?:post|peri)?-?menopaus\w*\b", re.IGNORECASE)),
    ("symptom", "asymptomatic", re.compile(r"\b(?:asymptomatic|symptom[- ]free|no symptoms?|(?:denies|reports no) (?:any )?(?:symptoms|complaints))\b", re.IGNORECASE)),
    ("history", "pneumonectomy", re.compile(r"\b(?:(?:left|right|rt|lt)\s+)?(?:total\s+|complete\s+)?pneumonectomy\b", re.IGNORECASE)),
)
_ASYMPTOMATIC = re.compile(r"asymptomatic|symptom[- ]free|no symptoms?|(?:denies|reports no) (?:any )?(?:symptoms|complaints)", re.IGNORECASE)


def scan_cues(text: str, note_id: str) -> list[ExtractedFact]:
    out: list[ExtractedFact] = []
    for ftype, value, pat in _CUES:
        for m in pat.finditer(text):
            if ftype == "demographic" and value == "pregnant" and _NEG_BEFORE.search(text[: m.start()]):
                continue
            polarity = "absent" if ftype == "symptom" else ("historical" if ftype == "history" else "present")
            out.append(ExtractedFact(note_id, (m.start(), m.end()), m.group(0), ftype, value, polarity))
    return sorted(out, key=lambda f: f.span)


def _merge_cues(facts: list[ExtractedFact], cues: list[ExtractedFact]) -> list[ExtractedFact]:
    """Cues win over any model fact they overlap (they carry the side word or the exact phrase)."""
    kept = [f for f in facts if not any(f.span[0] < c.span[1] and c.span[0] < f.span[1] for c in cues)]
    return sorted([*kept, *cues], key=lambda f: f.span)


def facts_with_cues(facts: list[ExtractedFact], text: str, note_id: str) -> list[ExtractedFact]:
    """The model's facts plus the closed-class cues found in the text itself (cues win where they overlap)."""
    return _merge_cues(facts, scan_cues(text, note_id))


def normalise_side(raw: str) -> Literal["left", "right"] | None:
    """'Rt', 'right-sided', "patient's left" -> a side; bilateral or no side -> None."""
    text = raw.replace("'s", "").replace("-sided", "")
    if _BOTH.search(text):
        return None
    right, left = bool(_RIGHT.search(text)), bool(_LEFT.search(text))
    if right == left:
        return None
    return "right" if right else "left"


def _finding_side(f: Finding) -> Literal["left", "right"] | None:
    for ev in f.image_evidence:
        if ev.region_name:
            side = normalise_side(ev.region_name)
            if side:
                return side
    return None


def _text(fact: ExtractedFact) -> str:
    return f"{fact.value} {fact.quote}".lower()


def _demographic_flags(facts: list[ExtractedFact], demo: Demographics) -> set[str]:
    flags: set[str] = set()
    for fact in facts:
        t = _text(fact)
        # the model may file pregnancy or menopause under history or symptom, so the type is not trusted
        if fact.fact_type in ("demographic", "history", "symptom"):
            if _PREGNANCY.search(t) and (demo.sex == "M" or (demo.age is not None and (demo.age < 10 or demo.age >= 60))):
                flags.add("demographic_implausible")
            if _MENOPAUSE.search(t) and (demo.sex == "M" or (demo.age is not None and demo.age < 35)):
                flags.add("demographic_implausible")
        if fact.fact_type != "demographic":
            continue
        m = _AGE_SEX.fullmatch(fact.quote.strip())
        if m:
            age, sex = int(m.group(1)), (m.group(2) or "").upper()
            if (demo.age is not None and abs(age - demo.age) > 2) or (sex and demo.sex and sex != demo.sex):
                flags.add("demographic_mismatch")
    return flags


def check(
    findings: list[Finding],
    facts: list[ExtractedFact],
    demo: Demographics | None = None,
    *,
    evidence_start: int = 1,
    note_text: str | None = None,
    note_id: str = "n",
) -> ContextResult:
    t0 = time.perf_counter()
    if note_text:
        facts = _merge_cues(facts, scan_cues(note_text, note_id))
    demo = demo or Demographics()
    study_flags = _demographic_flags(facts, demo)
    lat_facts = [f for f in facts if f.fact_type == "laterality"]
    note_sides = {s for f in lat_facts if (s := normalise_side(f.value) or normalise_side(f.quote))}
    asymptomatic = [f for f in facts if f.fact_type in ("symptom", "negation") and _ASYMPTOMATIC.search(_text(f))]
    counts: Counter[str] = Counter()
    n = evidence_start
    out: list[Finding] = []
    for f in findings:
        flags = list(f.flags) + sorted(study_flags)
        evidence: list[TextEvidence] = []

        def attach(fact: ExtractedFact, polarity: Polarity, _f: Finding = f, _ev: list[TextEvidence] = evidence) -> None:
            nonlocal n
            _ev.append(TextEvidence(evidence_id=f"te_{n}", note_id=fact.note_id, span=fact.span, quote=fact.quote,
                                    fact_type=fact.fact_type, polarity=polarity))
            n += 1

        side = _finding_side(f)
        conflict = bool(side and note_sides and side not in note_sides and len(note_sides) == 1)
        for fact in lat_facts:
            fside = normalise_side(fact.value) or normalise_side(fact.quote)
            if side is None or fside is None:
                attach(fact, "neutral")
            else:
                # agreement is consistency, not support: a side word says nothing about whether the finding is real
                attach(fact, "contradicts" if (conflict and fside != side) else "neutral")
        if conflict:
            flags.append("laterality_conflict")

        if f.modality == "cxr" and side:
            for fact in facts:
                if fact.fact_type == "history" and (m := _PNEUMONECTOMY.search(_text(fact))) and normalise_side(m.group(1)) == side:
                    flags.append("history_conflict")
                    attach(fact, "contradicts")

        keys = _SUPPORTING.get(f.label, ())
        # Plan D1: a region counts only if deleting it matters. When every image evidence item failed that test, a symptom
        # that fits the label is consistent with the finding but must not verify it (headaches fit any brain tumour class).
        image_failed = bool(f.image_evidence) and all(e.faithful is False for e in f.image_evidence)
        for fact in facts:
            if fact.fact_type == "symptom" and fact.polarity == "present":
                fits = any(k in _text(fact) for k in keys)
                attach(fact, "supports" if fits and not image_failed else "neutral")
            elif fact.fact_type == "negation":
                attach(fact, "neutral")

        if asymptomatic and f.label in _LARGE and f.tier in ("high", "moderate"):
            flags.append("symptom_finding_incoherent")
            for fact in asymptomatic:
                attach(fact, "contradicts")

        flags = list(dict.fromkeys(flags))
        counts.update(x for x in flags if x not in f.flags)
        out.append(f.model_copy(update={"flags": flags, "text_evidence": [*f.text_evidence, *evidence]}))

    missing: list[str] = []
    if not any(x.fact_type in ("symptom", "negation") for x in facts):
        missing.append("Doctor, the notes record no symptoms; consider adding the presenting complaint.")
    if not any(x.fact_type in ("history", "device", "medication") for x in facts):
        missing.append("Doctor, the notes record no relevant history; consider adding prior conditions, surgery or devices.")
    stage = StageResult(
        stage="context", ok=True, ms=int((time.perf_counter() - t0) * 1000),
        payload={"flags": dict(counts), "facts": len(facts), "missing_context": len(missing)},
    )
    return ContextResult(out, missing, stage)
