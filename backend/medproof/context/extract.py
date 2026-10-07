"""Span-grounded fact extraction from a clinical note.

The model returns facts with the exact quote only. Offsets are recovered here by exact string
search and never read from the model, so a fact cannot point at text that is not in the note.
Text flagged by the injection guard is blanked out before the model sees it, and no fact may
overlap a flagged span.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

from pydantic import BaseModel

from medproof.context.injection_guard import GuardResult
from medproof.core.schemas import TextEvidence
from medproof.llm.errors import LLMError
from medproof.llm.groq_pool import PoolResult, quote_data

FACT_TYPES = frozenset(
    {"symptom", "negation", "history", "device", "laterality", "lab", "medication", "demographic"}
)
REDACTION = "#"


class RawFact(BaseModel):
    type: str
    value: str
    polarity: Literal["present", "absent", "historical"]
    quote: str


class ExtractedFacts(BaseModel):
    facts: list[RawFact] = []


@dataclass(frozen=True)
class ExtractedFact:
    note_id: str
    span: tuple[int, int]
    quote: str
    fact_type: str
    value: str
    polarity: str


@dataclass
class ExtractionResult:
    ok: bool
    facts: list[ExtractedFact] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class _Pool(Protocol):
    def chat(
        self, service: str, prompt_name: str, user: str, schema: type[ExtractedFacts], **kw: object
    ) -> PoolResult[ExtractedFacts]: ...


def _redact(text: str, spans: list[tuple[int, int]]) -> str:
    """Blank flagged spans with a same-length filler so every offset stays valid."""
    chars = list(text)
    for s, e in spans:
        for i in range(max(0, s), min(len(chars), e)):
            if not chars[i].isspace():
                chars[i] = REDACTION
    return "".join(chars)


def _find_free(text: str, quote: str, taken: set[tuple[int, int]], flagged: list[tuple[int, int]]) -> tuple[int, int] | None:
    start = 0
    while True:
        at = text.find(quote, start)
        if at < 0:
            return None
        span = (at, at + len(quote))
        overlaps_flag = any(span[0] < fe and fs < span[1] for fs, fe in flagged)
        if span not in taken and not overlaps_flag:
            return span
        start = at + 1


def extract(
    text: str, note_id: str, pool: _Pool, guard: GuardResult | None = None, *, service: str = "extract"
) -> ExtractionResult:
    if not text.strip():
        return ExtractionResult(ok=True)
    flagged = guard.spans() if guard else []
    try:
        reply = pool.chat(service, "extract", quote_data(_redact(text, flagged)), ExtractedFacts).data
    except LLMError as exc:
        return ExtractionResult(ok=False, warnings=[f"fact extraction unavailable: {exc}"])
    result = ExtractionResult(ok=True)
    taken: set[tuple[int, int]] = set()
    for raw in reply.facts if reply else []:
        if raw.type not in FACT_TYPES:
            result.dropped.append(f"unknown fact type '{raw.type}': {raw.quote!r}")
            continue
        quote = raw.quote.strip()
        span = _find_free(text, quote, taken, flagged) if quote else None
        if span is None:
            result.dropped.append(f"quote not found in note: {raw.quote!r}")
            continue
        taken.add(span)
        result.facts.append(ExtractedFact(note_id, span, quote, raw.type, raw.value, raw.polarity))
    result.facts.sort(key=lambda f: f.span)
    return result


def to_text_evidence(facts: list[ExtractedFact], start: int = 1) -> list[TextEvidence]:
    """Contract objects for the extracted facts. Polarity stays neutral until a finding is compared."""
    return [
        TextEvidence(
            evidence_id=f"te_{start + i}",
            note_id=f.note_id,
            span=f.span,
            quote=f.quote,
            fact_type=f.fact_type,
            polarity="neutral",
        )
        for i, f in enumerate(facts)
    ]
