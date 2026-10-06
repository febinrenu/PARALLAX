"""Prompt-injection guard for clinical notes. Notes are data; this finds text that tries to be a command.

Three layers, cheapest first:
  1. regex rules on several normalised views of the text (homoglyphs, leetspeak, spaced letters,
     zero-width characters, base64), each mapped back to offsets in the original note;
  2. Llama Prompt Guard 2 on Groq (a score, localised to a sentence when it fires);
  3. a gpt-oss-20b classifier that runs only when layers 1 and 2 found nothing and returns the
     offending sentence, whose span is recovered by exact search (the model's offsets are never used).
Layers 2 and 3 are best effort: if Groq is unavailable the guard degrades to regex with a warning.
"""

from __future__ import annotations

import base64
import binascii
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Protocol

from pydantic import BaseModel

from medproof.llm.config import PROMPT_GUARD_MODEL
from medproof.llm.errors import LLMError
from medproof.llm.groq_pool import PoolResult, ScoreResult, quote_data

_ZERO_WIDTH = {"​", "‌", "‍", "⁠", "﻿", "­"}
_HOMOGLYPHS = str.maketrans(
    {"о": "o", "і": "i", "а": "a", "е": "e", "р": "p", "с": "c",
     "у": "y", "х": "x", "ѕ": "s", "ј": "j", "ο": "o", "Ι": "i"}
)
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})
_SPACED = re.compile(r"(?<![A-Za-z])(?:[A-Za-z] ){5,}[A-Za-z](?![A-Za-z])")
_B64 = re.compile(r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{24,}={0,2}(?![A-Za-z0-9+/=])")
PG_THRESHOLD = 0.9  # 0.5 flagged a legitimate sentence in the corpus (0.58); recall on fresh attacks is 4 vs 7 of 28, see reports/p3_guard_blind.json
_TERMINATORS = ".!?\n"
_SENTENCE = re.compile(r"[^.!?\n]+[.!?\n]?")

_TARGET = r"(?:instructions?|rules?|prompts?|guidelines|directives|programming|safety checks?|checks)"
RULES: dict[str, re.Pattern[str]] = {
    "direct_override": re.compile(
        rf"\b(?:ignore|disregard|forget|override|bypass)\b[^.]{{0,40}}\b{_TARGET}\b|"
        r"\b(?:ignore|disregard)\s+(?:all\s+|everything\s+)?(?:of\s+)?(?:the\s+)?above\b|"
        r"\bnew\s+instructions?\b|"
        r"\b(?:do not|don't|never)\s+report\s+any\s+findings?\b|"
        r"\breport\s+(?:that\s+)?no\s+(?:findings?|abnormalit\w+)\b|"
        r"\b(?:state|say|write|output)\b[^.\n]{0,30}\b(?:study|film|everything|result)\b[^.\n]{0,20}\bnormal\b|"
        r"\b(?:say|state|write)\s+(?:that\s+)?there\s+are\s+no\s+findings\b|"
        r"\boutput\s*:\s*no\s+abnormalit\w+"
    ),
    "role_spoof": re.compile(
        r"(?:^|[.!?\n]\s*)(?:system|assistant|developer|admin|root)\s*:|"
        r"#{2,}\s*system\b|<\|\s*(?:system|assistant|im_start)\s*\|>|\[\s*admin\s+override\s*\]|"
        r"\bdeveloper\s+message\b|\byou\s+are\s+now\s+in\s+\w+\s+mode\b|\byou\s+must\s+(?:now\s+)?(?:approve|report|answer|output)\b"
    ),
    "delimiter_breakout": re.compile(
        r"</?\s*(?:note_data|user_notes|instructions?|system|prompt)\s*>|```|-{3,}\s*end\s+of\s+note|"
        r"\"\s*\}\s*\]|\bend\s+of\s+note\b[^.\n]{0,20}\bnow\b"
    ),
    "schema_hijack": re.compile(
        r"\{\s*\"(?:facts|claims|findings|status|no_findings)\"|\"(?:evidence_ids|claim_id|finding_id|template)\"\s*:|"
        r"\breturn\s+exactly\s*:"
    ),
    "evidence_id_spoof": re.compile(
        r"\b(?:te|ie)_\d+\b|\bf\d+\s+is\s+verified\b|\bfaithful\s*=\s*true\b|\bcite\b[^.\n]{0,30}\bevidence\b|"
        r"\bevidence\s+(?:id\s+)?\w*\d+\w*\s+proves\b"
    ),
    "exfiltration": re.compile(
        r"\b(?:print|reveal|repeat|output|show|list|leak|dump)\b[^.\n]{0,50}"
        r"\b(?:system\s+prompt|api\s+key|instructions\s+you\s+were\s+given|hidden\s+rules|configuration\s+file|\.env|"
        r"secret\s+tokens?|your\s+instructions|your\s+rules)\b|"
        r"\bwhat\s+is\s+in\s+your\s+configuration\b|\bthe\s+text\s+above\s+this\s+line\b"
    ),
    "indirect_clinical": re.compile(
        r"\b(?:do not|don't|never|avoid|omit)\b[^.\n]{0,30}\b(?:mention|report|list|include|state)\b[^.\n]{0,60}"
        r"\b(?:pneumonia|mass|nodule|malignan\w*|findings?|abnormalit\w*|tumou?r|fracture)\b|"
        r"\bomit\s+any\b[^.\n]{0,40}\bfrom\s+the\s+(?:summary|report|impression)\b|"
        r"\bregardless\s+of\s+the\s+image\b|\bfilm\s+is\s+normal\b[^.\n]{0,40}\b(?:told|prefers?|report)\b|"
        r"\bkeep\s+the\s+impression\s+reassuring\b|\bno\s+malignancy\s+be\s+reported\b|"
        r"\bavoid\s+listing\s+abnormalit\w+\b|\balways\s+report\b[^.\n]{0,30}\bno\s+acute\s+findings?\b"
    ),
    "multilingual": re.compile(
        r"ignora\s+las\s+instrucciones|ignorez\s+les\s+instructions|ignoriere\s+die\s+anweisungen|"
        r"ignorez\s+toutes?\s+les\s+instructions|"
        r"निर्देशों\s+को\s+अनदेखा|"
        r"पिछले\s+निर्देश|"
        r"வழிமுறைகளை\s+புறக்கணி"
    ),
}


@dataclass(frozen=True)
class Flag:
    span: tuple[int, int]
    rule: str
    source: str  # "regex" | "prompt_guard" | "llm"


@dataclass
class GuardResult:
    text: str
    flags: list[Flag] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def flagged(self) -> bool:
        return bool(self.flags)

    def text_of(self, flag: Flag) -> str:
        return self.text[flag.span[0] : flag.span[1]]

    def spans(self) -> list[tuple[int, int]]:
        """Flag spans merged when they overlap or touch, for use as `StudyView.flagged`."""
        merged: list[list[int]] = []
        for s, e in sorted(f.span for f in self.flags):
            if merged and s <= merged[-1][1] + 1:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        return [(s, e) for s, e in merged]


class GuardVerdict(BaseModel):
    injection: bool
    quote: str = ""


class _Pool(Protocol):
    def score(self, service: str, model: str, text: str) -> ScoreResult: ...

    def chat(self, service: str, prompt_name: str, user: str, schema: type[GuardVerdict], **kw: object) -> PoolResult[GuardVerdict]: ...


View = tuple[str, list[int]]  # (text, index of each char in the original)


def _views(text: str) -> list[View]:
    """The text as the rules should see it: normalised, de-leeted and with spaced letters joined."""
    chars, idx = [], []
    for i, ch in enumerate(text):
        if ch in _ZERO_WIDTH:
            continue
        n = unicodedata.normalize("NFKC", ch)
        n = n if len(n) == 1 else ch
        chars.append(n.translate(_HOMOGLYPHS).lower())
        idx.append(i)
    base = "".join(chars)
    leet = base.translate(_LEET)
    views: list[View] = [(base, idx), (leet, idx)]
    for src in (base, leet):
        keep = [True] * len(src)
        for m in _SPACED.finditer(src):
            for j in range(m.start(), m.end()):
                if src[j] == " ":
                    keep[j] = False
        if not all(keep):
            views.append(("".join(c for c, k in zip(src, keep, strict=True) if k), [i for i, k in zip(idx, keep, strict=True) if k]))
    return views


def _sentence_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    """Widen a match to the sentence (or line) that contains it, terminator included."""
    s = start
    while s > 0 and text[s - 1] not in _TERMINATORS:
        s -= 1
    e = end
    if not (e > 0 and text[e - 1] in _TERMINATORS):
        while e < len(text) and text[e] not in _TERMINATORS:
            e += 1
        if e < len(text):
            e += 1
    while s < e and text[s].isspace():
        s += 1
    return s, e


def scan_regex(text: str) -> list[Flag]:
    flags: dict[tuple[int, int], Flag] = {}

    def add(a: int, b: int, rule: str) -> None:
        span = _sentence_bounds(text, a, b)
        flags.setdefault(span, Flag(span, rule, "regex"))

    for view, idx in _views(text):
        for rule, pat in RULES.items():
            for m in pat.finditer(view):
                if m.end() > m.start():
                    add(idx[m.start()], idx[m.end() - 1] + 1, rule)
    for m in _B64.finditer(text):
        try:
            decoded = base64.b64decode(m.group(0), validate=False).decode("ascii")
        except (binascii.Error, UnicodeDecodeError, ValueError):
            continue
        low = decoded.lower()
        if decoded.isprintable() and any(p.search(low) for p in RULES.values()):
            add(m.start(), m.end(), "base64")
    return sorted(flags.values(), key=lambda f: f.span)


class InjectionGuard:
    def __init__(
        self,
        pool: _Pool | None = None,
        *,
        pg_model: str = PROMPT_GUARD_MODEL,
        pg_threshold: float = PG_THRESHOLD,
        use_regex: bool = True,
        use_pg: bool = True,
        use_llm: bool = True,
        service: str = "judge",
    ) -> None:
        self.pool, self.pg_model, self.pg_threshold = pool, pg_model, pg_threshold
        self.use_regex, self.use_pg, self.use_llm, self.service = use_regex, use_pg, use_llm, service

    def check(self, text: str) -> GuardResult:
        res = GuardResult(text=text, flags=list(scan_regex(text)) if self.use_regex else [])
        if self.pool is None or not text.strip():
            return res
        if self.use_pg and not res.flagged:
            self._prompt_guard(text, res)
        if res.flagged or not self.use_llm:
            return res
        try:
            verdict = self.pool.chat(self.service, "guard", quote_data(text), GuardVerdict).data
        except LLMError as exc:
            res.warnings.append(f"llm guard unavailable: {exc}")
            return res
        if verdict is not None and verdict.injection:
            at = text.find(verdict.quote) if verdict.quote.strip() else -1
            span = (at, at + len(verdict.quote)) if at >= 0 else (0, len(text))
            res.flags.append(Flag(span, "llm_classifier", "llm"))
        return res

    def _prompt_guard(self, text: str, res: GuardResult) -> None:
        """Score each sentence on its own: a whole note dilutes the payload among ordinary clinical text."""
        assert self.pool is not None
        for m in _SENTENCE.finditer(text):
            sent = m.group(0).strip()
            if len(sent) < 8:
                continue
            try:
                score = self.pool.score(self.service, self.pg_model, sent).score
            except LLMError as exc:
                res.warnings.append(f"prompt guard unavailable: {exc}")
                return
            if score >= self.pg_threshold:
                start = m.start() + (len(m.group(0)) - len(m.group(0).lstrip()))
                res.flags.append(Flag((start, m.end()), "prompt_guard", "prompt_guard"))
