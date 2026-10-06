"""Slot grammar for report claims.

    template := (TEXT | SLOT)+          SLOT := "{" ref "." field "}"
    ref      := "f" N | "te_" N | "ie_" N

TEXT may hold only ASCII letters, spaces and , . ; : ( ) - ' so a digit, brace, markup or
look-alike character can never reach a report through the template.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

MAX_WORDS = 40

ALLOWED_FIELDS: dict[str, frozenset[str]] = {
    "f": frozenset({"label", "region", "tier", "status", "conformal_set", "second_read", "flag_phrase"}),
    "te": frozenset({"quote"}),
    "ie": frozenset({"region"}),
}

SLOT_RE = re.compile(r"\{((?:f|te_|ie_)\d+)\.([a-z_]+)\}")
TEXT_RE = re.compile(r"[A-Za-z ,.;:()'\-]*")


class TemplateSyntaxError(ValueError):
    """The template is outside the grammar (rule R1)."""


class SlotError(ValueError):
    """A slot names an unknown ref or a field the ref does not offer (rule R2)."""


@dataclass(frozen=True)
class Slot:
    ref: str  # "f1", "te_2", "ie_3"
    field: str

    @property
    def kind(self) -> str:
        return "f" if self.ref.startswith("f") else self.ref.split("_", 1)[0]


@dataclass(frozen=True)
class Parsed:
    parts: tuple[str | Slot, ...]

    @property
    def slots(self) -> list[Slot]:
        return [p for p in self.parts if isinstance(p, Slot)]

    @property
    def text(self) -> str:
        """The literal text only, slots removed, joined with a space."""
        return " ".join(p for p in self.parts if isinstance(p, str))


def parse_template(template: str) -> Parsed:
    if not isinstance(template, str) or not template.strip():
        raise TemplateSyntaxError("empty template")
    parts: list[str | Slot] = []
    pos = 0
    for m in SLOT_RE.finditer(template):
        if m.start() > pos:
            parts.append(template[pos : m.start()])
        parts.append(Slot(m.group(1), m.group(2)))
        pos = m.end()
    if pos < len(template):
        parts.append(template[pos:])
    for p in parts:
        if isinstance(p, str) and not TEXT_RE.fullmatch(p):
            raise TemplateSyntaxError("literal text has characters outside the allowed set")
    slots = [p for p in parts if isinstance(p, Slot)]
    if not any(s.kind == "f" for s in slots):
        raise TemplateSyntaxError("template names no finding slot")
    words = sum(len(p.split()) if isinstance(p, str) else 1 for p in parts)
    if words > MAX_WORDS:
        raise TemplateSyntaxError(f"template longer than {MAX_WORDS} words")
    return Parsed(tuple(parts))


def check_fields(parsed: Parsed) -> None:
    for s in parsed.slots:
        if s.field not in ALLOWED_FIELDS[s.kind]:
            raise SlotError(f"field '{s.field}' is not available on {s.ref}")
