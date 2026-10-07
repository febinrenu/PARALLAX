"""Read-only lookup tables the renderer and firewall need, built from findings and notes."""

from __future__ import annotations

from dataclasses import dataclass, field

from medproof.core.schemas import Finding, ImageEvidence, TextEvidence


@dataclass(frozen=True)
class StudyView:
    findings: dict[str, Finding]
    image_evidence: dict[str, ImageEvidence]
    text_evidence: dict[str, TextEvidence]
    owner: dict[str, str]  # evidence_id -> finding_id
    notes: dict[str, str]  # note_id -> original note text
    flagged: dict[str, list[tuple[int, int]]] = field(default_factory=dict)  # injection-guard spans

    @classmethod
    def build(
        cls,
        findings: list[Finding],
        notes: dict[str, str],
        flagged: dict[str, list[tuple[int, int]]] | None = None,
    ) -> StudyView:
        by_id: dict[str, Finding] = {}
        images: dict[str, ImageEvidence] = {}
        texts: dict[str, TextEvidence] = {}
        owner: dict[str, str] = {}
        for f in findings:
            by_id[f.finding_id] = f
            for e in f.image_evidence:
                images[e.evidence_id] = e
                owner[e.evidence_id] = f.finding_id
            for t in f.text_evidence:
                texts[t.evidence_id] = t
                owner[t.evidence_id] = f.finding_id
        return cls(by_id, images, texts, owner, dict(notes), dict(flagged or {}))

    def is_flagged(self, ev: TextEvidence) -> bool:
        s, e = ev.span
        return any(s < fe and fs < e for fs, fe in self.flagged.get(ev.note_id, []))
