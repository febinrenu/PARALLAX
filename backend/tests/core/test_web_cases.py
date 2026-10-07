"""The web app's baked sample cases must stay valid against the contract, and every text
evidence span must point at exactly its quote in the case's notes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from medproof.core.schemas import StudyResult

CASES_DIR = Path(__file__).resolve().parents[3] / "web" / "public" / "cases"
CASE_FILES = sorted(CASES_DIR.glob("*/case.json"))


def test_every_listed_case_exists():
    index = json.loads((CASES_DIR / "index.json").read_text(encoding="utf-8"))
    assert {c["id"] for c in index} == {p.parent.name for p in CASE_FILES}


@pytest.mark.parametrize("path", CASE_FILES, ids=lambda p: p.parent.name)
def test_case_study_validates_and_spans_match_notes(path: Path):
    case = json.loads(path.read_text(encoding="utf-8"))
    study = StudyResult.model_validate(case["study"])
    assert study.modality == case["modality"]
    for finding in study.findings:
        for te in finding.text_evidence:
            note = case["notes"][te.note_id]
            assert note[te.span[0] : te.span[1]] == te.quote
        for ev in finding.image_evidence:
            for ref in (ev.heatmap_ref, ev.mask_ref):
                if ref:
                    assert ref.startswith("/cases/") and (CASES_DIR.parent / ref.lstrip("/")).is_file()
    assert (CASES_DIR.parent / case["image"]["src"].lstrip("/")).is_file()
