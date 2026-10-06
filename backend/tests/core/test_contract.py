"""Contract tests: fixtures parse cleanly, and committed JSON Schemas match schemas.py.

The schema-drift check runs `Model.model_json_schema()` in-process and diffs it against the
checked-in file, so it catches "someone edited schemas.py but forgot to re-run
scripts/export_contract.py" without needing to invoke the script as a subprocess.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from medproof.core.schemas import Finding, StudyResult
from medproof.core.status_rule import compute_status

REPO_ROOT = Path(__file__).resolve().parents[3]
CONTRACTS_DIR = REPO_ROOT / "contracts"
FIXTURES_DIR = CONTRACTS_DIR / "fixtures"

FIXTURE_FILES = sorted(FIXTURES_DIR.glob("*.json"))


def test_fixtures_exist_for_every_core_modality():
    names = {p.stem for p in FIXTURE_FILES}
    assert names == {"cxr", "brain_mri", "skin_dermoscopy", "bone_xray"}


@pytest.mark.parametrize("path", FIXTURE_FILES, ids=lambda p: p.stem)
def test_fixture_parses_as_study_result(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    study = StudyResult.model_validate(data)
    assert study.modality == path.stem
    assert study.findings, "fixture should carry at least one finding"


@pytest.mark.parametrize("path", FIXTURE_FILES, ids=lambda p: p.stem)
def test_fixture_finding_status_matches_the_status_rule(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    study = StudyResult.model_validate(data)
    for finding in study.findings:
        assert compute_status(finding) == finding.status, (
            f"{path.stem}/{finding.finding_id}: fixture says status={finding.status!r} "
            f"but compute_status() says {compute_status(finding)!r}"
        )


@pytest.mark.parametrize("path", FIXTURE_FILES, ids=lambda p: p.stem)
def test_fixture_text_evidence_span_length_matches_quote(path: Path):
    study = StudyResult.model_validate(json.loads(path.read_text(encoding="utf-8")))
    for finding in study.findings:
        for te in finding.text_evidence:
            assert te.span[1] - te.span[0] == len(te.quote), f"{path.stem}/{te.evidence_id}"


def _strip_titles(node):
    """Mirrors scripts/export_contract.py::strip_titles. Keep the two in sync."""
    if isinstance(node, dict):
        return {k: _strip_titles(v) for k, v in node.items() if k != "title"}
    if isinstance(node, list):
        return [_strip_titles(v) for v in node]
    return node


@pytest.mark.parametrize(
    "filename,model",
    [("finding.schema.json", Finding), ("study_result.schema.json", StudyResult)],
)
def test_committed_schema_matches_model(filename: str, model):
    committed = json.loads((CONTRACTS_DIR / filename).read_text(encoding="utf-8"))
    current = _strip_titles(model.model_json_schema())
    assert committed == current, (
        f"{filename} is stale — run `python scripts/export_contract.py` and commit the result"
    )
