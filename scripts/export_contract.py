#!/usr/bin/env python3
"""Regenerate contracts/*.schema.json from backend/medproof/core/schemas.py.

Run from the repo root (or via `make contracts`). Deterministic: running twice produces byte-
identical output, so `backend/tests/core/test_contract.py` can detect drift between schemas.py
and the committed JSON without re-running this script in CI.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from medproof.core.schemas import Finding, StudyResult  # noqa: E402

CONTRACTS_DIR = ROOT / "contracts"

EXPORTS = {
    "finding.schema.json": Finding,
    "study_result.schema.json": StudyResult,
}


def strip_titles(node):
    """Drop pydantic's auto-generated per-field "title" keys.

    Pydantic titles every field (e.g. "Prob Raw" for `prob_raw`), which makes
    json-schema-to-typescript hoist each one into its own top-level named type. Titles aren't
    part of the contract (field names and types are), so both the committed schema and the
    generated TS are cleaner without them. `backend/tests/core/test_contract.py` strips titles
    the same way before diffing, so this is safe to change without touching the drift check.
    """
    if isinstance(node, dict):
        return {k: strip_titles(v) for k, v in node.items() if k != "title"}
    if isinstance(node, list):
        return [strip_titles(v) for v in node]
    return node


def export() -> None:
    CONTRACTS_DIR.mkdir(exist_ok=True)
    for filename, model in EXPORTS.items():
        schema = strip_titles(model.model_json_schema())
        text = json.dumps(schema, indent=2, sort_keys=True) + "\n"
        (CONTRACTS_DIR / filename).write_text(text, encoding="utf-8")
        print(f"wrote {CONTRACTS_DIR / filename}")


if __name__ == "__main__":
    export()
