"""Loads the synthetic note corpus straight from JSONL so backend tests need not import ml/."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[3] / "ml" / "data" / "notes" / "notes_v1.jsonl"


def load_notes() -> list[dict]:
    if not PATH.is_file():
        pytest.skip("run `python -m ml.data.gen_notes` first")
    return [json.loads(line) for line in PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
