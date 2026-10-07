"""System-level routes (P4.4): metrics, model cards, ledger verification.

Each route serves `reports/metrics.json` (P2.14) or `docs/model_cards/*.json` (P2.13) when present;
when absent they say so instead of fabricating numbers or cards.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Request

router = APIRouter()

REPO_ROOT = Path(__file__).resolve().parents[3]  # api/ -> medproof/ -> backend/ -> repo root
METRICS_PATH = REPO_ROOT / "reports" / "metrics.json"
MODEL_CARDS_DIR = REPO_ROOT / "docs" / "model_cards"


@router.get("/metrics")
async def get_metrics() -> dict:
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return {"available": False, "note": "see plan.md P2.14 (make eval)"}


@router.get("/model-cards")
async def get_model_cards() -> dict:
    if MODEL_CARDS_DIR.is_dir():
        cards = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(MODEL_CARDS_DIR.glob("*.json"))]
        return {"cards": cards}
    return {"cards": [], "note": "see plan.md P2.13 (model cards + datasheets)"}


@router.get("/ledger/verify")
async def ledger_verify(request: Request) -> dict:
    return request.app.state.ledger.verify()
