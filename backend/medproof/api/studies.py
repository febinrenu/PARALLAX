"""`/studies` routes (P4.4): upload, SSE stage stream, fetch, feedback, FHIR placeholder."""

from __future__ import annotations

import asyncio
import uuid
from typing import Literal

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from medproof.core.schemas import StageResult
from medproof.pipeline import run_study

router = APIRouter()


class CreateStudyResponse(BaseModel):
    study_id: str
    events_url: str
    status_url: str


class FeedbackIn(BaseModel):
    finding_id: str
    decision: Literal["accept", "reject"]
    actor: str = "clinician"
    note: str | None = None


class FeedbackOut(BaseModel):
    ledger_hash: str
    ledger_head: str


@router.post("/studies", status_code=202, response_model=CreateStudyResponse)
async def create_study(
    request: Request,
    image: UploadFile = File(...),  # noqa: B008 (FastAPI's own idiom for file uploads)
    notes: str | None = Form(None),
    modality_hint: str | None = Form(None),
) -> CreateStudyResponse:
    raw = await image.read()
    study_id = str(uuid.uuid4())

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    store = request.app.state.store
    ledger = request.app.state.ledger
    stages = request.app.state.stages
    config = request.app.state.config
    store.create(study_id, queue)

    def on_stage_result(result: StageResult) -> None:
        ledger.append("stage_completed", result.model_dump(mode="json"), study_id=study_id)
        loop.call_soon_threadsafe(store.push_stage_result, study_id, result)

    def worker() -> None:
        try:
            study, _ = run_study(
                raw,
                study_id=study_id,
                modality_hint=modality_hint,  # type: ignore[arg-type]  # validated loosely; unknown values just degrade
                notes=notes,
                stages=stages,
                config=config,
                on_stage_result=on_stage_result,
            )
            study.ledger_head = ledger.head()
            loop.call_soon_threadsafe(store.complete, study_id, study)
        except Exception as exc:  # run_study itself shouldn't raise; belt and suspenders
            loop.call_soon_threadsafe(store.fail, study_id, f"{type(exc).__name__}: {exc}")

    loop.run_in_executor(None, worker)

    return CreateStudyResponse(
        study_id=study_id,
        events_url=f"/studies/{study_id}/events",
        status_url=f"/studies/{study_id}",
    )


@router.get("/studies/{study_id}/events")
async def study_events(request: Request, study_id: str) -> EventSourceResponse:
    store = request.app.state.store
    if not store.exists(study_id):
        raise HTTPException(status_code=404, detail="unknown study_id")

    async def gen():
        async for kind, data in store.stream(study_id):
            if kind == "stage":
                payload = data.model_dump_json()
            elif kind == "done":
                payload = data.model_dump_json()
            else:
                payload = data  # error string
            yield {"event": kind, "data": payload}

    return EventSourceResponse(gen())


@router.get("/studies/{study_id}")
async def get_study(request: Request, study_id: str):
    store = request.app.state.store
    record = store.get(study_id)
    if record is None:
        raise HTTPException(status_code=404, detail="unknown study_id")
    if record.status == "done" and record.result is not None:
        return record.result
    if record.status == "failed":
        return {"study_id": study_id, "status": "failed", "error": record.error}
    return {
        "study_id": study_id,
        "status": "running",
        "stages_completed": [sr.stage for sr in record.stage_results],
    }


@router.post("/studies/{study_id}/feedback", response_model=FeedbackOut)
async def post_feedback(request: Request, study_id: str, body: FeedbackIn) -> FeedbackOut:
    store = request.app.state.store
    if not store.exists(study_id):
        raise HTTPException(status_code=404, detail="unknown study_id")
    ledger = request.app.state.ledger
    entry = ledger.append("finding_feedback", body.model_dump(), study_id=study_id, actor=body.actor)
    return FeedbackOut(ledger_hash=entry.hash, ledger_head=ledger.head())


@router.get("/studies/{study_id}/fhir")
async def get_fhir(request: Request, study_id: str) -> dict:
    store = request.app.state.store
    if not store.exists(study_id):
        raise HTTPException(status_code=404, detail="unknown study_id")
    return {"available": False, "note": "FHIR export not yet implemented, see plan.md P3.13"}
