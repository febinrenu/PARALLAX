"""`/studies` routes (P4.4, P4.6): upload, SSE stage stream, fetch, feedback, pixels for the
viewer, per-study artifacts and ledger, FHIR export, dictation."""

from __future__ import annotations

import asyncio
import gzip
import io
import re
import uuid
from typing import Literal

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from PIL import Image
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from medproof.core.schemas import StageResult, StudyResult
from medproof.intake.decode import DecodedImage, DecodeError, load_image, sha256_hex
from medproof.pipeline import run_study

router = APIRouter()

_ARTIFACT_NAME = re.compile(r"^[A-Za-z0-9_.-]+\.png$")
MAX_PIXEL_EDGE = 2048
MAX_AUDIO_BYTES = 25 * 1024 * 1024  # Groq's Whisper upload limit
_LUMA = np.array([0.299, 0.587, 0.114], np.float32)


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


# -- ref rewriting: no server filesystem path ever leaves the API ---------------------------------
def _artifact_url(ref: str | None, study_id: str) -> str | None:
    if not ref:
        return ref
    name = ref.replace("\\", "/").rsplit("/", 1)[-1]
    return f"/studies/{study_id}/artifacts/{name}"


def _sanitize_finding_dict(finding: dict, study_id: str) -> None:
    for ev in finding.get("image_evidence", []) or []:
        for key in ("heatmap_ref", "mask_ref"):
            if ev.get(key):
                ev[key] = _artifact_url(ev[key], study_id)


def _sanitize_stage(result: StageResult, study_id: str) -> StageResult:
    copy = result.model_copy(deep=True)
    for finding in copy.payload.get("findings", []) or []:
        _sanitize_finding_dict(finding, study_id)
    return copy


def _sanitize_study(study: StudyResult, study_id: str) -> None:
    # Precedent thumb_refs are left alone: the retrieval stage (P3.8) owns their URL scheme.
    for finding in study.findings:
        for ev in finding.image_evidence:
            ev.heatmap_ref = _artifact_url(ev.heatmap_ref, study_id)
            ev.mask_ref = _artifact_url(ev.mask_ref, study_id)


def _require(request: Request, study_id: str):
    record = request.app.state.store.get(study_id)
    if record is None:
        raise HTTPException(status_code=404, detail="unknown study_id")
    return record


async def _decoded(request: Request, study_id: str) -> DecodedImage:
    record = _require(request, study_id)
    if record.raw is None:
        raise HTTPException(status_code=404, detail="no image stored for this study")
    try:
        return await asyncio.to_thread(load_image, record.raw)
    except DecodeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


# -- routes ------------------------------------------------------------------------------------------
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
    store.create(study_id, queue, raw=raw)

    def on_stage_result(result: StageResult) -> None:
        clean = _sanitize_stage(result, study_id)
        ledger.append("stage_completed", clean.model_dump(mode="json"), study_id=study_id)
        loop.call_soon_threadsafe(store.push_stage_result, study_id, clean)

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
            _sanitize_study(study, study_id)
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


class TranscribeOut(BaseModel):
    ok: bool
    note: str
    transcript: str = ""
    language: str = ""
    duration_s: float | None = None
    warnings: list[str] = []


@router.post("/transcribe", response_model=TranscribeOut)
async def transcribe(
    request: Request,
    audio: UploadFile = File(...),  # noqa: B008
    typed: str | None = Form(None),
) -> TranscribeOut:
    """Dictation in, the typed note with the transcript appended out (P3's `dictation_to_note`). The
    audio is not stored and is not part of any study: the doctor reviews the merged text, and only
    what they then submit with a study is analysed, injection guard included."""
    data = await audio.read()
    if not data:
        raise HTTPException(status_code=422, detail="empty audio")
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="audio longer than the 25 MB transcription limit")
    from medproof.context.voice import dictation_to_note

    pool = request.app.state.audio_pool()
    out = await asyncio.to_thread(dictation_to_note, data, audio.filename or "dictation.webm", typed, pool)
    return TranscribeOut(ok=out.ok, note=out.note, transcript=out.transcript or "", language=out.language or "",
                         duration_s=out.duration_s, warnings=list(out.warnings))


@router.get("/studies/{study_id}/events")
async def study_events(request: Request, study_id: str) -> EventSourceResponse:
    store = request.app.state.store
    if not store.exists(study_id):
        raise HTTPException(status_code=404, detail="unknown study_id")

    async def gen():
        async for kind, data in store.stream(study_id):
            payload = data if kind == "error" else data.model_dump_json()
            yield {"event": kind, "data": payload}

    return EventSourceResponse(gen())


@router.get("/studies/{study_id}")
async def get_study(request: Request, study_id: str):
    record = _require(request, study_id)
    if record.status == "done" and record.result is not None:
        return record.result
    if record.status == "failed":
        return {"study_id": study_id, "status": "failed", "error": record.error}
    return {
        "study_id": study_id,
        "status": "running",
        "stages_completed": [sr.stage for sr in record.stage_results],
    }


@router.get("/studies/{study_id}/image")
async def get_image(request: Request, study_id: str) -> Response:
    """8-bit display copy at the original resolution: the viewer paints this first."""
    img = await _decoded(request, study_id)

    def encode() -> bytes:
        buf = io.BytesIO()
        Image.fromarray(img.display).save(buf, format="PNG")
        return buf.getvalue()

    return Response(await asyncio.to_thread(encode), media_type="image/png", headers={"Cache-Control": "private, max-age=3600"})


@router.get("/studies/{study_id}/pixels")
async def get_pixels(request: Request, study_id: str) -> Response:
    """Full-depth luminance as little-endian float16 in [0, 1], long edge <= 2048 px.

    The viewer works in normalised image coordinates, so overlays (which are on the original
    grid) line up regardless of this texture's resolution. Gzip is applied here only: a global
    gzip middleware could buffer the SSE stream.
    """
    img = await _decoded(request, study_id)

    def encode() -> tuple[bytes, int, int, int, int]:
        a = np.asarray(img.analysis, dtype=np.float32)
        if a.ndim == 3:
            a = a[..., :3] @ _LUMA
        h, w = a.shape
        scale = min(1.0, MAX_PIXEL_EDGE / max(h, w))
        if scale < 1.0:
            a = cv2.resize(a, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
        out = np.clip(a, 0.0, 1.0).astype("<f2")
        return gzip.compress(out.tobytes(), compresslevel=6), out.shape[1], out.shape[0], w, h

    body, tw, th, ow, oh = await asyncio.to_thread(encode)
    headers = {
        "Content-Encoding": "gzip",
        "X-Width": str(tw),
        "X-Height": str(th),
        "X-Original-Width": str(ow),
        "X-Original-Height": str(oh),
        "Cache-Control": "private, max-age=3600",
    }
    return Response(body, media_type="application/octet-stream", headers=headers)


@router.get("/studies/{study_id}/artifacts/{name}")
async def get_artifact(request: Request, study_id: str, name: str) -> FileResponse:
    record = _require(request, study_id)
    if not _ARTIFACT_NAME.fullmatch(name) or record.raw is None:
        raise HTTPException(status_code=404, detail="unknown artifact")
    # Artifacts are stored by input hash (see pipeline.run_study), so repeat uploads share them.
    base = (request.app.state.config.artifact_root / sha256_hex(record.raw)).resolve()
    path = (base / name).resolve()
    if not path.is_relative_to(base) or not path.is_file():
        raise HTTPException(status_code=404, detail="unknown artifact")
    return FileResponse(path, media_type="image/png")


@router.get("/studies/{study_id}/ledger")
async def get_study_ledger(request: Request, study_id: str) -> dict:
    _require(request, study_id)
    ledger = request.app.state.ledger
    return {"entries": ledger.entries(study_id), "head": ledger.head()}


@router.post("/studies/{study_id}/feedback", response_model=FeedbackOut)
async def post_feedback(request: Request, study_id: str, body: FeedbackIn) -> FeedbackOut:
    _require(request, study_id)
    ledger = request.app.state.ledger
    entry = ledger.append("finding_feedback", body.model_dump(), study_id=study_id, actor=body.actor)
    return FeedbackOut(ledger_hash=entry.hash, ledger_head=ledger.head())


@router.get("/studies/{study_id}/fhir")
async def get_fhir(request: Request, study_id: str) -> dict:
    """FHIR R4B Bundle of the finished study (P3.13's `build_bundle`): a preliminary
    DiagnosticReport with only reportable claims, Observations for supported findings, note quotes
    redacted, and the ledger head as an identifier."""
    record = _require(request, study_id)
    if record.result is None or record.completed_at is None:
        raise HTTPException(status_code=409, detail="study has not finished")
    try:
        from medproof.report.fhir import build_bundle
    except ImportError:
        return {"available": False, "note": "FHIR export needs backend[services] (fhir.resources)"}
    study = record.result
    return build_bundle(
        study.findings,
        study.claims,
        study_id=study.study_id,
        issued=record.completed_at,
        ledger_head=study.ledger_head,
    )
