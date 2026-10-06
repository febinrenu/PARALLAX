"""FastAPI wrapper: GET /health, POST /read.

Run: uvicorn services.medgemma.server:app --port 8001
The model loads lazily on the first /read so the process starts fast and /health answers.
"""

import io
import os
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from services.medgemma.reader import Backend, read_image
from services.medgemma.schema import Modality, ReadResult

MAX_BYTES = 25 * 1024 * 1024


def create_app(backend: Backend | None = None) -> FastAPI:
    app = FastAPI(title="medgemma-read", version="1")
    state: dict[str, Backend | None] = {"backend": backend}

    def get_backend() -> Backend:
        be = state["backend"]
        if be is None:
            from services.medgemma.loader import TransformersBackend

            be = state["backend"] = TransformersBackend.from_env()
        return be

    @app.get("/health")
    def health() -> dict:
        be = state["backend"]
        return {
            "loaded": be is not None,
            "model": getattr(be, "name", os.environ.get("MEDGEMMA_MODEL_ID", "")),
            "quant": getattr(be, "quant", os.environ.get("MEDGEMMA_QUANT", "nf4")),
        }

    @app.post("/read", response_model=ReadResult)
    async def read(
        image: Annotated[UploadFile, File()],
        modality: Annotated[Modality, Form()] = "cxr",
        prompt: Annotated[str | None, Form()] = None,
    ) -> ReadResult:
        data = await image.read()
        if len(data) > MAX_BYTES:
            raise HTTPException(413, "image too large")
        try:
            pil = Image.open(io.BytesIO(data))
            pil.load()
        except (UnidentifiedImageError, OSError) as exc:
            raise HTTPException(400, "upload is not a decodable image") from exc
        return read_image(get_backend(), pil.convert("RGB"), modality, prompt)

    return app


app = create_app()
