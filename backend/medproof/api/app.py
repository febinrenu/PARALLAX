"""FastAPI app factory (P4.4). `uvicorn medproof.api.app:app` runs it with real defaults.

Ledger and study store are constructor-injected onto `app.state`, not module-level singletons
(unlike `readers/cxr.py`'s `get_reader()` pattern) — the API layer specifically needs per-test
isolation, since many tests share one pytest process and a global ledger/store would leak
state between them.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import FastAPI

from medproof.api import studies, system
from medproof.core.config import PipelineConfig
from medproof.core.ledger import Ledger
from medproof.core.store import StudyStore
from medproof.pipeline import PIPELINE, StageSpec


def create_app(
    config: PipelineConfig | None = None,
    ledger: Ledger | None = None,
    store: StudyStore | None = None,
    stages: list[StageSpec] | None = None,
    audio_pool: Callable[[], Any] | None = None,
) -> FastAPI:
    config = config or PipelineConfig()
    app = FastAPI(title="Parallax / MedProof API")
    app.state.config = config
    app.state.ledger = ledger or Ledger(config.ledger_path)
    app.state.store = store or StudyStore()
    app.state.stages = stages if stages is not None else PIPELINE
    app.state.audio_pool = audio_pool or _default_audio_pool
    app.include_router(studies.router)
    app.include_router(system.router)
    return app


def _default_audio_pool() -> Any:
    """The shared Groq pool (it serves Whisper too), or None without keys or without P3's module."""
    try:
        from medproof.reasoning_stages import _default_llm_pool
    except ImportError:
        return None
    return _default_llm_pool()


app = create_app()
