"""FastAPI app factory (P4.4). `uvicorn medproof.api.app:app` runs it with real defaults.

Ledger and study store are constructor-injected onto `app.state`, not module-level singletons
(unlike `readers/cxr.py`'s `get_reader()` pattern) — the API layer specifically needs per-test
isolation, since many tests share one pytest process and a global ledger/store would leak
state between them.
"""

from __future__ import annotations

import threading
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from medproof.api import studies, system, warmup
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
    warmup_targets: tuple[str, ...] = (),
) -> FastAPI:
    config = config or PipelineConfig()
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Warm the models in a background thread: the server answers while it loads, and a failure never stops it.
        if warmup_targets:
            def run() -> None:
                try:
                    app.state.warmup = warmup.warm_up(warmup_targets, loaders=warmup.default_loaders(app.state))
                except BaseException as exc:  # noqa: BLE001 - even MemoryError must not take the server down
                    app.state.warmup = {"error": f"{type(exc).__name__}: {exc}"}

            app.state.warmup_thread = threading.Thread(target=run, name="warm-up", daemon=True)
            app.state.warmup_thread.start()
        yield

    app = FastAPI(title="Parallax / MedProof API", lifespan=lifespan)
    app.state.warmup = {}
    app.state.warmup_thread = None
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


app = create_app(warmup_targets=warmup.targets_from_env())
