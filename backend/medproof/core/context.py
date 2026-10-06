"""The in-process object a stage's `run(ctx)` reads from and writes to.

Not a contract type: it carries live `DecodedImage`/numpy data that has no business crossing a
wire boundary (that's what `StageResult`/`StudyResult` in `schemas.py` are for). Every field is
read via `getattr(ctx, name, default)` by existing stages (`intake/run.py`, `readers/cxr.py`),
and both of their test suites construct `ctx` as a bare `types.SimpleNamespace(...)` — this
class is a stricter, defaulted stand-in for the same shape, not a replacement for duck typing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from medproof.core.schemas import Modality
from medproof.intake.decode import DecodedImage


@dataclass
class StudyContext:
    study_id: str
    raw_bytes: bytes | None = None
    path: str | None = None
    modality_hint: Modality | None = None
    decoded: DecodedImage | None = None
    artifact_dir: Path | None = None
    finding_start: int = 1
    evidence_start: int = 1
    input_sha256: str | None = None  # set by the orchestrator before any stage runs
