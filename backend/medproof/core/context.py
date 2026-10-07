"""The in-process object a stage's `run(ctx)` reads from and writes to.

Not a contract type: it carries live `DecodedImage`/numpy data that has no business crossing a
wire boundary (that's what `StageResult`/`StudyResult` in `schemas.py` are for). Every field is
read via `getattr(ctx, name, default)` by existing stages (`intake/run.py`, `readers/cxr.py`),
and both of their test suites construct `ctx` as a bare `types.SimpleNamespace(...)` — this
class is a stricter, defaulted stand-in for the same shape, not a replacement for duck typing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from medproof.core.config import PipelineConfig
from medproof.core.schemas import Finding, Modality, StageResult
from medproof.intake.decode import DecodedImage


@dataclass
class StudyContext:
    study_id: str
    raw_bytes: bytes | None = None
    path: str | None = None
    modality_hint: Modality | None = None
    notes: str | None = None  # raw uploaded note text; the P3 context and report stages read it
    decoded: DecodedImage | None = None
    artifact_dir: Path | None = None
    finding_start: int = 1
    evidence_start: int = 1
    input_sha256: str | None = None  # set by the orchestrator before any stage runs
    # Every StageResult so far, in run order, appended by the orchestrator right after each stage
    # (cache hits included). Later stages read earlier findings from here (P3-1).
    stage_results: list[StageResult] = field(default_factory=list)
    config: PipelineConfig | None = None  # the run's config, for stages that need model paths or budgets
    # Set by the router stage (P1.4) when it runs: body part for the bone reader, flags (ood,
    # router_uncertain) that the reader step copies onto every finding.
    body_part: str | None = None
    router_flags: list[str] = field(default_factory=list)
    # Left by every reader (P1's `_share`) so faithfulness and stability can re-score the same image:
    # the live reader object, its raw `ReaderOutput`, and its contract findings.
    reader: Any = None
    reader_output: Any = None
    findings: list[Finding] | None = None
