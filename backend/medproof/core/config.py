"""Pipeline-wide settings read from the environment. Mirrors `readers/cxr_config.py`'s
`_env()` pattern: model ids and tunables live in config, not in code paths (CLAUDE.md).
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class PipelineConfig:
    # Never defaults into the repo: a fresh clone must not accumulate cache/artifact files
    # under version control just by running the pipeline.
    cache_dir: Path = field(
        default_factory=lambda: Path(_env("MEDPROOF_CACHE_DIR", str(Path(tempfile.gettempdir()) / "medproof_cache")))
    )
    artifact_root: Path = field(
        default_factory=lambda: Path(
            _env("MEDPROOF_ARTIFACT_DIR", str(Path(tempfile.gettempdir()) / "medproof_artifacts"))
        )
    )
    default_timeout_s: float = field(default_factory=lambda: float(_env("MEDPROOF_STAGE_TIMEOUT_S", "20.0")))
    ledger_path: Path = field(
        default_factory=lambda: Path(
            _env("MEDPROOF_LEDGER_PATH", str(Path(tempfile.gettempdir()) / "medproof_ledger.jsonl"))
        )
    )
