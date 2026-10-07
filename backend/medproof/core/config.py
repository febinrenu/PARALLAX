"""Pipeline-wide settings read from the environment. Mirrors `readers/cxr_config.py`'s
`_env()` pattern: model ids and tunables live in config, not in code paths (CLAUDE.md).
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]  # core/ -> medproof/ -> backend/ -> repo root


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
    # P2's model bundles (`<name>/model_meta.json` + git-ignored `weights.pt`), one per reader.
    models_root: Path = field(default_factory=lambda: Path(_env("MEDPROOF_MODELS_DIR", str(REPO_ROOT / "ml" / "artifacts"))))
    # Faithfulness costs ~14 s per finding on CPU (16 random placements), so only the most probable
    # findings are tested; the rest stay untested and therefore never count as verified evidence.
    faithfulness_top_k: int = field(default_factory=lambda: int(_env("MEDPROOF_FAITHFULNESS_TOP_K", "3")))
