"""Disk-backed stage cache, keyed by `(input_sha256, stage, version)` (plan.md section 4:
"Stages must be idempotent and cacheable by (input_sha256, stage, model_version)").

A thin wrapper over `diskcache.Cache` (plan.md section 5.5, pre-approved, Apache-2.0) so
callers never touch the on-disk format directly; `StageResult` is stored as its own JSON, not
pickled, so the cache survives a schema version bump without pickling across process/Python
version boundaries.
"""

from __future__ import annotations

from pathlib import Path

import diskcache

from medproof.core.schemas import StageResult


def cache_key(input_sha256: str, stage: str, version: str) -> str:
    return f"{input_sha256}:{stage}:{version}"


class StageCache:
    def __init__(self, directory: str | Path):
        self._cache = diskcache.Cache(str(directory))

    def get(self, key: str) -> StageResult | None:
        raw = self._cache.get(key)
        if raw is None:
            return None
        return StageResult.model_validate_json(raw)

    def set(self, key: str, result: StageResult) -> None:
        self._cache.set(key, result.model_dump_json())

    def close(self) -> None:
        self._cache.close()
