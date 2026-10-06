"""In-memory per-study bookkeeping for the API layer (P4.4). Not persisted — a known,
stated limitation, not an oversight: studies vanish on a server restart. Persistence across
restarts isn't asked for anywhere in plan.md's P4.4 Done-when; the ledger (P4.5) is the
durable record, this is just live request/response/SSE plumbing.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from typing import Literal

from medproof.core.schemas import StageResult, StudyResult

Status = Literal["running", "done", "failed"]


@dataclass
class StudyRecord:
    study_id: str
    status: Status = "running"
    stage_results: list[StageResult] = field(default_factory=list)
    result: StudyResult | None = None
    error: str | None = None
    queue: asyncio.Queue | None = None  # live subscribers; None once terminal and drained


class StudyStore:
    def __init__(self) -> None:
        self._records: dict[str, StudyRecord] = {}
        self._lock = threading.Lock()

    def create(self, study_id: str, queue: asyncio.Queue) -> None:
        with self._lock:
            self._records[study_id] = StudyRecord(study_id=study_id, queue=queue)

    def exists(self, study_id: str) -> bool:
        with self._lock:
            return study_id in self._records

    def get(self, study_id: str) -> StudyRecord | None:
        with self._lock:
            return self._records.get(study_id)

    def push_stage_result(self, study_id: str, result: StageResult) -> None:
        """Called via `loop.call_soon_threadsafe` from the worker thread — safe to touch the
        asyncio.Queue because this only ever runs on the event-loop thread."""
        with self._lock:
            record = self._records.get(study_id)
            if record is None:
                return
            record.stage_results.append(result)
            if record.queue is not None:
                record.queue.put_nowait(("stage", result))

    def complete(self, study_id: str, result: StudyResult) -> None:
        with self._lock:
            record = self._records.get(study_id)
            if record is None:
                return
            record.status = "done"
            record.result = result
            if record.queue is not None:
                record.queue.put_nowait(("done", result))
                record.queue.put_nowait(None)  # sentinel: stop streaming

    def fail(self, study_id: str, error: str) -> None:
        with self._lock:
            record = self._records.get(study_id)
            if record is None:
                return
            record.status = "failed"
            record.error = error
            if record.queue is not None:
                record.queue.put_nowait(("error", error))
                record.queue.put_nowait(None)

    async def stream(self, study_id: str):
        """Yields `(kind, data)` tuples: `("stage", StageResult)`, then a terminal
        `("done", StudyResult)` or `("error", str)`. Every event the study has ever produced
        sits in its queue until something drains it — including a subscriber connecting after
        the study already finished, since `complete()`/`fail()` only ever *add* to the queue,
        never discard it — so draining unconditionally handles "connected late" correctly with
        no special-casing. Once the queue is fully drained (the sentinel seen), it's cleared
        from the record; a second, later subscriber then falls back to the stored terminal
        state instead of hanging on an empty queue (this store fans a study's events out to at
        most one live drainer at a time, which matches today's single-viewer-per-study UI)."""
        record = self.get(study_id)
        if record is None:
            return
        queue = record.queue
        if queue is None:
            if record.status == "done" and record.result is not None:
                yield ("done", record.result)
            elif record.status == "failed":
                yield ("error", record.error)
            return
        while True:
            item = await queue.get()
            if item is None:
                with self._lock:
                    record.queue = None
                return
            yield item
