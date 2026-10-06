"""The evidence ledger (plan.md D17 / P4.5): a hash-chained, append-only JSONL audit trail.

One global ledger, not per-study — matches `GET /ledger/verify`'s single endpoint. Entries
carry `study_id` so a future per-study audit view can filter, but no filtering endpoint exists
yet (not in P4.4's route list). The hash covers more than the payload (`ts`, `actor`, `event`,
`study_id` too), so tampering any one of those fields alone is still caught by `verify()`.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

GENESIS_HASH = "0" * 64


@dataclass(frozen=True)
class LedgerEntry:
    index: int
    prev_hash: str
    hash: str
    ts: str
    actor: str
    event: str
    study_id: str | None
    payload_hash: str
    payload: dict = field(default_factory=dict)


def _payload_hash(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _entry_hash(prev_hash: str, event: str, payload_hash: str, ts: str, actor: str, study_id: str | None) -> str:
    material = f"{prev_hash}:{event}:{payload_hash}:{ts}:{actor}:{study_id}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


class Ledger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()
        self._lock = threading.Lock()

    def _lines(self) -> list[str]:
        return [ln for ln in self.path.read_text(encoding="utf-8").splitlines() if ln.strip()]

    def head(self) -> str:
        lines = self._lines()
        if not lines:
            return GENESIS_HASH
        return json.loads(lines[-1])["hash"]

    def append(self, event: str, payload: dict, *, study_id: str | None = None, actor: str = "system") -> LedgerEntry:
        with self._lock:
            lines = self._lines()
            prev_hash = json.loads(lines[-1])["hash"] if lines else GENESIS_HASH
            index = len(lines)
            ts = datetime.now(UTC).isoformat()
            payload_hash = _payload_hash(payload)
            entry_hash = _entry_hash(prev_hash, event, payload_hash, ts, actor, study_id)
            entry = LedgerEntry(
                index=index, prev_hash=prev_hash, hash=entry_hash, ts=ts, actor=actor,
                event=event, study_id=study_id, payload_hash=payload_hash, payload=payload,
            )
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(entry), sort_keys=True, default=str) + "\n")
            return entry

    def verify(self) -> dict:
        prev_hash = GENESIS_HASH
        lines = self._lines()
        for i, line in enumerate(lines):
            entry = json.loads(line)
            if entry["prev_hash"] != prev_hash:
                return {"ok": False, "entries": len(lines), "broken_at": i, "reason": "prev_hash mismatch"}
            expected_payload_hash = _payload_hash(entry["payload"])
            if entry["payload_hash"] != expected_payload_hash:
                return {"ok": False, "entries": len(lines), "broken_at": i, "reason": "payload_hash mismatch"}
            expected_hash = _entry_hash(
                prev_hash, entry["event"], entry["payload_hash"], entry["ts"], entry["actor"], entry["study_id"]
            )
            if entry["hash"] != expected_hash:
                return {"ok": False, "entries": len(lines), "broken_at": i, "reason": "hash mismatch"}
            prev_hash = entry["hash"]
        return {"ok": True, "entries": len(lines), "head": prev_hash}
