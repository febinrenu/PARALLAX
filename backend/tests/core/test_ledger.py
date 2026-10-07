"""Tests for the hash-chained evidence ledger (plan.md P4.5)."""

from __future__ import annotations

import json

from medproof.core.ledger import GENESIS_HASH, Ledger


def test_empty_ledger_verifies_and_has_genesis_head(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    assert ledger.head() == GENESIS_HASH
    assert ledger.verify() == {"ok": True, "entries": 0, "head": GENESIS_HASH}


def test_append_chains_prev_hash_and_advances_head(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    e1 = ledger.append("stage_completed", {"stage": "intake"}, study_id="s1")
    assert e1.prev_hash == GENESIS_HASH
    e2 = ledger.append("stage_completed", {"stage": "reader"}, study_id="s1")
    assert e2.prev_hash == e1.hash
    assert ledger.head() == e2.hash
    result = ledger.verify()
    assert result == {"ok": True, "entries": 2, "head": e2.hash}


def test_payload_key_order_does_not_affect_the_hash(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    e1 = ledger.append("x", {"a": 1, "b": 2})
    ledger2 = Ledger(tmp_path / "ledger2.jsonl")
    e2 = ledger2.append("x", {"b": 2, "a": 1})
    assert e1.payload_hash == e2.payload_hash


def test_tampering_payload_on_disk_fails_verification(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path)
    ledger.append("stage_completed", {"stage": "intake", "ok": True}, study_id="s1")
    ledger.append("stage_completed", {"stage": "reader", "ok": True}, study_id="s1")

    lines = path.read_text(encoding="utf-8").splitlines()
    tampered = json.loads(lines[0])
    tampered["payload"]["ok"] = False  # flip one field after the fact
    lines[0] = json.dumps(tampered)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = Ledger(path).verify()
    assert result["ok"] is False and result["broken_at"] == 0


def test_tampering_actor_without_touching_payload_still_fails(tmp_path):
    """The hash covers actor/ts/event too, not just the payload."""
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path)
    ledger.append("finding_feedback", {"finding_id": "f1"}, actor="dr_jane")

    lines = path.read_text(encoding="utf-8").splitlines()
    tampered = json.loads(lines[0])
    tampered["actor"] = "dr_evil"  # payload_hash still matches; entry hash should not
    lines[0] = json.dumps(tampered)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = Ledger(path).verify()
    assert result["ok"] is False and result["reason"] == "hash mismatch"


def test_reopening_an_existing_ledger_continues_the_chain(tmp_path):
    path = tmp_path / "ledger.jsonl"
    Ledger(path).append("a", {})
    second_open = Ledger(path)
    e2 = second_open.append("b", {})
    assert e2.index == 1
    assert Ledger(path).verify()["ok"] is True
