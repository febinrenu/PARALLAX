"""Tests for the `/studies` API + SSE (P4.4) and its ledger wiring (P4.5)."""

from __future__ import annotations

import io
import json
import time

import pytest
from fastapi.testclient import TestClient

from medproof.api import system
from medproof.api.app import create_app
from medproof.core.config import PipelineConfig
from medproof.core.ledger import Ledger
from medproof.core.schemas import StageResult, StudyResult
from medproof.core.status_rule import compute_status
from medproof.pipeline import PIPELINE, StageSpec
from medproof.readers import cxr as cxr_reader
from tests.conftest import png_bytes, to_u8
from tests.readers.test_cxr import _image, make_reader

torch = pytest.importorskip("torch")  # make_reader needs torch, same as tests/test_pipeline.py


def _stub_stages() -> list[StageSpec]:
    reader = make_reader()
    return [
        PIPELINE[0],
        StageSpec(name="reader", run=lambda ctx: cxr_reader.run(ctx, reader=reader)),
    ]


def _client(tmp_path, *, stages=None) -> TestClient:
    config = PipelineConfig(
        cache_dir=tmp_path / "cache",
        artifact_root=tmp_path / "artifacts",
        ledger_path=tmp_path / "ledger.jsonl",
    )
    app = create_app(config=config, stages=stages)
    return TestClient(app)


def _upload(client: TestClient, *, modality_hint="cxr", notes=None):
    files = {"image": ("phantom.png", io.BytesIO(png_bytes(to_u8(_image()))), "image/png")}
    data = {}
    if modality_hint is not None:
        data["modality_hint"] = modality_hint
    if notes is not None:
        data["notes"] = notes
    return client.post("/studies", files=files, data=data)


def _poll_until_done(client: TestClient, study_id: str, *, timeout_s: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        body = client.get(f"/studies/{study_id}").json()
        if body.get("status") in ("failed",) or "findings" in body:
            return body
        time.sleep(0.02)
    raise AssertionError("study did not complete in time")


def test_happy_path_assembles_findings_with_real_status(tmp_path):
    client = _client(tmp_path, stages=_stub_stages())
    resp = _upload(client)
    assert resp.status_code == 202
    body = resp.json()

    result = _poll_until_done(client, body["study_id"])
    study = StudyResult.model_validate(result)
    assert study.modality == "cxr" and study.findings
    for f in study.findings:
        assert f.status == compute_status(f)


def test_sse_stream_emits_one_stage_event_per_stage_then_done(tmp_path):
    client = _client(tmp_path, stages=_stub_stages())
    study_id = _upload(client).json()["study_id"]

    kinds = []
    with client.stream("GET", f"/studies/{study_id}/events") as response:
        assert response.status_code == 200
        event_kind = None
        for line in response.iter_lines():
            if line.startswith("event:"):
                event_kind = line.split(":", 1)[1].strip()
            elif line.startswith("data:") and event_kind:
                kinds.append(event_kind)
                if event_kind == "done":
                    data = json.loads(line.split(":", 1)[1].strip())
                    StudyResult.model_validate(data)
                event_kind = None

    assert kinds.count("stage") == 2  # intake, reader
    assert kinds[-1] == "done"


def test_unsupported_modality_still_completes_through_the_real_api(tmp_path):
    client = _client(tmp_path)  # real PIPELINE, no stub needed for the degraded case
    study_id = _upload(client, modality_hint="brain_mri").json()["study_id"]
    result = _poll_until_done(client, study_id)
    assert result["findings"] == []


def test_unknown_study_id_is_404_on_every_route(tmp_path):
    client = _client(tmp_path)
    for path in ["/studies/nope", "/studies/nope/events", "/studies/nope/fhir"]:
        assert client.get(path).status_code == 404
    assert client.post("/studies/nope/feedback", json={"finding_id": "f1", "decision": "accept"}).status_code == 404


def test_metrics_and_model_cards_are_honest_placeholders(tmp_path, monkeypatch):
    # Point at paths that don't exist: P2 publishes the real files, and once it has, the routes
    # serve them instead (covered by the next test).
    monkeypatch.setattr(system, "METRICS_PATH", tmp_path / "missing" / "metrics.json")
    monkeypatch.setattr(system, "MODEL_CARDS_DIR", tmp_path / "missing" / "model_cards")
    client = _client(tmp_path, stages=_stub_stages())

    metrics = client.get("/metrics").json()
    assert metrics.get("available") is False

    cards = client.get("/model-cards").json()
    assert cards["cards"] == []


def test_fhir_exports_the_finished_study_as_a_validated_bundle(tmp_path):
    pytest.importorskip("fhir.resources")
    claim = {"claim_id": "c1", "template": "{f1.label}", "evidence_ids": ["e1"], "rendered": "Doctor, consider a finding.", "entailed": True}
    stages = [*_stub_stages(), StageSpec(name="report", run=lambda ctx: StageResult(stage="report", ok=True, ms=0, payload={"claims": [claim]}))]
    client = _client(tmp_path, stages=stages)
    study_id = _upload(client).json()["study_id"]
    study = _poll_until_done(client, study_id)
    assert [c["claim_id"] for c in study["claims"]] == ["c1"]

    resp = client.get(f"/studies/{study_id}/fhir")
    assert resp.status_code == 200
    bundle = resp.json()
    assert bundle["resourceType"] == "Bundle"
    report = bundle["entry"][0]["resource"]
    assert report["resourceType"] == "DiagnosticReport"
    assert report["status"] == "preliminary"
    assert {"system": "urn:parallax:ledger-head", "value": study["ledger_head"]} in report["identifier"]
    assert "Doctor, consider a finding." in report["conclusion"]


def test_fhir_is_refused_until_the_study_finishes(tmp_path):
    # TestClient's POST returns only after the pipeline finishes, so register a running study
    # directly instead of racing the worker.
    client = _client(tmp_path, stages=_stub_stages())
    client.app.state.store.create("still-running", queue=None)
    assert client.get("/studies/still-running/fhir").status_code == 409


def test_metrics_serves_the_published_report(tmp_path, monkeypatch):
    report = tmp_path / "metrics.json"
    report.write_text('{"bootstrap_resamples": 1000}', encoding="utf-8")
    monkeypatch.setattr(system, "METRICS_PATH", report)
    client = _client(tmp_path, stages=_stub_stages())
    assert client.get("/metrics").json() == {"bootstrap_resamples": 1000}


def test_feedback_is_logged_and_ledger_still_verifies(tmp_path):
    client = _client(tmp_path, stages=_stub_stages())
    study_id = _upload(client).json()["study_id"]
    _poll_until_done(client, study_id)

    before = client.get("/ledger/verify").json()
    resp = client.post(
        f"/studies/{study_id}/feedback",
        json={"finding_id": "f1", "decision": "accept", "actor": "dr_jane"},
    )
    assert resp.status_code == 200
    after = client.get("/ledger/verify").json()
    assert after["ok"] is True
    assert after["entries"] == before["entries"] + 1


def test_tampering_the_ledger_on_disk_fails_verification_end_to_end(tmp_path):
    client = _client(tmp_path, stages=_stub_stages())
    study_id = _upload(client).json()["study_id"]
    _poll_until_done(client, study_id)
    client.post(f"/studies/{study_id}/feedback", json={"finding_id": "f1", "decision": "reject"})

    assert client.get("/ledger/verify").json()["ok"] is True

    ledger_path = tmp_path / "ledger.jsonl"
    lines = ledger_path.read_text(encoding="utf-8").splitlines()
    tampered = json.loads(lines[0])
    tampered["actor"] = "someone else"
    lines[0] = json.dumps(tampered)
    ledger_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    assert client.get("/ledger/verify").json()["ok"] is False


def test_ledger_path_is_shared_correctly(tmp_path):
    """Sanity check on the test fixture itself: the app's ledger reads the same file we poke."""
    path = tmp_path / "ledger.jsonl"
    config = PipelineConfig(cache_dir=tmp_path / "c", artifact_root=tmp_path / "a", ledger_path=path)
    ledger = Ledger(path)
    ledger.append("probe", {})
    app = create_app(config=config, ledger=ledger)
    client = TestClient(app)
    assert client.get("/ledger/verify").json()["entries"] == 1
