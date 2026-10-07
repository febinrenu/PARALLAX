from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from fhir.resources.R4B.bundle import Bundle

from medproof.report.drafts import template_drafts, to_claim
from medproof.report.fhir import DISCLAIMER, build_bundle, validate_bundle
from medproof.report.firewall import run
from tests.report.conftest import build_findings, make_view

ISSUED = datetime(2026, 10, 7, 12, 0, tzinfo=timezone.utc)


def claims_for(view):
    out, _ = run([to_claim(d, i, view) for i, d in enumerate(template_drafts(view).drafts, 1)], view)
    return out


def bundle(**kw):
    view = make_view()
    args = dict(study_id="study-0001", issued=ISSUED, ledger_head="ab" * 32)
    args.update(kw)
    return build_bundle(build_findings(), claims_for(view), **args)


def by_type(b, kind):
    return [e["resource"] for e in b["entry"] if e["resource"]["resourceType"] == kind]


def test_bundle_is_valid_fhir_r4b_and_round_trips():
    b = bundle()
    model = Bundle.model_validate(b)
    assert model.type == "collection" and validate_bundle(b) is not None
    assert json.loads(json.dumps(b)) == b  # plain JSON


def test_one_diagnostic_report_that_is_preliminary_and_marked_decision_support():
    (dr,) = by_type(bundle(), "DiagnosticReport")
    assert dr["status"] == "preliminary"
    assert dr["category"][0]["coding"][0]["code"] == "RAD"
    assert dr["code"]["coding"][0]["system"] == "http://loinc.org"
    assert DISCLAIMER in dr["text"]["div"] and "Doctor," in dr["conclusion"]
    assert dr["subject"]["display"] == "De-identified study"
    assert {i["system"]: i["value"] for i in dr["identifier"]} == {
        "urn:parallax:study": "study-0001", "urn:parallax:ledger-head": "ab" * 32,
    }
    assert dr["issued"].startswith("2026-10-07T12:00:00")


def test_only_supported_non_rejected_findings_become_observations():
    obs = by_type(bundle(), "Observation")
    labels = {o["code"]["text"] for o in obs}
    assert labels == {"Consolidation", "Nodule", "Atelectasis", "Hernia", "Pneumothorax"}
    assert "Effusion" not in labels  # rejected
    assert "Mass" not in labels and "Edema" not in labels and "Fibrosis" not in labels  # unsupported
    assert all(o["status"] == "preliminary" for o in obs)
    assert all(o["category"][0]["coding"][0]["code"] == "imaging" for o in obs)


def test_every_result_reference_resolves_to_a_bundle_entry_and_ids_are_stable():
    b = bundle()
    urls = {e["fullUrl"] for e in b["entry"]}
    (dr,) = by_type(b, "DiagnosticReport")
    assert dr["result"] and all(r["reference"] in urls for r in dr["result"])
    assert len(urls) == len(b["entry"])
    assert [e["fullUrl"] for e in bundle()["entry"]] == [e["fullUrl"] for e in b["entry"]]
    assert [e["fullUrl"] for e in bundle(study_id="study-0002")["entry"]] != [e["fullUrl"] for e in b["entry"]]


def test_observation_carries_probability_tier_status_and_region():
    obs = {o["code"]["text"]: o for o in by_type(bundle(), "Observation")}
    c = obs["Consolidation"]
    comp = {x["code"]["text"]: x for x in c["component"]}
    assert comp["calibrated probability"]["valueQuantity"]["value"] == pytest.approx(0.8)
    assert comp["tier"]["valueString"] == "high" and comp["status"]["valueString"] == "verified"
    assert c["bodySite"]["text"] == "right lower zone"
    assert c["valueString"].startswith("Doctor,")
    assert "bodySite" not in obs["Pneumothorax"]  # that finding has no region name


def test_blocked_claims_never_reach_the_conclusion():
    view = make_view()
    claims = claims_for(view)
    claims[0] = claims[0].model_copy(update={"blocked_reason": "entailment: not supported", "entailed": False})
    b = build_bundle(build_findings(), claims, study_id="s", issued=ISSUED, ledger_head="00" * 32)
    (dr,) = by_type(b, "DiagnosticReport")
    assert claims[0].rendered not in dr["conclusion"]
    assert claims[1].rendered in dr["conclusion"]


def test_note_quotes_are_left_out_unless_asked_for():
    plain = json.dumps(bundle())
    assert "fever" not in plain and "[note text omitted]" in plain
    with_q = json.dumps(bundle(include_quotes=True))
    assert "fever" in with_q
    Bundle.model_validate(bundle(include_quotes=True))


def test_narrative_escapes_markup_so_it_stays_valid_xhtml():
    view = make_view()
    claims = claims_for(view)
    claims[0] = claims[0].model_copy(update={"rendered": 'Doctor, consider <script>alert("x")</script> & more.'})
    b = build_bundle(build_findings(), claims, study_id="s", issued=ISSUED, ledger_head="00" * 32)
    (dr,) = by_type(b, "DiagnosticReport")
    assert "<script>" not in dr["text"]["div"] and "&lt;script&gt;" in dr["text"]["div"]
    Bundle.model_validate(b)


def test_empty_study_still_gives_a_valid_report_that_says_nothing_was_reported():
    b = build_bundle([], [], study_id="s", issued=ISSUED, ledger_head="00" * 32)
    Bundle.model_validate(b)
    (dr,) = by_type(b, "DiagnosticReport")
    assert "no findings" in dr["conclusion"].lower() and "result" not in dr


def test_structurally_invalid_resources_are_rejected_by_validation():
    b = bundle()
    obs = next(e for e in b["entry"] if e["resource"]["resourceType"] == "Observation")
    del obs["resource"]["code"]  # required by FHIR
    with pytest.raises(Exception):  # noqa: B017 - pydantic ValidationError from the FHIR model
        validate_bundle(b)
    b2 = bundle()
    b2["entry"][0]["resource"]["issued"] = "not-a-date"
    with pytest.raises(Exception):  # noqa: B017
        validate_bundle(b2)


def test_study_id_is_sanitised_for_the_fhir_id_field():
    b = build_bundle([], [], study_id="Study 1/ä?", issued=ISSUED, ledger_head="00" * 32)
    Bundle.model_validate(b)
    assert all(c.isalnum() or c in "-." for c in b["id"])
