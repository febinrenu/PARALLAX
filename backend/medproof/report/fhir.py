"""FHIR export (D20): a collection Bundle holding one DiagnosticReport and one Observation per finding.

Validated with `fhir.resources` against R4B, the nearest R4 release the library ships (the resources
used here are unchanged between R4 and R4B). Everything is marked preliminary decision support.

Only what the report itself would show is exported: findings that are not rejected and are supported
by evidence, and claims that passed the firewall and were not removed by the entailment judge. Note
text is left out unless `include_quotes=True`, because a quote can carry patient information.
"""

from __future__ import annotations

import html
import json
import re
import uuid
from datetime import UTC, datetime
from typing import Any

from fhir.resources.R4B.bundle import Bundle

from medproof.core.schemas import Claim, Finding
from medproof.report import lexicon
from medproof.report.firewall import is_supported

DISCLAIMER = "Decision support only. Not a diagnosis and not a medical device; a clinician must review every finding."
_OBS_CATEGORY = {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "imaging", "display": "Imaging"}]}
_UCUM = "http://unitsofmeasure.org"


def _uuid(study_id: str, key: str) -> str:
    return "urn:uuid:" + str(uuid.uuid5(uuid.NAMESPACE_URL, f"urn:parallax:{study_id}/{key}"))


def _fhir_id(study_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9\-.]", "-", study_id)[:64] or "study"


_QUOTE = re.compile(r'"[^"]*"')
QUOTE_OMITTED = "[note text omitted]"


def _scrub(text: str, include_quotes: bool) -> str:
    """Rendered sentences quote the note; drop those quotes unless the caller opted in."""
    return text if include_quotes else _QUOTE.sub(QUOTE_OMITTED, text)


def _reportable(c: Claim) -> bool:
    return bool(c.rendered) and c.blocked_reason is None and c.entailed is not False


def _region(f: Finding) -> str | None:
    return next((e.region_name for e in f.image_evidence if e.region_name), None)


def _observation(f: Finding, claims: list[Claim], include_quotes: bool) -> dict[str, Any]:
    sentence = next((c.rendered for c in claims if _reportable(c) and "{" + f.finding_id + "." in c.template and c.rendered), None)
    text = _scrub(sentence or f"Doctor, consider {f.label.replace('_', ' ')} ({lexicon.TIER_PHRASE[f.tier]}).", include_quotes)
    comps: list[dict[str, Any]] = [
        {"code": {"text": "calibrated probability"},
         "valueQuantity": {"value": round(f.prob_calibrated, 3), "unit": "1", "system": _UCUM, "code": "1"}},
        {"code": {"text": "tier"}, "valueString": f.tier},
        {"code": {"text": "status"}, "valueString": f.status},
    ]
    if f.conformal_set:
        comps.append({"code": {"text": "conformal set"}, "valueString": ", ".join(f.conformal_set)})
    if f.flags:
        comps.append({"code": {"text": "flags"}, "valueString": ", ".join(f.flags)})
    if f.second_read is not None:
        verdict = {True: "agrees", False: "disagrees", None: "inconclusive"}[f.second_read.agrees]
        comps.append({"code": {"text": "second reader"}, "valueString": f"{f.second_read.model}: {verdict}"})
    obs: dict[str, Any] = {
        "resourceType": "Observation",
        "status": "preliminary",
        "category": [_OBS_CATEGORY],
        "code": {"coding": [{"system": "urn:parallax:finding", "code": re.sub(r"\W+", "-", f.label.lower()), "display": f.label}], "text": f.label},
        "valueString": text,
        "component": comps,
    }
    if (region := _region(f)):
        obs["bodySite"] = {"text": region}
    if f.image_evidence:
        e = f.image_evidence[0]
        obs["method"] = {"text": f"{e.method}; {e.source_model}"}
    if include_quotes and f.text_evidence:
        obs["note"] = [{"text": "Note evidence: " + "; ".join(f'"{t.quote}" ({t.polarity})' for t in f.text_evidence)}]
    return obs


def build_bundle(
    findings: list[Finding],
    claims: list[Claim],
    *,
    study_id: str,
    issued: datetime,
    ledger_head: str,
    include_quotes: bool = False,
    disclaimer: str = DISCLAIMER,
) -> dict[str, Any]:
    exported = [f for f in findings if f.status != "rejected" and is_supported(f)]
    entries: list[dict[str, Any]] = []
    refs: list[dict[str, str]] = []
    for f in exported:
        url = _uuid(study_id, f.finding_id)
        entries.append({"fullUrl": url, "resource": _observation(f, claims, include_quotes)})
        refs.append({"reference": url})
    lines = [_scrub(c.rendered, include_quotes) for c in claims if _reportable(c) and c.rendered]
    conclusion = " ".join(lines) if lines else "No findings were reported by this decision-support system."
    body = "".join(f"<p>{html.escape(x)}</p>" for x in (lines or [conclusion]))
    report: dict[str, Any] = {
        "resourceType": "DiagnosticReport",
        "status": "preliminary",
        "meta": {"tag": [{"system": "urn:parallax:flags", "code": "decision-support-only"}]},
        "text": {"status": "generated", "div": f'<div xmlns="http://www.w3.org/1999/xhtml">{body}<p><b>{html.escape(disclaimer)}</b></p></div>'},
        "identifier": [
            {"system": "urn:parallax:study", "value": study_id},
            {"system": "urn:parallax:ledger-head", "value": ledger_head},
        ],
        "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/v2-0074", "code": "RAD", "display": "Radiology"}]}],
        "code": {"coding": [{"system": "http://loinc.org", "code": "18748-4", "display": "Diagnostic imaging study"}]},
        "subject": {"display": "De-identified study"},
        "effectiveDateTime": issued.astimezone(UTC).isoformat(timespec="seconds"),
        "issued": issued.astimezone(UTC).isoformat(timespec="seconds"),
        "conclusion": conclusion,
    }
    if refs:
        report["result"] = refs
    entries.insert(0, {"fullUrl": _uuid(study_id, "report"), "resource": report})
    bundle = {"resourceType": "Bundle", "id": _fhir_id(study_id), "type": "collection", "entry": entries}
    return validate_bundle(bundle)


def validate_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    """Validate with fhir.resources and return the normalised JSON; raises pydantic.ValidationError."""
    model = Bundle.model_validate(bundle)
    normalised: dict[str, Any] = json.loads(model.model_dump_json(exclude_none=True))
    return normalised
