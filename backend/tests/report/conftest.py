"""A small synthetic study: findings, evidence and one note, shared by the report tests."""

from __future__ import annotations

import pytest

from medproof.core.schemas import Claim, Finding, ImageEvidence, TextEvidence
from medproof.report.study_view import StudyView

NOTE = "63F c/o fever and cough x3 days. No chest pain."
NOTE_ID = "n_1"


def span(sub: str, text: str = NOTE) -> tuple[int, int]:
    i = text.index(sub)
    return (i, i + len(sub))


def ie(eid: str, faithful: bool | None, region: str | None = "right lower zone") -> ImageEvidence:
    return ImageEvidence(
        evidence_id=eid,
        kind="bbox",
        bbox_xyxy=(1.0, 2.0, 30.0, 40.0),
        source_model="xrv-densenet121-all@abcd1234",
        method="gradcam++",
        region_name=region,
        faithful=faithful,
    )


def te(eid: str, sub: str, polarity: str = "supports", quote: str | None = None) -> TextEvidence:
    return TextEvidence(
        evidence_id=eid,
        note_id=NOTE_ID,
        span=span(sub),
        quote=quote if quote is not None else sub,
        fact_type="symptom",
        polarity=polarity,  # type: ignore[arg-type]
    )


def finding(fid: str, label: str, tier: str, status: str, image=None, text=None, **kw) -> Finding:
    return Finding(
        finding_id=fid,
        modality="cxr",
        label=label,
        prob_raw=0.8,
        prob_calibrated=0.8,
        conformal_set=kw.pop("conformal_set", []),
        tier=tier,  # type: ignore[arg-type]
        status=status,  # type: ignore[arg-type]
        image_evidence=image or [],
        text_evidence=text or [],
        **kw,
    )


def build_findings() -> list[Finding]:
    return [
        # f1: well supported, high tier, verified
        finding("f1", "Consolidation", "high", "verified", [ie("ie_1", True)], [te("te_1", "fever")]),
        # f2: rejected, must never be referenced
        finding("f2", "Effusion", "low", "rejected", [ie("ie_2", True)]),
        # f3: low tier, uncertain, supported only by text
        finding("f3", "Nodule", "low", "uncertain", [ie("ie_3", None)], [te("te_3", "cough")]),
        # f4: discordant but with faithful image evidence
        finding(
            "f4", "Atelectasis", "moderate", "discordant", [ie("ie_4", True)],
            second_read=None, flags=["laterality_conflict"],
        ),
        # f5: image evidence not faithful, no text: unsupported
        finding("f5", "Mass", "moderate", "uncertain", [ie("ie_5", False)]),
        # f6: faithful unknown, only contradicting text
        finding("f6", "Edema", "moderate", "uncertain", [ie("ie_6", None)], [te("te_6", "No chest pain", "contradicts")]),
        # f7: faithful unknown, only neutral text
        finding("f7", "Fibrosis", "moderate", "uncertain", [ie("ie_7", None)], [te("te_7", "x3 days", "neutral")]),
        # f8: abstain tier but text support
        finding("f8", "Hernia", "abstain", "uncertain", [ie("ie_8", None)], [te("te_8", "fever")]),
        # f9: moderate, verified by faithful image, no region name
        finding("f9", "Pneumothorax", "moderate", "verified", [ie("ie_9", True, region=None)]),
    ]


def make_view(extra_text=(), flagged=None, notes=None) -> StudyView:
    findings = build_findings()
    if extra_text:
        findings[0] = findings[0].model_copy(update={"text_evidence": [*findings[0].text_evidence, *extra_text]})
    return StudyView.build(findings, notes if notes is not None else {NOTE_ID: NOTE}, flagged or {})


@pytest.fixture
def view() -> StudyView:
    return make_view()


def claim(template: str, ids: list[str], cid: str = "c1") -> Claim:
    return Claim(claim_id=cid, template=template, evidence_ids=ids)
