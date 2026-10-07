from __future__ import annotations

import pytest

from medproof.llm.errors import LLMUnavailable
from medproof.report.drafts import (
    ClaimDraft,
    ClaimDrafts,
    build_input,
    draft_claims,
    template_drafts,
    to_claim,
)
from medproof.report.firewall import run
from tests.report.conftest import make_view


class FakePool:
    def __init__(self, drafts=None, exc=None):
        self.drafts, self.exc, self.calls = drafts, exc, []

    def chat(self, service, prompt_name, user, schema, *, fallback=None, **kw):
        self.calls.append((service, prompt_name, user))
        if self.exc:
            if fallback is None:
                raise self.exc

            class R:
                data = fallback()
                source = "fallback"
                ok = False
                warning = str(self.exc)

            return R()

        class R:  # noqa: F811
            data = self.drafts
            source = "live"
            ok = True
            warning = None

        return R()


def test_template_drafts_skip_rejected_findings_and_pick_sensible_frames(view):
    drafts = template_drafts(view)
    by_finding = {d.finding: d for d in drafts.drafts}
    assert "f2" not in by_finding  # rejected
    assert by_finding["f4"].frame == "discordant_review"
    assert by_finding["f8"].frame == "abstain_note"
    assert by_finding["f1"].frame == "supported_by_note" and by_finding["f1"].note_ref == "te_1"
    assert by_finding["f9"].frame == "consider"


def test_template_claims_all_pass_the_firewall(view):
    claims = [to_claim(d, i, view) for i, d in enumerate(template_drafts(view).drafts, 1)]
    out, stage = run(claims, view)
    assert len(out) == 5 and stage.payload["blocked"] == 0 and stage.payload["rendered"] == 5
    assert all(c.rendered.startswith("Doctor,") for c in out)


def test_to_claim_builds_a_slot_template_with_the_findings_evidence(view):
    c = to_claim(ClaimDraft(frame="supported_by_note", finding="f1", note_ref="te_1"), 3, view)
    assert c.claim_id == "c3"
    assert c.template == "Doctor, consider {f1.label} ({f1.tier}); the note states {te_1.quote}."
    assert c.evidence_ids == ["ie_1", "te_1"]
    plain = to_claim(ClaimDraft(frame="consider", finding="f1"), 1, view)
    assert plain.template == "Doctor, consider {f1.label} in the {f1.region} ({f1.tier})." and plain.evidence_ids == ["ie_1"]


def test_llm_chooses_frames_but_the_firewall_still_decides(view):
    bad = ClaimDrafts(drafts=[
        ClaimDraft(frame="consider", finding="f8"),  # abstain tier cannot be 'consider'
        ClaimDraft(frame="supported_by_note", finding="f1", note_ref="te_1"),
        ClaimDraft(frame="consider", finding="f2"),  # rejected finding
        ClaimDraft(frame="consider", finding="f99"),  # does not exist
        ClaimDraft(frame="supported_by_note", finding="f1", note_ref="te_99"),  # unknown note ref
    ])
    pool = FakePool(bad)
    claims, stage = draft_claims(view, pool)
    rendered = [c.rendered for c in claims if c.rendered]
    assert len(rendered) == 1 and "the note states" in rendered[0]
    assert stage.stage == "report" and stage.payload["blocked"] == 4
    assert pool.calls[0][0] == "report"


def test_llm_input_has_ids_and_states_but_no_free_text(view):
    text = build_input(view)
    assert "f1" in text and "te_1" in text and "tier" in text
    assert "fever" not in text and "Consolidation" not in text  # quotes and labels never reach the model


def test_offline_fallback_when_groq_is_unavailable(view):
    claims, stage = draft_claims(view, FakePool(exc=LLMUnavailable("down")))
    assert claims and all(c.rendered for c in claims[:6])
    assert any("template" in w for w in stage.warnings)


def test_no_pool_uses_templates_directly(view):
    claims, stage = draft_claims(view, None)
    assert claims and any("template" in w for w in stage.warnings)


def test_draft_schema_only_allows_known_frames():
    with pytest.raises(ValueError):
        ClaimDraft(frame="the patient has", finding="f1")


def test_empty_study_gives_no_claims_and_a_clear_stage():
    claims, stage = draft_claims(make_view().__class__.build([], {}, {}), None)
    assert claims == [] and stage.ok


def test_template_report_cites_a_clinical_quote_rather_than_a_side_word():
    from medproof.core.schemas import TextEvidence
    from medproof.report.drafts import template_drafts
    from medproof.report.study_view import StudyView
    from tests.report.conftest import NOTE, NOTE_ID, finding, ie

    side = TextEvidence(evidence_id="te_1", note_id=NOTE_ID, span=(0, 3), quote=NOTE[0:3], fact_type="laterality", polarity="supports")
    fever = TextEvidence(evidence_id="te_2", note_id=NOTE_ID, span=(NOTE.index("fever"), NOTE.index("fever") + 5), quote="fever",
                         fact_type="symptom", polarity="supports")
    f = finding("f1", "Pneumonia", "moderate", "verified", [ie("ie_1", True)], [side, fever])
    d = template_drafts(StudyView.build([f], {NOTE_ID: NOTE}, {})).drafts[0]
    assert d.frame == "supported_by_note" and d.note_ref == "te_2"


def test_template_report_does_not_use_the_note_frame_when_only_a_side_word_supports():
    from medproof.core.schemas import TextEvidence
    from medproof.report.drafts import template_drafts
    from medproof.report.study_view import StudyView
    from tests.report.conftest import NOTE, NOTE_ID, finding, ie

    side = TextEvidence(evidence_id="te_1", note_id=NOTE_ID, span=(0, 3), quote=NOTE[0:3], fact_type="laterality", polarity="supports")
    f = finding("f1", "Pneumonia", "moderate", "verified", [ie("ie_1", True)], [side])
    assert template_drafts(StudyView.build([f], {NOTE_ID: NOTE}, {})).drafts[0].frame == "consider"


# ---- wording a doctor reads: names, not dataset codes, and no phrase about a region nobody named

@pytest.mark.parametrize("modality,code,spoken", [
    ("skin_dermoscopy", "mel", "melanoma"),
    ("skin_dermoscopy", "nv", "melanocytic nevus"),
    ("skin_dermoscopy", "bcc", "basal cell carcinoma"),
    ("skin_dermoscopy", "akiec", "actinic keratosis"),
    ("skin_dermoscopy", "bkl", "benign keratosis"),
    ("skin_dermoscopy", "df", "dermatofibroma"),
    ("skin_dermoscopy", "vasc", "vascular lesion"),
    ("brain_mri", "pituitary", "pituitary tumour"),
    ("brain_mri", "glioma", "glioma"),
    ("cxr", "Pleural_Thickening", "pleural thickening"),
    ("bone_xray", "fracture", "fracture"),
])
def test_findings_are_named_for_a_doctor_not_by_dataset_code(modality, code, spoken):
    from medproof.report.render import display_label

    assert display_label(modality, code).lower() == spoken


def test_a_claim_names_the_finding_in_words_and_skips_an_unnamed_region():
    from medproof.core.schemas import Finding, ImageEvidence
    from medproof.report import firewall
    from medproof.report.drafts import ClaimDraft, to_claim
    from medproof.report.study_view import StudyView

    ev = ImageEvidence(evidence_id="ie_1", kind="heatmap", source_model="m@abcd1234", method="g", faithful=True)  # no region name
    f = Finding(finding_id="f1", modality="skin_dermoscopy", label="bcc", prob_raw=0.9, prob_calibrated=0.9, conformal_set=[], tier="high",
                status="verified", image_evidence=[ev])
    view = StudyView.build([f], {}, {})
    claims, _ = firewall.run([to_claim(ClaimDraft(frame="consider", finding="f1"), 1, view)], view)
    text = claims[0].rendered or ""
    assert claims[0].blocked_reason is None, claims[0].blocked_reason
    assert text == "Doctor, consider basal cell carcinoma (high confidence)."


def test_a_named_region_is_still_used():
    from medproof.report import firewall
    from medproof.report.drafts import ClaimDraft, to_claim
    from medproof.report.study_view import StudyView
    from tests.report.conftest import finding, ie

    f = finding("f1", "Pneumonia", "moderate", "verified", [ie("ie_1", True)])
    view = StudyView.build([f], {}, {})
    claims, _ = firewall.run([to_claim(ClaimDraft(frame="consider", finding="f1"), 1, view)], view)
    assert "right lower zone" in (claims[0].rendered or "")


# ---- a note and an image that disagree about the side must be said, not reported at full confidence

def _conflicted(frame="consider"):
    from medproof.report.drafts import ClaimDraft, to_claim
    from medproof.report.study_view import StudyView
    from tests.report.conftest import finding, ie

    f = finding("f1", "Pneumonia", "high", "verified", [ie("ie_1", True)]).model_copy(update={"flags": ["laterality_conflict"]})
    view = StudyView.build([f], {}, {})
    return to_claim(ClaimDraft(frame=frame, finding="f1"), 1, view), view


@pytest.mark.parametrize("frame", ["consider", "supported_by_note"])
def test_a_laterality_conflict_replaces_the_confident_sentence_with_a_request_to_review(frame):
    from medproof.report import firewall

    claim, view = _conflicted(frame)
    claims, _ = firewall.run([claim], view)
    text = claims[0].rendered or ""
    assert claims[0].blocked_reason is None, claims[0].blocked_reason
    assert "disagree about the side" in text and "review the image" in text
    assert "confidence" not in text  # it must not repeat the model's tier as if the conflict did not exist


def test_a_finding_without_a_conflict_is_unchanged():
    from medproof.report import firewall
    from medproof.report.drafts import ClaimDraft, to_claim
    from medproof.report.study_view import StudyView
    from tests.report.conftest import finding, ie

    f = finding("f1", "Pneumonia", "high", "verified", [ie("ie_1", True)])
    view = StudyView.build([f], {}, {})
    claims, _ = firewall.run([to_claim(ClaimDraft(frame="consider", finding="f1"), 1, view)], view)
    assert "high confidence" in (claims[0].rendered or "")
