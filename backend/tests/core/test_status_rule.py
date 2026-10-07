"""Exhaustive truth-table tests for the status rule (plan.md section 4)."""

from __future__ import annotations

from medproof.core.schemas import Finding, ImageEvidence, SecondRead, Stability, TextEvidence
from medproof.core.status_rule import compute_status


def _finding(**overrides) -> Finding:
    base = {
        "finding_id": "f1",
        "modality": "cxr",
        "label": "Pneumonia",
        "prob_raw": 0.8,
        "prob_calibrated": 0.8,
        "conformal_set": [],
        "tier": "high",
        "status": "uncertain",  # input value is irrelevant: compute_status ignores it
    }
    base.update(overrides)
    return Finding(**base)


def _evidence(**overrides) -> ImageEvidence:
    base = {"evidence_id": "ie_1", "kind": "heatmap", "source_model": "m@sha", "method": "gradcam++"}
    base.update(overrides)
    return ImageEvidence(**base)


def test_no_evidence_is_rejected():
    assert compute_status(_finding()) == "rejected"


def test_faithful_image_evidence_alone_is_verified():
    f = _finding(image_evidence=[_evidence(faithful=True)])
    assert compute_status(f) == "verified"


def test_unfaithful_image_evidence_alone_is_rejected():
    f = _finding(image_evidence=[_evidence(faithful=False)])
    assert compute_status(f) == "rejected"


def test_unset_faithfulness_is_rejected():
    f = _finding(image_evidence=[_evidence(faithful=None)])
    assert compute_status(f) == "rejected"


def test_supporting_text_evidence_alone_is_verified():
    te = TextEvidence(evidence_id="te_1", note_id="n1", span=(0, 5), quote="fever", fact_type="symptom", polarity="supports")
    assert compute_status(_finding(text_evidence=[te])) == "verified"


def test_contradicting_text_evidence_alone_is_rejected():
    te = TextEvidence(evidence_id="te_1", note_id="n1", span=(0, 5), quote="fever", fact_type="symptom", polarity="contradicts")
    assert compute_status(_finding(text_evidence=[te])) == "rejected"


def test_second_reader_disagreement_is_discordant_even_with_valid_evidence():
    f = _finding(
        image_evidence=[_evidence(faithful=True)],
        second_read=SecondRead(model="medgemma", label="Atelectasis", agrees=False, box_iou=None, raw_ref="r1"),
    )
    assert compute_status(f) == "discordant"


def test_low_box_iou_is_discordant():
    f = _finding(
        image_evidence=[_evidence(faithful=True)],
        second_read=SecondRead(model="medgemma", label="Pneumonia", agrees=True, box_iou=0.05, raw_ref="r1"),
    )
    assert compute_status(f) == "discordant"


def test_sufficient_box_iou_does_not_block_verified():
    f = _finding(
        image_evidence=[_evidence(faithful=True)],
        second_read=SecondRead(model="medgemma", label="Pneumonia", agrees=True, box_iou=0.5, raw_ref="r1"),
    )
    assert compute_status(f) == "verified"


def test_high_flip_rate_is_uncertain():
    f = _finding(image_evidence=[_evidence(faithful=True)], stability=Stability(tests=8, flip_rate=0.5, worst_perturbation="blur"))
    assert compute_status(f) == "uncertain"


def test_flip_rate_at_limit_is_still_verified():
    f = _finding(image_evidence=[_evidence(faithful=True)], stability=Stability(tests=8, flip_rate=0.25, worst_perturbation=None))
    assert compute_status(f) == "verified"


def test_ood_flag_is_uncertain():
    f = _finding(image_evidence=[_evidence(faithful=True)], flags=["ood"])
    assert compute_status(f) == "uncertain"


def test_large_conformal_set_is_uncertain():
    f = _finding(image_evidence=[_evidence(faithful=True)], conformal_set=["a", "b", "c"])
    assert compute_status(f) == "uncertain"


def test_conformal_set_at_limit_is_still_verified():
    f = _finding(image_evidence=[_evidence(faithful=True)], conformal_set=["a", "b"])
    assert compute_status(f) == "verified"


def test_low_tier_is_uncertain():
    f = _finding(image_evidence=[_evidence(faithful=True)], tier="low")
    assert compute_status(f) == "uncertain"


def test_rejected_beats_everything():
    """No valid evidence outranks a second reader that happens to agree."""
    f = _finding(second_read=SecondRead(model="medgemma", label="Pneumonia", agrees=True, box_iou=0.9, raw_ref="r1"))
    assert compute_status(f) == "rejected"


def test_discordant_beats_otherwise_clean_signals():
    """Good evidence, good stability, small conformal set, high tier — still discordant."""
    f = _finding(
        image_evidence=[_evidence(faithful=True)],
        stability=Stability(tests=8, flip_rate=0.0, worst_perturbation=None),
        second_read=SecondRead(model="medgemma", label="Atelectasis", agrees=False, box_iou=None, raw_ref="r1"),
    )
    assert compute_status(f) == "discordant"


# P3-2: a note can lift a finding to uncertain, but only a region that passed the deletion test verifies it.
_SUPPORT = TextEvidence(evidence_id="te_1", note_id="n1", span=(0, 5), quote="fever", fact_type="symptom", polarity="supports")


def test_untested_region_with_supporting_note_is_capped_at_uncertain():
    f = _finding(image_evidence=[_evidence(faithful=None)], text_evidence=[_SUPPORT])
    assert compute_status(f) == "uncertain"


def test_failed_region_with_supporting_note_is_capped_at_uncertain():
    f = _finding(image_evidence=[_evidence(faithful=False)], text_evidence=[_SUPPORT])
    assert compute_status(f) == "uncertain"


def test_passed_region_with_supporting_note_is_verified():
    f = _finding(image_evidence=[_evidence(faithful=True)], text_evidence=[_SUPPORT])
    assert compute_status(f) == "verified"


def test_one_passed_region_is_enough_among_several():
    f = _finding(image_evidence=[_evidence(faithful=False), _evidence(evidence_id="ie_2", kind="mask", faithful=True)])
    assert compute_status(f) == "verified"
