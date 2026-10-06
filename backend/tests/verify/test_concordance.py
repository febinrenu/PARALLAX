from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from medproof.core.schemas import Finding, ImageEvidence
from medproof.readers.generalist import GeneralistRead
from medproof.verify.concordance import apply, box_iou, cohen_kappa
from medproof.verify.report_labels import labels_from_report


def finding(label="Effusion", box=(40.0, 40.0, 200.0, 160.0), modality="cxr", fid="f1") -> Finding:
    ev = ImageEvidence(
        evidence_id="ie_1", kind="bbox", bbox_xyxy=box, source_model="xrv@abcd1234", method="gradcam++",
        region_name="right lower zone",
    )
    return Finding(
        finding_id=fid, modality=modality, label=label, prob_raw=0.8, prob_calibrated=0.8, conformal_set=[],
        tier="high", status="uncertain", image_evidence=[ev] if box else [],
    )


def read(text: str, modality="cxr", boxes=None, ok=True) -> GeneralistRead:
    parsed = labels_from_report(text, modality)
    return GeneralistRead(
        ok=ok, modality=modality, source="live" if ok else "unavailable", model="google/medgemma-1.5-4b-it",
        findings_text=text, labels=parsed.labels, says_normal=parsed.says_normal, boxes=boxes or [], raw_ref="medgemma:abc",
    )


def test_iou_basics_and_symmetry():
    a, b = (0.0, 0.0, 10.0, 10.0), (5.0, 5.0, 15.0, 15.0)
    assert box_iou(a, a) == pytest.approx(1.0)
    assert box_iou(a, (20.0, 20.0, 30.0, 30.0)) == 0.0
    assert box_iou(a, b) == pytest.approx(25 / 175) == pytest.approx(box_iou(b, a))
    assert box_iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0.0


@given(
    st.tuples(*[st.floats(0, 500, allow_nan=False)] * 4),
    st.tuples(*[st.floats(0, 500, allow_nan=False)] * 4),
)
def test_iou_is_bounded_and_symmetric(a, b):
    ba = (min(a[0], a[2]), min(a[1], a[3]), max(a[0], a[2]), max(a[1], a[3]))
    bb = (min(b[0], b[2]), min(b[1], b[3]), max(b[0], b[2]), max(b[1], b[3]))
    v = box_iou(ba, bb)
    assert 0.0 <= v <= 1.0 and v == pytest.approx(box_iou(bb, ba))


def test_agreement_when_second_reader_reports_the_same_label():
    out, stage = apply([finding()], read("Small right pleural effusion."))
    sr = out[0].second_read
    assert sr is not None and sr.agrees is True and sr.label == "Effusion" and sr.raw_ref == "medgemma:abc"
    assert stage.stage == "concordance" and stage.ok and stage.payload["agree"] == 1


def test_disagreement_when_explicitly_negated_or_report_is_normal_or_names_something_else():
    for text in ["No pleural effusion. Lungs are clear.", "The lungs are clear. Normal chest X-ray.", "There is a right lower lobe consolidation."]:
        out, _ = apply([finding()], read(text))
        assert out[0].second_read.agrees is False, text
        assert "second_reader_disagrees" in out[0].flags


def test_silence_is_inconclusive_not_disagreement():
    out, _ = apply([finding()], read("The study was reviewed."))
    assert out[0].second_read.agrees is None and "second_reader_disagrees" not in out[0].flags


def test_hedged_second_read_still_counts_as_agreement():
    out, _ = apply([finding()], read("Possible small effusion."))
    assert out[0].second_read.agrees is True


def test_box_iou_is_computed_against_the_matching_box_only():
    boxes = [("Effusion", (40.0, 40.0, 200.0, 160.0)), ("Nodule", (0.0, 0.0, 5.0, 5.0))]
    out, _ = apply([finding()], read("Small right pleural effusion.", boxes=boxes))
    assert out[0].second_read.box_iou == pytest.approx(1.0)
    far = [("Effusion", (300.0, 300.0, 350.0, 350.0))]
    out2, _ = apply([finding()], read("Small right pleural effusion.", boxes=far))
    assert out2[0].second_read.box_iou == 0.0 and "second_box_mismatch" in out2[0].flags


def test_no_box_means_no_iou():
    out, _ = apply([finding()], read("Small right pleural effusion."))
    assert out[0].second_read.box_iou is None


def test_unavailable_second_read_leaves_findings_and_says_why():
    out, stage = apply([finding()], read("", ok=False))
    assert out[0].second_read is None and "second_read_unavailable" in out[0].flags
    assert stage.ok is False and stage.stage == "concordance" and stage.warnings


def test_skin_single_label_agreement_and_conflict():
    f = finding(label="mel", box=None, modality="skin_dermoscopy")
    ok, _ = apply([f], read("Suspicious for melanoma.", "skin_dermoscopy"))
    assert ok[0].second_read.agrees is True
    bad, _ = apply([f], read("Consistent with a benign nevus.", "skin_dermoscopy"))
    assert bad[0].second_read.agrees is False and bad[0].second_read.label == "nv"


def test_input_findings_are_not_mutated():
    f = finding()
    apply([f], read("No pleural effusion."))
    assert f.second_read is None and f.flags == []


def test_cohen_kappa_known_values():
    assert cohen_kappa([1, 1, 0, 0], [1, 1, 0, 0]) == pytest.approx(1.0)
    assert cohen_kappa([1, 1, 0, 0], [0, 0, 1, 1]) == pytest.approx(-1.0)
    assert cohen_kappa([1, 0, 1, 0], [1, 0, 0, 1]) == pytest.approx(0.0)
    # textbook example: po=0.7, pe=0.5 -> 0.4
    a = [1] * 20 + [1] * 5 + [0] * 10 + [0] * 15
    b = [1] * 20 + [0] * 5 + [1] * 10 + [0] * 15
    assert cohen_kappa(a, b) == pytest.approx(0.4, abs=1e-9)
    assert cohen_kappa(["a", "b", "c", "a"], ["a", "b", "c", "a"]) == pytest.approx(1.0)
    assert cohen_kappa([1, 1, 1], [1, 1, 1]) == 1.0  # single class everywhere: perfect agreement
    with pytest.raises(ValueError):
        cohen_kappa([1], [1, 0])
