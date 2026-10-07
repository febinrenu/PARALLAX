from __future__ import annotations

import pytest

from medproof.verify.report_labels import labels_from_report


def state(text: str, modality: str) -> dict[str, str]:
    """label -> 'present' | 'hedged' | 'absent' for every label the report mentions."""
    out = {}
    for lab in labels_from_report(text, modality).labels:
        out[lab.label] = "absent" if not lab.present else ("hedged" if lab.hedged else "present")
    return out


REAL_NORMAL = (
    "FINDINGS: The lungs are clear. The heart size is normal. There is no pleural effusion or pneumothorax. "
    "No consolidation or mass is seen. IMPRESSION: Normal chest X-ray."
)


def test_real_medgemma_normal_report_has_no_positives_and_says_normal():
    r = labels_from_report(REAL_NORMAL, "cxr")
    assert not [x for x in r.labels if x.present]
    assert r.says_normal
    assert {x.label for x in r.labels if not x.present} >= {"Effusion", "Pneumothorax", "Consolidation", "Mass"}


@pytest.mark.parametrize(
    "text,expected",
    [
        ("There is a right lower lobe consolidation. Small left pleural effusion.", {"Consolidation": "present", "Effusion": "present"}),
        ("Possible nodule in the left upper zone.", {"Nodule": "hedged"}),
        ("No evidence of pneumothorax, but there is mild cardiomegaly.", {"Pneumothorax": "absent", "Cardiomegaly": "present"}),
        ("The heart is enlarged.", {"Cardiomegaly": "present"}),
        ("No pleural effusion, pneumothorax, or consolidation.", {"Effusion": "absent", "Pneumothorax": "absent", "Consolidation": "absent"}),
        ("Pulmonary edema is present.", {"Edema": "present"}),
        ("A previously seen effusion has resolved.", {"Effusion": "absent"}),
        ("Cannot exclude pneumonia.", {"Pneumonia": "hedged"}),
        ("Suspicious for a mass in the right hilum.", {"Mass": "hedged"}),
        ("Hyperinflated lungs with flattened diaphragms.", {"Emphysema": "hedged"}),
        ("Linear atelectasis at the left base.", {"Atelectasis": "present"}),
        ("Airspace opacity in the right mid zone.", {"Lung Opacity": "present"}),
        ("Pleural thickening on the left.", {"Pleural_Thickening": "present"}),
        ("Widened mediastinum.", {"Enlarged Cardiomediastinum": "present"}),
        ("There is no acute fracture.", {"Fracture": "absent"}),
    ],
)
def test_cxr_phrases(text, expected):
    got = state(text, "cxr")
    for label, st in expected.items():
        assert got.get(label) == st, (text, got)


def test_says_normal_for_global_negative_statements():
    for t in ["No acute cardiopulmonary abnormality.", "Impression: normal chest radiograph.", "The lungs are clear."]:
        assert labels_from_report(t, "cxr").says_normal, t
    assert not labels_from_report("Right lower lobe consolidation.", "cxr").says_normal


def test_negation_scope_stops_at_but_and_sentence_end():
    got = state("No pneumothorax. Consolidation is present at the base.", "cxr")
    assert got == {"Pneumothorax": "absent", "Consolidation": "present"}


@pytest.mark.parametrize(
    "text,expected",
    [
        ("No acute fracture or dislocation.", "absent"),
        ("Transverse fracture of the distal radius.", "present"),
        ("Cortical irregularity suspicious for a nondisplaced fracture.", "hedged"),
        ("The bone is broken at the mid shaft.", "present"),
        ("There is no evidence of fracture.", "absent"),
        ("A fracture line is seen through the femoral neck.", "present"),
        ("No fracture is identified.", "absent"),
    ],
)
def test_bone_fracture_phrases(text, expected):
    assert state(text, "bone_xray").get("fracture") == expected, text


def test_bone_normal_report():
    assert labels_from_report("FINDINGS: The bones are intact. IMPRESSION: Normal wrist radiograph.", "bone_xray").says_normal


@pytest.mark.parametrize(
    "text,label,st",
    [
        ("Findings are consistent with a melanocytic nevus.", "nv", "hedged"),
        ("Irregular pigment network, suspicious for melanoma.", "mel", "hedged"),
        ("Features of seborrheic keratosis.", "bkl", "present"),
        ("Arborizing vessels typical of basal cell carcinoma.", "bcc", "present"),
        ("Actinic keratosis.", "akiec", "present"),
        ("A dermatofibroma with central white patch.", "df", "present"),
        ("Cherry hemangioma.", "vasc", "present"),
        ("No features of melanoma.", "mel", "absent"),
        ("Benign nevus.", "nv", "present"),
        ("A solar lentigo.", "bkl", "present"),
    ],
)
def test_skin_phrases(text, label, st):
    assert state(text, "skin_dermoscopy").get(label) == st, text


@pytest.mark.parametrize(
    "text,label,st",
    [
        ("A well-circumscribed extra-axial mass consistent with meningioma.", "meningioma", "hedged"),
        ("Pituitary macroadenoma in the sella.", "pituitary", "present"),
        ("There is a glioma in the left frontal lobe.", "glioma", "present"),
        ("No tumor is seen.", "notumor", "present"),
        ("No intracranial mass lesion.", "notumor", "present"),
        ("Normal brain MRI.", "notumor", "present"),
    ],
)
def test_brain_phrases(text, label, st):
    assert state(text, "brain_mri").get(label) == st, text


def test_other_modality_and_empty_text_yield_nothing():
    assert labels_from_report("", "cxr").labels == []
    assert labels_from_report("Anything at all", "other").labels == []


def test_evidence_sentence_is_returned_for_audit():
    lab = labels_from_report("The heart is normal. There is a small effusion on the right.", "cxr").labels[0]
    assert "effusion" in lab.evidence.lower()
