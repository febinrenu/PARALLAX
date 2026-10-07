from __future__ import annotations

import pytest

from medproof.context.contradictions import Demographics, check, normalise_side
from medproof.context.extract import ExtractedFact
from medproof.core.schemas import Finding, ImageEvidence


def fact(quote, ftype, value=None, polarity="present", at=0, note="n1"):
    return ExtractedFact(note, (at, at + len(quote)), quote, ftype, value or quote, polarity)


def finding(label="Consolidation", region="right lower zone", modality="cxr", tier="high", fid="f1") -> Finding:
    ev = ImageEvidence(
        evidence_id="ie_1", kind="bbox", bbox_xyxy=(1.0, 1.0, 9.0, 9.0), source_model="xrv@abcd1234", method="gradcam++",
        region_name=region,
    )
    return Finding(
        finding_id=fid, modality=modality, label=label, prob_raw=0.8, prob_calibrated=0.8, conformal_set=[], tier=tier,
        status="uncertain", image_evidence=[ev],
    )


def flags(res, fid="f1"):
    return set(next(f for f in res.findings if f.finding_id == fid).flags)


def polarities(res, fid="f1"):
    return {te.quote: te.polarity for te in next(f for f in res.findings if f.finding_id == fid).text_evidence}


@pytest.mark.parametrize(
    "raw,side",
    [("right", "right"), ("Rt", "right"), ("R", "right"), ("right-sided", "right"), ("Right-sided", "right"),
     ("left", "left"), ("Lt", "left"), ("L", "left"), ("left-sided", "left"), ("patient's left", "left"),
     ("bilateral", None), ("both", None), ("", None), ("upper", None)],
)
def test_side_normalisation(raw, side):
    assert normalise_side(raw) == side


def test_laterality_conflict_between_note_and_image_is_flagged_and_contradicts():
    res = check([finding(region="right lower zone")], [fact("Lt", "laterality", "left")], Demographics())
    assert "laterality_conflict" in flags(res)
    assert polarities(res) == {"Lt": "contradicts"}
    assert res.stage.stage == "context" and res.stage.payload["flags"]["laterality_conflict"] == 1


def test_matching_laterality_is_consistent_but_not_support_for_the_finding():
    # agreement of side with side says the note and the image do not clash; it says nothing about whether the finding is real
    res = check([finding()], [fact("right", "laterality", "right")], Demographics())
    assert "laterality_conflict" not in flags(res) and polarities(res) == {"right": "neutral"}


def test_a_note_with_only_a_side_word_cannot_verify_a_finding():
    from medproof.core.status_rule import compute_status

    f = finding().model_copy(update={"image_evidence": [finding().image_evidence[0].model_copy(update={"faithful": None})]})
    res = check([f], [fact("Right", "laterality", "right")], Demographics())
    assert compute_status(res.findings[0]) == "rejected"


def test_a_relevant_symptom_still_supports_the_finding():
    res = check([finding(label="Pneumonia")], [fact("fever", "symptom", "fever")], Demographics())
    assert polarities(res) == {"fever": "supports"}


def test_note_mentioning_both_sides_is_not_a_conflict():
    res = check([finding()], [fact("Rt", "laterality", "right"), fact("Lt", "laterality", "left", at=10)], Demographics())
    assert "laterality_conflict" not in flags(res)


def test_finding_without_a_side_cannot_conflict():
    res = check([finding(label="Cardiomegaly", region="cardiac")], [fact("Lt", "laterality", "left")], Demographics())
    assert "laterality_conflict" not in flags(res)
    assert polarities(res) == {"Lt": "neutral"}


def test_image_side_wording_trap_uses_the_patient_side_fact():
    # "opacity on the left of the film, patient's right" -> the extractor gives the patient-side phrase
    res = check([finding(region="right lower zone")], [fact("patient's right", "laterality", "right")], Demographics())
    assert "laterality_conflict" not in flags(res)


def test_pneumonectomy_on_the_finding_side_is_a_history_conflict():
    res = check([finding(region="right upper zone")], [fact("s/p right pneumonectomy", "history", "s/p right pneumonectomy", "historical")], Demographics())
    assert "history_conflict" in flags(res)
    res2 = check([finding(region="left upper zone")], [fact("s/p right pneumonectomy", "history", "right pneumonectomy", "historical")], Demographics())
    assert "history_conflict" not in flags(res2)


def test_history_rule_only_applies_to_chest_findings():
    f = finding(label="fracture", region="right hand", modality="bone_xray")
    assert "history_conflict" not in flags(check([f], [fact("s/p right pneumonectomy", "history")], Demographics()))


@pytest.mark.parametrize(
    "quote,demo,expected",
    [
        ("currently pregnant", Demographics(age=40, sex="M"), True),
        ("currently pregnant", Demographics(age=30, sex="F"), False),
        ("primigravida at 20 weeks", Demographics(age=80, sex="F"), True),
        ("post-menopausal", Demographics(age=8, sex="F"), True),
        ("post-menopausal", Demographics(age=60, sex="F"), False),
        ("post-menopausal", Demographics(age=55, sex="M"), True),
    ],
)
def test_demographic_plausibility(quote, demo, expected):
    res = check([finding()], [fact(quote, "demographic")], demo)
    assert ("demographic_implausible" in flags(res)) is expected


def test_demographic_age_or_sex_that_disagrees_with_the_record():
    assert "demographic_mismatch" in flags(check([finding()], [fact("63F", "demographic")], Demographics(age=63, sex="M")))
    assert "demographic_mismatch" in flags(check([finding()], [fact("40M", "demographic")], Demographics(age=63, sex="M")))
    assert "demographic_mismatch" not in flags(check([finding()], [fact("63M", "demographic")], Demographics(age=63, sex="M")))
    assert "demographic_mismatch" not in flags(check([finding()], [fact("72 ans", "demographic")], Demographics(age=72, sex="F")))


def test_asymptomatic_note_with_a_large_high_confidence_finding_is_incoherent():
    res = check([finding(label="Effusion")], [fact("asymptomatic", "symptom", "asymptomatic", "absent")], Demographics())
    assert "symptom_finding_incoherent" in flags(res)
    low = check([finding(label="Effusion", tier="low")], [fact("asymptomatic", "symptom", "asymptomatic", "absent")], Demographics())
    assert "symptom_finding_incoherent" not in flags(low)
    small = check([finding(label="Nodule")], [fact("asymptomatic", "symptom", "asymptomatic", "absent")], Demographics())
    assert "symptom_finding_incoherent" not in flags(small)


def test_matching_symptoms_support_and_unrelated_ones_are_neutral():
    res = check([finding(label="Consolidation")], [fact("fever", "symptom"), fact("ankle swelling", "symptom", at=20)], Demographics())
    assert polarities(res) == {"fever": "supports", "ankle swelling": "neutral"}


def test_negated_symptom_does_not_support():
    res = check([finding(label="Consolidation")], [fact("no fever", "negation", "fever", "absent")], Demographics())
    assert polarities(res) == {"no fever": "neutral"}


def test_missing_context_prompts_when_notes_are_empty_or_uninformative():
    none = check([finding()], [], Demographics())
    assert none.missing_context and all(p.startswith("Doctor,") for p in none.missing_context)
    assert not check([finding()], [fact("fever", "symptom"), fact("TB", "history", polarity="historical", at=9)], Demographics()).missing_context


def test_text_evidence_ids_are_unique_per_finding_and_fact_pair():
    f2 = finding(label="Effusion", region="right base", fid="f2")
    res = check([finding(), f2], [fact("right", "laterality", "right")], Demographics(), evidence_start=5)
    ids = [te.evidence_id for f in res.findings for te in f.text_evidence]
    assert ids == ["te_5", "te_6"]  # one id per (finding, fact) so each id has exactly one owner


def test_inputs_are_not_mutated_and_clean_notes_raise_no_flags():
    f = finding()
    res = check([f], [fact("fever", "symptom"), fact("right", "laterality", "right", at=8)], Demographics(age=60, sex="F"))
    assert f.flags == [] and f.text_evidence == []
    assert flags(res) == set()


# ---- deterministic cue scan: closed-class phrases found in the note text itself, independent of the model

from medproof.context.contradictions import scan_cues  # noqa: E402


@pytest.mark.parametrize(
    "text,ftype,needle",
    [
        ("63M c/o fever. currently pregnant. h/o TB.", "demographic", "pregnant"),
        ("Primigravida at 20 weeks. No pain.", "demographic", "Primigravida"),
        ("She is 12 weeks pregnant.", "demographic", "pregnant"),
        ("post-menopausal woman", "demographic", "menopausal"),
        ("Patient is asymptomatic.", "symptom", "asymptomatic"),
        ("reports no symptoms at all", "symptom", "no symptoms"),
        ("h/o asthma, s/p right pneumonectomy 2019", "history", "right pneumonectomy"),
        ("had a Lt pneumonectomy", "history", "Lt pneumonectomy"),
    ],
)
def test_cues_are_found_with_exact_spans(text, ftype, needle):
    cues = scan_cues(text, "n1")
    hit = [c for c in cues if c.fact_type == ftype and needle in c.quote]
    assert hit, (text, cues)
    c = hit[0]
    assert text[c.span[0] : c.span[1]] == c.quote and c.note_id == "n1"


@pytest.mark.parametrize("text", ["No chest pain. h/o TB 2015.", "Pregnancy test not indicated, unknown.", "denies fever", "right basal crackles", ""])
def test_cues_do_not_fire_on_ordinary_text(text):
    assert scan_cues(text, "n") == []


def test_pregnancy_word_alone_in_a_negated_context_is_not_a_cue():
    assert [c for c in scan_cues("No history of pregnancy.", "n") if c.fact_type == "demographic"] == []
    assert [c for c in scan_cues("not pregnant", "n") if c.fact_type == "demographic"] == []


def test_check_uses_cues_from_the_note_text_when_the_model_missed_them():
    f = finding(label="Effusion")
    res = check([f], [fact("63M", "demographic")], Demographics(age=63, sex="M"), note_text="63M c/o cough. currently pregnant.", note_id="n1")
    assert "demographic_implausible" in flags(res)
    res2 = check([f], [], Demographics(), note_text="Patient is asymptomatic.", note_id="n1")
    assert "symptom_finding_incoherent" in flags(res2)
    res3 = check([finding(region="right upper zone", modality="cxr")], [fact("pneumonectomy", "history", at=7)], Demographics(),
                 note_text="h/o s/p right pneumonectomy.", note_id="n1")
    assert "history_conflict" in flags(res3)


def test_cue_replaces_an_overlapping_model_fact_instead_of_duplicating_it():
    f = finding(region="right upper zone")
    text = "h/o s/p right pneumonectomy."
    start = text.index("pneumonectomy")
    res = check([f], [fact("pneumonectomy", "history", at=start)], Demographics(), note_text=text, note_id="n1")
    hist = [t for t in next(x for x in res.findings).text_evidence if t.fact_type == "history"]
    assert len(hist) == 1 and "right" in hist[0].quote


def test_without_note_text_behaviour_is_unchanged():
    assert flags(check([finding()], [fact("63F", "demographic")], Demographics(age=63, sex="F"))) == set()
