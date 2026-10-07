import pytest
from pydantic import ValidationError

from ml.data.notes_schema import (
    INJECTION_FAMILIES,
    CorpusRequirements,
    Injection,
    NoteContext,
    NoteFact,
    SyntheticNote,
    check_corpus,
    read_jsonl,
    write_jsonl,
)

TEXT = "63F c/o fever and cough x3 days. No chest pain. h/o TB 2015. Right-sided crackles."


def span(sub: str) -> tuple[int, int]:
    i = TEXT.index(sub)
    return (i, i + len(sub))


def fact(fid, ftype, sub, polarity="present", value=None):
    return NoteFact(
        fact_id=fid, type=ftype, value=value or sub, polarity=polarity, span=span(sub), quote=sub
    )


def note(nid="n_0001", **kw):
    base = dict(
        note_id=nid,
        seed=1,
        text=TEXT,
        context=NoteContext(modality="cxr", age=63, sex="F", seed_finding="consolidation", side="right"),
        facts=[fact("f1", "symptom", "fever"), fact("f2", "history", "TB 2015", "historical")],
    )
    base.update(kw)
    return SyntheticNote(**base)


def test_valid_note_builds():
    assert note().facts[0].quote == "fever"


def test_quote_must_equal_span():
    wrong = NoteFact(fact_id="f", type="symptom", value="x", polarity="present", span=(0, 3), quote="xyz")
    with pytest.raises(ValidationError):
        note(facts=[wrong])
    with pytest.raises(ValidationError):  # quote length differs from span length
        NoteFact(fact_id="f", type="symptom", value="x", polarity="present", span=(0, 3), quote="fe")


def test_span_out_of_bounds_rejected():
    n = len(TEXT)
    tail = NoteFact(
        fact_id="f", type="symptom", value="x", polarity="present", span=(n - 2, n + 5), quote=TEXT[-2:] + "xxxxx"
    )
    with pytest.raises(ValidationError):
        note(facts=[tail])


def test_same_type_overlap_rejected_but_cross_type_allowed():
    a = fact("f1", "symptom", "fever and cough")
    b = fact("f2", "symptom", "cough x3 days")
    with pytest.raises(ValidationError):
        note(facts=[a, b])
    c = fact("f3", "laterality", "Right-sided crackles")
    d = fact("f4", "symptom", "crackles")
    assert note(facts=[c, d])


def test_duplicate_fact_ids_rejected():
    with pytest.raises(ValidationError):
        note(facts=[fact("f1", "symptom", "fever"), fact("f1", "symptom", "cough")])


def test_contradiction_must_reference_known_facts():
    with pytest.raises(ValidationError):
        note(contradiction={"kind": "laterality", "expected_flag": "laterality_conflict", "fact_ids": ["nope"]})
    ok = note(contradiction={"kind": "laterality", "expected_flag": "laterality_conflict", "fact_ids": ["f1"]})
    assert ok.contradiction.kind == "laterality"


def test_injection_span_checked_and_family_enumerated():
    s = span("No chest pain.")
    inj = Injection(kind="direct_override", span=s, quote="No chest pain.", expected_guard="flag")
    assert note(injection=inj).injection.expected_effect == "none"
    with pytest.raises(ValidationError):
        Injection(kind="not_a_family", span=s, quote="No chest pain.", expected_guard="flag")
    with pytest.raises(ValidationError):
        note(injection=Injection(kind="direct_override", span=s, quote="wrong", expected_guard="flag"))


def test_benign_lookalike_expects_pass():
    s = span("No chest pain.")
    with pytest.raises(ValidationError):
        Injection(kind="benign_lookalike", span=s, quote="No chest pain.", expected_guard="flag")


def test_jsonl_round_trip(tmp_path):
    p = tmp_path / "n.jsonl"
    write_jsonl([note("n_1"), note("n_2")], p)
    assert [n.note_id for n in read_jsonl(p)] == ["n_1", "n_2"]


def inj_note(i, kind):
    s = span("No chest pain.")
    expected = "pass" if kind == "benign_lookalike" else "flag"
    return note(
        f"n_{i:04d}",
        injection=Injection(kind=kind, span=s, quote="No chest pain.", expected_guard=expected),
    )


def test_corpus_checks():
    notes = [note(f"n_{i:04d}") for i in range(3)]
    req = CorpusRequirements(min_notes=3, min_contradictions=0, min_injections=0, min_per_family=0, min_benign=0)
    assert check_corpus(notes, req) == []
    assert check_corpus(notes + [notes[0]], req)  # duplicate id
    big = CorpusRequirements()
    errs = check_corpus(notes, big)
    assert any("notes" in e for e in errs) and any("injection" in e for e in errs)


def test_full_corpus_requirements_satisfiable():
    notes = []
    i = 0
    for fam in INJECTION_FAMILIES:
        for _ in range(5):
            notes.append(inj_note(i, fam)); i += 1
    for _ in range(10):
        notes.append(inj_note(i, "benign_lookalike")); i += 1
    for _ in range(60):
        notes.append(note(f"n_{i:04d}", contradiction={"kind": "history", "expected_flag": "history_conflict", "fact_ids": ["f1"]})); i += 1
    while len(notes) < 200:
        notes.append(note(f"n_{i:04d}")); i += 1
    assert check_corpus(notes, CorpusRequirements()) == []
