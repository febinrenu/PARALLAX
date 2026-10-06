import csv
import json
import re

import pytest

from ml.data.gen_notes import Seeds, generate, load_seeds, main
from ml.data.notes_schema import (
    INJECTION_FAMILIES,
    CorpusRequirements,
    check_corpus,
    read_jsonl,
)


@pytest.fixture(scope="module")
def notes():
    return generate(seed=1, seeds=Seeds.defaults())


def test_corpus_meets_the_targets(notes):
    assert check_corpus(notes, CorpusRequirements()) == []
    assert len(notes) == 200


def test_generation_is_deterministic_per_seed(notes):
    again = generate(seed=1, seeds=Seeds.defaults())
    assert [n.model_dump() for n in again] == [n.model_dump() for n in notes]
    other = generate(seed=2, seeds=Seeds.defaults())
    assert [n.text for n in other] != [n.text for n in notes]


def test_every_attack_has_a_clean_twin_differing_only_by_the_injection(notes):
    by_id = {n.note_id: n for n in notes}
    attacks = [n for n in notes if n.injection and n.injection.kind != "benign_lookalike"]
    assert len(attacks) >= 48
    for n in attacks:
        assert n.twin_of in by_id, n.note_id
        twin = by_id[n.twin_of]
        assert twin.injection is None and twin.contradiction is None
        removed = n.text.replace(" " + n.injection.quote, "", 1).replace("\n" + n.injection.quote, "", 1)
        assert removed == twin.text, n.note_id
        assert n.context == twin.context


def test_injections_never_overlap_facts(notes):
    for n in notes:
        if not n.injection:
            continue
        s, e = n.injection.span
        for f in n.facts:
            assert f.span[1] <= s or f.span[0] >= e, n.note_id
            assert n.injection.quote not in f.quote


def test_each_family_has_at_least_six_distinct_payloads(notes):
    for fam in INJECTION_FAMILIES:
        payloads = {n.injection.quote for n in notes if n.injection and n.injection.kind == fam}
        assert len(payloads) >= 6, fam


def test_laterality_contradictions_really_contradict(notes):
    lat = [n for n in notes if n.contradiction and n.contradiction.kind == "laterality"]
    assert len(lat) >= 15
    for n in lat:
        facts = {f.fact_id: f for f in n.facts}
        stated = [facts[i].value for i in n.contradiction.fact_ids]
        assert n.context.side in ("left", "right")
        assert all(v in ("left", "right") and v != n.context.side for v in stated), n.note_id


def test_clean_notes_agree_with_the_image_side(notes):
    for n in notes:
        if n.contradiction or n.injection:
            continue
        for f in n.facts:
            if f.type == "laterality" and n.context.side in ("left", "right"):
                assert f.value == n.context.side, n.note_id


def test_contradiction_mix_and_clean_set_sizes(notes):
    kinds = {}
    for n in notes:
        if n.contradiction:
            kinds[n.contradiction.kind] = kinds.get(n.contradiction.kind, 0) + 1
    assert sum(kinds.values()) == 60
    assert set(kinds) >= {"laterality", "history", "demographic", "symptom"}
    clean = [n for n in notes if not n.contradiction and not n.injection]
    assert len(clean) >= 60


def test_includes_hard_negatives_other_languages_and_empty_notes(notes):
    assert any(n.lang != "en" for n in notes)
    assert any(n.text == "" for n in notes)
    assert any(re.search(r"patient'?s? (right|left)", n.text) for n in notes if not n.contradiction)


def test_facts_are_found_at_their_spans_and_unique_per_note(notes):
    for n in notes:
        for f in n.facts:
            assert n.text[f.span[0] : f.span[1]] == f.quote
        ids = [f.fact_id for f in n.facts]
        assert len(ids) == len(set(ids))


def test_seeds_come_from_the_split_files(tmp_path):
    with (tmp_path / "ham10000.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image_id", "label", "age", "sex", "site"])
        w.writerows([["i1", "mel", "77.0", "female", "scalp"], ["i2", "nv", "77.0", "female", "scalp"]])
    with (tmp_path / "fracatlas.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image_id", "label", "body_part", "hardware", "view"])
        w.writerows([["b1", "fracture", "hand", "1", "frontal"], ["b2", "no_fracture", "hand", "0", "frontal"]])
    seeds = load_seeds(tmp_path)
    assert seeds.skin and seeds.skin[0]["age"] == 77 and seeds.skin[0]["site"] == "scalp"
    assert seeds.bone[0]["body_part"] == "hand"
    notes = generate(seed=3, seeds=seeds)
    skin = [n for n in notes if n.context.modality == "skin_dermoscopy"]
    assert skin and all(n.context.age == 77 and n.context.sex == "F" for n in skin if not n.lang != "en")


def test_missing_split_files_fall_back_to_defaults(tmp_path):
    seeds = load_seeds(tmp_path)
    assert seeds.skin and seeds.bone


def test_cli_writes_valid_jsonl(tmp_path):
    out = tmp_path / "notes.jsonl"
    assert main(["--seed", "5", "--out", str(out), "--splits-dir", str(tmp_path)]) == 0
    loaded = read_jsonl(out)
    assert len(loaded) == 200
    json.loads(out.read_text(encoding="utf-8").splitlines()[0])


def test_notes_do_not_contradict_themselves_by_accident(notes):
    for n in notes:
        present = [f.quote.lower() for f in n.facts if f.type == "symptom" and f.polarity == "present"]
        for f in n.facts:
            if f.type == "negation":
                assert not any(f.value in p for p in present), (n.note_id, f.value, present)


def test_asymptomatic_notes_have_no_symptom_complaint(notes):
    for n in notes:
        if any(f.value == "asymptomatic" for f in n.facts):
            assert not [f for f in n.facts if f.type == "symptom" and f.polarity == "present"], n.note_id
            assert not re.search(r"\d+[MF] is ", n.text), n.text
