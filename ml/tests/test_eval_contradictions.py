from ml.eval_p3.eval_contradictions import evaluate
from ml.data.notes_schema import read_jsonl
from ml.eval_p3.eval_contradictions import NOTES


def test_rules_meet_the_plan_targets_on_the_planted_contradictions():
    # plan 9.5: precision >= 0.9, recall >= 0.8 on 60 planted contradictions plus clean notes (gold facts)
    r = evaluate(read_jsonl(NOTES))
    assert r["contradiction_notes"] == 60 and r["clean_notes"] >= 60
    assert r["precision"] >= 0.9 and r["recall"] >= 0.8
    assert r["clean_notes_flagged"] == 0
