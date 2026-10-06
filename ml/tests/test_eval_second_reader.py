import json

import pytest

pytest.importorskip("scipy")

from ml.eval_p3.eval_second_reader import (  # noqa: E402
    bone_prediction,
    eval_bone,
    eval_skin,
    main,
    skin_prediction,
    specialist_concordance,
)


def row(image, label, text, ok=True, group=None):
    return {"image": image, "label": label, "group": group or image, "ok": ok, "findings_text": text, "impression": "", "labels": []}


def test_bone_prediction_hedged_vs_strict():
    assert bone_prediction(row("a", "fracture", "Transverse fracture of the radius."), strict=True) == 1
    assert bone_prediction(row("a", "fracture", "Cortical irregularity suspicious for a fracture."), strict=True) == 0
    assert bone_prediction(row("a", "fracture", "Cortical irregularity suspicious for a fracture."), strict=False) == 1
    assert bone_prediction(row("a", "no_fracture", "No acute fracture."), strict=False) == 0
    assert bone_prediction(row("a", "no_fracture", "Normal hand."), strict=False) == 0


def test_skin_prediction_prefers_definite_then_clinical_priority():
    assert skin_prediction(row("a", "mel", "Suspicious for melanoma.")) == "mel"
    assert skin_prediction(row("a", "nv", "Seborrheic keratosis and a melanocytic nevus.")) == "bkl"  # priority bkl < nv
    assert skin_prediction(row("a", "nv", "Nothing to report.")) == "none"
    assert skin_prediction(row("a", "nv", "Possible melanoma. A benign nevus is present.")) == "nv"  # definite beats hedged


def make_bone_rows():
    rows = []
    for i in range(30):
        pos = i % 3 == 0
        rows.append(row(f"i{i}", "fracture" if pos else "no_fracture",
                        ("Transverse fracture." if i % 6 == 0 else "No acute abnormality.") if pos else ("No acute fracture." if i % 5 else "A fracture line is seen.")))
    return rows


def test_bone_metrics_have_confidence_intervals():
    out = eval_bone(make_bone_rows())
    s = out["hedged_counts_as_positive"]
    assert out["n_fracture"] == 10 and out["n_ok"] == 30
    assert 0.0 <= s["sensitivity"]["lo"] <= s["sensitivity"]["point"] <= s["sensitivity"]["hi"] <= 1.0
    assert s["sensitivity"]["point"] == pytest.approx(5 / 10)
    assert s["specificity"]["point"] == pytest.approx(16 / 20)  # 4 of 20 negatives flagged
    assert "kappa" in s and "accuracy" in s


def test_failed_reads_are_counted_but_not_scored():
    rows = make_bone_rows() + [row("bad", "fracture", "", ok=False)]
    out = eval_bone(rows)
    assert out["n_reads"] == 31 and out["n_ok"] == 30 and out["ok_rate"] == pytest.approx(30 / 31, abs=1e-4)


def test_too_few_reads_gives_a_note_not_a_crash():
    assert "note" in eval_bone([row("a", "fracture", "Fracture."), row("b", "no_fracture", "Normal.")])
    assert "note" in eval_skin([row("a", "mel", "Melanoma.")])


def test_skin_metrics_count_unmapped_as_wrong():
    rows = []
    for i in range(35):
        c = ["mel", "nv", "bkl", "bcc", "akiec", "df", "vasc"][i % 7]
        text = {"mel": "Melanoma.", "nv": "Benign nevus.", "bkl": "Seborrheic keratosis.", "bcc": "Basal cell carcinoma.",
                "akiec": "Actinic keratosis.", "df": "Dermatofibroma.", "vasc": "Cherry hemangioma."}[c]
        rows.append(row(f"s{i}", c, text if i % 7 != 2 else "Nothing notable."))  # every bkl is unmapped
    out = eval_skin(rows)
    assert out["coverage"] == pytest.approx(30 / 35, abs=1e-4)
    assert out["balanced_accuracy"]["point"] == pytest.approx(6 / 7, abs=1e-4)
    assert out["melanoma_sensitivity"]["point"] == 1.0
    assert out["prediction_counts"]["none"] == 5


def test_specialist_concordance_reports_kappa():
    rows = [row(f"i{i}", "fracture", "Fracture." if i < 5 else "No fracture.") for i in range(10)]
    spec = {f"i{i}": int(i < 5) for i in range(10)}
    out = specialist_concordance(spec, rows, "bone_xray")
    assert out["kappa"] == pytest.approx(1.0) and out["n"] == 10
    assert "note" in specialist_concordance({}, rows, "bone_xray")


def test_main_writes_the_report_from_cached_reads(tmp_path):
    d = tmp_path / "reads"
    d.mkdir()
    (d / "fracatlas.jsonl").write_text("\n".join(json.dumps(r) for r in make_bone_rows()), encoding="utf-8")
    out = tmp_path / "concordance.json"
    assert main(["--reads-dir", str(d), "--out", str(out)]) == 0
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert "fracatlas" in rep["datasets"] and "ham10000" not in rep["datasets"]
    assert rep["datasets"]["fracatlas"]["n_ok"] == 30
