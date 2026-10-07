import json

import numpy as np
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
    for i in range(70):
        c = ["mel", "nv", "bkl", "bcc", "akiec", "df", "vasc"][i % 7]
        text = {"mel": "Melanoma.", "nv": "Benign nevus.", "bkl": "Seborrheic keratosis.", "bcc": "Basal cell carcinoma.",
                "akiec": "Actinic keratosis.", "df": "Dermatofibroma.", "vasc": "Cherry hemangioma."}[c]
        rows.append(row(f"s{i}", c, text if i % 7 != 2 else "Nothing notable."))  # every bkl is unmapped
    out = eval_skin(rows)
    assert out["coverage"] == pytest.approx(60 / 70, abs=1e-4)
    assert out["balanced_accuracy"]["point"] == pytest.approx(6 / 7, abs=1e-4)
    assert out["melanoma_sensitivity"]["point"] == 1.0
    assert out["prediction_counts"]["none"] == 10


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


# ---- specialist vs second reader on skin, from the specialist's saved predictions

def _make_npz(path, n=120):
    import numpy as np

    classes = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    ids = np.array([f"ISIC_{i:07d}" for i in range(n)])
    y = np.array([i % 7 for i in range(n)])
    logits = np.full((n, 7), -1.0, dtype="float32")
    for i in range(n):
        # the specialist is right except on every 5th image, where it picks the next class
        logits[i, y[i] if i % 5 else (y[i] + 1) % 7] = 3.0
    np.savez(path, ids=ids, logits=logits, y=y, group=np.array([f"g{i}" for i in range(n)]))
    return classes


def _skin_rows(n=120, medgemma_right_when_specialist_right=True):
    texts = {"akiec": "Actinic keratosis.", "bcc": "Basal cell carcinoma.", "bkl": "Seborrheic keratosis.", "df": "Dermatofibroma.",
             "mel": "Melanoma.", "nv": "Benign nevus.", "vasc": "Cherry hemangioma."}
    classes = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    rows = []
    for i in range(n):
        truth = classes[i % 7]
        # MedGemma is right whenever the specialist is right, and picks the specialist's wrong class when it is wrong
        said = truth if i % 5 else classes[(i % 7 + 1) % 7]
        rows.append(row(f"ISIC_{i:07d}", truth, texts[said], group=f"g{i}"))
    return rows


def test_specialist_comparison_measures_agreement_and_whether_it_predicts_correctness(tmp_path):
    from ml.eval_p3.eval_second_reader import skin_specialist_comparison

    npz = tmp_path / "official_test.npz"
    _make_npz(npz)
    out = skin_specialist_comparison(_skin_rows(), npz)
    assert out["n_paired"] == 120 and out["medgemma_coverage"] == 1.0
    assert out["agreement_rate"]["point"] == pytest.approx(1.0)  # both readers err together in this construction
    assert out["kappa"]["point"] == pytest.approx(1.0)
    assert out["specialist_accuracy_when_agree"]["point"] == pytest.approx(96 / 120)
    assert "specialist_accuracy_when_disagree" in out


def test_disagreement_predicts_specialist_errors_when_the_second_reader_is_independent(tmp_path):
    from ml.eval_p3.eval_second_reader import skin_specialist_comparison

    npz = tmp_path / "official_test.npz"
    _make_npz(npz)
    rows = _skin_rows()
    texts = {"akiec": "Actinic keratosis.", "bcc": "Basal cell carcinoma.", "bkl": "Seborrheic keratosis.", "df": "Dermatofibroma.",
             "mel": "Melanoma.", "nv": "Benign nevus.", "vasc": "Cherry hemangioma."}
    classes = list(texts)
    for i, r in enumerate(rows):  # MedGemma now always says the truth, so it disagrees exactly where the specialist errs
        r["findings_text"] = texts[classes[i % 7]]
    out = skin_specialist_comparison(rows, npz)
    assert out["specialist_accuracy_when_agree"]["point"] == pytest.approx(1.0)
    assert out["specialist_accuracy_when_disagree"]["point"] == pytest.approx(0.0)
    assert out["accuracy_gap_agree_minus_disagree"]["point"] == pytest.approx(1.0)
    assert out["accuracy_gap_agree_minus_disagree"]["lo"] > 0.5


def test_unmapped_second_reads_are_excluded_from_agreement_and_counted_in_coverage(tmp_path):
    from ml.eval_p3.eval_second_reader import skin_specialist_comparison

    npz = tmp_path / "official_test.npz"
    _make_npz(npz)
    rows = _skin_rows()
    for r in rows[:30]:
        r["findings_text"] = "Nothing notable."
    out = skin_specialist_comparison(rows, npz)
    assert out["medgemma_coverage"] == pytest.approx(90 / 120) and out["n_compared"] == 90


def test_comparison_needs_enough_pairs_and_a_missing_file_is_not_an_error(tmp_path):
    from ml.eval_p3.eval_second_reader import skin_specialist_comparison

    npz = tmp_path / "official_test.npz"
    _make_npz(npz)
    assert "note" in skin_specialist_comparison(_skin_rows(10), npz)
    assert "note" in skin_specialist_comparison(_skin_rows(), tmp_path / "missing.npz")


def _make_bone_npz(path, n=100, misses=6, false_alarms=6):
    import numpy as np

    ids = np.array([f"IMG{i:07d}" for i in range(n)])
    truth = np.array([i % 4 == 0 for i in range(n)])
    det, off, gt, goff = [], [0], [], [0]
    pos_seen = neg_seen = 0
    for i in range(n):
        if truth[i]:
            pos_seen += 1
            score = 0.1 if pos_seen <= misses else 0.9
            gt.append([10, 10, 50, 50])
        else:
            neg_seen += 1
            score = 0.9 if neg_seen <= false_alarms else 0.1
        det.append([[10, 10, 50, 50, score]])
        off.append(off[-1] + 1)
        goff.append(goff[-1] + int(truth[i]))
    np.savez(path, ids=ids, groups=np.array([f"p{i}" for i in range(n)]), body_part=np.array(["hand"] * n),
             det_rows=np.array(det, dtype=float).reshape(-1, 5), det_offsets=np.array(off),
             gt_rows=np.array(gt, dtype=float).reshape(-1, 4), gt_offsets=np.array(goff))


def _bone_calibration(path):
    from medproof.calibrate.binary import BinaryCalibrator

    BinaryCalibrator("fracture", 1.0, 0.0, 0.1, 0.5, 0.4, 0.6).save(path)


def _bone_rows(n=100):
    rows = []
    for i in range(n):
        truth = i % 4 == 0  # MedGemma says the truth, so it disagrees with the specialist exactly where that errs
        said = truth and i not in (0, 4)  # except two misses it shares with the specialist, so unflagged cases also err
        rows.append(row(f"IMG{i:07d}", "fracture" if truth else "no_fracture", "Transverse fracture." if said else "No acute fracture."))
    return rows


def test_bone_discordance_predicts_specialist_errors_when_the_second_reader_is_independent(tmp_path):
    pytest.importorskip("pandas")
    from ml.eval_p3.eval_second_reader import bone_specialist_comparison

    _make_bone_npz(tmp_path / "test.npz")
    _bone_calibration(tmp_path / "cal.json")
    out = bone_specialist_comparison(_bone_rows(), tmp_path / "test.npz", tmp_path / "cal.json")
    assert out["n_paired"] == 100
    assert out["n_disagree"] == 10 and out["n_agree"] == 90
    assert out["specialist_accuracy_when_agree"]["point"] == pytest.approx(88 / 90, abs=1e-3)
    assert out["specialist_accuracy_when_disagree"]["point"] == pytest.approx(0.0)
    sig = out["discordant_signal"]
    assert sig["n_flagged"] == 10 and sig["predicts_errors"] is True
    hit = out["correct_positive_reports_flagged"]
    assert hit["n_correct_positive_reports"] == 19 and hit["share_flagged_discordant"] == 0.0


def test_bone_comparison_degrades_to_a_note(tmp_path):
    from ml.eval_p3.eval_second_reader import bone_specialist_comparison

    assert "note" in bone_specialist_comparison(_bone_rows(), tmp_path / "missing.npz", tmp_path / "cal.json")
    _make_bone_npz(tmp_path / "test.npz")
    _bone_calibration(tmp_path / "cal.json")
    assert "note" in bone_specialist_comparison(_bone_rows(10), tmp_path / "test.npz", tmp_path / "cal.json")


def test_skin_comparison_reports_the_discordant_signal(tmp_path):
    pytest.importorskip("pandas")
    from ml.eval_p3.eval_second_reader import skin_specialist_comparison

    npz = tmp_path / "official_test.npz"
    _make_npz(npz, n=300)
    rows = _skin_rows(300)
    classes = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    texts = {"akiec": "Actinic keratosis.", "bcc": "Basal cell carcinoma.", "bkl": "Seborrheic keratosis.", "df": "Dermatofibroma.",
             "mel": "Melanoma.", "nv": "Benign nevus.", "vasc": "Cherry hemangioma."}
    for i, r in enumerate(rows):
        r["findings_text"] = texts[classes[i % 7]] if i not in (0, 5, 10) else texts[classes[(i % 7 + 1) % 7]]  # three shared errors
    out = skin_specialist_comparison(rows, npz)
    assert out["discordant_signal"]["n_flagged"] == 57 and out["discordant_signal"]["predicts_errors"] is True


def test_signal_check_survives_data_with_no_unflagged_errors():
    from ml.eval_p3.eval_second_reader import discordant_signal

    err = np.array([True] * 12 + [False] * 88)
    out = discordant_signal(err, err)  # flagged exactly where wrong: no errors outside the flag
    assert "n_flagged" in out and ("note" in out or "predicts_errors" in out)
