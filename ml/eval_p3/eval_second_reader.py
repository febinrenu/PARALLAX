"""Scores the MedGemma second reader against ground truth on the cached batch reads (P3.2).

    python -m ml.eval_p3.eval_second_reader                 # reads ml/artifacts/medgemma_reads/*.jsonl
    python -m ml.eval_p3.eval_second_reader --dataset fracatlas

Writes reports/concordance.json. Everything is recomputed from the cached reads, so it needs no GPU.
The specialist-vs-MedGemma kappa (D3) is added once P2's weights exist: `specialist_concordance`
takes the specialist's per-image labels and returns the same metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))

from medproof.verify.concordance import cohen_kappa  # noqa: E402
from medproof.verify.report_labels import labels_from_report  # noqa: E402

from ml.eval.bootstrap import ci  # noqa: E402

READS = ROOT / "ml" / "artifacts" / "medgemma_reads"
SKIN_PRIORITY = ["mel", "bcc", "akiec", "bkl", "df", "vasc", "nv"]  # clinical risk order, used to break ties
SKIN_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
MIN_READS_BONE = 20  # below these, intervals are too wide to mean anything and are not reported
MIN_READS_SKIN = 50


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def report_text(row: dict) -> str:
    names = [x["name"] for x in row.get("labels", []) if x.get("present", True)]
    parts = [row.get("findings_text", ""), ("Findings: " + "; ".join(names) + ".") if names else "", row.get("impression", "")]
    return " ".join(p for p in parts if p)


def bone_prediction(row: dict, strict: bool) -> int:
    state = labels_from_report(report_text(row), "bone_xray").state("fracture")
    return int(state == "present" or (state == "hedged" and not strict))


def skin_prediction(row: dict) -> str:
    labs = [lab for lab in labels_from_report(report_text(row), "skin_dermoscopy").labels if lab.present]
    if not labs:
        return "none"
    definite = [lab.label for lab in labs if not lab.hedged]
    pool = definite or [lab.label for lab in labs]
    return min(pool, key=SKIN_PRIORITY.index)


def _binary_stats(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    sens = float(p[y == 1].mean()) if (y == 1).any() else float("nan")
    spec = float((1 - p[y == 0]).mean()) if (y == 0).any() else float("nan")
    return {"sensitivity": sens, "specificity": spec, "accuracy": float((p == y).mean()),
            "kappa": float(cohen_kappa(list(y), list(p)))}


def _ci(stat: str, y: np.ndarray, p: np.ndarray, groups: np.ndarray, strata: np.ndarray) -> dict:
    return ci({"y": y, "p": p}, lambda y, p: _binary_stats(y, p)[stat], groups=groups, strata=strata, B=1000)


def eval_bone(rows: list[dict]) -> dict:
    good = [r for r in rows if r.get("ok")]
    y = np.array([int(r["label"] == "fracture") for r in good])
    out: dict = {"n_reads": len(rows), "n_ok": len(good), "ok_rate": round(len(good) / len(rows), 4) if rows else None,
                 "n_fracture": int(y.sum()), "n_no_fracture": int((1 - y).sum())}
    if len(good) < MIN_READS_BONE or y.sum() == 0 or y.sum() == len(y):
        out["note"] = "too few reads of both classes for statistics yet"
        return out
    groups = np.array([r.get("group") or r["image"] for r in good], dtype=object)
    for name, strict in (("hedged_counts_as_positive", False), ("definite_only", True)):
        p = np.array([bone_prediction(r, strict) for r in good])
        stats = _binary_stats(y, p)
        block = {}
        for k, v in stats.items():
            c = _ci(k, y, p, groups, y)
            block[k] = {"point": round(v, 4), "lo": round(c["lo"], 4), "hi": round(c["hi"], 4)}
        out[name] = block
    return out


def eval_skin(rows: list[dict]) -> dict:
    good = [r for r in rows if r.get("ok")]
    truth = np.array([r["label"] for r in good], dtype=object)
    pred = np.array([skin_prediction(r) for r in good], dtype=object)
    out: dict = {"n_reads": len(rows), "n_ok": len(good), "ok_rate": round(len(good) / len(rows), 4) if rows else None,
                 "class_counts": dict(Counter(truth)), "prediction_counts": dict(Counter(pred))}
    if len(good) < MIN_READS_SKIN:
        out["note"] = "too few reads for statistics yet"
        return out
    groups = np.array([r.get("group") or r["image"] for r in good], dtype=object)

    def bal_acc(t, p):  # unmapped predictions count as wrong
        rec = [float((p[t == c] == c).mean()) for c in SKIN_CLASSES if (t == c).any()]
        return float(np.mean(rec))

    def mel_sens(t, p):
        return float((p[t == "mel"] == "mel").mean()) if (t == "mel").any() else float("nan")

    def kap(t, p):
        return float(cohen_kappa(list(t), list(p)))

    def acc_mapped(t, p):
        m = p != "none"
        return float((t[m] == p[m]).mean()) if m.any() else float("nan")

    arrays = {"t": truth, "p": pred}
    out["coverage"] = round(float((pred != "none").mean()), 4)
    for name, fn in (("balanced_accuracy", bal_acc), ("melanoma_sensitivity", mel_sens), ("kappa", kap),
                     ("accuracy_on_mapped", acc_mapped)):
        c = ci(arrays, fn, groups=groups, strata=truth, B=1000)
        out[name] = {k: round(c[k], 4) for k in ("point", "lo", "hi")}
    return out



SKIN_NPZ = ROOT / "ml" / "artifacts" / "skin_cls" / "predictions" / "official_test.npz"
BONE_NPZ = ROOT / "ml" / "artifacts" / "bone_det" / "predictions" / "test.npz"
BONE_CAL = ROOT / "ml" / "artifacts" / "bone_det" / "calibration.json"
MIN_PAIRS = 50


def _mean_or_nan(x: np.ndarray) -> float:
    return float(x.mean()) if len(x) else float("nan")


def discordant_signal(error: np.ndarray, discordant: np.ndarray) -> dict:
    """Plan 9.4 / D13: is the specialist wrong more often on images where the second reader disagrees?

    Uses P2's `signals.compare` so the verdict and its wording match the validation page. The
    bootstrap there resamples images, not groups, which is slightly optimistic for grouped data.
    """
    try:
        import pandas as pd

        from ml.eval.signals import compare
    except ImportError as exc:
        return {"note": f"signal check needs pandas and the P2 eval package ({type(exc).__name__})"}
    frame = pd.DataFrame({"error": np.asarray(error, dtype=bool), "discordant": np.asarray(discordant, dtype=bool)})
    try:
        return compare(frame, "discordant")
    except (IndexError, ValueError):  # the risk-ratio bootstrap is undefined when no unflagged case is wrong
        return {"n_flagged": int(frame.discordant.sum()), "n_unflagged": int((~frame.discordant).sum()),
                "note": "bootstrap undefined: no errors among unflagged cases, or no flagged cases"}


def skin_specialist_comparison(rows: list[dict], npz_path: Path = SKIN_NPZ) -> dict:
    """Specialist (P2's saved official-test predictions) against MedGemma on the same images.

    Reports agreement and kappa, and the trust-signal question from plan 9.4: is the specialist right
    more often when the second reader agrees with it than when it disagrees?
    """
    if not Path(npz_path).is_file():
        return {"note": f"specialist predictions not found at {npz_path}"}
    z = np.load(npz_path, allow_pickle=False)
    index = {str(i): k for k, i in enumerate(z["ids"])}
    pairs = [(r, index[r["image"]]) for r in rows if r.get("ok") and r["image"] in index]
    if len(pairs) < MIN_PAIRS:
        return {"n_paired": len(pairs), "note": f"need at least {MIN_PAIRS} paired reads for statistics"}
    spec = np.array([SKIN_CLASSES[int(np.argmax(z["logits"][k]))] for _, k in pairs], dtype=object)
    truth = np.array([SKIN_CLASSES[int(z["y"][k])] for _, k in pairs], dtype=object)
    mg = np.array([skin_prediction(r) for r, _ in pairs], dtype=object)
    groups = np.array([str(z["group"][k]) for _, k in pairs], dtype=object)
    mapped = mg != "none"
    out: dict = {"n_paired": len(pairs), "medgemma_coverage": round(float(mapped.mean()), 4), "n_compared": int(mapped.sum())}
    if mapped.sum() < MIN_PAIRS:
        out["note"] = f"fewer than {MIN_PAIRS} images where MedGemma named a lesion type"
        return out
    s_, t_, m_, g_ = spec[mapped], truth[mapped], mg[mapped], groups[mapped]
    out["specialist_accuracy_on_compared"] = round(float((s_ == t_).mean()), 4)
    out["medgemma_accuracy_on_compared"] = round(float((m_ == t_).mean()), 4)

    def ci_of(fn, **arrays) -> dict:
        c = ci(arrays, fn, groups=g_, strata=None, B=1000)
        return {k: round(c[k], 4) for k in ("point", "lo", "hi")}

    out["agreement_rate"] = ci_of(lambda s, m: float((s == m).mean()), s=s_, m=m_)
    out["kappa"] = ci_of(lambda s, m: float(cohen_kappa(list(s), list(m))), s=s_, m=m_)
    agree = s_ == m_
    correct = s_ == t_
    out["n_agree"], out["n_disagree"] = int(agree.sum()), int((~agree).sum())
    out["specialist_accuracy_when_agree"] = ci_of(lambda c, a: _mean_or_nan(c[a]), c=correct, a=agree)
    if (~agree).sum() >= 5:
        out["specialist_accuracy_when_disagree"] = ci_of(lambda c, a: _mean_or_nan(c[~a]), c=correct, a=agree)
        out["accuracy_gap_agree_minus_disagree"] = ci_of(lambda c, a: _mean_or_nan(c[a]) - _mean_or_nan(c[~a]), c=correct, a=agree)
    else:
        out["specialist_accuracy_when_disagree"] = {"note": "fewer than 5 disagreements"}
    out["discordant_signal"] = discordant_signal(~correct, ~agree)
    return out


def bone_specialist_comparison(rows: list[dict], npz_path: Path = BONE_NPZ, cal_path: Path = BONE_CAL) -> dict:
    """The same comparison for bone: P2's detector (image score = top box, reported at the calibrated
    FNR-controlled threshold) against MedGemma, plus the discordant-signal check."""
    if not Path(npz_path).is_file() or not Path(cal_path).is_file():
        return {"note": f"specialist predictions or calibration not found ({npz_path.name}, {cal_path.name})"}
    from medproof.calibrate.binary import BinaryCalibrator

    z = np.load(npz_path, allow_pickle=False)
    index = {str(i): k for k, i in enumerate(z["ids"])}
    pairs = [(r, index[r["image"]]) for r in rows if r.get("ok") and r["image"] in index]
    if len(pairs) < MIN_PAIRS:
        return {"n_paired": len(pairs), "note": f"need at least {MIN_PAIRS} paired reads for statistics"}
    det, doff, goff = z["det_rows"], z["det_offsets"], z["gt_offsets"]
    score = np.array([float(det[doff[k] : doff[k + 1], 4].max()) if doff[k + 1] > doff[k] else 0.0 for _, k in pairs])
    truth = np.array([goff[k + 1] > goff[k] for _, k in pairs])
    spec = np.asarray(BinaryCalibrator.load(cal_path).decide(score)["report"], dtype=bool)
    mg = np.array([bool(bone_prediction(r, False)) for r, _ in pairs])
    groups = np.array([str(z["groups"][k]) for _, k in pairs], dtype=object)

    def ci_of(fn, **arrays) -> dict:
        c = ci(arrays, fn, groups=groups, strata=truth.astype(int), B=1000)
        return {k: round(c[k], 4) for k in ("point", "lo", "hi")}

    agree, correct = spec == mg, spec == truth
    out: dict = {"n_paired": len(pairs), "specialist_positive_rate": round(float(spec.mean()), 4), "medgemma_positive_rate": round(float(mg.mean()), 4)}
    out["agreement_rate"] = ci_of(lambda s, m: float((s == m).mean()), s=spec, m=mg)
    out["kappa"] = ci_of(lambda s, m: float(cohen_kappa(list(s), list(m))), s=spec, m=mg)
    out["n_agree"], out["n_disagree"] = int(agree.sum()), int((~agree).sum())
    out["specialist_accuracy_when_agree"] = ci_of(lambda c, a: _mean_or_nan(c[a]), c=correct, a=agree)
    if (~agree).sum() >= 5:
        out["specialist_accuracy_when_disagree"] = ci_of(lambda c, a: _mean_or_nan(c[~a]), c=correct, a=agree)
    else:
        out["specialist_accuracy_when_disagree"] = {"note": "fewer than 5 disagreements"}
    out["discordant_signal"] = discordant_signal(~correct, ~agree)
    # A flag can predict errors and still be unusable: if the second reader rarely says "fracture", the flag
    # mostly means "the specialist reported one", and downgrading on it would demote correct findings too.
    hit = spec & truth
    out["correct_positive_reports_flagged"] = {
        "n_correct_positive_reports": int(hit.sum()),
        "share_flagged_discordant": round(float((~agree)[hit].mean()), 4) if hit.any() else None,
        "note": "share of the specialist's correct fracture reports that a discordant-downgrade rule would also demote",
    }
    return out


def specialist_concordance(specialist: dict[str, str | int], rows: list[dict], modality: str) -> dict:
    """D3 kappa between a specialist and MedGemma on the same images; `specialist` maps image id -> label."""
    pairs = [(specialist[r["image"]], bone_prediction(r, False) if modality == "bone_xray" else skin_prediction(r))
             for r in rows if r.get("ok") and r["image"] in specialist]
    if len(pairs) < 4:
        return {"n": len(pairs), "note": "too few paired reads"}
    a, b = zip(*pairs, strict=True)
    return {"n": len(pairs), "kappa": round(cohen_kappa(list(a), list(b)), 4), "agreement": round(float(np.mean([x == y for x, y in pairs])), 4)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["fracatlas", "ham10000"], default=None)
    ap.add_argument("--reads-dir", type=Path, default=READS)
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "concordance.json")
    args = ap.parse_args(argv)
    report: dict = {
        "what": "MedGemma 1.5 4B (4-bit) as second reader, scored against ground truth on P2's eval batches",
        "specialist_vs_medgemma": "from P2's saved test predictions: see datasets.<name>.specialist_vs_medgemma (skin official test, bone FracAtlas test)",
        "datasets": {},
    }
    for name, fn in (("fracatlas", eval_bone), ("ham10000", eval_skin)):
        path = args.reads_dir / f"{name}.jsonl"
        if (args.dataset in (None, name)) and path.is_file():
            rows = read_rows(path)
            report["datasets"][name] = fn(rows)
            if name == "ham10000":
                report["datasets"][name]["specialist_vs_medgemma"] = skin_specialist_comparison(rows)
            else:
                report["datasets"][name]["specialist_vs_medgemma"] = bone_specialist_comparison(rows)
            planned = json.loads((ROOT / "ml" / "data" / "eval_index.json").read_text(encoding="utf-8"))["datasets"][name]["n_batch"]
            report["datasets"][name]["progress"] = {
                "reads": len(rows), "planned": planned,
                "status": "complete" if len(rows) >= planned else "partial: a random prefix of the planned batch (fixed shuffle)",
            }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
