"""P2.8 / P2.9: fit and evaluate calibration for the classifiers from their cached predictions.

    python ml/eval/calibration.py            # skin_cls and brain_cls, writes ml/artifacts/<model>/calibration.json and reports/calibration.json

Temperature is fitted on the validation split, conformal thresholds and confidence tiers on the calibration split,
and every number is reported on held-out test splits only. Coverage carries an exact Clopper-Pearson interval.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for p in (REPO, REPO / "backend"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from medproof.calibrate import conformal, selective, temperature  # noqa: E402
from medproof.calibrate.calibrator import Calibrator  # noqa: E402

from ml.eval import bootstrap as bs  # noqa: E402
from ml.eval import metrics as mt  # noqa: E402

ART = REPO / "ml" / "artifacts"
ALPHAS = (0.10, 0.05)
MODELS = {  # directory -> (class order source, test splits)
    "skin_cls": ["official_test", "test"],
    "brain_cls": ["test"],
}


def _load(model_dir: Path, split: str) -> dict:
    z = np.load(model_dir / "predictions" / f"{split}.npz", allow_pickle=False)
    return {k: z[k] for k in z.files}


def reliability_bins(y: np.ndarray, probs: np.ndarray, bins: int = 15) -> list[dict]:
    conf = probs.max(1)
    correct = (probs.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    out = []
    for b in range(bins):
        m = which == b
        out.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "n": int(m.sum()), "confidence": float(conf[m].mean()) if m.any() else None, "accuracy": float(correct[m].mean()) if m.any() else None})
    return out


def _ece_ci(y, p, groups, B):
    return bs.ci({"y": y, "p": p}, lambda y, p: mt.ece(y, p, 15), groups=groups, strata=y, B=B)


def calibrate_classifier(name: str, test_splits: list[str], B: int = 1000) -> dict:
    mdir = ART / name
    meta = json.loads((mdir / "model_meta.json").read_text())
    classes = meta["classes"]
    val, cal = _load(mdir, "val"), _load(mdir, "cal")
    cal_obj = Calibrator.fit(classes, val["logits"], val["y"], cal["logits"], cal["y"], alpha=ALPHAS[0], mondrian=True)
    report: dict = {"model_id": meta["model_id"], "classes": classes, "temperature": cal_obj.temperature, "fit": {"temperature_on": "val", "conformal_on": "cal", "n_val": int(len(val["y"])), "n_cal": int(len(cal["y"]))},
                    "tier_rule": cal_obj.tier_rule.to_dict(), "conformal": {}, "splits": {}}
    p_cal = temperature.softmax(cal["logits"], cal_obj.temperature)
    for alpha in ALPHAS:
        report["conformal"][f"alpha_{alpha}"] = {"alpha": alpha, "coverage_target": 1 - alpha, "qhat": conformal.fit_aps(p_cal, cal["y"], alpha),
                                                 "mondrian_qhat": conformal.fit_aps_mondrian(p_cal, cal["y"], alpha, len(classes))}
    for split in test_splits:
        d = _load(mdir, split)
        y, logits, groups = d["y"], d["logits"], d["group"] if "group" in d else np.arange(len(d["y"]))
        p_raw, p_cal_t = temperature.softmax(logits, 1.0), temperature.softmax(logits, cal_obj.temperature)
        u = Calibrator._u(logits)  # the same per-input draw the serving code uses
        blk: dict = {"n": int(len(y)), "n_groups": int(len(np.unique(groups)))}
        blk["ece_raw"] = _ece_ci(y, p_raw, groups, B)
        blk["ece_calibrated"] = _ece_ci(y, p_cal_t, groups, B)
        blk["nll_raw"], blk["nll_calibrated"] = mt.nll(y, p_raw), mt.nll(y, p_cal_t)
        blk["brier_raw"], blk["brier_calibrated"] = mt.brier(y, p_raw), mt.brier(y, p_cal_t)
        blk["reliability_raw"], blk["reliability_calibrated"] = reliability_bins(y, p_raw), reliability_bins(y, p_cal_t)
        blk["conformal"] = {}
        for alpha in ALPHAS:
            q = report["conformal"][f"alpha_{alpha}"]["qhat"]
            sets = conformal.aps_sets(p_cal_t, q, u)
            hit = np.array([int(t) in s for s, t in zip(sets, y)])
            lo, hi = conformal.clopper_pearson(int(hit.sum()), len(y))
            sizes = np.array([len(s) for s in sets])
            per_class = {classes[c]: {"n": int((y == c).sum()), "coverage": float(hit[y == c].mean())} for c in range(len(classes)) if (y == c).any()}
            msets = conformal.aps_sets_mondrian(p_cal_t, report["conformal"][f"alpha_{alpha}"]["mondrian_qhat"], u)
            mhit = np.array([int(t) in s for s, t in zip(msets, y)])
            blk["conformal"][f"alpha_{alpha}"] = {
                "coverage_target": 1 - alpha, "coverage": float(hit.mean()), "coverage_ci95": [lo, hi], "within_3_points_of_target": bool(abs(hit.mean() - (1 - alpha)) <= 0.03),
                "mean_set_size": float(sizes.mean()), "singleton_fraction": float((sizes == 1).mean()), "empty_or_wide_fraction": float((sizes > 2).mean()),
                "per_class_coverage": per_class, "mondrian_coverage": float(mhit.mean()), "mondrian_mean_set_size": float(np.mean([len(s) for s in msets])),
                "mondrian_per_class_coverage": {classes[c]: float(mhit[y == c].mean()) for c in range(len(classes)) if (y == c).any()},
            }
        out = cal_obj.calibrate(logits)
        correct = out.probs.argmax(1) == y
        tiers = np.array(out.tiers)
        blk["tiers"] = {t: {"n": int((tiers == t).sum()), "share": float((tiers == t).mean()), "accuracy": float(correct[tiers == t].mean()) if (tiers == t).any() else None} for t in ("high", "moderate", "low", "abstain")}
        conf = out.confidence
        blk["selective"] = {
            "aurc": bs.ci({"c": conf, "k": correct.astype(float)}, lambda c, k: selective.aurc(c, k), groups=groups, B=B),
            "oracle_aurc": selective.oracle_aurc(correct), "accuracy_full_coverage": float(correct.mean()),
            "coverage_at_5pct_error": selective.coverage_at_risk(conf, correct, 0.05), "coverage_at_10pct_error": selective.coverage_at_risk(conf, correct, 0.10),
            "curve": [{"coverage": float(c), "risk": float(r)} for c, r in zip(*[a[:: max(1, len(y) // 200)] for a in selective.risk_coverage(conf, correct)])],
        }
        report["splits"][split] = blk
    cal_obj.meta.update({"model_id": meta["model_id"], "fit_splits": {"temperature": "val", "conformal_and_tiers": "cal"}})
    cal_obj.save(mdir / "calibration.json")
    return report


def main() -> int:
    out = {"generated_by": "ml/eval/calibration.py", "seed": bs.SEED, "models": {}}
    for name, splits in MODELS.items():
        if not (ART / name / "model_meta.json").is_file():
            print(f"skipping {name}: no artifacts")
            continue
        out["models"][name] = calibrate_classifier(name, splits)
        for s, b in out["models"][name]["splits"].items():
            c = b["conformal"]["alpha_0.1"]
            print(f"{name}/{s}: T={out['models'][name]['temperature']:.3f} ECE {b['ece_raw']['point']:.3f}->{b['ece_calibrated']['point']:.3f}  coverage@90 {c['coverage']:.3f} {c['coverage_ci95']} set size {c['mean_set_size']:.2f}")
    p = REPO / "reports" / "calibration.json"
    old = json.loads(p.read_text()) if p.is_file() else {"models": {}}
    old["models"].update(out["models"])
    old.update({k: v for k, v in out.items() if k != "models"})
    p.write_text(json.dumps(old, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
