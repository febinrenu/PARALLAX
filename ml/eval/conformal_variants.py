"""Optional extra: randomised APS (shipped) against class-conditional (Mondrian) APS and RAPS on the skin and brain classifiers.

    python ml/eval/conformal_variants.py        # writes reports/conformal_variants.json

All variants are fitted on the calibration split after temperature scaling on the validation split and scored on held-out test splits.
Reported: marginal coverage with an exact interval, the lowest per-class coverage (what Mondrian is for), mean set size.
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

from medproof.calibrate import conformal, temperature  # noqa: E402
from medproof.calibrate.calibrator import Calibrator  # noqa: E402

ART = REPO / "ml" / "artifacts"
ALPHA = 0.1
JOBS = {"skin_cls": ["official_test", "test", "milk10k"], "brain_cls": ["test", "bdneuro"]}


def _row(sets, y, classes):
    hit = np.array([int(t) in s for s, t in zip(sets, y)])
    lo, hi = conformal.clopper_pearson(int(hit.sum()), len(y))
    per = {classes[c]: float(hit[y == c].mean()) for c in range(len(classes)) if (y == c).any()}
    return {"coverage": float(hit.mean()), "coverage_ci95": [lo, hi], "mean_set_size": conformal.mean_set_size(sets), "min_class_coverage": min(per.values()), "worst_class": min(per, key=per.get)}


def main() -> int:
    res = {"alpha": ALPHA, "target_coverage": 1 - ALPHA, "models": {}}
    for name, splits in JOBS.items():
        meta = json.loads((ART / name / "model_meta.json").read_text())
        classes = meta["classes"]
        val, cal = (np.load(ART / name / "predictions" / f"{s}.npz", allow_pickle=False) for s in ("val", "cal"))
        T = temperature.fit_temperature(val["logits"], val["y"])
        pc = temperature.softmax(cal["logits"], T)
        q_aps = conformal.fit_aps(pc, cal["y"], ALPHA)
        q_mon = conformal.fit_aps_mondrian(pc, cal["y"], ALPHA, len(classes))
        q_raps = {(k, lam): conformal.fit_raps(pc, cal["y"], ALPHA, k_reg=k, lam=lam) for k, lam in ((1, 0.01), (2, 0.05))}
        res["models"][name] = {}
        for s in splits:
            p = ART / name / "predictions" / f"{s}.npz"
            if not p.is_file():
                continue
            z = np.load(p, allow_pickle=False)
            pt, y = temperature.softmax(z["logits"], T), z["y"]
            u = Calibrator._u(z["logits"])
            blk = {"randomised APS (shipped)": _row(conformal.aps_sets(pt, q_aps, u), y, classes), "Mondrian APS": _row(conformal.aps_sets_mondrian(pt, q_mon, u), y, classes)}
            for (k, lam), q in q_raps.items():
                blk[f"RAPS (k_reg={k}, lambda={lam})"] = _row(conformal.raps_sets(pt, q, k, lam), y, classes)
            res["models"][name][s] = blk
            for v, r in blk.items():
                print(f"{name:9s} {s:14s} {v:30s} coverage {r['coverage']:.3f}  min class {r['min_class_coverage']:.2f} ({r['worst_class']})  size {r['mean_set_size']:.2f}")
    (REPO / "reports" / "conformal_variants.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
