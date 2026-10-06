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
        "specialist_vs_medgemma": "pending: needs P2's trained skin/bone weights (see specialist_concordance)",
        "datasets": {},
    }
    for name, fn in (("fracatlas", eval_bone), ("ham10000", eval_skin)):
        path = args.reads_dir / f"{name}.jsonl"
        if (args.dataset in (None, name)) and path.is_file():
            rows = read_rows(path)
            report["datasets"][name] = fn(rows)
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
