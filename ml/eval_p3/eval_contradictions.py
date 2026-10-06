"""Scores the contradiction rules (P3.7) on the synthetic notes, using the gold facts.

Gold facts isolate the rules from extraction error (extraction is scored separately in
eval_context.py). Offline and deterministic.

    python -m ml.eval_p3.eval_contradictions
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))

from medproof.context.contradictions import Demographics, check  # noqa: E402
from medproof.context.extract import ExtractedFact  # noqa: E402
from medproof.core.schemas import Finding, ImageEvidence  # noqa: E402

from ml.data.notes_schema import SyntheticNote, read_jsonl  # noqa: E402

NOTES = ROOT / "ml" / "data" / "notes" / "notes_v1.jsonl"
RULE_FLAGS = {"laterality_conflict", "history_conflict", "demographic_implausible", "demographic_mismatch", "symptom_finding_incoherent"}
CXR_LABEL = {"consolidation": "Consolidation", "effusion": "Effusion", "pneumothorax": "Pneumothorax", "nodule": "Nodule",
             "cardiomegaly": "Cardiomegaly", "atelectasis": "Atelectasis", "edema": "Edema"}


def finding_for(note: SyntheticNote) -> Finding:
    c = note.context
    label = CXR_LABEL.get(c.seed_finding or "", c.seed_finding or "Consolidation") if c.modality == "cxr" else (c.seed_finding or "fracture")
    region = f"{c.side} lower zone" if c.side in ("left", "right") else "cardiac"
    ev = ImageEvidence(evidence_id="ie_1", kind="bbox", bbox_xyxy=(1.0, 1.0, 9.0, 9.0), source_model="gold", method="gold", region_name=region)
    return Finding(finding_id="f1", modality=c.modality, label=label, prob_raw=0.9, prob_calibrated=0.9, conformal_set=[],
                   tier="high", status="uncertain", image_evidence=[ev])


def facts_for(note: SyntheticNote) -> list[ExtractedFact]:
    return [ExtractedFact(note.note_id, f.span, f.quote, f.type, f.value, f.polarity) for f in note.facts]


def evaluate(notes: list[SyntheticNote]) -> dict:
    tp: Counter[str] = Counter()
    fn: Counter[str] = Counter()
    fp_flags: Counter[str] = Counter()
    clean = clean_flagged = contradictions = 0
    misses: list[dict] = []
    false_alarms: list[dict] = []
    for n in notes:
        if n.injection:
            continue
        res = check([finding_for(n)], facts_for(n), Demographics(n.context.age, n.context.sex if n.context.sex in ("M", "F") else None))  # type: ignore[arg-type]
        raised = set(res.findings[0].flags) & RULE_FLAGS
        if n.contradiction:
            contradictions += 1
            exp = n.contradiction.expected_flag
            if exp in raised:
                tp[n.contradiction.kind] += 1
            else:
                fn[n.contradiction.kind] += 1
                misses.append({"note_id": n.note_id, "kind": n.contradiction.kind, "expected": exp, "raised": sorted(raised)})
            for extra in raised - {exp}:
                fp_flags[extra] += 1
                false_alarms.append({"note_id": n.note_id, "kind": n.contradiction.kind, "unexpected": extra})
        else:
            clean += 1
            if raised:
                clean_flagged += 1
                for fl in raised:
                    fp_flags[fl] += 1
                false_alarms.append({"note_id": n.note_id, "kind": "clean", "unexpected": sorted(raised)})
    tp_all, fn_all, fp_all = sum(tp.values()), sum(fn.values()), sum(fp_flags.values())
    prec = tp_all / (tp_all + fp_all) if tp_all + fp_all else 0.0
    rec = tp_all / (tp_all + fn_all) if tp_all + fn_all else 0.0
    per_kind: dict[str, dict] = defaultdict(dict)
    for k in sorted(set(tp) | set(fn)):
        per_kind[k] = {"recall": round(tp[k] / (tp[k] + fn[k]), 4), "tp": tp[k], "fn": fn[k]}
    return {
        "contradiction_notes": contradictions, "clean_notes": clean, "precision": round(prec, 4), "recall": round(rec, 4),
        "clean_notes_flagged": clean_flagged, "unexpected_flags": dict(fp_flags), "per_kind": dict(per_kind),
        "misses": misses[:10], "false_alarms": false_alarms[:10], "facts_used": "gold (extraction scored separately)",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "p3_contradictions.json")
    args = ap.parse_args(argv)
    result = evaluate(read_jsonl(NOTES))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
