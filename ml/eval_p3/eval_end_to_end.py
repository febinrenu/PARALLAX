"""Guard -> extraction -> contradiction rules on the synthetic notes, with extracted instead of gold facts.

The rules scored 1.00 / 1.00 on gold facts (eval_contradictions.py). This measures what that becomes
once the facts come from the language model, on the same notes, so the drop is the cost of extraction.

    python -m ml.eval_p3.eval_end_to_end --split test       # cached Groq answers are free
    python -m ml.eval_p3.eval_end_to_end --split dev        # needs fresh extraction calls
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))

from medproof.context.contradictions import Demographics, check  # noqa: E402
from medproof.context.extract import extract  # noqa: E402
from medproof.context.injection_guard import InjectionGuard  # noqa: E402
from medproof.llm.config import LLMConfig  # noqa: E402
from medproof.llm.groq_pool import GroqPool  # noqa: E402

from ml.data.notes_schema import read_jsonl  # noqa: E402
from ml.eval_p3.eval_context import load_env  # noqa: E402
from ml.eval_p3.eval_contradictions import NOTES, RULE_FLAGS, facts_for, finding_for  # noqa: E402


def score(rows: list[tuple[set[str], str | None]]) -> dict:
    """rows: (flags raised, expected flag or None for a clean note)."""
    tp = fn = fp = clean = clean_flagged = 0
    per: Counter[str] = Counter()
    for raised, expected in rows:
        if expected is None:
            clean += 1
            clean_flagged += int(bool(raised))
            fp += len(raised)
        else:
            hit = expected in raised
            tp += int(hit)
            fn += int(not hit)
            fp += len(raised - {expected})
            per[expected + (":hit" if hit else ":miss")] += 1
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "tp": tp, "fn": fn, "unexpected_flags": fp,
            "clean_notes": clean, "clean_notes_flagged": clean_flagged, "by_flag": dict(sorted(per.items()))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test", "all"], default="test")
    args = ap.parse_args(argv)
    load_env()
    notes = [n for n in read_jsonl(NOTES) if (args.split == "all" or n.split == args.split) and not n.injection and n.text]
    pool = GroqPool(LLMConfig(cache_dir=ROOT / ".cache" / "llm", deadline_s=300.0, max_attempts=6))
    guard = InjectionGuard(None)
    gold_rows: list[tuple[set[str], str | None]] = []
    ext_rows: list[tuple[set[str], str | None]] = []
    cue_rows: list[tuple[set[str], str | None]] = []
    only_rows: list[tuple[set[str], str | None]] = []
    failed = 0
    for n in notes:
        demo = Demographics(n.context.age, n.context.sex if n.context.sex in ("M", "F") else None)  # type: ignore[arg-type]
        expected = n.contradiction.expected_flag if n.contradiction else None
        finding = finding_for(n)
        res = extract(n.text, n.note_id, pool, guard.check(n.text))
        if not res.ok:
            failed += 1
            continue
        gold = check([finding], facts_for(n), demo).findings[0].flags
        got = check([finding], res.facts, demo).findings[0].flags
        plus = check([finding], res.facts, demo, note_text=n.text, note_id=n.note_id).findings[0].flags
        cue_rows.append((set(plus) & RULE_FLAGS, expected))
        alone = check([finding], [], demo, note_text=n.text, note_id=n.note_id).findings[0].flags
        only_rows.append((set(alone) & RULE_FLAGS, expected))
        gold_rows.append((set(gold) & RULE_FLAGS, expected))
        ext_rows.append((set(got) & RULE_FLAGS, expected))
    out = {
        "split": args.split, "notes_scored": len(ext_rows), "extraction_failures": failed,
        "contradiction_notes": sum(1 for _, e in ext_rows if e), "gold_facts": score(gold_rows), "model_facts_only": score(ext_rows),
        "model_facts_plus_cue_scan": score(cue_rows),
        "cue_scan_only_no_model": score(only_rows),
    }
    path = ROOT / "reports" / f"p3_end_to_end_{args.split}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=1))
    pool.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
