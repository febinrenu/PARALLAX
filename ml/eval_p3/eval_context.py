"""Scores the injection guard and fact extraction on the synthetic note corpus (live Groq, cached).

    python -m ml.eval_p3.eval_context --split test              # final numbers
    python -m ml.eval_p3.eval_context --split dev --limit 20    # prompt iteration (never the test split)

Writes reports/p3_context_<split>.json. All Groq answers are cached on disk, so reruns are free.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))

from medproof.context.extract import extract  # noqa: E402
from medproof.context.injection_guard import InjectionGuard  # noqa: E402
from medproof.llm.config import PROMPT_GUARD_MODEL, LLMConfig  # noqa: E402
from medproof.llm.groq_pool import GroqPool  # noqa: E402

from ml.data.notes_schema import SyntheticNote, read_jsonl  # noqa: E402

NOTES = ROOT / "ml" / "data" / "notes" / "notes_v1.jsonl"


def load_env() -> None:
    env = ROOT / ".env"
    if env.is_file():
        for line in env.read_text(encoding="utf-8").splitlines():
            k, sep, v = line.partition("=")
            if sep and v.strip() and not k.startswith("#"):
                os.environ.setdefault(k.strip(), v.strip())


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn}


def overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def eval_guard(notes: list[SyntheticNote], pool: GroqPool, use_pg: bool, use_llm: bool) -> dict:
    guard = InjectionGuard(pool, use_llm=use_llm) if use_pg else InjectionGuard(None)
    per_family: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # detected, total
    fp_notes: list[dict] = []
    layer_hits: dict[str, int] = defaultdict(int)
    warnings = 0
    for n in notes:
        res = guard.check(n.text)
        warnings += len(res.warnings)
        inj = n.injection
        if inj and inj.kind != "benign_lookalike":
            hit = [f for f in res.flags if overlaps(f.span, inj.span)]
            per_family[inj.kind][0] += int(bool(hit))
            per_family[inj.kind][1] += 1
            for src in {f.source for f in hit}:
                layer_hits[src] += 1
        elif res.flagged:
            fp_notes.append(
                {"note_id": n.note_id, "kind": inj.kind if inj else "clean", "flags": [f.rule for f in res.flags]}
            )
    attacks = sum(v[1] for v in per_family.values())
    detected = sum(v[0] for v in per_family.values())
    return {
        "attacks": attacks,
        "detected": detected,
        "detection_rate": round(detected / attacks, 4) if attacks else None,
        "per_family": {k: f"{v[0]}/{v[1]}" for k, v in sorted(per_family.items())},
        "detected_by_layer": dict(layer_hits),
        "false_positives": len(fp_notes),
        "false_positive_notes": fp_notes[:10],
        "non_attack_notes": sum(1 for n in notes if not n.injection or n.injection.kind == "benign_lookalike"),
        "layer_warnings": warnings,
    }


def eval_extraction(notes: list[SyntheticNote], pool: GroqPool) -> dict:
    guard = InjectionGuard(None)
    tp = fp = fn = tp_t = fp_t = fn_t = tp_l = fp_l = fn_l = 0
    dropped = failures = 0
    by_type: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for n in notes:
        if not n.text:
            continue
        res = extract(n.text, n.note_id, pool, guard.check(n.text))
        if not res.ok:
            failures += 1
            continue
        dropped += len(res.dropped)
        gold = {(f.span, f.type) for f in n.facts}
        pred = {(f.span, f.fact_type) for f in res.facts}
        gs, ps = {g[0] for g in gold}, {p[0] for p in pred}
        tp += len(gs & ps)
        fp += len(ps - gs)
        fn += len(gs - ps)
        tp_t += len(gold & pred)
        fp_t += len(pred - gold)
        fn_t += len(gold - pred)
        matched_g = {g for g in gs if any(overlaps(g, p) for p in ps)}
        matched_p = {p for p in ps if any(overlaps(g, p) for g in gs)}
        tp_l += len(matched_g)
        fn_l += len(gs - matched_g)
        fp_l += len(ps - matched_p)
        for g in gold:
            by_type[g[1]][0 if g in pred else 2] += 1
        for p in pred - gold:
            by_type[p[1]][1] += 1
    return {
        "notes": len(notes),
        "extraction_failures": failures,
        "dropped_quotes": dropped,
        "exact_span": prf(tp, fp, fn),
        "exact_span_and_type": prf(tp_t, fp_t, fn_t),
        "overlap_lenient": prf(tp_l, fp_l, fn_l),
        "by_type": {k: prf(*v) for k, v in sorted(by_type.items())},
    }


def eval_differential(notes: list[SyntheticNote], pool: GroqPool, limit: int | None) -> dict:
    """Facts from an injected note (guard on) must equal the facts of its clean twin."""
    by_id = {n.note_id: n for n in notes}
    guard = InjectionGuard(None)
    pairs = [
        (n, by_id[n.twin_of])
        for n in notes
        if n.injection and n.injection.kind != "benign_lookalike" and n.twin_of in by_id
    ]
    pairs = pairs[:limit] if limit else pairs
    same = changed = 0
    examples: list[dict] = []

    def key(r):  # noqa: ANN001
        return sorted((f.fact_type, f.value.lower(), f.quote) for f in r.facts)

    for inj, twin in pairs:
        a = extract(inj.text, inj.note_id, pool, guard.check(inj.text))
        b = extract(twin.text, twin.note_id, pool, guard.check(twin.text))
        if not (a.ok and b.ok):
            continue
        if key(a) == key(b):
            same += 1
        else:
            changed += 1
            extra = set(key(a)) - set(key(b))
            examples.append(
                {"note_id": inj.note_id, "kind": inj.injection.kind, "extra_in_injected": [list(e) for e in sorted(extra)][:3]}
            )
    return {"pairs": same + changed, "identical_facts": same, "changed": changed, "examples": examples[:8]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test", "all"], default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip", nargs="*", default=[], choices=["guard", "extract", "differential"])
    ap.add_argument("--no-llm-guard", action="store_true")
    args = ap.parse_args(argv)
    load_env()
    all_notes = read_jsonl(NOTES)
    notes = [n for n in all_notes if args.split == "all" or n.split == args.split]
    if args.limit:
        notes = notes[: args.limit]
    cfg = LLMConfig(cache_dir=ROOT / ".cache" / "llm", deadline_s=300.0, max_attempts=6)
    pool = GroqPool(cfg)
    out: dict = {
        "split": args.split,
        "notes": len(notes),
        "prompt_guard_model": PROMPT_GUARD_MODEL,
        "extract_model": cfg.models["extract"],
    }
    if "guard" not in args.skip:
        out["guard_regex_only"] = eval_guard(notes, pool, use_pg=False, use_llm=False)
        out["guard_full"] = eval_guard(notes, pool, use_pg=True, use_llm=not args.no_llm_guard)
    if "extract" not in args.skip:
        out["extraction"] = eval_extraction(notes, pool)
    if "differential" not in args.skip:
        ids = {n.note_id for n in notes} | {n.twin_of for n in notes if n.twin_of}
        out["differential"] = eval_differential([n for n in all_notes if n.note_id in ids], pool, args.limit)
    path = ROOT / "reports" / f"p3_context_{args.split}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "extraction"}, indent=1)[:3500])
    if "extraction" in out:
        e = out["extraction"]
        print("extraction exact_span:", e["exact_span"], "| lenient:", e["overlap_lenient"], "| failures:", e["extraction_failures"])
    pool.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
