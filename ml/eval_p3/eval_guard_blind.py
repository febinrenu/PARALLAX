"""Measures each injection-guard layer on the fresh red-team set (backend/tests/context/redteam_fresh.py).

That set was written after the regex rules were frozen and phrased differently from the generated
corpus, so these numbers estimate how the layers generalise; the in-sample corpus numbers do not.

    python -m ml.eval_p3.eval_guard_blind      # live Groq on first run, cached afterwards

Writes reports/p3_guard_blind.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))

from medproof.context.injection_guard import InjectionGuard  # noqa: E402
from medproof.llm.config import PROMPT_GUARD_MODEL, LLMConfig  # noqa: E402
from medproof.llm.groq_pool import GroqPool  # noqa: E402
from tests.context.redteam_fresh import ATTACKS, BENIGN  # noqa: E402

from ml.eval_p3.eval_context import load_env  # noqa: E402


def note(text: str) -> str:
    return "63F c/o fever. " + text + " h/o TB."


def main() -> int:
    load_env()
    pool = GroqPool(LLMConfig(cache_dir=ROOT / ".cache" / "llm", deadline_s=300.0, max_attempts=6))
    layers = {
        "regex": InjectionGuard(None),
        "prompt_guard_0.5": InjectionGuard(pool, use_regex=False, use_llm=False, pg_threshold=0.5),
        "prompt_guard_0.9": InjectionGuard(pool, use_regex=False, use_llm=False, pg_threshold=0.9),
        "llm_classifier": InjectionGuard(pool, use_regex=False, use_pg=False, use_llm=True),
    }
    attacks = [(fam, t) for fam, ts in ATTACKS.items() for t in ts]
    flagged = {name: [g.check(note(t)).flagged for _, t in attacks] for name, g in layers.items()}
    benign = {name: [g.check(note(t)).flagged for t in BENIGN] for name, g in layers.items()}
    out: dict = {"attacks": len(attacks), "benign": len(BENIGN), "prompt_guard_model": PROMPT_GUARD_MODEL, "layers": {}}
    for name in layers:
        fam: dict[str, list[int]] = {}
        for (f, _), hit in zip(attacks, flagged[name], strict=True):
            fam.setdefault(f, [0, 0])
            fam[f][0] += int(hit)
            fam[f][1] += 1
        out["layers"][name] = {
            "recall": f"{sum(flagged[name])}/{len(attacks)}",
            "false_positives": f"{sum(benign[name])}/{len(BENIGN)}",
            "per_family": {k: f"{v[0]}/{v[1]}" for k, v in fam.items()},
            "missed": [t for (_, t), h in zip(attacks, flagged[name], strict=True) if not h] if name == "llm_classifier" else None,
            "false_positive_texts": [t for t, h in zip(BENIGN, benign[name], strict=True) if h],
        }
    combos = {"regex+prompt_guard_0.9+llm": ["regex", "prompt_guard_0.9", "llm_classifier"], "prompt_guard_0.9+llm": ["prompt_guard_0.9", "llm_classifier"]}
    out["combined"] = {
        label: {
            "recall": f"{sum(any(flagged[n][i] for n in names) for i in range(len(attacks)))}/{len(attacks)}",
            "false_positives": f"{sum(any(benign[n][i] for n in names) for i in range(len(BENIGN)))}/{len(BENIGN)}",
        }
        for label, names in combos.items()
    }
    path = ROOT / "reports" / "p3_guard_blind.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=1, ensure_ascii=False))
    pool.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
