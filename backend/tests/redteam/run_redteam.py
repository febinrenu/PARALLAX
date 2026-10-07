"""Runs the P3 red-team cases and writes reports/redteam_p3.json.

    cd backend && python -m tests.redteam.run_redteam

The summary is per category with the failing cases listed, so the validation page can show it directly.
The pixel-level cases of plan 9.7 (blank image, rotated film, wrong modality) are P1/P4's and not included.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from tests.redteam.cases import CASES

OUT = Path(__file__).resolve().parents[3] / "reports" / "redteam_p3.json"


def main() -> int:
    by_cat: dict[str, dict] = defaultdict(lambda: {"cases": 0, "passed": 0, "failed": []})
    for c in CASES:
        try:
            ok, detail = c.run()
        except Exception as exc:  # noqa: BLE001 - a crash is a failed case, not a stopped run
            ok, detail = False, f"raised {type(exc).__name__}: {exc}"
        row = by_cat[c.category]
        row["cases"] += 1
        row["passed"] += int(ok)
        if not ok:
            row["failed"].append({"id": c.id, "expected": c.expected, "got": detail})
    total = sum(r["cases"] for r in by_cat.values())
    passed = sum(r["passed"] for r in by_cat.values())
    report = {"scope": "P3 components, offline, language models replaced by fakes", "cases": total, "passed": passed, "categories": dict(by_cat)}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: (v if k != "categories" else {c: f"{r['passed']}/{r['cases']}" for c, r in v.items()}) for k, v in report.items()}, indent=1))
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
