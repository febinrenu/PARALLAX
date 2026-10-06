"""make eval is deterministic, needs only committed files, and fits the 2-minute budget."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def test_metrics_json_is_reproduced_exactly_within_the_time_budget():
    t0 = time.time()
    r = subprocess.run([sys.executable, str(REPO / "ml" / "eval" / "run_all.py"), "--check"], cwd=REPO, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    assert time.time() - t0 < 180, "make eval should recompute everything in about two minutes"
