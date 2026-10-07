"""Stability score: does a finding survive realistic image perturbations?

Each of the eight perturbations (noise, contrast, gamma, JPEG, small rotation, downsample, blur,
crop) is applied at a fixed severity and seed, the image is re-scored, and a finding "flips" when
it crosses the positive threshold in either direction. The flip rate is flips / tests. All labels
share the same forward passes, so a study costs eight extra model calls.

`score` is any callable image -> probabilities over all labels; the CXR reader's `.score` fits.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from medproof.core.schemas import Finding, StageResult, Stability
from medproof.verify import perturbations as P

Score = Callable[[np.ndarray], np.ndarray]
UNSTABLE_FLIP_RATE = 0.25  # status rule in plan.md section 4: above this a finding is not "verified"


@dataclass(frozen=True)
class StabilityConfig:
    perturbations: tuple = P.NAMES
    severity: int = 2  # "realistic" rather than destructive; P2.12 sweeps all five
    seeds: tuple = (0,)
    positive_threshold: float = 0.5
    unstable_flip_rate: float = UNSTABLE_FLIP_RATE

    def __post_init__(self):
        if self.severity not in P.SEVERITIES:
            raise ValueError(f"severity must be one of {P.SEVERITIES}")
        bad = [n for n in self.perturbations if n not in P.NAMES]
        if bad:
            raise ValueError(f"unknown perturbations: {bad}")


@dataclass
class StabilityResult:
    tests: int
    flips: int
    flip_rate: float
    worst_perturbation: str | None  # the perturbation with the most flips (ties: largest mean shift); None if no flips
    per_perturbation: dict = field(default_factory=dict)  # name -> {"flipped", "flips", "mean_shift", "mean_prob"}
    base_prob: float = 0.0

    def to_dict(self) -> dict:
        return {"tests": self.tests, "flips": self.flips, "flip_rate": round(self.flip_rate, 5),
                "worst_perturbation": self.worst_perturbation, "base_prob": round(float(self.base_prob), 5),
                "per_perturbation": {k: {kk: (round(float(vv), 5) if isinstance(vv, float) else vv) for kk, vv in v.items()}
                                     for k, v in self.per_perturbation.items()}}


def _probs_under_perturbations(score: Score, img: np.ndarray, cfg: StabilityConfig):
    base = np.asarray(score(img), np.float32)
    runs = {n: [np.asarray(score(P.apply(n, img, cfg.severity, s)), np.float32) for s in cfg.seeds] for n in cfg.perturbations}
    return base, runs


def _summarise(base: np.ndarray, runs: dict, idx: int, cfg: StabilityConfig) -> StabilityResult:
    thr = cfg.positive_threshold
    base_pos = bool(base[idx] >= thr)
    per, flips = {}, 0
    for name, probs in runs.items():
        ps = np.array([p[idx] for p in probs], np.float32)
        f = int(((ps >= thr) != base_pos).sum())
        flips += f
        per[name] = {"flipped": bool(f > 0), "flips": f, "mean_shift": float(np.abs(ps - base[idx]).mean()), "mean_prob": float(ps.mean())}
    tests = sum(len(v) for v in runs.values())
    worst = None
    if flips:
        worst = max(per, key=lambda n: (per[n]["flips"], per[n]["mean_shift"]))
    return StabilityResult(tests, flips, flips / max(1, tests), worst, per, float(base[idx]))


def assess_labels(score: Score, img: np.ndarray, label_indices: list[int], cfg: StabilityConfig | None = None) -> dict[int, StabilityResult]:
    cfg = cfg or StabilityConfig()
    base, runs = _probs_under_perturbations(score, img, cfg)
    return {i: _summarise(base, runs, i, cfg) for i in label_indices}


def assess_label(score: Score, img: np.ndarray, label_idx: int, cfg: StabilityConfig | None = None) -> StabilityResult:
    return assess_labels(score, img, [label_idx], cfg)[label_idx]


def assess_output(reader, img, out, cfg: StabilityConfig | None = None) -> dict[str, StabilityResult]:
    """Stability of every finding in a ReaderOutput. `img` must be the image the reader saw."""
    arr = img.analysis if hasattr(img, "analysis") else np.asarray(img, np.float32)
    idx = {rf.label: out.labels.index(rf.label) for rf in out.findings}
    by_index = assess_labels(reader.score, arr, sorted(set(idx.values())), cfg)
    return {label: by_index[i] for label, i in idx.items()}


def apply_to_findings(findings: list[Finding], results: dict[str, StabilityResult], cfg: StabilityConfig | None = None) -> list[Finding]:
    """Copies of `findings` with the contract Stability object filled in. Inputs stay untouched."""
    cfg = cfg or StabilityConfig()
    updated = []
    for f in findings:
        r = results.get(f.label)
        if r is None:
            updated.append(f)
            continue
        flags = list(f.flags)
        if r.flip_rate > cfg.unstable_flip_rate and "unstable" not in flags:
            flags.append("unstable")
        updated.append(f.model_copy(update={"stability": Stability(tests=r.tests, flip_rate=r.flip_rate, worst_perturbation=r.worst_perturbation), "flags": flags}))
    return updated


def run(ctx, reader=None, cfg: StabilityConfig | None = None) -> StageResult:
    """Stage: `ctx` needs `.decoded`, `.reader_output` and `.findings`. Never raises."""
    t0 = time.perf_counter()

    def done(ok, payload, warnings):
        return StageResult(stage="stability", ok=ok, ms=int((time.perf_counter() - t0) * 1000), payload=payload, warnings=warnings)

    img, out, findings = (getattr(ctx, n, None) for n in ("decoded", "reader_output", "findings"))
    if img is None or out is None or findings is None or reader is None:
        return done(False, {"error": "missing decoded image, reader output, findings or reader"}, ["stability skipped: missing inputs"])
    try:
        results = assess_output(reader, img, out, cfg)
        updated = apply_to_findings(findings, results, cfg)
    except Exception as exc:  # a model failure must not lose the study
        return done(False, {"error": "stability_failed"}, [f"stability failed ({type(exc).__name__})"])
    return done(True, {"results": {k: v.to_dict() for k, v in results.items()}, "findings": [f.model_dump() for f in updated]}, [])
