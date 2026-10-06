"""Confidence tiers and the abstention band.

Tiers come from calibrated confidence, not raw softmax. Thresholds are fitted on the calibration split so that
the "high" tier reaches a target accuracy and "moderate" a lower one. A model that cannot reach a target
(the skin classifier never reaches 95% accuracy at any usable coverage) gets `None`, and that tier is simply never used.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

TIERS = ("high", "moderate", "low", "abstain")


@dataclass
class TierRule:
    t_high: float | None
    t_moderate: float | None
    t_low: float = 0.5  # below this calibrated confidence the reader abstains
    max_set_size: int = 2  # a conformal set wider than this also means abstain

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TierRule":
        return cls(**d)


def _threshold_for_accuracy(conf: np.ndarray, correct: np.ndarray, target: float, min_count: int) -> float | None:
    """Smallest confidence t such that accuracy among {conf >= t} is at least target, with at least min_count items."""
    order = np.argsort(-conf)
    c, ok = conf[order], correct[order].astype(float)
    acc = np.cumsum(ok) / np.arange(1, len(c) + 1)
    valid = np.nonzero((acc >= target) & (np.arange(1, len(c) + 1) >= min_count))[0]
    if len(valid) == 0:
        return None
    return float(c[valid[-1]])  # the deepest point that still meets the target


def fit_tiers(conf: np.ndarray, correct: np.ndarray, acc_high: float = 0.95, acc_moderate: float = 0.85, min_count: int = 50, t_low: float = 0.5, max_set_size: int = 2) -> TierRule:
    conf, correct = np.asarray(conf, dtype=np.float64), np.asarray(correct).astype(bool)
    t_high = _threshold_for_accuracy(conf, correct, acc_high, min_count)
    t_mod = _threshold_for_accuracy(conf, correct, acc_moderate, min_count)
    if t_high is not None and t_mod is not None and t_mod > t_high:
        t_mod = t_high
    return TierRule(t_high=t_high, t_moderate=t_mod, t_low=t_low, max_set_size=max_set_size)


def tier(conf: float, set_size: int, rule: TierRule) -> str:
    if set_size > rule.max_set_size or conf < rule.t_low:
        return "abstain"
    if rule.t_high is not None and conf >= rule.t_high and set_size == 1:
        return "high"
    if rule.t_moderate is not None and conf >= rule.t_moderate:
        return "moderate"
    return "low"
