"""A fitted calibrator for one multi-class model: temperature, conformal threshold, tier rule. Saved as JSON next to the weights."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from medproof.calibrate import abstain, conformal, temperature


@dataclass
class CalibratedOutput:
    probs: np.ndarray  # (N, K) temperature-scaled
    conformal_sets: list[list[str]]  # class names, most likely first
    tiers: list[str]
    confidence: np.ndarray  # calibrated probability of the top class


@dataclass
class Calibrator:
    classes: list[str]
    temperature: float
    alpha: float
    qhat: float
    tier_rule: abstain.TierRule
    mondrian_qhat: list[float] | None = None
    method: str = "aps_randomized"
    meta: dict = field(default_factory=dict)

    @classmethod
    def fit(cls, classes: list[str], logits_val: np.ndarray, y_val: np.ndarray, logits_cal: np.ndarray, y_cal: np.ndarray, alpha: float = 0.1, mondrian: bool = False) -> "Calibrator":
        """Temperature on the validation split, conformal threshold and tiers on the calibration split (never the same data)."""
        T = temperature.fit_temperature(logits_val, y_val)
        p_cal = temperature.softmax(logits_cal, T)
        q = conformal.fit_aps(p_cal, y_cal, alpha)
        mq = conformal.fit_aps_mondrian(p_cal, y_cal, alpha, len(classes)) if mondrian else None
        rule = abstain.fit_tiers(p_cal.max(1), p_cal.argmax(1) == np.asarray(y_cal))
        return cls(list(classes), T, alpha, q, rule, mq, "aps_randomized", {"n_val": int(len(y_val)), "n_cal": int(len(y_cal))})

    @staticmethod
    def _u(logits: np.ndarray) -> np.ndarray:
        """Per-input uniform draw derived from the logits themselves: random-looking, but the same input always gets the same set."""
        return np.array([int.from_bytes(hashlib.sha256(np.ascontiguousarray(row, dtype=np.float32).tobytes()).digest()[:8], "big") / 2.0**64 for row in logits])

    def calibrate(self, logits: np.ndarray) -> CalibratedOutput:
        logits = np.atleast_2d(logits)
        p = temperature.softmax(logits, self.temperature)
        sets_idx = conformal.aps_sets(p, self.qhat, self._u(logits) if self.method == "aps_randomized" else None)
        conf = p.max(1)
        tiers = [abstain.tier(float(c), len(s), self.tier_rule) for c, s in zip(conf, sets_idx)]
        return CalibratedOutput(p, [[self.classes[i] for i in s] for s in sets_idx], tiers, conf)

    def to_dict(self) -> dict:
        return {"classes": self.classes, "temperature": self.temperature, "alpha": self.alpha, "qhat": self.qhat, "method": self.method,
                "coverage_target": 1 - self.alpha, "tier_rule": self.tier_rule.to_dict(), "mondrian_qhat": self.mondrian_qhat, "meta": self.meta}

    def save(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "Calibrator":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(d["classes"], d["temperature"], d["alpha"], d["qhat"], abstain.TierRule.from_dict(d["tier_rule"]), d.get("mondrian_qhat"), d.get("method", "aps_randomized"), d.get("meta", {}))
