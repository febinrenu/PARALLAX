"""Calibration for one sigmoid output (a chest X-ray label, or a detector's image-level fracture score).

Platt scaling p = sigmoid(a * logit(score) + b) corrects both sharpness and base rate, which temperature alone cannot:
the pretrained chest models output operating-point-scaled scores whose prevalence differs from the target population.
A false-negative-rate threshold (conformal risk control for one label) and an abstention band are fitted on a
separate split from the Platt parameters, so neither sees the data that tuned the other.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from medproof.calibrate import conformal, temperature


@dataclass
class BinaryCalibrator:
    label: str
    a: float
    b: float
    alpha: float  # target false-negative rate
    thr_fnr: float  # calibrated probability at or above which the label is reported so that at most alpha of true cases are missed
    abstain_low: float  # calibrated probability band around the decision point where the reader should abstain
    abstain_high: float
    input_kind: str = "probability"  # what `score` means: a probability in [0, 1]
    meta: dict = field(default_factory=dict)

    @staticmethod
    def _z(score: np.ndarray, kind: str = "probability") -> np.ndarray:
        return temperature.logit(score) if kind == "probability" else np.asarray(score, dtype=np.float64)

    @classmethod
    def fit(cls, label: str, score_a: np.ndarray, y_a: np.ndarray, score_b: np.ndarray, y_b: np.ndarray, alpha: float = 0.1, input_kind: str = "probability", abstain_halfwidth: float = 0.1) -> "BinaryCalibrator":
        """Platt on split A; FNR threshold on split B (needs about 1/alpha positives there to certify anything)."""
        a, b = temperature.fit_platt(cls._z(score_a, input_kind), y_a)
        p_b = temperature.sigmoid(a * cls._z(score_b, input_kind) + b)
        thr = conformal.fit_fnr_thresholds(p_b[:, None], np.asarray(y_b)[:, None], alpha)[0]
        prev = float(np.mean(y_b))
        centre = 0.5
        return cls(label, a, b, alpha, float(thr), centre - abstain_halfwidth, centre + abstain_halfwidth, input_kind, {"n_a": int(len(y_a)), "n_b": int(len(y_b)), "prevalence_b": prev, "positives_b": int(np.sum(y_b))})

    def prob(self, score: np.ndarray) -> np.ndarray:
        return temperature.sigmoid(self.a * self._z(score, self.input_kind) + self.b)

    def decide(self, score: np.ndarray) -> dict:
        p = self.prob(score)
        return {"prob_calibrated": p, "report": p >= self.thr_fnr, "abstain": (p >= self.abstain_low) & (p <= self.abstain_high)}

    def save(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "BinaryCalibrator":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))
