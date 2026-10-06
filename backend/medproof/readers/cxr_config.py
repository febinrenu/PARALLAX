"""Settings for the chest X-ray reader. Model ids and weight names live here, not in code paths."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class CxrConfig:
    # TorchXRayVision weight set. "densenet121-res224-all" saw RSNA, so use "-chex" or
    # "-mimic_ch" when validating on RSNA (plan.md section 5.2).
    weights: str = field(default_factory=lambda: _env("MEDPROOF_CXR_WEIGHTS", "densenet121-res224-all"))
    cam_method: str = field(default_factory=lambda: _env("MEDPROOF_CXR_CAM", "gradcam++"))
    device: str = field(default_factory=lambda: _env("MEDPROOF_DEVICE", "cpu"))
    spec: str = "cxr_xrv"  # preprocessing spec in intake.preprocess

    positive_threshold: float = 0.5  # labels below this are not reported as findings
    tier_high: float = 0.8
    tier_moderate: float = 0.65  # between positive_threshold and this is "low"
    max_findings: int = 5  # one CAM per finding, so this bounds latency
    cam_peak_frac: float = 0.5  # region = connected area at or above this share of the CAM peak

    @property
    def model_name(self) -> str:
        """Short name used in evidence, e.g. "xrv-densenet121-all"."""
        return "xrv-" + self.weights.replace("-res224", "")

    def tier(self, prob: float) -> str:
        if prob >= self.tier_high:
            return "high"
        if prob >= self.tier_moderate:
            return "moderate"
        if prob >= self.positive_threshold:
            return "low"
        return "abstain"
