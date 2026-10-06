"""Common output shape for every modality reader (CXR, brain, skin, bone)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass
class ReaderFinding:
    label: str  # controlled vocabulary term for the modality
    logit: float
    prob_raw: float
    heatmap: np.ndarray | None = None  # float32 HxW in [0, 1], original-image grid
    mask: np.ndarray | None = None  # bool HxW
    boxes_xyxy: list[tuple[float, float, float, float]] = field(default_factory=list)
    box_scores: list[float] = field(default_factory=list)
    method: str = ""  # "gradcam++", "unet", "yolo", "medsam"
    region_name: str | None = None


@dataclass
class ReaderOutput:
    modality: str
    model_id: str  # "name@<sha8>"
    labels: list[str]  # full label order for `logits`
    logits: np.ndarray  # float32 (n_labels,)
    findings: list[ReaderFinding] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class Reader(Protocol):
    def predict(self, image: np.ndarray) -> ReaderOutput: ...
