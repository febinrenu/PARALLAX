"""Common output shape for every modality reader (CXR, brain, skin, bone).

Coordinates: every box, mask and heatmap is on the grid of the decoded display image
(`DecodedImage.display`), in pixels, x to the right and y downwards. Boxes are xyxy with
the max edge exclusive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass
class ReaderFinding:
    label: str  # controlled vocabulary term for the modality
    logit: float  # raw model output for this label (see ReaderOutput.output_kind)
    prob_raw: float
    heatmap: np.ndarray | None = None  # float32 HxW in [0, 1], original-image grid
    mask: np.ndarray | None = None  # bool HxW
    boxes_xyxy: list[tuple[float, float, float, float]] = field(default_factory=list)
    box_scores: list[float] = field(default_factory=list)
    method: str = ""  # "gradcam++", "unet", "yolo", "medsam"
    region_name: str | None = None  # patient-side, e.g. "right lower zone"
    flags: list[str] = field(default_factory=list)


@dataclass
class ReaderOutput:
    modality: str
    model_id: str  # "name@<sha8>"
    labels: list[str]  # full label order for `logits` and `probs`
    logits: np.ndarray  # float32 (n_labels,), raw model outputs
    findings: list[ReaderFinding] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    probs: np.ndarray | None = None  # float32 (n_labels,) in [0, 1]
    output_kind: str = "logit"  # "logit" or "probability": what `logits` holds
    image_hw: tuple[int, int] | None = None
    flags: list[str] = field(default_factory=list)  # study-level, e.g. "laterality_uncertain"


class Reader(Protocol):
    def predict(self, image: np.ndarray) -> ReaderOutput: ...
