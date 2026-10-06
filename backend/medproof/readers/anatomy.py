"""Chest anatomy masks (lungs, heart) from the TorchXRayVision ChestX-Det PSPNet.

The segmenter sees the same central square as the classifier, and its masks are pasted back
onto the full original grid, so they line up with CAM regions and boxes. Masks are labelled by
the patient's side: PSPNet's "Left Lung" is the patient's left lung (verified on a real image,
see progress.md Verified facts).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from medproof.intake.decode import DecodedImage
from medproof.intake.preprocess import center_crop_box, prepare
from medproof.readers.cam import resize_cam
from medproof.readers.zones import Anatomy

SPEC = "cxr_anatomy"
MASK_THRESHOLD = 0.5  # on sigmoid(logit)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class AnatomySegmenter:
    def __init__(self, model: Any, device: str = "cpu", threshold: float = MASK_THRESHOLD):
        self.model = model.eval()
        self.device = device
        self.threshold = threshold
        targets = list(model.targets)
        self._idx = {n: targets.index(n) for n in ("Left Lung", "Right Lung", "Heart")}

    def __call__(self, img: DecodedImage | np.ndarray) -> Anatomy:
        import torch

        hw = img.shape if isinstance(img, DecodedImage) else tuple(np.asarray(img).shape[:2])
        h, w = int(hw[0]), int(hw[1])
        x = torch.from_numpy(prepare(img, SPEC))[None].to(self.device)
        with torch.no_grad():
            logits = self.model(x)[0].cpu().numpy()  # (14, 512, 512) raw logits
        y0, x0, side = center_crop_box(h, w)

        def mask(name: str) -> np.ndarray:
            prob = _sigmoid(logits[self._idx[name]])
            full = np.zeros((h, w), bool)
            full[y0 : y0 + side, x0 : x0 + side] = resize_cam(prob.astype(np.float32), (side, side)) >= self.threshold
            return full

        return Anatomy(right_lung=mask("Right Lung"), left_lung=mask("Left Lung"), heart=mask("Heart"))

    @classmethod
    def try_load(cls, device: str = "cpu") -> AnatomySegmenter | None:
        """Load the pretrained PSPNet, or return None if the package or weights are unavailable."""
        try:
            import torchxrayvision as xrv

            model = xrv.baseline_models.chestx_det.PSPNet()
            model.to(device)
            return cls(model, device)
        except Exception:  # missing package, offline first run, or a corrupt download
            return None
