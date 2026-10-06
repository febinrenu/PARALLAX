"""Box-prompted segmentation with MedSAM (ViT-B), cached by (image hash, box).

`MedSAM(predictor)` takes any callable (uint8 RGB image, box xyxy) -> float probability map on the
image grid, so the logic is testable without the model. `MedSAM.load()` builds the real predictor
from `flaviagiammarino/medsam-vit-base` through transformers (ungated, Apache-2.0, CPU is fine).

Boxes are xyxy in image pixels with an exclusive max edge, the same convention as the readers.
"""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from medproof.core.schemas import StageResult

MODEL_ID = os.environ.get("MEDPROOF_MEDSAM_MODEL", "flaviagiammarino/medsam-vit-base")
MASK_THRESHOLD = 0.5
FAILURE_JACCARD = 0.65  # ISIC 2018 Task 1 counts a lesion below this as a failure

Box = tuple[float, float, float, float]
Predictor = Callable[[np.ndarray, Box], np.ndarray]


@dataclass
class SegmentResult:
    mask: np.ndarray  # bool HxW on the input image grid
    box: Box  # the box actually used (clipped to the image)
    score: float  # mean predicted probability inside the mask, 0 if empty


def _to_rgb_u8(image: np.ndarray) -> np.ndarray:
    a = np.asarray(image)
    if a.dtype != np.uint8:
        a = np.rint(np.clip(a.astype(np.float32), 0.0, 1.0) * 255.0).astype(np.uint8)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError("image must be HxW or HxWx3")
    return a


def _clip_box(box, h: int, w: int) -> Box:
    b = np.asarray(box, np.float64)
    if b.shape != (4,) or not np.isfinite(b).all():
        raise ValueError("box must be four finite numbers (x0, y0, x1, y1)")
    x0, y0 = max(0.0, b[0]), max(0.0, b[1])
    x1, y1 = min(float(w), b[2]), min(float(h), b[3])
    if x1 - x0 < 1.0 or y1 - y0 < 1.0:
        raise ValueError("box is empty or outside the image")
    return (float(x0), float(y0), float(x1), float(y1))


class MedSAM:
    def __init__(self, predictor: Predictor, *, cache_dir: str | Path | None = None, model_id: str = MODEL_ID):
        self.predictor = predictor
        self.model_id = model_id
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._mem: dict[str, SegmentResult] = {}

    def _key(self, rgb: np.ndarray, box: Box) -> str:
        h = hashlib.sha256(rgb.tobytes())
        h.update(repr((rgb.shape, box, self.model_id)).encode())
        return h.hexdigest()

    def segment(self, image: np.ndarray, box) -> SegmentResult:
        rgb = _to_rgb_u8(image)
        b = _clip_box(box, rgb.shape[0], rgb.shape[1])
        key = self._key(rgb, b)
        if key in self._mem:
            return self._mem[key]
        path = self.cache_dir / f"{key}.npz" if self.cache_dir else None
        if path is not None and path.is_file():
            try:
                with np.load(path, allow_pickle=False) as f:
                    res = SegmentResult(f["mask"].astype(bool), b, float(f["score"]))
                self._mem[key] = res
                return res
            except (OSError, ValueError, KeyError):
                pass  # a damaged cache entry is a miss
        prob = np.asarray(self.predictor(rgb, b), np.float32)
        mask = prob >= MASK_THRESHOLD
        res = SegmentResult(mask, b, float(prob[mask].mean()) if mask.any() else 0.0)
        self._mem[key] = res
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, mask=mask, score=np.float32(res.score))
        return res

    @classmethod
    def load(cls, cache_dir: str | Path | None = None, model_id: str = MODEL_ID, threads: int = 1) -> MedSAM:
        """Build the real predictor (downloads the weights on first use)."""
        import torch
        from PIL import Image
        from transformers import SamModel, SamProcessor

        torch.set_num_threads(max(1, threads))
        model = SamModel.from_pretrained(model_id).eval()
        processor = SamProcessor.from_pretrained(model_id)

        def predict(rgb: np.ndarray, box: Box) -> np.ndarray:
            inputs = processor(Image.fromarray(rgb), input_boxes=[[list(box)]], return_tensors="pt")
            with torch.no_grad():
                out = model(**inputs, multimask_output=False)
            masks = processor.image_processor.post_process_masks(
                out.pred_masks.cpu(),
                inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu(), binarize=False,
            )
            return torch.sigmoid(masks[0][0, 0]).numpy().astype(np.float32)

        return cls(predict, cache_dir=cache_dir, model_id=model_id)


# ---- overlap metrics and helpers -----------------------------------------------------------
def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(np.logical_and(a, b).sum() / union)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    total = a.sum() + b.sum()
    return 1.0 if total == 0 else float(2 * np.logical_and(a, b).sum() / total)


def box_from_mask(mask: np.ndarray) -> Box | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def summarise(pairs: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    """Mean Jaccard and Dice of (ground truth, prediction) pairs and the share below the ISIC failure line."""
    if not pairs:
        return {"n": 0}
    j = np.array([jaccard(g, p) for g, p in pairs])
    d = np.array([dice(g, p) for g, p in pairs])
    return {"n": len(pairs), "mean_jaccard": float(j.mean()), "mean_dice": float(d.mean()),
            "failure_rate": float((j < FAILURE_JACCARD).mean())}


def run_box(segmenter, image: np.ndarray, box) -> StageResult:
    """Stage-style wrapper that never raises."""
    t0 = time.perf_counter()
    try:
        r = segmenter.segment(image, box)
    except Exception as exc:
        msg = str(exc) if isinstance(exc, ValueError) else f"segmentation unavailable ({type(exc).__name__})"
        return StageResult(stage="segment", ok=False, ms=int((time.perf_counter() - t0) * 1000), payload={"error": msg}, warnings=[msg])
    payload = {"box": list(r.box), "score": round(r.score, 5), "area_px": int(r.mask.sum()), "shape": list(r.mask.shape)}
    return StageResult(stage="segment", ok=True, ms=int((time.perf_counter() - t0) * 1000), payload=payload, warnings=[])
