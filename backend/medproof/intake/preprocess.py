"""Model-input preprocessing shared by training notebooks and inference.

One function, one spec per model family, so a model sees identical tensors at train time
and at serve time. Specs are plain data and can be overridden from config.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from medproof.intake.decode import DecodedImage


@dataclass(frozen=True)
class PreprocSpec:
    name: str
    size: int  # square output side in pixels
    channels: int  # 1 or 3
    mean: tuple[float, ...]
    std: tuple[float, ...]
    # "stretch" resizes straight to size x size; "pad" keeps aspect with zero padding;
    # "center_crop" crops the central square first (what TorchXRayVision trained with).
    resize: str = "stretch"
    # Scale of the value range before mean/std: [0, 1] or a custom range like xrv's [-1024, 1024].
    value_range: tuple[float, float] = (0.0, 1.0)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SPECS: dict[str, PreprocSpec] = {
    # TorchXRayVision expects a single channel scaled to [-1024, 1024] at 224 px, no mean/std.
    "cxr_xrv": PreprocSpec(
        "cxr_xrv", 224, 1, (0.0,), (1.0,), resize="center_crop", value_range=(-1024.0, 1024.0)
    ),
    # Same crop and range as cxr_xrv at the 512 px the anatomy segmenter expects.
    "cxr_anatomy": PreprocSpec(
        "cxr_anatomy", 512, 1, (0.0,), (1.0,), resize="center_crop", value_range=(-1024.0, 1024.0)
    ),
    "brain_effnet": PreprocSpec("brain_effnet", 224, 3, IMAGENET_MEAN, IMAGENET_STD),
    "skin_cls": PreprocSpec("skin_cls", 224, 3, IMAGENET_MEAN, IMAGENET_STD),
    "bone_yolo": PreprocSpec("bone_yolo", 640, 3, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), resize="pad"),
}


def get_spec(name: str) -> PreprocSpec:
    try:
        return SPECS[name]
    except KeyError:
        raise KeyError(f"unknown preprocessing spec {name!r}; choose from {sorted(SPECS)}") from None


def center_crop_box(h: int, w: int) -> tuple[int, int, int]:
    """(y0, x0, side) of the central square crop, using the same rule as TorchXRayVision."""
    side = min(h, w)
    return h // 2 - side // 2, w // 2 - side // 2, side


def _resize(a: np.ndarray, size: int, mode: str) -> np.ndarray:
    if mode == "center_crop":
        y0, x0, side = center_crop_box(*a.shape[:2])
        a = a[y0 : y0 + side, x0 : x0 + side]
        mode = "stretch"
    h, w = a.shape[:2]
    interp = cv2.INTER_AREA if max(h, w) > size else cv2.INTER_LINEAR
    if mode == "stretch":
        return cv2.resize(a, (size, size), interpolation=interp)
    scale = size / max(h, w)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    r = cv2.resize(a, (nw, nh), interpolation=interp)
    out = np.zeros((size, size) + a.shape[2:], dtype=a.dtype)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    out[y0 : y0 + nh, x0 : x0 + nw] = r
    return out


def prepare(img: DecodedImage | np.ndarray, spec: PreprocSpec | str) -> np.ndarray:
    """Return a float32 CHW array ready for a model.

    Accepts a DecodedImage (uses its analysis copy) or a float32 [0, 1] array, HxW or HxWx3.
    """
    if isinstance(spec, str):
        spec = get_spec(spec)
    a = img.analysis if isinstance(img, DecodedImage) else img
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 3 and spec.channels == 1:
        a = a.mean(axis=2)
    if a.ndim == 2 and spec.channels == 3:
        a = np.repeat(a[..., None], 3, axis=2)
    a = _resize(a, spec.size, spec.resize)
    lo, hi = spec.value_range
    a = a * (hi - lo) + lo
    if a.ndim == 2:
        a = a[None, ...]
    else:
        a = np.transpose(a, (2, 0, 1))
    mean = np.asarray(spec.mean, dtype=np.float32).reshape(-1, 1, 1)
    std = np.asarray(spec.std, dtype=np.float32).reshape(-1, 1, 1)
    return ((a - mean) / std).astype(np.float32)
