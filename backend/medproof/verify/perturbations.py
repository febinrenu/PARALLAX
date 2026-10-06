"""Eight seeded, realistic image perturbations with severities 1 to 5.

Shared by the quality-gate fixtures, the stability score (flip rate) and the corruption
benchmark, so all three describe "degraded" the same way.

Images are float32 arrays in [0, 1], HxW or HxWx3. Output has the same shape and dtype.
Every perturbation is deterministic for a given (name, severity, seed).
"""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np

SEVERITIES = (1, 2, 3, 4, 5)

# Parameters per severity (index 0 is severity 1).
_NOISE_SIGMA = (0.01, 0.02, 0.04, 0.07, 0.10)
_CONTRAST = (0.90, 0.80, 0.65, 0.50, 0.35)
_GAMMA = (1.15, 1.35, 1.65, 2.1, 2.7)
_JPEG_QUALITY = (60, 40, 25, 15, 8)
_ROTATE_DEG = (1.5, 3.0, 4.5, 6.0, 7.0)
_DOWNSAMPLE = (0.8, 0.6, 0.45, 0.33, 0.25)
_BLUR_SIGMA = (0.8, 1.4, 2.2, 3.2, 4.5)
_CROP_FRAC = (0.03, 0.06, 0.09, 0.12, 0.16)


def _check(severity: int) -> int:
    if severity not in SEVERITIES:
        raise ValueError(f"severity must be one of {SEVERITIES}, got {severity}")
    return severity - 1


def _rng(name: str, severity: int, seed: int) -> np.random.Generator:
    key = sum(ord(c) * (i + 1) for i, c in enumerate(name))
    return np.random.default_rng([seed, key, severity])


def noise(img: np.ndarray, severity: int, seed: int = 0) -> np.ndarray:
    s = _NOISE_SIGMA[_check(severity)]
    g = _rng("noise", severity, seed).normal(0.0, s, img.shape).astype(np.float32)
    return np.clip(img + g, 0.0, 1.0).astype(np.float32)


def contrast(img: np.ndarray, severity: int, seed: int = 0) -> np.ndarray:
    f = _CONTRAST[_check(severity)]
    mean = float(img.mean())
    return np.clip((img - mean) * f + mean, 0.0, 1.0).astype(np.float32)


def gamma(img: np.ndarray, severity: int, seed: int = 0) -> np.ndarray:
    g = _GAMMA[_check(severity)]
    return np.power(np.clip(img, 0.0, 1.0), g).astype(np.float32)


def jpeg(img: np.ndarray, severity: int, seed: int = 0) -> np.ndarray:
    q = _JPEG_QUALITY[_check(severity)]
    u8 = np.rint(np.clip(img, 0, 1) * 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", u8, [cv2.IMWRITE_JPEG_QUALITY, q])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    flag = cv2.IMREAD_GRAYSCALE if img.ndim == 2 else cv2.IMREAD_COLOR
    out = cv2.imdecode(buf, flag)
    return (out.astype(np.float32) / 255.0).astype(np.float32)


def rotate(img: np.ndarray, severity: int, seed: int = 0) -> np.ndarray:
    deg = _ROTATE_DEG[_check(severity)]
    sign = 1.0 if _rng("rotate", severity, seed).random() < 0.5 else -1.0
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), sign * deg, 1.0)
    out = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return out.astype(np.float32)


def downsample(img: np.ndarray, severity: int, seed: int = 0) -> np.ndarray:
    f = _DOWNSAMPLE[_check(severity)]
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(2, int(w * f)), max(2, int(h * f))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def blur(img: np.ndarray, severity: int, seed: int = 0) -> np.ndarray:
    s = _BLUR_SIGMA[_check(severity)]
    return cv2.GaussianBlur(img, (0, 0), s).astype(np.float32)


def crop(img: np.ndarray, severity: int, seed: int = 0) -> np.ndarray:
    """Crop a margin off each side, then resize back to the original shape."""
    frac = _CROP_FRAC[_check(severity)]
    h, w = img.shape[:2]
    dy, dx = max(1, int(h * frac)), max(1, int(w * frac))
    out = cv2.resize(img[dy : h - dy, dx : w - dx], (w, h), interpolation=cv2.INTER_LINEAR)
    return out.astype(np.float32)


PERTURBATIONS: dict[str, Callable[[np.ndarray, int, int], np.ndarray]] = {
    "noise": noise,
    "contrast": contrast,
    "gamma": gamma,
    "jpeg": jpeg,
    "rotate": rotate,
    "downsample": downsample,
    "blur": blur,
    "crop": crop,
}
NAMES = tuple(PERTURBATIONS)


def apply(name: str, img: np.ndarray, severity: int = 3, seed: int = 0) -> np.ndarray:
    """Apply one named perturbation. Raises KeyError for an unknown name."""
    if name not in PERTURBATIONS:
        raise KeyError(f"unknown perturbation {name!r}; choose from {NAMES}")
    if img.dtype != np.float32:
        img = img.astype(np.float32)
    out = PERTURBATIONS[name](img, severity, seed)
    assert out.shape == img.shape
    return out
