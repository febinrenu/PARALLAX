"""Name an image region by anatomy, using patient-side laterality.

Masks are labelled by the patient's side, not the image side. On a standard PA or AP
display the patient's right is on the image left, so a finding over the image-left lung
is a "right" finding. Pure numpy; no model needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MIN_OVERLAP_FRAC = 0.2  # share of the region that must sit on named anatomy


@dataclass
class Anatomy:
    """Boolean HxW masks on the original-image grid, labelled by the patient's side."""

    right_lung: np.ndarray
    left_lung: np.ndarray
    heart: np.ndarray


@dataclass
class ZoneResult:
    name: str | None
    side: str | None  # "right" | "left" | "cardiac" | None
    vertical: str | None  # "upper" | "middle" | "lower" | None


_NONE = ZoneResult(None, None, None)


def _centroid_x(mask: np.ndarray) -> float | None:
    cols = np.flatnonzero(mask.any(axis=0))
    if cols.size == 0:
        return None
    xs = np.nonzero(mask)[1]
    return float(xs.mean())


def laterality_ok(a: Anatomy) -> bool | None:
    """True if the heart sits on the image-right of the lung midline (standard orientation).

    Returns None when it cannot be judged. False suggests a flipped image or dextrocardia,
    in which case patient-side names should not be trusted.
    """
    hx = _centroid_x(a.heart)
    both = a.right_lung | a.left_lung
    mx = _centroid_x(both)
    if hx is None or mx is None:
        return None
    return hx > mx


def _vertical(lung: np.ndarray, region: np.ndarray) -> str | None:
    rows = np.flatnonzero(lung.any(axis=1))
    if rows.size == 0:
        return None
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    ys = np.nonzero(region & lung)[0]
    if ys.size == 0:
        ys = np.nonzero(region)[0]
    frac = (float(ys.mean()) + 0.5 - y0) / max(1, y1 - y0)
    if frac < 1 / 3:
        return "upper"
    if frac < 2 / 3:
        return "middle"
    return "lower"


def name_region(region: np.ndarray, a: Anatomy) -> ZoneResult:
    """Zone name for a boolean region mask, e.g. "right lower zone" or "cardiac region"."""
    region = np.asarray(region, dtype=bool)
    size = int(region.sum())
    if size == 0:
        return _NONE
    # Segmenter lung masks often include the retrocardiac lung, so a pixel inside the heart
    # counts as heart, not lung. Vertical thirds below still use each full lung extent.
    ov_r = int((region & a.right_lung & ~a.heart).sum())
    ov_l = int((region & a.left_lung & ~a.heart).sum())
    ov_h = int((region & a.heart).sum())
    best = max(ov_r, ov_l, ov_h)
    if best / size < MIN_OVERLAP_FRAC:
        return _NONE
    if ov_h > max(ov_r, ov_l):
        return ZoneResult("cardiac region", "cardiac", None)
    side, lung = ("right", a.right_lung) if ov_r >= ov_l else ("left", a.left_lung)
    vertical = _vertical(lung, region)
    if vertical is None:
        return _NONE
    return ZoneResult(f"{side} {vertical} zone", side, vertical)
