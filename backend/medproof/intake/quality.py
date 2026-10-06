"""Image quality gate with a human-readable fix for every failure.

Checks run on a copy resized to a fixed long side so thresholds do not depend on the
upload's resolution. A failed check never blocks inference: it adds the `low_quality` flag
and a reason telling the user what to change. Thresholds live in `QualityConfig` and can be
re-fitted on real validation images.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from medproof.intake.decode import DecodedImage

NORM_SIDE = 512
SYMMETRY_SIDE = 128


@dataclass(frozen=True)
class QualityConfig:
    min_side_fail: int = 224  # model input size; below this we would upsample
    min_side_warn: int = 384
    # Laplacian variance divided by intensity variance (contrast-invariant sharpness).
    blur_warn: float = 0.060
    blur_fail: float = 0.020
    clip_warn: float = 0.02  # fraction of pixels at 0 or 255 (the larger side)
    clip_fail: float = 0.08
    min_range_fail: float = 0.45  # p1 to p99 spread of the display image
    dark_p99_fail: float = 0.55  # whole image sits in the dark half
    bright_p1_fail: float = 0.45  # whole image sits in the bright half
    noise_warn: float = 0.030  # Immerkaer sigma divided by p1-p99 range
    noise_fail: float = 0.060
    tilt_warn_deg: float = 10.0
    tilt_fail_deg: float = 20.0
    tilt_min_gain: float = 0.03  # symmetry must improve this much before we call it a tilt
    aspect_min: float = 0.6
    aspect_max: float = 1.7
    margin_warn: float = 0.35  # share of the frame that is flat padding


@dataclass
class QualityReason:
    code: str
    level: str  # "warn" | "fail"
    message: str
    fix: str

    def to_dict(self) -> dict:
        return {"code": self.code, "level": self.level, "message": self.message, "fix": self.fix}


@dataclass
class QualityReport:
    metrics: dict[str, float] = field(default_factory=dict)
    reasons: list[QualityReason] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not any(r.level == "fail" for r in self.reasons)

    @property
    def flags(self) -> list[str]:
        return [] if self.passed else ["low_quality"]

    def codes(self) -> set[str]:
        return {r.code for r in self.reasons}

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "flags": self.flags,
            "metrics": {k: round(float(v), 5) for k, v in self.metrics.items()},
            "reasons": [r.to_dict() for r in self.reasons],
        }


def _gray01(img: DecodedImage) -> np.ndarray:
    d = img.display.astype(np.float32) / 255.0
    return d.mean(axis=2) if d.ndim == 3 else d


def _norm_copy(g: np.ndarray) -> np.ndarray:
    h, w = g.shape
    s = NORM_SIDE / max(h, w)
    if abs(s - 1.0) < 1e-6:
        return g
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    return cv2.resize(g, (max(2, round(w * s)), max(2, round(h * s))), interpolation=interp)


def blur_score(g: np.ndarray) -> float:
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    return float(lap.var() / (g.var() + 1e-6))


def noise_sigma(g: np.ndarray) -> float:
    """Immerkaer (1996) fast single-image noise estimate."""
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    h, w = g.shape
    conv = cv2.filter2D(g, cv2.CV_32F, k)[1:-1, 1:-1]
    return float(np.sqrt(np.pi / 2.0) / (6.0 * (w - 2) * (h - 2)) * np.abs(conv).sum())


def _mirror_corr(g: np.ndarray, max_shift: int = 0) -> float:
    """Best correlation between g and its mirror, allowing the spine to sit off-centre."""
    flipped = g[:, ::-1]
    best = -2.0
    a = g - g.mean()
    for dx in range(-max_shift, max_shift + 1):
        b = np.roll(flipped, dx, axis=1)
        b = b - b.mean()
        c = float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))
        best = max(best, c)
    return best


def tilt_degrees(g: np.ndarray, max_deg: float = 30.0, step: float = 1.0) -> tuple[float, float]:
    """Rotation that maximises left-right mirror symmetry. Returns (angle, gain over zero).

    A frontal chest film is close to mirror-symmetric about the spine, so the rotation that
    best symmetrises it estimates how far the film is turned. The plan's lung-mask axis
    replaces this once the anatomy segmenter exists.
    """
    h, w = g.shape
    s = SYMMETRY_SIDE / max(h, w)
    small = cv2.resize(g, (max(2, round(w * s)), max(2, round(h * s))), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape
    # Mask a central disc so the rotated corners do not dominate the correlation.
    yy, xx = np.mgrid[0:sh, 0:sw]
    disc = ((yy - sh / 2) ** 2 + (xx - sw / 2) ** 2) <= (min(sh, sw) / 2) ** 2
    best_angle, best_corr, corr0 = 0.0, -2.0, 0.0
    for ang in np.arange(-max_deg, max_deg + 1e-6, step):
        m = cv2.getRotationMatrix2D((sw / 2.0, sh / 2.0), float(ang), 1.0)
        r = cv2.warpAffine(small, m, (sw, sh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        r = np.where(disc, r, float(small.mean()))
        c = _mirror_corr(r, max_shift=max(1, sw // 8))
        if abs(ang) < 1e-9:
            corr0 = c
        if c > best_corr:
            best_corr, best_angle = c, float(ang)
    return best_angle, best_corr - corr0


def margin_fraction(g: np.ndarray, flat_std: float = 0.01) -> float:
    """Share of rows/columns, counted in from each edge, that are flat padding."""

    def run(profile: np.ndarray) -> int:
        n = 0
        for v in profile:
            if v >= flat_std:
                break
            n += 1
        return n

    rows, cols = g.std(axis=1), g.std(axis=0)
    h, w = g.shape
    vert = (run(rows) + run(rows[::-1])) / h
    horiz = (run(cols) + run(cols[::-1])) / w
    return float(max(vert, horiz))


def assess(
    img: DecodedImage, modality: str | None = None, cfg: QualityConfig | None = None
) -> QualityReport:
    """Run every check. `modality="cxr"` also enables the rotation check."""
    cfg = cfg or QualityConfig()
    rep = QualityReport()
    h, w = img.shape
    g_full = _gray01(img)
    g = _norm_copy(g_full)

    side = min(h, w)
    rep.metrics["min_side"] = side
    if side < cfg.min_side_fail:
        rep.reasons.append(QualityReason(
            "low_resolution", "fail", f"Resolution is only {w}x{h}.",
            "Export or scan at the original resolution (at least 224 px on the short side)."))
    elif side < cfg.min_side_warn:
        rep.reasons.append(QualityReason(
            "low_resolution", "warn", f"Resolution is low ({w}x{h}).",
            "Use the original export if one exists; fine detail may be lost."))

    sharp = blur_score(g)
    rep.metrics["blur"] = sharp
    if sharp < cfg.blur_fail:
        rep.reasons.append(QualityReason(
            "blurred", "fail", "The image looks blurred.",
            "Check focus and patient motion, or re-acquire the image."))
    elif sharp < cfg.blur_warn:
        rep.reasons.append(QualityReason(
            "blurred", "warn", "The image is somewhat soft.",
            "Re-acquire if fine detail matters for this study."))

    p1, p99 = (float(v) for v in np.percentile(g, (1, 99)))
    spread = p99 - p1
    clip = float(max((img.display == 0).mean(), (img.display == 255).mean()))
    rep.metrics.update(p1=p1, p99=p99, range=spread, clip=clip)
    if p99 < cfg.dark_p99_fail:
        rep.reasons.append(QualityReason(
            "underexposed", "fail", "The image is underexposed (too dark).",
            "Re-acquire with a higher dose or longer exposure, or re-export with a wider window."))
    elif p1 > cfg.bright_p1_fail:
        rep.reasons.append(QualityReason(
            "overexposed", "fail", "The image is overexposed (washed out).",
            "Re-acquire with a lower dose, or re-export with a narrower window."))
    elif spread < cfg.min_range_fail:
        rep.reasons.append(QualityReason(
            "low_contrast", "fail", "The image has very little contrast.",
            "Re-export with a suitable window, or re-acquire."))
    if clip > cfg.clip_fail:
        rep.reasons.append(QualityReason(
            "clipped", "fail", f"{clip:.0%} of pixels are fully black or white.",
            "Re-acquire or re-export so highlights and shadows are not clipped."))
    elif clip > cfg.clip_warn:
        rep.reasons.append(QualityReason(
            "clipped", "warn", f"{clip:.0%} of pixels are fully black or white.",
            "Check the window settings; some detail may be lost."))

    sigma = noise_sigma(g)
    rel_noise = sigma / max(spread, 1e-3)
    rep.metrics["noise_rel"] = rel_noise
    if rel_noise > cfg.noise_fail:
        rep.reasons.append(QualityReason(
            "noisy", "fail", "The image is very noisy.",
            "This looks like a low-dose or high-gain acquisition; re-acquire if possible."))
    elif rel_noise > cfg.noise_warn:
        rep.reasons.append(QualityReason(
            "noisy", "warn", "The image is noisy.",
            "Findings near the noise floor may be unreliable."))

    aspect = w / h
    margin = margin_fraction(g)
    rep.metrics.update(aspect=aspect, margin=margin)
    if not (cfg.aspect_min <= aspect <= cfg.aspect_max) or margin > cfg.margin_warn:
        rep.reasons.append(QualityReason(
            "unusual_framing", "warn", "The framing is unusual (extreme crop or large flat margins).",
            "Upload the full image, not a screenshot or a tight crop."))

    if modality == "cxr":
        angle, gain = tilt_degrees(g)
        rep.metrics.update(tilt_deg=angle, tilt_gain=gain)
        if gain >= cfg.tilt_min_gain:
            if abs(angle) > cfg.tilt_fail_deg:
                rep.reasons.append(QualityReason(
                    "rotated", "fail", f"The image appears rotated by about {abs(angle):.0f} degrees.",
                    "Re-orient the image so the spine is vertical, then upload again."))
            elif abs(angle) > cfg.tilt_warn_deg:
                rep.reasons.append(QualityReason(
                    "rotated", "warn", f"The image appears tilted by about {abs(angle):.0f} degrees.",
                    "Straighten the image if possible."))
    return rep
