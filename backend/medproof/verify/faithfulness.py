"""Faithfulness of a heatmap: does the model really depend on the region it highlights?

Deletion test: blur the top k% of the heatmap, re-score, and record the probability drop.
A heatmap passes only if its drop beats random placements of a region with the same shape and
area (permutation test), so a random region passes at most `alpha` of the time by construction.
The insertion curve (reveal the most salient pixels first, from a blurred image) is reported as
supporting evidence and does not decide the pass.

Findings that carry only a box (the bone detector) get the same test on the box region: the box is
blurred and the drop is compared with equally sized boxes placed elsewhere (`assess_box`).

`score` is any callable image -> probabilities over all labels; the CXR reader's `.score` fits.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import cv2
import numpy as np

from medproof.core.schemas import Finding, StageResult

Score = Callable[[np.ndarray], np.ndarray]
FRACTIONS = (0.05, 0.10, 0.20)
DECISION_FRACTION = 0.10
BASELINES = ("blur", "median")


@dataclass(frozen=True)
class FaithfulnessConfig:
    fractions: tuple = FRACTIONS
    decision_fraction: float = DECISION_FRACTION
    n_random: int = 16  # random placements for the permutation test (resolution 1 / (n + 1))
    alpha: float = 0.10  # random regions may pass at most this often
    min_drop: float = 0.05  # absolute probability drop needed on top of beating chance
    insertion_steps: int = 8
    # What replaces deleted pixels: "blur" (plan) keeps overall shape, so it cannot remove silhouette
    # evidence such as heart size; "median" fills with the image median and does.
    baseline: str = "blur"

    def __post_init__(self):
        # with n placements the smallest possible p is 1 / (n + 1); below alpha nothing could ever pass
        if self.baseline not in BASELINES:
            raise ValueError(f"baseline must be one of {BASELINES}")
        if self.n_random and 1.0 / (self.n_random + 1) > self.alpha:
            raise ValueError(f"n_random={self.n_random} cannot reach alpha={self.alpha}; use at least {int(np.ceil(1 / self.alpha)) - 1}")


@dataclass
class FaithfulnessResult:
    faithful: bool | None
    drop: float | None  # probability drop at the decision fraction
    drops: dict = field(default_factory=dict)  # fraction -> drop
    p_random: float | None = None  # share of random placements with drop >= the heatmap's
    insertion_auc: float | None = None
    reason: str = ""

    def to_dict(self) -> dict:
        r = lambda v: None if v is None else round(float(v), 5)  # noqa: E731
        return {
            "faithful": self.faithful, "drop": r(self.drop), "drops": {str(k): r(v) for k, v in self.drops.items()},
            "p_random": r(self.p_random), "insertion_auc": r(self.insertion_auc), "reason": self.reason,
        }


# ---- image operations ----------------------------------------------------------------------
def _top_mask(heat: np.ndarray, frac: float) -> np.ndarray:
    k = max(1, int(round(frac * heat.size)))
    thresh = np.partition(heat.ravel(), heat.size - k)[heat.size - k]
    mask = heat >= thresh
    if mask.sum() > k:  # ties: keep exactly k pixels in a stable order
        idx = np.flatnonzero(mask.ravel())
        order = np.argsort(-heat.ravel()[idx], kind="stable")[:k]
        mask = np.zeros(heat.size, bool)
        mask[idx[order]] = True
        mask = mask.reshape(heat.shape)
    return mask


def _replacement(img: np.ndarray, baseline: str) -> np.ndarray:
    if baseline == "median":
        return np.full_like(img, float(np.median(img)))
    return _blurred(img)


def _blurred(img: np.ndarray) -> np.ndarray:
    sigma = max(img.shape[:2]) / 16.0
    return cv2.GaussianBlur(img, (0, 0), sigma)


def blur_top_region(img: np.ndarray, heat: np.ndarray, frac: float, baseline: str = "blur") -> np.ndarray:
    """Copy of `img` with the top `frac` of `heat` replaced by a heavy blur (or the image median)."""
    if not 0.0 < frac <= 1.0:
        raise ValueError("frac must be in (0, 1]")
    if heat.shape != img.shape[:2]:
        raise ValueError(f"heatmap {heat.shape} does not match image {img.shape[:2]}")
    mask = _top_mask(heat, frac)
    return _apply(img, _replacement(img, baseline), mask)


def _apply(img: np.ndarray, replacement: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = img.copy()
    out[mask] = replacement[mask]
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def shifted_region(heat: np.ndarray, frac: float, rng: np.random.Generator, max_iou: float = 0.25) -> np.ndarray:
    """The top-`frac` mask moved to a random place (wrapped), keeping its exact shape and area.

    Placements that overlap the real region by more than `max_iou` are redrawn, so the baseline is
    "a region of the same size elsewhere", not a near-copy of the evidence.
    """
    mask = _top_mask(heat, frac)
    moved = mask
    for _ in range(50):
        dy, dx = int(rng.integers(0, heat.shape[0])), int(rng.integers(0, heat.shape[1]))
        moved = np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
        inter = np.logical_and(moved, mask).sum()
        if inter / max(1, np.logical_or(moved, mask).sum()) <= max_iou:
            break
    return moved


# ---- tests ---------------------------------------------------------------------------------
def deletion_drop(score: Score, img: np.ndarray, heat: np.ndarray, label_idx: int, frac: float, baseline: str = "blur") -> float:
    base = float(score(img)[label_idx])
    return base - float(score(blur_top_region(img, heat, frac, baseline))[label_idx])


def insertion_auc(score: Score, img: np.ndarray, heat: np.ndarray, label_idx: int, steps: int = 8, baseline: str = "blur") -> float:
    """Area under p(label) as the most salient pixels are revealed in `steps` equal increments."""
    blurred = _replacement(img, baseline)
    order = np.argsort(-heat.ravel(), kind="stable")
    ps = []
    for i in range(steps + 1):
        n = int(round(i / steps * heat.size))
        mask = np.zeros(heat.size, bool)
        mask[order[:n]] = True
        ps.append(float(score(_reveal(img, blurred, mask.reshape(heat.shape)))[label_idx]))
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return float(trapezoid(ps, dx=1.0 / steps))


def _reveal(img: np.ndarray, blurred: np.ndarray, revealed: np.ndarray) -> np.ndarray:
    out = blurred.copy()
    out[revealed] = img[revealed]
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def assess_heatmap(
    score: Score, img: np.ndarray, heat: np.ndarray | None, label_idx: int, cfg: FaithfulnessConfig | None = None, seed: int = 0
) -> FaithfulnessResult:
    cfg = cfg or FaithfulnessConfig()
    if heat is None:
        return FaithfulnessResult(None, None, reason="no heatmap")
    if float(heat.max() - heat.min()) <= 1e-6:
        return FaithfulnessResult(None, None, reason="flat heatmap: no region to test")
    drops = {f: deletion_drop(score, img, heat, label_idx, f, cfg.baseline) for f in cfg.fractions}
    k = cfg.decision_fraction
    base = float(score(img)[label_idx])
    rng = np.random.default_rng(seed)
    blurred = _replacement(img, cfg.baseline)
    random_drops = []
    for _ in range(cfg.n_random):
        mask = shifted_region(heat, k, rng)
        random_drops.append(base - float(score(_apply(img, blurred, mask))[label_idx]))
    drop = drops[k]
    p = float((np.sum(np.asarray(random_drops) >= drop) + 1) / (cfg.n_random + 1)) if cfg.n_random else 1.0
    faithful = bool(p <= cfg.alpha and drop >= cfg.min_drop)
    auc = insertion_auc(score, img, heat, label_idx, cfg.insertion_steps, cfg.baseline)
    why = "" if faithful else ("drop no larger than random regions" if p > cfg.alpha else f"drop below {cfg.min_drop}")
    return FaithfulnessResult(faithful, drop, drops, p, auc, why)


# ---- box-only findings -------------------------------------------------------------------
MAX_BOX_AREA_FRAC = 0.35  # above this a same-sized box cannot be placed away from the real one


def box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def random_box_like(box, hw, rng: np.random.Generator, max_iou: float = 0.25) -> tuple:
    """A box with the same width and height as `box`, placed at random inside the image and off the real box."""
    h, w = hw
    bw, bh = box[2] - box[0], box[3] - box[1]
    best = None
    for _ in range(100):
        x0 = float(rng.uniform(0, max(0.0, w - bw)))
        y0 = float(rng.uniform(0, max(0.0, h - bh)))
        cand = (x0, y0, x0 + bw, y0 + bh)
        iou = box_iou(cand, box)
        if best is None or iou < best[0]:
            best = (iou, cand)
        if iou <= max_iou:
            break
    return best[1]


def _box_mask(box, hw) -> np.ndarray:
    h, w = hw
    x0, y0 = max(0, int(round(box[0]))), max(0, int(round(box[1])))
    x1, y1 = min(w, int(round(box[2]))), min(h, int(round(box[3])))
    mask = np.zeros((h, w), bool)
    mask[y0:y1, x0:x1] = True
    return mask


def assess_box(score: Score, img: np.ndarray, box, label_idx: int, cfg: FaithfulnessConfig | None = None, seed: int = 0) -> FaithfulnessResult:
    """Deletion test on a box: blur it, compare the probability drop with equally sized boxes elsewhere."""
    cfg = cfg or FaithfulnessConfig()
    if box is None:
        return FaithfulnessResult(None, None, reason="no box")
    hw = img.shape[:2]
    mask = _box_mask(box, hw)
    area = int(mask.sum())
    if area == 0 or not (box[2] > box[0] and box[3] > box[1]):
        return FaithfulnessResult(None, None, reason="empty or degenerate box")
    if area / mask.size > MAX_BOX_AREA_FRAC:
        return FaithfulnessResult(None, None, reason="box too large to compare with other regions")
    base = float(score(img)[label_idx])
    blurred = _replacement(img, cfg.baseline)
    drop = base - float(score(_apply(img, blurred, mask))[label_idx])
    rng = np.random.default_rng(seed)
    random_drops = []
    for _ in range(cfg.n_random):
        rb = random_box_like(box, hw, rng)
        random_drops.append(base - float(score(_apply(img, blurred, _box_mask(rb, hw)))[label_idx]))
    p = float((np.sum(np.asarray(random_drops) >= drop) + 1) / (cfg.n_random + 1)) if cfg.n_random else 1.0
    faithful = bool(p <= cfg.alpha and drop >= cfg.min_drop)
    why = "" if faithful else ("drop no larger than random boxes" if p > cfg.alpha else f"drop below {cfg.min_drop}")
    return FaithfulnessResult(faithful, drop, {"box": drop}, p, None, why)


# ---- reader integration ------------------------------------------------------------------
def assess_output(reader, img, out, cfg: FaithfulnessConfig | None = None, seed: int = 0) -> dict[str, FaithfulnessResult]:
    """Assess every finding of a ReaderOutput. `img` must be the image the reader saw."""
    arr = img.analysis if hasattr(img, "analysis") else np.asarray(img, np.float32)
    results = {}
    for i, rf in enumerate(out.findings):
        idx = out.labels.index(rf.label)
        if rf.heatmap is None and rf.boxes_xyxy:  # box-only finding (bone detector): test the box region
            results[rf.label] = assess_box(reader.score, arr, rf.boxes_xyxy[0], idx, cfg, seed + i)
        else:
            results[rf.label] = assess_heatmap(reader.score, arr, rf.heatmap, idx, cfg, seed + i)
    return results


def apply_to_findings(findings: list[Finding], results: dict[str, FaithfulnessResult]) -> list[Finding]:
    """Copies of `findings` with faithfulness filled into the first image evidence. Inputs stay untouched."""
    updated = []
    for f in findings:
        res = results.get(f.label)
        if res is None or not f.image_evidence:
            updated.append(f)
            continue
        faithful = res.faithful
        if faithful and "heatmap_off_target" in f.flags:
            faithful = False  # causal for the model, but not on the segmented lesion: it cannot prove the finding
        ev = f.image_evidence[0].model_copy(update={"faithful": faithful, "faithfulness_drop": res.drop})
        flags = list(f.flags)
        if faithful is False and "unfaithful" not in flags:
            flags.append("unfaithful")
        updated.append(f.model_copy(update={"image_evidence": [ev, *f.image_evidence[1:]], "flags": flags}))
    return updated


def pass_rate_report(model_pass: list[bool], random_pass: list[bool]) -> dict:
    """Pass rates of real heatmaps against random-region baselines (plan section 9.3)."""
    def rate(xs):
        return None if not xs else float(np.mean(xs))

    return {"model_pass_rate": rate(model_pass), "random_pass_rate": rate(random_pass),
            "n_model": len(model_pass), "n_random": len(random_pass)}


def run(ctx, reader=None, cfg: FaithfulnessConfig | None = None, seed: int = 0) -> StageResult:
    """Stage: `ctx` needs `.decoded`, `.reader_output` and `.findings`. Never raises."""
    t0 = time.perf_counter()

    def done(ok, payload, warnings):
        return StageResult(stage="faithfulness", ok=ok, ms=int((time.perf_counter() - t0) * 1000), payload=payload, warnings=warnings)

    img, out, findings = (getattr(ctx, n, None) for n in ("decoded", "reader_output", "findings"))
    if img is None or out is None or findings is None:
        return done(False, {"error": "missing decoded image, reader output or findings"}, ["faithfulness skipped: missing inputs"])
    try:
        results = assess_output(reader, img, out, cfg, seed)
        updated = apply_to_findings(findings, results)
    except Exception as exc:  # a model failure must not lose the study
        return done(False, {"error": "faithfulness_failed"}, [f"faithfulness failed ({type(exc).__name__})"])
    warnings = [f"{k}: not assessable ({v.reason})" for k, v in results.items() if v.faithful is None]
    payload = {"results": {k: v.to_dict() for k, v in results.items()}, "findings": [f.model_dump() for f in updated]}
    return done(True, payload, warnings)
