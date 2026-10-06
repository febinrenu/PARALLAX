"""Single-class detection metrics computed from saved detections, so `make eval` never needs the detector.

Boxes are normalised xyxy in [0, 1]. A detection row is (x1, y1, x2, y2, confidence).
AP uses all-point interpolation (the VOC2010+/Ultralytics convention) at a fixed IoU threshold.
"""

from __future__ import annotations

import numpy as np


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def average_precision(dets, gts, iou_thr: float = 0.5) -> float:
    """AP over a set of images. `dets[i]` is (n, 5), `gts[i]` is (m, 4). NaN when there are no ground-truth boxes."""
    n_gt = int(sum(len(np.asarray(g).reshape(-1, 4)) for g in gts))
    if n_gt == 0:
        return float("nan")
    rows = [(float(d[4]), i, d[:4]) for i, dd in enumerate(dets) for d in np.asarray(dd, float).reshape(-1, 5)]
    rows.sort(key=lambda r: -r[0])
    taken = [np.zeros(len(np.asarray(g).reshape(-1, 4)), bool) for g in gts]
    tp = np.zeros(len(rows))
    for k, (_, i, box) in enumerate(rows):
        g = np.asarray(gts[i], float).reshape(-1, 4)
        if len(g) == 0:
            continue
        ious = iou_matrix(box[None], g)[0]
        ious = np.where(taken[i], -1.0, ious)  # each GT box can be claimed once
        j = int(np.argmax(ious))
        if ious[j] >= iou_thr:
            tp[k] = 1
            taken[i][j] = True
    if len(rows) == 0:
        return 0.0
    ctp = np.cumsum(tp)
    recall = ctp / n_gt
    precision = ctp / np.arange(1, len(rows) + 1)
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[1.0], precision, [0.0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    idx = np.nonzero(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def image_scores(dets) -> np.ndarray:
    """Image-level fracture score = highest detection confidence (0 when nothing was detected)."""
    return np.array([float(np.max(np.asarray(d, float).reshape(-1, 5)[:, 4])) if len(np.asarray(d).reshape(-1, 5)) else 0.0 for d in dets])
