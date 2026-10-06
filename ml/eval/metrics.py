"""Metric functions shared by the training notebooks and the eval harness (numpy + scikit-learn only).

Every function takes plain arrays so it can be dropped into `bootstrap.ci(..., stat_fn=...)`.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def softmax(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y) == np.asarray(pred)))


def per_class_recall(y: np.ndarray, pred: np.ndarray, n_classes: int) -> np.ndarray:
    y, pred = np.asarray(y), np.asarray(pred)
    out = np.full(n_classes, np.nan)
    for c in range(n_classes):
        m = y == c
        if m.any():
            out[c] = np.mean(pred[m] == c)
    return out


def balanced_accuracy(y: np.ndarray, pred: np.ndarray, n_classes: int) -> float:
    """Mean per-class recall over the classes present: the official ISIC 2018 Task 3 metric."""
    return float(np.nanmean(per_class_recall(y, pred, n_classes)))


def macro_f1(y: np.ndarray, pred: np.ndarray, n_classes: int) -> float:
    y, pred = np.asarray(y), np.asarray(pred)
    f = []
    for c in range(n_classes):
        tp = np.sum((pred == c) & (y == c))
        fp = np.sum((pred == c) & (y != c))
        fn = np.sum((pred != c) & (y == c))
        if tp + fp + fn == 0:
            continue
        f.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def nll(y: np.ndarray, probs: np.ndarray) -> float:
    p = np.clip(np.asarray(probs)[np.arange(len(y)), np.asarray(y)], 1e-12, 1)
    return float(-np.mean(np.log(p)))


def brier(y: np.ndarray, probs: np.ndarray) -> float:
    probs = np.asarray(probs)
    onehot = np.eye(probs.shape[1])[np.asarray(y)]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def ece(y: np.ndarray, probs: np.ndarray, bins: int = 15, adaptive: bool = False) -> float:
    """Top-label expected calibration error: equal-width bins, or equal-mass bins when adaptive."""
    probs = np.asarray(probs)
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == np.asarray(y)).astype(float)
    if adaptive:
        order = np.argsort(conf)
        parts = np.array_split(order, bins)
    else:
        edges = np.linspace(0, 1, bins + 1)
        which = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
        parts = [np.nonzero(which == b)[0] for b in range(bins)]
    n = len(conf)
    return float(sum(len(p) / n * abs(correct[p].mean() - conf[p].mean()) for p in parts if len(p)))


def auroc_binary(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def auroc_ovr_macro(y: np.ndarray, probs: np.ndarray) -> float:
    probs, y = np.asarray(probs), np.asarray(y)
    a = [auroc_binary(y == c, probs[:, c]) for c in range(probs.shape[1]) if (y == c).any() and (y != c).any()]
    return float(np.mean(a))


def threshold_at_specificity(y: np.ndarray, score: np.ndarray, spec: float = 0.9) -> float:
    """Smallest threshold t such that P(score >= t | negative) <= 1 - spec. Fit this on validation data only."""
    neg = np.sort(np.asarray(score)[np.asarray(y) == 0])
    k = int(np.floor(spec * len(neg)))
    if k >= len(neg):
        return float(neg[-1] + 1e-9)
    return float(np.nextafter(neg[k - 1], np.inf)) if k > 0 else float(neg[0])


def sensitivity_at_threshold(y: np.ndarray, score: np.ndarray, thr: float) -> float:
    pos = np.asarray(score)[np.asarray(y) == 1]
    return float(np.mean(pos >= thr)) if len(pos) else float("nan")


def specificity_at_threshold(y: np.ndarray, score: np.ndarray, thr: float) -> float:
    neg = np.asarray(score)[np.asarray(y) == 0]
    return float(np.mean(neg < thr)) if len(neg) else float("nan")


def dice(tp: float, fp: float, fn: float) -> float:
    d = 2 * tp + fp + fn
    return float(2 * tp / d) if d > 0 else 1.0  # empty prediction on an empty mask counts as perfect
