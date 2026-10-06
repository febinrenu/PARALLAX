"""Split conformal prediction: APS and RAPS sets, class-conditional (Mondrian) APS, and per-label FNR control.

Guarantee (exchangeable calibration and test data): P(true label in set) >= 1 - alpha. The deterministic APS used
here is conservative (it can over-cover a little) and never returns an empty set. Everything takes plain arrays.

References: Romano, Sesia, Candes 2020 (APS); Angelopoulos et al. 2021 (RAPS); Vovk 2003 (class-conditional).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import beta


def _quantile_level(n: int, alpha: float) -> float:
    return min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    if n == 0:
        return float("inf")
    level = _quantile_level(n, alpha)
    return float(np.quantile(scores, level, method="higher"))


# ----------------------------------------------------------------------------- APS


def _sorted(probs: np.ndarray):
    probs = np.asarray(probs, dtype=np.float64)
    order = np.argsort(-probs, axis=1, kind="stable")
    srt = np.take_along_axis(probs, order, axis=1)
    return order, srt, np.cumsum(srt, axis=1) - srt  # order, sorted probs, mass strictly before each rank


def aps_scores(probs: np.ndarray, y: np.ndarray, u: np.ndarray | None = None) -> np.ndarray:
    """APS score of the true class: mass of the classes ranked above it plus u times its own mass.

    u ~ Uniform(0, 1) gives the randomised APS, whose coverage is close to 1 - alpha on average. u = None uses u = 1
    (the deterministic, conservative variant), which over-covers badly when the model is confident, because wrong
    cases then all sit near score 1 and the quantile jumps over the gap.
    """
    probs = np.asarray(probs, dtype=np.float64)
    order, srt, before = _sorted(probs)
    pos = np.argmax(order == np.asarray(y)[:, None], axis=1)
    rows = np.arange(len(y))
    uu = 1.0 if u is None else np.asarray(u)
    score = before[rows, pos] + uu * srt[rows, pos]
    # The set always contains the top class (a reader must name something), so a true top class is covered whatever q is:
    # its conformity score is 0. Without this the randomised rule drops confident correct answers and the forced
    # top class then makes coverage overshoot the target by several points.
    return np.where(pos == 0, 0.0, score)


def fit_aps(probs: np.ndarray, y: np.ndarray, alpha: float = 0.1, randomized: bool = True, seed: int = 20261006) -> float:
    u = np.random.default_rng(seed).uniform(size=len(y)) if randomized else None
    return _conformal_quantile(aps_scores(probs, y, u), alpha)


def aps_sets(probs: np.ndarray, qhat: float, u: np.ndarray | None = None) -> list[list[int]]:
    """Class indices per row, most likely first: class c is kept when its APS score (mass above it + u * its own mass) is within qhat.

    This is {y : score(y) <= qhat}, so coverage is at least 1 - alpha. If nothing qualifies the top class is kept, so a
    reader always names something (that only adds coverage). Pass the same kind of u (random or None) used to fit qhat.
    """
    order, srt, before = _sorted(probs)
    uu = 1.0 if u is None else np.asarray(u)[:, None]
    keep = before + uu * srt <= qhat + 1e-12
    keep[:, 0] = True
    return [[int(c) for c in order[i][keep[i]]] for i in range(len(order))]


# ----------------------------------------------------------------------------- Mondrian (class-conditional)


def fit_aps_mondrian(probs: np.ndarray, y: np.ndarray, alpha: float, n_classes: int, randomized: bool = True, seed: int = 20261006) -> list[float]:
    """One threshold per true class, so coverage holds for each class, not just on average."""
    u = np.random.default_rng(seed).uniform(size=len(y)) if randomized else None
    s = aps_scores(probs, y, u)
    y = np.asarray(y)
    return [_conformal_quantile(s[y == c], alpha) for c in range(n_classes)]


def aps_sets_mondrian(probs: np.ndarray, qhats: list[float], u: np.ndarray | None = None) -> list[list[int]]:
    """Class c is in the set when its APS score is within its own class threshold."""
    order, srt, before = _sorted(probs)
    uu = 1.0 if u is None else np.asarray(u)[:, None]
    score_sorted = before + uu * srt  # score of the class at each rank
    q_sorted = np.asarray(qhats)[order]
    keep = score_sorted <= q_sorted + 1e-12
    out = []
    for i in range(len(order)):
        k = [int(c) for c in order[i][keep[i]]]
        out.append(k or [int(order[i][0])])
    return out


# ----------------------------------------------------------------------------- RAPS


def _raps_cum(p: np.ndarray, k_reg: int, lam: float) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-p, kind="stable")
    ranks = np.arange(1, len(p) + 1)
    reg = lam * np.maximum(ranks - k_reg, 0)
    return order, np.cumsum(p[order]) + reg


def raps_scores(probs: np.ndarray, y: np.ndarray, k_reg: int, lam: float) -> np.ndarray:
    out = np.empty(len(y))
    for i, (p, t) in enumerate(zip(np.asarray(probs, dtype=np.float64), np.asarray(y))):
        order, cum = _raps_cum(p, k_reg, lam)
        out[i] = cum[int(np.nonzero(order == t)[0][0])]
    return out


def fit_raps(probs: np.ndarray, y: np.ndarray, alpha: float = 0.1, k_reg: int = 2, lam: float = 0.01) -> float:
    return _conformal_quantile(raps_scores(probs, y, k_reg, lam), alpha)


def raps_sets(probs: np.ndarray, qhat: float, k_reg: int = 2, lam: float = 0.01) -> list[list[int]]:
    out = []
    for p in np.asarray(probs, dtype=np.float64):
        order, cum = _raps_cum(p, k_reg, lam)
        keep = cum <= qhat + 1e-12
        keep[0] = True
        out.append([int(c) for c in order[keep]])
    return out


# ----------------------------------------------------------------------------- evaluation helpers


def coverage(sets: list[list[int]], y: np.ndarray) -> float:
    y = np.asarray(y)
    return float(np.mean([int(t) in s for s, t in zip(sets, y)])) if len(y) else float("nan")


def mean_set_size(sets: list[list[int]]) -> float:
    return float(np.mean([len(s) for s in sets]))


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact binomial interval for an empirical coverage k / n."""
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


# ----------------------------------------------------------------------------- multi-label FNR control


def fit_fnr_thresholds(scores: np.ndarray, labels: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """Per-label threshold t_j so that P(score >= t_j | label j present) >= 1 - alpha, with a finite-sample correction.

    Choose t_j as the floor((n_pos + 1) * alpha)-th smallest positive score; calling a label present when
    score >= t_j then misses at most a fraction alpha of true positives. With too few positives to certify the
    rate (n_pos < 1/alpha - 1) the threshold is -inf, meaning "report everything" rather than a false guarantee.
    """
    scores, labels = np.asarray(scores, dtype=np.float64), np.asarray(labels)
    out = np.full(scores.shape[1], -np.inf)
    for j in range(scores.shape[1]):
        pos = np.sort(scores[labels[:, j] == 1, j])
        k = math.floor((len(pos) + 1) * alpha)
        if k >= 1:
            out[j] = pos[k - 1]
    return out
