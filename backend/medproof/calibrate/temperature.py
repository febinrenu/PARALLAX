"""Temperature scaling (Guo et al. 2017) and Platt scaling.

Temperature scaling divides the logits by one scalar T fitted by minimising negative log-likelihood on a
held-out split. It never changes the predicted class, so accuracy is untouched and only confidence moves.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, minimize_scalar


def softmax(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / T
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def nll(probs: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(np.asarray(probs)[np.arange(len(y)), np.asarray(y)], 1e-12, 1.0)
    return float(-np.mean(np.log(p)))


def fit_temperature(logits: np.ndarray, y: np.ndarray, bounds: tuple[float, float] = (0.05, 20.0)) -> float:
    """Single temperature for a multi-class model. Optimises log T, so T stays positive."""
    logits, y = np.asarray(logits, dtype=np.float64), np.asarray(y)

    def loss(logT: float) -> float:
        z = logits / np.exp(logT)
        z = z - z.max(axis=1, keepdims=True)
        lse = np.log(np.exp(z).sum(axis=1))
        return float(np.mean(lse - z[np.arange(len(y)), y]))

    r = minimize_scalar(loss, bounds=(np.log(bounds[0]), np.log(bounds[1])), method="bounded", options={"xatol": 1e-6})
    return float(np.exp(r.x))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return np.log(p / (1 - p))


def fit_temperature_binary(z: np.ndarray, y: np.ndarray, bounds: tuple[float, float] = (0.05, 50.0)) -> float:
    """Temperature for a single sigmoid output: p = sigmoid(z / T)."""
    z, y = np.asarray(z, dtype=np.float64), np.asarray(y, dtype=np.float64)

    def loss(logT: float) -> float:
        s = z / np.exp(logT)
        return float(np.mean(np.logaddexp(0, s) - y * s))

    r = minimize_scalar(loss, bounds=(np.log(bounds[0]), np.log(bounds[1])), method="bounded", options={"xatol": 1e-6})
    return float(np.exp(r.x))


def fit_platt(score: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Platt scaling p = sigmoid(a * score + b). Use when a score has no logit meaning (e.g. a detector's max box confidence)."""
    s, y = np.asarray(score, dtype=np.float64), np.asarray(y, dtype=np.float64)

    def loss(ab: np.ndarray) -> float:
        z = ab[0] * s + ab[1]
        return float(np.mean(np.logaddexp(0, z) - y * z) + 1e-9 * (ab[0] ** 2))

    r = minimize(loss, x0=np.array([1.0, 0.0]), method="L-BFGS-B")
    return float(r.x[0]), float(r.x[1])
