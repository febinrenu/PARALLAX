"""Selective prediction: risk-coverage curves and AURC.

Order cases by confidence, answer the most confident first, and track the error rate (risk) among the answered
cases as coverage grows. A confidence score is useful exactly when risk stays low until coverage is high.
"""

from __future__ import annotations

import numpy as np


def risk_coverage(conf: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(coverage, risk) after answering the k most confident cases, k = 1..n. Ties are broken by input order."""
    conf, correct = np.asarray(conf, dtype=np.float64), np.asarray(correct).astype(float)
    order = np.argsort(-conf, kind="stable")
    errs = 1.0 - correct[order]
    n = len(conf)
    k = np.arange(1, n + 1)
    return k / n, np.cumsum(errs) / k


def aurc(conf: np.ndarray, correct: np.ndarray) -> float:
    """Area under the risk-coverage curve (mean risk over all coverage levels). Lower is better."""
    return float(np.mean(risk_coverage(conf, correct)[1]))


def oracle_aurc(correct: np.ndarray) -> float:
    """AURC of a perfect ranking: every correct case before every wrong one."""
    correct = np.asarray(correct).astype(float)
    return aurc(correct + 0.0, correct)


def excess_aurc(conf: np.ndarray, correct: np.ndarray) -> float:
    """AURC minus the oracle: how much of the achievable improvement the confidence ranking leaves unused."""
    return aurc(conf, correct) - oracle_aurc(correct)


def coverage_at_risk(conf: np.ndarray, correct: np.ndarray, target_risk: float) -> float:
    """Largest coverage whose risk is at most target_risk (0.0 when even the most confident case is wrong)."""
    cov, risk = risk_coverage(conf, correct)
    ok = np.nonzero(risk <= target_risk + 1e-12)[0]
    return float(cov[ok[-1]]) if len(ok) else 0.0
