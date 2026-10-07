from __future__ import annotations

import numpy as np
import pytest

from ml.eval import bootstrap as bs
from ml.eval import metrics as mt


def _mean(x):
    return float(np.mean(x))


def test_percentile_ci_matches_normal_theory_for_iid_mean():
    rng = np.random.default_rng(0)
    x = rng.normal(5, 2, 400)
    c = bs.ci({"x": x}, _mean, B=1500)
    half = 1.96 * x.std(ddof=1) / np.sqrt(len(x))
    assert abs((c["hi"] - c["lo"]) / 2 - half) < 0.25 * half
    assert c["lo"] < 5 < c["hi"] and c["method"] == "percentile" and c["n_groups"] == 400


def test_cluster_bootstrap_is_wider_than_iid_when_groups_are_correlated():
    rng = np.random.default_rng(1)
    g = np.repeat(np.arange(40), 10)
    x = rng.normal(0, 1, 40)[g] + rng.normal(0, 0.1, 400)  # almost all variance is between groups
    iid = bs.ci({"x": x}, _mean, B=800)
    clu = bs.ci({"x": x}, _mean, groups=g, B=800)
    assert (clu["hi"] - clu["lo"]) > 2.5 * (iid["hi"] - iid["lo"])
    assert clu["n_groups"] == 40


def test_stratified_resampling_keeps_class_counts_fixed():
    y = np.array([0] * 90 + [1] * 8 + [2] * 2)
    seen = set()

    def stat(y):
        seen.add(tuple(np.bincount(y, minlength=3)))
        return 0.0

    bs.ci({"y": y}, stat, strata=y, B=60)
    assert seen == {(90, 8, 2)}


def test_deterministic_for_a_seed_and_different_across_seeds():
    x = np.random.default_rng(2).normal(size=100)
    a, b, c = bs.ci({"x": x}, _mean, B=200, seed=1), bs.ci({"x": x}, _mean, B=200, seed=1), bs.ci({"x": x}, _mean, B=200, seed=2)
    assert a == b and (a["lo"], a["hi"]) != (c["lo"], c["hi"])


def test_bca_differs_from_percentile_on_skewed_small_sample_and_brackets_point():
    x = np.random.default_rng(3).lognormal(0, 1.2, 11)  # like 11 per-patient Dice scores
    p = bs.ci({"x": x}, _mean, B=1500, method="percentile")
    b = bs.ci({"x": x}, _mean, B=1500, method="bca")
    assert b["lo"] < b["point"] < b["hi"]
    assert (b["lo"], b["hi"]) != (p["lo"], p["hi"]) and b["hi"] >= p["hi"] - 1e-9  # skew pushes the upper end out


def test_needs_two_groups():
    with pytest.raises(ValueError):
        bs.ci({"x": np.array([1.0, 2.0])}, _mean, groups=np.array([7, 7]))


def test_difference_ci_of_independent_estimates():
    rng = np.random.default_rng(4)
    a_pt, a_vals, _ = bs.bootstrap_values({"x": rng.normal(0.99, 0.05, 300)}, _mean, B=500)
    b_pt, b_vals, _ = bs.bootstrap_values({"x": rng.normal(0.90, 0.05, 300)}, _mean, B=500, seed=9)
    d = bs.diff_ci(a_vals, b_vals, a_pt, b_pt)
    assert d["lo"] > 0.05 and d["lo"] < d["point"] < d["hi"]


# ----------------------------------------------------------------------------- metrics


def test_balanced_accuracy_is_mean_recall_not_accuracy():
    y = np.array([0] * 90 + [1] * 10)
    pred = np.zeros(100, int)
    assert mt.accuracy(y, pred) == 0.9
    assert mt.balanced_accuracy(y, pred, 2) == 0.5


def test_balanced_accuracy_ignores_absent_classes():
    assert mt.balanced_accuracy(np.array([0, 0, 1]), np.array([0, 1, 1]), 4) == 0.75


def test_macro_f1_perfect_and_worst():
    y = np.array([0, 1, 2, 2])
    assert mt.macro_f1(y, y, 3) == 1.0
    assert mt.macro_f1(y, np.array([1, 2, 0, 0]), 3) == 0.0


def test_nll_brier_known_values():
    p = np.array([[0.8, 0.2], [0.3, 0.7]])
    y = np.array([0, 1])
    assert mt.nll(y, p) == pytest.approx(-(np.log(0.8) + np.log(0.7)) / 2)
    assert mt.brier(y, p) == pytest.approx(((0.2**2 + 0.2**2) + (0.3**2 + 0.3**2)) / 2)


def test_ece_zero_when_calibrated_and_large_when_overconfident():
    rng = np.random.default_rng(5)
    n = 20000
    conf = rng.uniform(0.5, 1.0, n)
    correct = rng.uniform(size=n) < conf  # calibrated by construction
    y = np.where(correct, 0, 1)  # class 0 is always the top class, with probability conf
    probs = np.stack([conf, 1 - conf], 1)
    assert mt.ece(y, probs) < 0.02
    over = np.stack([np.full(n, 0.99), np.full(n, 0.01)], 1)
    y_half = (rng.uniform(size=n) < 0.6).astype(int) ^ 1  # right 60% of the time
    assert mt.ece(y_half, over) == pytest.approx(0.39, abs=0.02)
    assert abs(mt.ece(y, probs, adaptive=True) - mt.ece(y, probs)) < 0.02


def test_sensitivity_at_90_specificity_threshold_comes_from_reference_data():
    rng = np.random.default_rng(6)
    yv = np.r_[np.zeros(1000), np.ones(1000)]
    sv = np.r_[rng.normal(0, 1, 1000), rng.normal(2, 1, 1000)]
    thr = mt.threshold_at_specificity(yv, sv, 0.9)
    assert mt.specificity_at_threshold(yv, sv, thr) >= 0.9 - 1e-9
    assert mt.specificity_at_threshold(yv, sv, thr) < 0.915
    yt = np.r_[np.zeros(1000), np.ones(1000)]
    st = np.r_[rng.normal(0, 1, 1000), rng.normal(2, 1, 1000)]
    assert 0.70 < mt.sensitivity_at_threshold(yt, st, thr) < 0.85  # theory: Phi(2 - 1.2816) = 0.76


def test_auroc_ovr_perfect_and_chance():
    y = np.array([0, 1, 2] * 50)
    probs = np.eye(3)[y]
    assert mt.auroc_ovr_macro(y, probs) == 1.0
    assert abs(mt.auroc_ovr_macro(y, np.full((150, 3), 1 / 3) + np.random.default_rng(0).normal(0, 1e-3, (150, 3))) - 0.5) < 0.15


def test_dice_from_counts():
    assert mt.dice(10, 0, 0) == 1.0 and mt.dice(0, 0, 0) == 1.0 and mt.dice(0, 5, 5) == 0.0
    assert mt.dice(8, 2, 2) == pytest.approx(0.8)


def test_ci_returns_nan_interval_when_statistic_is_undefined_everywhere():
    c = bs.ci({"x": np.arange(10.0)}, lambda x: float("nan"), B=20)
    assert c["n_valid"] == 0 and np.isnan(c["lo"]) and np.isnan(c["hi"])


def test_signal_compare_survives_zero_errors_in_the_unflagged_group():
    import pandas as pd
    from ml.eval import signals as sg

    df = pd.DataFrame({"error": [1] * 20 + [0] * 40, "flag": [True] * 20 + [False] * 40})
    out = sg.compare(df, "flag")
    assert out["predicts_errors"] is True and out["risk_ratio"]["n_valid"] == 0 and "risk_ratio_note" in out
