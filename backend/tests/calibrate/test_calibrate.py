from __future__ import annotations

import json

import numpy as np
import pytest

from medproof.calibrate import abstain, conformal, selective, temperature
from medproof.calibrate.calibrator import Calibrator


def _overconfident(n=20000, k=5, T_true=2.5, seed=0):
    """Logits whose softmax is calibrated only after dividing by T_true."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 2.0, (n, k))  # latent logits
    p = np.exp(z - z.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    y = np.array([rng.choice(k, p=row) for row in p])
    return z * T_true, y  # the model reports logits stretched by T_true


def test_temperature_recovers_the_stretch_factor():
    logits, y = _overconfident()
    T = temperature.fit_temperature(logits, y)
    assert T == pytest.approx(2.5, rel=0.06)
    p1 = temperature.softmax(logits, 1.0)
    pT = temperature.softmax(logits, T)
    assert temperature.nll(pT, y) < temperature.nll(p1, y)


def test_temperature_leaves_a_calibrated_model_alone():
    logits, y = _overconfident(T_true=1.0)
    assert temperature.fit_temperature(logits, y) == pytest.approx(1.0, abs=0.08)


def test_temperature_scaling_never_changes_the_predicted_class():
    logits, y = _overconfident(2000)
    T = temperature.fit_temperature(logits, y)
    assert (temperature.softmax(logits, T).argmax(1) == logits.argmax(1)).all()


def test_binary_temperature_and_platt():
    rng = np.random.default_rng(1)
    z = rng.normal(0, 1.5, 30000)
    y = (rng.uniform(size=z.size) < 1 / (1 + np.exp(-z))).astype(int)
    T = temperature.fit_temperature_binary(z * 3.0, y)
    assert T == pytest.approx(3.0, rel=0.08)
    a, b = temperature.fit_platt(z * 3.0 + 1.0, y)  # shifted and stretched scores
    p = 1 / (1 + np.exp(-(a * (z * 3.0 + 1.0) + b)))
    assert abs(p.mean() - y.mean()) < 0.01


# ----------------------------------------------------------------------------- conformal


def _calibrated_probs(n, k=6, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.dirichlet(np.ones(k) * 0.4, n)
    y = np.array([rng.choice(k, p=row) for row in p])
    return p, y


@pytest.mark.parametrize("alpha", [0.1, 0.05])
def test_aps_coverage_meets_target_on_exchangeable_data(alpha):
    pc, yc = _calibrated_probs(4000, seed=1)
    pt, yt = _calibrated_probs(20000, seed=2)
    q = conformal.fit_aps(pc, yc, alpha)
    u = np.random.default_rng(99).uniform(size=len(yt))
    sets = conformal.aps_sets(pt, q, u)
    cov = conformal.coverage(sets, yt)
    assert 1 - alpha - 0.012 <= cov <= 1 - alpha + 0.012  # randomised APS is close to exact
    assert all(len(s) >= 1 for s in sets)
    qd = conformal.fit_aps(pc, yc, alpha, randomized=False)  # the deterministic variant is valid but conservative
    assert conformal.coverage(conformal.aps_sets(pt, qd), yt) >= 1 - alpha - 0.01


def test_randomised_aps_is_much_tighter_than_deterministic_for_a_confident_model():
    rng = np.random.default_rng(10)
    n = 6000
    y = rng.integers(0, 4, n)
    wrong = rng.uniform(size=n) < 0.09
    top = np.where(wrong, (y + 1) % 4, y)
    p = np.full((n, 4), 0.01)
    p[np.arange(n), top] = 0.97
    q_r = conformal.fit_aps(p[:3000], y[:3000], 0.1)
    q_d = conformal.fit_aps(p[:3000], y[:3000], 0.1, randomized=False)
    u = rng.uniform(size=3000)
    s_r = conformal.aps_sets(p[3000:], q_r, u)
    s_d = conformal.aps_sets(p[3000:], q_d)
    assert conformal.mean_set_size(s_r) <= conformal.mean_set_size(s_d) + 1e-9
    assert conformal.coverage(s_d, y[3000:]) >= conformal.coverage(s_r, y[3000:]) - 0.01
    assert abs(conformal.coverage(s_r, y[3000:]) - 0.9) < 0.03


def test_sets_are_larger_for_harder_inputs():
    pc, yc = _calibrated_probs(4000, seed=3)
    q = conformal.fit_aps(pc, yc, 0.1)
    sharp = np.array([[0.97, 0.01, 0.01, 0.005, 0.003, 0.002]])
    flat = np.full((1, 6), 1 / 6)
    assert len(conformal.aps_sets(sharp, q)[0]) < len(conformal.aps_sets(flat, q)[0])


def test_mondrian_gives_class_conditional_coverage():
    rng = np.random.default_rng(4)
    n = 30000
    y = rng.choice(3, size=n, p=[0.8, 0.15, 0.05])
    z = rng.normal(0, 1, (n, 3)) + 2.0 * np.eye(3)[y]
    p = temperature.softmax(z)
    split = n // 2
    qs = conformal.fit_aps_mondrian(p[:split], y[:split], 0.1, 3)
    sets = conformal.aps_sets_mondrian(p[split:], qs, rng.uniform(size=n - split))
    for c in range(3):
        m = y[split:] == c
        assert conformal.coverage([s for s, keep in zip(sets, m) if keep], y[split:][m]) >= 0.87


def test_raps_sets_are_not_larger_than_aps_on_peaked_data():
    pc, yc = _calibrated_probs(5000, seed=5)
    pt, yt = _calibrated_probs(5000, seed=6)
    q_aps = conformal.fit_aps(pc, yc, 0.1, randomized=False)
    q_raps = conformal.fit_raps(pc, yc, 0.1, k_reg=2, lam=0.05)
    s_raps = conformal.raps_sets(pt, q_raps, k_reg=2, lam=0.05)
    assert conformal.coverage(s_raps, yt) >= 0.9 - 0.012
    assert conformal.mean_set_size(s_raps) <= conformal.mean_set_size(conformal.aps_sets(pt, q_aps)) + 0.3


def test_clopper_pearson_interval():
    lo, hi = conformal.clopper_pearson(90, 100)
    assert 0.82 < lo < 0.84 and 0.94 < hi < 0.96
    assert conformal.clopper_pearson(0, 10)[0] == 0.0 and conformal.clopper_pearson(10, 10)[1] == 1.0


def test_fnr_thresholds_control_the_miss_rate_per_label():
    rng = np.random.default_rng(7)
    n = 40000
    labels = (rng.uniform(size=(n, 2)) < [0.3, 0.05]).astype(int)
    scores = np.clip(rng.normal(0.35 + 0.35 * labels, 0.2), 0, 1)
    half = n // 2
    thr = conformal.fit_fnr_thresholds(scores[:half], labels[:half], 0.1)
    for j in range(2):
        pos = labels[half:, j] == 1
        fnr = np.mean(scores[half:][pos, j] < thr[j])
        assert fnr <= 0.1 + 0.02, (j, fnr)


def test_fnr_threshold_with_too_few_positives_accepts_everything():
    thr = conformal.fit_fnr_thresholds(np.array([[0.9], [0.1]]), np.array([[1], [0]]), 0.1)
    assert thr[0] <= 0.0 or np.isneginf(thr[0])  # cannot certify a miss rate from one positive


# ----------------------------------------------------------------------------- abstention and selective


def test_tier_thresholds_reach_their_accuracy_targets_or_are_disabled():
    rng = np.random.default_rng(8)
    conf = rng.uniform(0.3, 1.0, 20000)
    y_ok = rng.uniform(size=conf.size) < conf  # calibrated
    rule = abstain.fit_tiers(conf, y_ok, acc_high=0.95, acc_moderate=0.85, min_count=100)
    # for calibrated confidence the accuracy among {conf >= t} is about (1 + t) / 2, so 95% needs t = 0.9 and 85% needs t = 0.7
    assert rule.t_high == pytest.approx(0.90, abs=0.03)
    assert rule.t_moderate == pytest.approx(0.70, abs=0.04)
    never = abstain.fit_tiers(np.full(500, 0.6), np.zeros(500, bool))
    assert never.t_high is None and never.t_moderate is None


def test_tier_assignment_uses_set_size_for_abstention():
    rule = abstain.TierRule(t_high=0.95, t_moderate=0.85, t_low=0.5)
    assert abstain.tier(0.97, 1, rule) == "high"
    assert abstain.tier(0.9, 1, rule) == "moderate"
    assert abstain.tier(0.6, 2, rule) == "low"
    assert abstain.tier(0.97, 3, rule) == "abstain"  # a wide conformal set overrides a confident top class
    assert abstain.tier(0.3, 1, rule) == "abstain"


def test_risk_coverage_and_aurc():
    conf = np.array([0.9, 0.8, 0.7, 0.6])
    correct = np.array([1, 1, 0, 0])
    cov, risk = selective.risk_coverage(conf, correct)
    assert cov.tolist() == [0.25, 0.5, 0.75, 1.0] and risk.tolist() == [0.0, 0.0, 1 / 3, 0.5]
    assert selective.aurc(conf, correct) == pytest.approx(np.mean(risk))
    assert selective.aurc(conf, correct) == pytest.approx(selective.oracle_aurc(correct))  # already perfectly ranked
    assert selective.aurc(conf[::-1], correct) > selective.oracle_aurc(correct)
    assert selective.coverage_at_risk(conf, correct, 0.0) == 0.5


# ----------------------------------------------------------------------------- calibrator


def test_calibrator_end_to_end_and_json_roundtrip(tmp_path):
    logits, y = _overconfident(6000, k=4, T_true=2.0, seed=9)
    cal = Calibrator.fit(["a", "b", "c", "d"], logits[:2000], y[:2000], logits[2000:4000], y[2000:4000], alpha=0.1)
    out = cal.calibrate(logits[4000:])
    assert out.probs.shape == (2000, 4) and np.allclose(out.probs.sum(1), 1)
    assert len(out.conformal_sets) == 2000 and set(out.tiers) <= {"high", "moderate", "low", "abstain"}
    cov = conformal.coverage([[cal.classes.index(c) for c in s] for s in out.conformal_sets], y[4000:])
    assert cov >= 0.9 - 0.03
    p = tmp_path / "cal.json"
    cal.save(p)
    cal2 = Calibrator.load(p)
    out2 = cal2.calibrate(logits[4000:])
    assert out2.conformal_sets == out.conformal_sets and np.allclose(out2.probs, out.probs)
    assert json.loads(p.read_text())["temperature"] == pytest.approx(cal.temperature)


# ----------------------------------------------------------------------------- binary


def test_binary_calibrator_fixes_base_rate_and_controls_fnr(tmp_path):
    from medproof.calibrate.binary import BinaryCalibrator

    rng = np.random.default_rng(11)
    n = 40000
    y = (rng.uniform(size=n) < 0.2).astype(int)
    z = rng.normal(-1.5 + 3.0 * y, 1.2)
    score = 1 / (1 + np.exp(-z / 2.5))  # sharpened-away, wrong base rate
    h = n // 3
    cal = BinaryCalibrator.fit("Lung Opacity", score[:h], y[:h], score[h : 2 * h], y[h : 2 * h], alpha=0.1)
    p = cal.prob(score[2 * h :])
    yt = y[2 * h :]
    assert abs(p.mean() - yt.mean()) < 0.015
    miss = np.mean(p[yt == 1] < cal.thr_fnr)
    assert miss <= 0.1 + 0.02
    cal.save(tmp_path / "b.json")
    again = BinaryCalibrator.load(tmp_path / "b.json")
    assert np.allclose(again.prob(score[:5]), cal.prob(score[:5]))
    d = cal.decide(score[:50])
    assert set(d) == {"prob_calibrated", "report", "abstain"}
