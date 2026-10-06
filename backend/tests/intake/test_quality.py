import cv2
import numpy as np
import pytest

from medproof.intake.decode import load_image
from medproof.intake.quality import QualityConfig, assess
from medproof.verify import perturbations as P
from tests.conftest import make_phantom, png_bytes, to_u8


def _decoded(arr: np.ndarray):
    return load_image(png_bytes(to_u8(arr)))


def _rotated(a: np.ndarray, deg: float) -> np.ndarray:
    h, w = a.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(a, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=0.06)


# (fixture name, builder, reason code the gate must give)
DEGRADED = [
    ("low_resolution", lambda p: cv2.resize(p, (160, 160), interpolation=cv2.INTER_AREA), "low_resolution"),
    ("blurred", lambda p: P.apply("blur", p, 5), "blurred"),
    ("underexposed", lambda p: p * 0.35, "underexposed"),
    ("overexposed", lambda p: np.clip(p * 0.5 + 0.55, 0, 1), "overexposed"),
    ("noisy", lambda p: P.apply("noise", p, 5), "noisy"),
    ("rotated", lambda p: _rotated(p, 25.0), "rotated"),
]


def test_clean_phantom_passes_with_no_reasons(phantom):
    rep = assess(_decoded(phantom), "cxr")
    assert rep.passed and rep.reasons == [], rep.to_dict()
    assert rep.flags == []


@pytest.mark.parametrize("name,build,code", DEGRADED, ids=[d[0] for d in DEGRADED])
def test_each_degradation_gives_the_right_reason(phantom, name, build, code):
    rep = assess(_decoded(build(phantom)), "cxr")
    assert code in rep.codes(), (name, rep.to_dict())
    match = next(r for r in rep.reasons if r.code == code)
    assert match.level == "fail"
    assert not rep.passed and rep.flags == ["low_quality"]
    assert len(match.fix) > 15  # every failure says how to fix it


def test_rotation_only_checked_for_cxr(phantom):
    img = _decoded(_rotated(phantom, 25.0))
    assert "rotated" in assess(img, "cxr").codes()
    assert "rotated" not in assess(img, "skin_dermoscopy").codes()
    assert "rotated" not in assess(img, None).codes()


def test_small_tilt_is_a_warning_not_a_failure(phantom):
    rep = assess(_decoded(_rotated(phantom, 14.0)), "cxr")
    rot = [r for r in rep.reasons if r.code == "rotated"]
    assert rot and rot[0].level == "warn" and rep.passed


def test_straight_image_has_no_tilt_reason_even_if_off_centre(phantom):
    shifted = np.roll(phantom, 20, axis=1)
    assert "rotated" not in assess(_decoded(shifted), "cxr").codes()


def test_metrics_are_reported_and_json_safe(phantom):
    d = assess(_decoded(phantom), "cxr").to_dict()
    for k in ("blur", "noise_rel", "range", "clip", "min_side", "tilt_deg"):
        assert k in d["metrics"]


def test_thresholds_are_configurable(phantom):
    strict = QualityConfig(min_side_warn=2000, min_side_fail=1000)
    rep = assess(_decoded(phantom), "cxr", strict)
    assert "low_resolution" in rep.codes() and not rep.passed


def test_resolution_independent_scores(phantom):
    big = cv2.resize(phantom, (1024, 1024), interpolation=cv2.INTER_CUBIC)
    a = assess(_decoded(phantom), "cxr").metrics
    b = assess(_decoded(big), "cxr").metrics
    assert abs(a["blur"] - b["blur"]) / a["blur"] < 0.35


def test_flat_margins_warn(phantom):
    padded = np.zeros((512, 1100), np.float32)
    padded[:, 300:812] = phantom
    assert "unusual_framing" in assess(_decoded(padded), None).codes()


def test_sweep_clean_flag_rate_low_and_severe_recall_high():
    """Documents the operating point: few false alarms on clean variants, high recall at severity >= 4."""
    clean_total = clean_flagged = 0
    for seed in range(8):
        p = make_phantom(512, seed=seed)
        clean_total += 1
        clean_flagged += not assess(_decoded(p), "cxr").passed
    assert clean_flagged / clean_total <= 0.05

    hits = total = 0
    for seed in range(3):
        p = make_phantom(512, seed=seed)
        for name in ("noise", "blur", "contrast"):
            for sev in (4, 5):
                total += 1
                hits += not assess(_decoded(P.apply(name, p, sev, seed)), "cxr").passed
    assert hits / total >= 0.9, (hits, total)
