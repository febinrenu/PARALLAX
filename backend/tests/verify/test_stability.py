import numpy as np
import pytest

from medproof.core.schemas import Finding, Stability
from medproof.readers.base import ReaderFinding, ReaderOutput
from medproof.readers.cxr_config import CxrConfig
from medproof.readers.evidence import to_findings
from medproof.verify import perturbations as P
from medproof.verify import stability as S
from tests.conftest import make_phantom

IMG = make_phantom(128)


def _const(p):
    return lambda img: np.array([p, 0.2], np.float32)


def _only_if_unchanged(original):
    """Confident only on the exact original pixels: any perturbation should flip it."""
    return lambda img: np.array([0.9 if np.array_equal(img, original) else 0.1], np.float32)


def _sharpness(thresh):
    import cv2

    def f(img):
        v = float(cv2.Laplacian(img.astype(np.float32), cv2.CV_32F).var())
        return np.array([0.9 if v >= thresh else 0.1], np.float32)

    return f


def test_a_score_that_never_changes_is_perfectly_stable():
    r = S.assess_label(_const(0.9), IMG, 0, S.StabilityConfig())
    assert r.tests == 8 and r.flips == 0 and r.flip_rate == 0.0 and r.worst_perturbation is None


def test_a_score_that_depends_on_exact_pixels_flips_under_every_perturbation():
    r = S.assess_label(_only_if_unchanged(IMG), IMG, 0, S.StabilityConfig())
    assert r.flips == 8 and r.flip_rate == 1.0
    assert r.worst_perturbation in P.NAMES


def test_only_perturbations_that_matter_flip_the_result():
    import cv2

    base = float(cv2.Laplacian(IMG, cv2.CV_32F).var())
    cfg = S.StabilityConfig(severity=4)
    r = S.assess_label(_sharpness(0.5 * base), IMG, 0, cfg)
    assert r.per_perturbation["blur"]["flipped"] is True
    assert r.per_perturbation["rotate"]["flipped"] is False or r.flips < 8
    assert 0 < r.flip_rate < 1.0
    assert r.per_perturbation[r.worst_perturbation]["flipped"] is True


def test_negative_baseline_flips_when_it_turns_positive():
    base_img = IMG
    score = lambda img: np.array([0.1 if np.array_equal(img, base_img) else 0.9], np.float32)  # noqa: E731
    r = S.assess_label(score, IMG, 0, S.StabilityConfig())
    assert r.flip_rate == 1.0


def test_threshold_is_respected():
    r = S.assess_label(_const(0.55), IMG, 0, S.StabilityConfig(positive_threshold=0.5))
    assert r.flips == 0
    r = S.assess_label(_const(0.55), IMG, 0, S.StabilityConfig(positive_threshold=0.6))
    assert r.flips == 0  # negative before and after: no flip either way


def test_deterministic_for_fixed_seeds_and_seeds_multiply_tests():
    cfg = S.StabilityConfig(seeds=(0, 1))
    a, b = S.assess_label(_only_if_unchanged(IMG), IMG, 0, cfg), S.assess_label(_only_if_unchanged(IMG), IMG, 0, cfg)
    assert a == b and a.tests == 16


def test_perturbed_images_are_fed_to_the_scorer_once_per_test():
    calls = []

    def score(img):
        calls.append(img.shape)
        return np.array([0.9], np.float32)

    S.assess_label(score, IMG, 0, S.StabilityConfig())
    assert len(calls) == 1 + 8  # the baseline plus eight perturbations
    assert all(c == IMG.shape for c in calls)


def test_all_labels_share_the_same_eight_forward_passes():
    calls = []

    def score(img):
        calls.append(1)
        return np.array([0.9, 0.8, 0.7], np.float32)

    res = S.assess_labels(score, IMG, [0, 1, 2], S.StabilityConfig())
    assert len(calls) == 1 + 8 and set(res) == {0, 1, 2}


def _output():
    f = ReaderFinding("Pneumonia", 0.9, 0.9, heatmap=np.ones((128, 128), np.float32), boxes_xyxy=[(10.0, 10.0, 50.0, 50.0)], method="gradcam++")
    g = ReaderFinding("Effusion", 0.8, 0.8, heatmap=np.ones((128, 128), np.float32), boxes_xyxy=[(60.0, 60.0, 90.0, 90.0)], method="gradcam++")
    return ReaderOutput(modality="cxr", model_id="m@1", labels=["Pneumonia", "Effusion"], logits=np.array([0.9, 0.8], np.float32),
                        probs=np.array([0.9, 0.8], np.float32), findings=[f, g], image_hw=(128, 128))


class _Reader:
    def __init__(self, score):
        self.score = score
        self.cfg = CxrConfig()


def test_contract_stability_object_and_unstable_flag():
    out = _output()
    findings, _ = to_findings(out, CxrConfig())
    reader = _Reader(lambda img: np.array([0.9 if np.array_equal(img, IMG) else 0.1, 0.8], np.float32))
    res = S.assess_output(reader, IMG, out, S.StabilityConfig())
    updated = S.apply_to_findings(findings, res, S.StabilityConfig())
    by = {f.label: f for f in updated}
    assert isinstance(by["Pneumonia"].stability, Stability)
    assert by["Pneumonia"].stability.tests == 8 and by["Pneumonia"].stability.flip_rate == 1.0
    assert by["Pneumonia"].stability.worst_perturbation in P.NAMES and "unstable" in by["Pneumonia"].flags
    assert by["Effusion"].stability.flip_rate == 0.0 and "unstable" not in by["Effusion"].flags
    assert by["Effusion"].stability.worst_perturbation is None
    for f in updated:
        Finding.model_validate(f.model_dump())
    assert findings[0].stability is None  # inputs untouched


def test_run_stage_and_failure_path():
    from types import SimpleNamespace

    out = _output()
    findings, _ = to_findings(out, CxrConfig())
    ctx = SimpleNamespace(decoded=IMG, reader_output=out, findings=findings)
    res = S.run(ctx, reader=_Reader(_const(0.9) if False else (lambda img: np.array([0.9, 0.8], np.float32))))
    assert res.stage == "stability" and res.ok and set(res.payload["results"]) == {"Pneumonia", "Effusion"}
    assert res.payload["findings"][0]["stability"]["flip_rate"] == 0.0
    assert S.run(SimpleNamespace(), reader=None).ok is False

    class Boom:
        def score(self, x):
            raise RuntimeError("gone")

    assert S.run(ctx, reader=Boom()).ok is False


def test_config_validation():
    with pytest.raises(ValueError):
        S.StabilityConfig(severity=9)
    with pytest.raises(ValueError):
        S.StabilityConfig(perturbations=("nope",))
