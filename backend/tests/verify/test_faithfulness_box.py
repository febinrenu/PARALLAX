"""Deletion test for findings that carry a box and no heatmap (bone detector)."""

import numpy as np

from medproof.readers.base import ReaderFinding, ReaderOutput
from medproof.readers.cxr_config import CxrConfig
from medproof.readers.evidence import to_findings
from medproof.verify import faithfulness as F

HW = (128, 128)
BOX = (20.0, 30.0, 60.0, 70.0)


def _img():
    img = np.full(HW, 0.3, np.float32)
    img += np.random.default_rng(1).normal(0, 0.05, HW).astype(np.float32)
    img[30:70, 20:60] = 0.9  # the lesion
    return np.clip(img, 0, 1)


def _depends_on_lesion(img):
    return np.array([float(img[30:70, 20:60].mean())], np.float32)  # falls when the box is blurred away


def _ignores_lesion(img):
    return np.array([float(img[100:128, 100:128].mean())], np.float32)  # looks elsewhere


def test_box_the_model_depends_on_is_faithful():
    r = F.assess_box(_depends_on_lesion, _img(), BOX, 0, F.FaithfulnessConfig())
    assert r.faithful is True and r.drop > 0.05 and r.p_random <= 0.10


def test_box_the_model_ignores_is_not_faithful():
    r = F.assess_box(_ignores_lesion, _img(), BOX, 0, F.FaithfulnessConfig())
    assert r.faithful is False


def test_random_boxes_have_the_same_size_and_stay_off_the_real_box():
    rng = np.random.default_rng(0)
    for _ in range(40):
        b = F.random_box_like(BOX, HW, rng)
        assert abs((b[2] - b[0]) - 40) < 1 and abs((b[3] - b[1]) - 40) < 1
        assert 0 <= b[0] and b[2] <= HW[1] and 0 <= b[1] and b[3] <= HW[0]
        assert F.box_iou(b, BOX) <= 0.25 + 1e-9


def test_a_box_covering_most_of_the_image_cannot_be_tested():
    r = F.assess_box(_depends_on_lesion, _img(), (0.0, 0.0, 120.0, 120.0), 0, F.FaithfulnessConfig())
    assert r.faithful is None and "too large" in r.reason


def test_degenerate_or_missing_box_is_not_assessable():
    assert F.assess_box(_depends_on_lesion, _img(), None, 0).faithful is None
    assert F.assess_box(_depends_on_lesion, _img(), (10.0, 10.0, 10.0, 40.0), 0).faithful is None


def test_deterministic_for_a_fixed_seed():
    a = F.assess_box(_depends_on_lesion, _img(), BOX, 0, seed=3)
    b = F.assess_box(_depends_on_lesion, _img(), BOX, 0, seed=3)
    assert a.to_dict() == b.to_dict()


def test_boxes_the_model_ignores_rarely_pass():
    img = _img()
    score = lambda im: np.array([float(im[100:128, 100:128].mean())], np.float32)  # noqa: E731
    rng = np.random.default_rng(5)
    passes = tested = 0
    for s in range(20):
        y, x = int(rng.integers(0, 60)), int(rng.integers(0, 60))
        tested += 1
        passes += bool(F.assess_box(score, img, (float(x), float(y), x + 40.0, y + 40.0), 0, seed=s).faithful)
    assert tested == 20 and passes <= 2


def test_assess_output_uses_the_box_when_there_is_no_heatmap_and_fills_the_finding():
    f = ReaderFinding("fracture", 0.9, 0.9, heatmap=None, boxes_xyxy=[BOX], method="yolo", region_name="hand")
    out = ReaderOutput(modality="bone_xray", model_id="bone@1", labels=["fracture"], logits=np.zeros(1, np.float32),
                       probs=np.array([0.9], np.float32), findings=[f], image_hw=HW)

    class Reader:
        score = staticmethod(_depends_on_lesion)

    res = F.assess_output(Reader(), _img(), out, F.FaithfulnessConfig())
    assert res["fracture"].faithful is True
    findings, _ = to_findings(out, CxrConfig())
    updated = F.apply_to_findings(findings, res)
    ev = updated[0].image_evidence[0]
    assert ev.kind == "bbox" and ev.faithful is True and ev.faithfulness_drop > 0.05
    assert findings[0].image_evidence[0].faithful is None  # inputs untouched


def test_heatmap_findings_still_use_the_heatmap():
    heat = np.zeros(HW, np.float32)
    heat[30:70, 20:60] = 1.0
    f = ReaderFinding("x", 0.9, 0.9, heatmap=heat, boxes_xyxy=[BOX], method="gradcam++")
    out = ReaderOutput(modality="cxr", model_id="m", labels=["x"], logits=np.zeros(1, np.float32),
                       probs=np.array([0.9], np.float32), findings=[f], image_hw=HW)

    class Reader:
        score = staticmethod(_depends_on_lesion)

    r = F.assess_output(Reader(), _img(), out, F.FaithfulnessConfig(n_random=9))["x"]
    assert r.insertion_auc is not None  # only the heatmap path computes the insertion curve
