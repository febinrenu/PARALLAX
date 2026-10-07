"""MedSAM wrapper logic tests with a stand-in predictor. Real-model checks are in test_medsam_real.py."""

import numpy as np
import pytest

from medproof.segment import medsam as M

IMG = (np.random.default_rng(0).random((120, 160)) * 255).astype(np.uint8)


class BoxPredictor:
    """Stand-in for the model: returns a probability map that is high inside the box. Counts calls."""

    def __init__(self):
        self.calls = 0

    def __call__(self, image_rgb, box):
        self.calls += 1
        h, w = image_rgb.shape[:2]
        x0, y0, x1, y1 = (int(round(v)) for v in box)
        m = np.zeros((h, w), np.float32)
        m[y0:y1, x0:x1] = 0.95
        return m


def _seg(tmp_path=None, **kw):
    return M.MedSAM(BoxPredictor(), cache_dir=tmp_path, **kw)


def test_mask_has_image_shape_and_covers_the_box():
    s = _seg()
    r = s.segment(IMG, (20, 30, 80, 90))
    assert r.mask.shape == IMG.shape and r.mask.dtype == bool
    assert r.mask[60, 50] and not r.mask[5, 5]
    assert r.box == (20.0, 30.0, 80.0, 90.0) and 0.0 < r.score <= 1.0


def test_box_is_clipped_to_the_image_and_degenerate_boxes_are_rejected():
    s = _seg()
    r = s.segment(IMG, (-10, -5, 500, 500))
    assert r.box == (0.0, 0.0, 160.0, 120.0)
    for bad in [(10, 10, 10, 50), (50, 50, 20, 80), (200, 200, 300, 300), (np.nan, 0, 5, 5)]:
        with pytest.raises(ValueError):
            s.segment(IMG, bad)


def test_same_image_and_box_hit_the_cache():
    s = _seg()
    a = s.segment(IMG, (20, 30, 80, 90))
    b = s.segment(IMG, (20, 30, 80, 90))
    assert s.predictor.calls == 1
    np.testing.assert_array_equal(a.mask, b.mask)
    s.segment(IMG, (21, 30, 80, 90))
    assert s.predictor.calls == 2  # a different box is a different entry
    s.segment(IMG[::-1].copy(), (20, 30, 80, 90))
    assert s.predictor.calls == 3  # a different image is a different entry


def test_disk_cache_survives_a_new_instance(tmp_path):
    a = _seg(tmp_path)
    r1 = a.segment(IMG, (20, 30, 80, 90))
    b = _seg(tmp_path)
    r2 = b.segment(IMG, (20, 30, 80, 90))
    assert b.predictor.calls == 0
    np.testing.assert_array_equal(r1.mask, r2.mask)
    assert r1.score == pytest.approx(r2.score)


def test_cache_key_includes_the_model_name(tmp_path):
    a = M.MedSAM(BoxPredictor(), cache_dir=tmp_path, model_id="model-a")
    a.segment(IMG, (20, 30, 80, 90))
    b = M.MedSAM(BoxPredictor(), cache_dir=tmp_path, model_id="model-b")
    b.segment(IMG, (20, 30, 80, 90))
    assert b.predictor.calls == 1


def test_gray_and_colour_inputs_and_float_images():
    s = _seg()
    rgb = np.stack([IMG] * 3, axis=2)
    assert s.segment(rgb, (20, 30, 80, 90)).mask.shape == IMG.shape
    flt = IMG.astype(np.float32) / 255.0
    assert s.segment(flt, (20, 30, 80, 90)).mask.shape == IMG.shape


def test_empty_prediction_gives_an_empty_mask_with_zero_score():
    s = M.MedSAM(lambda img, box: np.zeros(img.shape[:2], np.float32))
    r = s.segment(IMG, (20, 30, 80, 90))
    assert not r.mask.any() and r.score == 0.0


def test_overlap_metrics_known_values():
    a = np.zeros((10, 10), bool)
    a[:5] = True
    b = np.zeros((10, 10), bool)
    b[:5, :5] = True  # half of a
    assert M.jaccard(a, a) == 1.0 and M.dice(a, a) == 1.0
    assert M.jaccard(a, b) == pytest.approx(0.5) and M.dice(a, b) == pytest.approx(2 * 25 / 75)
    assert M.jaccard(a, np.zeros_like(a)) == 0.0
    assert M.jaccard(np.zeros_like(a), np.zeros_like(a)) == 1.0  # both empty agree


def test_box_from_mask_round_trip():
    m = np.zeros((50, 60), bool)
    m[10:20, 5:40] = True
    assert M.box_from_mask(m) == (5.0, 10.0, 40.0, 20.0)
    assert M.box_from_mask(np.zeros((5, 5), bool)) is None


def test_summary_statistics_over_pairs():
    gt = np.zeros((20, 20), bool)
    gt[5:15, 5:15] = True
    pairs = [(gt, gt), (gt, np.zeros_like(gt)), (gt, gt)]
    s = M.summarise(pairs)
    assert s["n"] == 3 and s["mean_jaccard"] == pytest.approx(2 / 3) and s["failure_rate"] == pytest.approx(1 / 3)
    assert M.summarise([])["n"] == 0


def test_stage_helpers_never_raise():
    s = _seg()
    ok = M.run_box(s, IMG, (20, 30, 80, 90))
    assert ok.ok and ok.stage == "segment" and ok.payload["box"] == [20.0, 30.0, 80.0, 90.0]
    bad = M.run_box(s, IMG, (10, 10, 10, 10))
    assert bad.ok is False and bad.warnings

    class Boom:
        def segment(self, image, box):
            raise RuntimeError("model gone")

    assert M.run_box(Boom(), IMG, (20, 30, 80, 90)).ok is False
