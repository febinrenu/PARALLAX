import numpy as np
import pytest

torch = pytest.importorskip("torch")

from medproof.core.schemas import Finding  # noqa: E402
from medproof.readers.evidence import to_findings  # noqa: E402
from medproof.verify import faithfulness as F  # noqa: E402
from tests.readers.test_cxr import BLOB, HW, CxrConfig, _image, make_reader  # noqa: E402


def _heat_on(region, value=1.0):
    h = np.zeros(HW, np.float32)
    h[region] = value
    return h


def _smooth(h):
    import cv2

    return cv2.GaussianBlur(h, (0, 0), 6).astype(np.float32)


# ---- blur_top_region ---------------------------------------------------------------------
def test_blur_changes_only_the_top_fraction_of_pixels():
    img = _image()
    heat = _smooth(_heat_on(BLOB))
    out = F.blur_top_region(img, heat, 0.10)
    changed = out != img
    area = changed.mean()
    assert 0.03 < area <= 0.10 + 1e-6  # only pixels in the region, never more than the fraction
    top = heat >= np.quantile(heat, 0.90)
    assert not changed[~top].any()
    assert out.shape == img.shape and out.dtype == np.float32 and out.min() >= 0 and out.max() <= 1


def test_blur_leaves_the_input_untouched_and_supports_colour():
    img = _image()
    before = img.copy()
    F.blur_top_region(img, _smooth(_heat_on(BLOB)), 0.2)
    np.testing.assert_array_equal(img, before)
    rgb = np.stack([img] * 3, axis=2)
    assert F.blur_top_region(rgb, _smooth(_heat_on(BLOB)), 0.2).shape == rgb.shape


def test_blur_rejects_bad_fraction_and_shape():
    with pytest.raises(ValueError):
        F.blur_top_region(_image(), _heat_on(BLOB), 0.0)
    with pytest.raises(ValueError):
        F.blur_top_region(_image(), np.zeros((5, 5), np.float32), 0.1)


# ---- deletion drop and random baseline -------------------------------------------------------
def test_deleting_the_evidence_drops_the_probability_a_lot():
    r = make_reader(sharpness=6.0)
    img, heat = _image(), _smooth(_heat_on(BLOB))
    drop = F.deletion_drop(r.score, img, heat, 0, 0.10)
    assert drop > 0.5


def test_deleting_an_irrelevant_region_does_nothing():
    r = make_reader(sharpness=6.0)
    img = _image()
    wrong = _smooth(_heat_on((slice(200, 260), slice(300, 380))))  # empty background, inside the crop
    assert abs(F.deletion_drop(r.score, img, wrong, 0, 0.10)) < 0.05


def test_random_placements_keep_shape_and_area_of_the_region():
    heat = _smooth(_heat_on(BLOB))
    rng = np.random.default_rng(0)
    for _ in range(5):
        moved = F.shifted_region(heat, 0.10, rng)
        assert moved.sum() == round(0.10 * heat.size)  # exactly the requested area, ties included


def test_random_placements_stay_away_from_the_real_region():
    heat = _smooth(_heat_on(BLOB))
    top = F._top_mask(heat, 0.10)
    rng = np.random.default_rng(1)
    for _ in range(40):
        moved = F.shifted_region(heat, 0.10, rng)
        iou = np.logical_and(moved, top).sum() / np.logical_or(moved, top).sum()
        assert iou <= 0.25


def test_permutation_p_value_is_small_for_good_evidence_and_large_for_bad():
    r = make_reader(sharpness=6.0)
    img = _image()
    good = F.assess_heatmap(r.score, img, _smooth(_heat_on(BLOB)), 0, F.FaithfulnessConfig(n_random=24), seed=0)
    bad = F.assess_heatmap(r.score, img, _smooth(_heat_on((slice(200, 260), slice(300, 380)))), 0, F.FaithfulnessConfig(n_random=24), seed=0)
    assert good.faithful is True and good.p_random <= 0.10 and good.drop > 0.5
    assert bad.faithful is False and bad.p_random > 0.10


def test_assessment_is_deterministic_per_seed():
    r = make_reader(sharpness=6.0)
    args = (r.score, _image(), _smooth(_heat_on(BLOB)), 0, F.FaithfulnessConfig(n_random=9))
    a, b = F.assess_heatmap(*args, seed=3), F.assess_heatmap(*args, seed=3)
    assert a == b


def test_all_three_deletion_fractions_are_reported():
    r = make_reader(sharpness=6.0)
    res = F.assess_heatmap(r.score, _image(), _smooth(_heat_on(BLOB)), 0, F.FaithfulnessConfig(n_random=9), seed=0)
    assert set(res.drops) == {0.05, 0.10, 0.20}
    assert res.drops[0.10] == pytest.approx(res.drop)


# ---- insertion curve ---------------------------------------------------------------------
def test_insertion_curve_is_higher_for_the_right_heatmap_than_a_wrong_one():
    r = make_reader(sharpness=6.0)
    img = _image()
    good = F.insertion_auc(r.score, img, _smooth(_heat_on(BLOB)), 0, steps=8)
    bad = F.insertion_auc(r.score, img, _smooth(_heat_on((slice(200, 260), slice(300, 380)))), 0, steps=8)
    assert good > bad + 0.1
    assert 0.0 <= bad <= good <= 1.0


# ---- degenerate input ----------------------------------------------------------------------
def test_flat_or_missing_heatmap_cannot_be_assessed_and_is_not_called_faithful():
    r = make_reader(sharpness=6.0)
    for heat in (np.zeros(HW, np.float32), np.full(HW, 0.5, np.float32)):
        res = F.assess_heatmap(r.score, _image(), heat, 0, F.FaithfulnessConfig(n_random=9), seed=0)
        assert res.faithful is None and res.reason
    assert F.assess_heatmap(r.score, _image(), None, 0, F.FaithfulnessConfig(n_random=9), seed=0).faithful is None


# ---- reader output -> contract findings -------------------------------------------------------
def test_assess_reader_output_updates_contract_evidence_and_flags():
    r = make_reader(sharpness=6.0)
    img = _image()
    out = r.predict(img)
    cfg = CxrConfig()
    findings, _ = to_findings(out, cfg)
    results = F.assess_output(r, img, out, F.FaithfulnessConfig(n_random=12), seed=0)
    assert set(results) == {f.label for f in findings}
    updated = F.apply_to_findings(findings, results)
    for f in updated:
        Finding.model_validate(f.model_dump())
        ev = f.image_evidence[0]
        assert ev.faithful is results[f.label].faithful
        assert ev.faithfulness_drop == pytest.approx(results[f.label].drop)
        assert ("unfaithful" in f.flags) == (results[f.label].faithful is False)
    assert findings[0].image_evidence[0].faithful is None  # inputs are not mutated
    assert updated[0].image_evidence[0].faithful is True  # the leading finding is real evidence in this fixture


def test_pass_rate_report_compares_model_maps_with_random_regions():
    rep = F.pass_rate_report(model_pass=[True, True, False, True], random_pass=[False, False, False, True, False, False, False, False, False, False])
    assert rep["model_pass_rate"] == 0.75 and rep["random_pass_rate"] == 0.1 and rep["n_model"] == 4
    assert F.pass_rate_report([], [])["model_pass_rate"] is None


def test_run_stage_returns_stage_result_and_never_raises():
    from types import SimpleNamespace

    r = make_reader(sharpness=6.0)
    img = _image()
    out = r.predict(img)
    findings, _ = to_findings(out, CxrConfig())
    ctx = SimpleNamespace(decoded=img, reader_output=out, findings=findings)
    res = F.run(ctx, reader=r, cfg=F.FaithfulnessConfig(n_random=12))
    assert res.stage == "faithfulness" and res.ok
    assert set(res.payload["results"]) == {f.label for f in findings}
    assert [f["image_evidence"][0]["faithful"] for f in res.payload["findings"]][0] is True
    bad = F.run(SimpleNamespace(), reader=r)
    assert bad.ok is False and bad.warnings

    class Boom:
        def score(self, x):
            raise RuntimeError("no model")

    assert F.run(ctx, reader=Boom()).ok is False


def test_config_rejects_a_sample_too_small_to_ever_pass():
    with pytest.raises(ValueError):
        F.FaithfulnessConfig(n_random=5, alpha=0.10)
    F.FaithfulnessConfig(n_random=9, alpha=0.10)  # smallest valid for alpha 0.10


def test_median_baseline_removes_a_silhouette_that_blur_cannot():
    """A big bright object keeps its shape under blur; filling with the median removes it."""
    img = np.zeros((128, 128), np.float32)
    img[32:96, 32:96] = 1.0
    heat = np.zeros_like(img)
    heat[32:96, 32:96] = 1.0
    area = lambda x: float((x > 0.5).mean())  # noqa: E731
    score = lambda x: np.array([area(x)], np.float32)  # noqa: E731
    blur_drop = F.deletion_drop(score, img, heat, 0, 0.25, baseline="blur")
    median_drop = F.deletion_drop(score, img, heat, 0, 0.25, baseline="median")
    assert median_drop > blur_drop + 0.05


def test_baseline_validation():
    with pytest.raises(ValueError):
        F.FaithfulnessConfig(baseline="inpaint")
    r = make_reader(sharpness=6.0)
    res = F.assess_heatmap(r.score, _image(), _smooth(_heat_on(BLOB)), 0, F.FaithfulnessConfig(n_random=12, baseline="median"), seed=0)
    assert res.faithful is True
