from __future__ import annotations

import pytest

pytest.importorskip("pandas")

from ml.eval_p3.eval_medgemma_boxes import (  # noqa: E402
    BASELINE_BOX,
    evaluate,
    parse_boxes,
    pointing_hit,
    to_pixels,
)


def test_parse_boxes_reads_box_2d_as_y_x_and_returns_x_y_order():
    reply = '```json\n[{"label": "opacity", "box_2d": [100, 200, 300, 400]}]\n```'
    assert parse_boxes(reply) == [(200.0, 100.0, 400.0, 300.0)]


@pytest.mark.parametrize("reply", [
    "[]", "No opacity is seen.", "", "[{\"label\": \"x\"}]",
    '[{"label": "x", "box_2d": [300, 400, 100, 200]}]',  # inverted
    '[{"label": "x", "box_2d": [0, 0, 5000, 10]}]',  # outside the 0-1000 grid
    '[{"label": "x", "box_2d": [1, 2, 3]}]',  # wrong length
    '[{"label": "x", "box_2d": ["a", 2, 3, 4]}]',  # not numbers
])
def test_parse_boxes_drops_anything_unusable_and_never_raises(reply):
    assert parse_boxes(reply) == []


def test_parse_boxes_survives_a_truncated_reply():
    assert parse_boxes('[{"label": "opacity", "box_2d": [100, 200, 300, 40') == []


def test_to_pixels_scales_the_grid_to_the_image():
    assert to_pixels((100.0, 200.0, 500.0, 600.0), 1024, 1024) == pytest.approx((102.4, 204.8, 512.0, 614.4))
    assert to_pixels((0.0, 0.0, 1000.0, 1000.0), 200, 100) == pytest.approx((0.0, 0.0, 200.0, 100.0))


def test_pointing_hit_uses_the_centre_of_the_prediction():
    gt = [(100.0, 100.0, 200.0, 200.0)]
    assert pointing_hit((120.0, 120.0, 180.0, 180.0), gt)
    assert not pointing_hit((300.0, 300.0, 400.0, 400.0), gt)
    assert not pointing_hit((50.0, 50.0, 100.0, 100.0), gt)  # centre (75, 75) is outside
    assert pointing_hit((90.0, 90.0, 130.0, 130.0), gt)  # centre (110, 110) is inside


def _rows():
    return [
        {"id": "a", "label": "Lung_Opacity", "gt": [(100, 100, 300, 300)], "pred": [(110, 110, 290, 290)]},
        {"id": "b", "label": "Lung_Opacity", "gt": [(100, 100, 300, 300)], "pred": [(600, 600, 800, 800)]},
        {"id": "c", "label": "Lung_Opacity", "gt": [(100, 100, 300, 300)], "pred": []},
        {"id": "d", "label": "Normal", "gt": [], "pred": [(100, 100, 200, 200)]},
        {"id": "e", "label": "Normal", "gt": [], "pred": []},
    ]


def test_evaluate_counts_boxes_overlap_and_pointing_on_opacity_images_and_false_alarms_on_normals():
    out = evaluate(_rows(), size=1000, n_boot=50)
    op = out["opacity_images"]
    assert op["n"] == 3
    assert op["any_box"]["point"] == pytest.approx(2 / 3, abs=1e-3)
    assert op["pointing_hit"]["point"] == pytest.approx(1 / 3, abs=1e-3)
    assert op["iou_ge_0.5"]["point"] == pytest.approx(1 / 3, abs=1e-3)
    assert out["normal_images"]["n"] == 2 and out["normal_images"]["any_box"]["point"] == pytest.approx(0.5)


def test_evaluate_includes_the_fixed_central_box_baseline():
    out = evaluate(_rows(), size=1000, n_boot=50)
    assert "baseline_central_box" in out and set(out["baseline_central_box"]) >= {"pointing_hit", "iou_ge_0.1"}
    assert len(BASELINE_BOX) == 4 and BASELINE_BOX[0] < BASELINE_BOX[2] and BASELINE_BOX[1] < BASELINE_BOX[3]


def test_evaluate_with_no_images_is_not_an_error():
    assert evaluate([], size=1000, n_boot=10)["opacity_images"]["n"] == 0
