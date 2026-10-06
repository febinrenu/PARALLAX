from __future__ import annotations

import numpy as np
import pytest

from ml.eval import bootstrap as bs
from ml.eval import detection as det


def _box(x, y, w=0.2, h=0.2):
    return [x, y, x + w, y + h]


def test_iou_known_values():
    a = np.array([[0, 0, 1, 1]])
    assert det.iou_matrix(a, np.array([[0, 0, 1, 1]]))[0, 0] == 1.0
    assert det.iou_matrix(a, np.array([[0.5, 0, 1.5, 1]]))[0, 0] == pytest.approx(1 / 3)
    assert det.iou_matrix(a, np.array([[2, 2, 3, 3]]))[0, 0] == 0.0


def test_perfect_detections_give_ap_one():
    gts = [np.array([_box(0.1, 0.1)]), np.array([_box(0.5, 0.5)])]
    dets = [np.array([_box(0.1, 0.1) + [0.9]]), np.array([_box(0.5, 0.5) + [0.8]])]
    assert det.average_precision(dets, gts) == pytest.approx(1.0)


def test_false_positive_ranked_first_halves_ap():
    gts = [np.array([_box(0.1, 0.1)])]
    dets = [np.array([_box(0.7, 0.7) + [0.9], _box(0.1, 0.1) + [0.5]])]
    assert det.average_precision(dets, gts) == pytest.approx(0.5)


def test_missed_box_caps_recall_and_duplicates_count_as_false_positives():
    gts = [np.array([_box(0.1, 0.1), _box(0.6, 0.6)])]
    dets = [np.array([_box(0.1, 0.1) + [0.9], _box(0.1, 0.1) + [0.8]])]  # second one duplicates the first GT
    assert det.average_precision(dets, gts) == pytest.approx(0.5)  # recall 0.5, precision 1 until the duplicate


def test_no_gt_is_nan_and_no_dets_is_zero():
    assert np.isnan(det.average_precision([np.zeros((0, 5))], [np.zeros((0, 4))]))
    assert det.average_precision([np.zeros((0, 5))], [np.array([_box(0, 0)])]) == 0.0


def test_image_scores_use_max_confidence():
    s = det.image_scores([np.array([[0, 0, 1, 1, 0.3], [0, 0, 1, 1, 0.7]]), np.zeros((0, 5))])
    assert s.tolist() == [0.7, 0.0]


def test_ap_works_as_a_bootstrap_statistic_over_object_arrays():
    rng = np.random.default_rng(0)
    dets, gts = np.empty(60, object), np.empty(60, object)
    for i in range(60):
        g = np.array([_box(*rng.uniform(0.1, 0.6, 2))])
        gts[i] = g
        hit = rng.uniform() < 0.8
        dets[i] = np.array([list(g[0] + rng.normal(0, 0.01, 4)) + [rng.uniform(0.4, 1)]]) if hit else np.zeros((0, 5))
    c = bs.ci({"dets": dets, "gts": gts}, lambda dets, gts: det.average_precision(list(dets), list(gts)), B=100)
    assert 0.6 < c["point"] < 0.95 and c["lo"] < c["point"] < c["hi"]
