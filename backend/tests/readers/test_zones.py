import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from medproof.readers.zones import Anatomy, laterality_ok, name_region

H = W = 200


def _box(y0, y1, x0, x1):
    m = np.zeros((H, W), bool)
    m[y0:y1, x0:x1] = True
    return m


def anatomy():
    """Standard frontal view: patient's right lung is on the image LEFT, heart toward image right."""
    return Anatomy(
        right_lung=_box(20, 170, 20, 90),
        left_lung=_box(20, 170, 110, 180),
        heart=_box(100, 160, 85, 130),
    )


def test_image_left_blob_is_patient_right():
    r = name_region(_box(25, 40, 30, 60), anatomy())
    assert r.name == "right upper zone" and r.side == "right"


def test_image_right_blob_is_patient_left():
    r = name_region(_box(140, 160, 130, 170), anatomy())
    assert r.name == "left lower zone" and r.side == "left"


@pytest.mark.parametrize(
    "y0,y1,vertical",
    [(25, 45, "upper"), (85, 105, "middle"), (140, 165, "lower")],
)
def test_vertical_thirds_follow_lung_extent(y0, y1, vertical):
    assert name_region(_box(y0, y1, 30, 70), anatomy()).vertical == vertical


def test_thirds_use_each_lungs_own_extent():
    a = Anatomy(right_lung=_box(20, 80, 20, 90), left_lung=_box(60, 190, 110, 180), heart=_box(0, 0, 0, 0))
    # y=70 is the lower part of the short right lung but the upper part of the tall left lung
    assert name_region(_box(66, 74, 30, 60), a).vertical == "lower"
    assert name_region(_box(66, 74, 130, 160), a).vertical == "upper"


def test_cardiac_wins_when_heart_overlap_is_larger():
    r = name_region(_box(110, 150, 95, 128), anatomy())
    assert r.name == "cardiac region" and r.side == "cardiac" and r.vertical is None


def test_region_outside_all_anatomy_has_no_name():
    r = name_region(_box(0, 10, 0, 10), anatomy())
    assert r.name is None and r.side is None


def test_empty_region_and_missing_masks_are_safe():
    assert name_region(np.zeros((H, W), bool), anatomy()).name is None
    empty = Anatomy(np.zeros((H, W), bool), np.zeros((H, W), bool), np.zeros((H, W), bool))
    assert name_region(_box(20, 40, 20, 40), empty).name is None


def test_region_straddling_both_lungs_takes_the_larger_share():
    r = name_region(_box(30, 50, 60, 130), anatomy())  # 30 columns in right lung, 20 in left
    assert r.side == "right"


def test_laterality_ok_for_standard_view():
    assert laterality_ok(anatomy()) is True


def test_laterality_flags_flipped_image():
    flipped = Anatomy(
        right_lung=anatomy().right_lung[:, ::-1],
        left_lung=anatomy().left_lung[:, ::-1],
        heart=anatomy().heart[:, ::-1],
    )
    assert laterality_ok(flipped) is False


def test_laterality_unknown_without_heart():
    a = anatomy()
    assert laterality_ok(Anatomy(a.right_lung, a.left_lung, np.zeros((H, W), bool))) is None


@settings(max_examples=40, deadline=None)
@given(st.integers(0, 150), st.integers(0, 120))
def test_mirroring_keeps_vertical_zone_and_breaks_laterality_property(y, x):
    reg = _box(y, y + 20, x, x + 20)
    a = anatomy()
    m = Anatomy(a.right_lung[:, ::-1], a.left_lung[:, ::-1], a.heart[:, ::-1])
    r1, r2 = name_region(reg, a), name_region(reg[:, ::-1], m)
    assert r1.vertical == r2.vertical and r1.side == r2.side
    assert laterality_ok(a) and laterality_ok(m) is False


def test_heart_pixels_inside_a_lung_mask_count_as_heart():
    """Segmenter lung masks overlap the heart; a region over the heart must be cardiac."""
    a = anatomy()
    a = Anatomy(a.right_lung, a.left_lung | a.heart, a.heart)  # left lung mask swallows the heart
    r = name_region(_box(110, 150, 95, 128), a)
    assert r.name == "cardiac region"
    # a region in the lung proper, away from the heart, is still named by its lung
    assert name_region(_box(30, 50, 140, 170), a).name == "left upper zone"
