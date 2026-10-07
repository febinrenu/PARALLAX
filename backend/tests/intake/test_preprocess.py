import numpy as np
import pytest

from medproof.intake.decode import load_image
from medproof.intake.preprocess import SPECS, get_spec, prepare
from tests.conftest import png_bytes, to_u8


@pytest.mark.parametrize("name", list(SPECS))
def test_every_spec_produces_expected_shape(phantom, name):
    spec = get_spec(name)
    out = prepare(load_image(png_bytes(to_u8(phantom))), spec)
    assert out.shape == (spec.channels, spec.size, spec.size)
    assert out.dtype == np.float32 and np.isfinite(out).all()


def test_same_input_gives_same_tensor_train_and_serve(phantom):
    """Array input (training loader) and DecodedImage input (serving) must agree."""
    img = load_image(png_bytes(to_u8(phantom)))
    np.testing.assert_array_equal(prepare(img, "brain_effnet"), prepare(img.analysis, "brain_effnet"))


def test_xrv_range_is_minus1024_to_1024(phantom):
    out = prepare(phantom, "cxr_xrv")
    assert out.min() >= -1024.001 and out.max() <= 1024.001 and out.min() < -500 and out.max() > 500


def test_pad_mode_keeps_aspect(phantom):
    wide = phantom[:, :256]  # 512 high, 256 wide
    out = prepare(wide, "bone_yolo")
    assert out.shape == (3, 640, 640)
    assert np.all(out[:, :, :100] == 0) and np.all(out[:, :, -100:] == 0)  # padded columns


def test_grayscale_input_to_three_channels_and_back(phantom):
    assert prepare(phantom, "skin_cls").shape[0] == 3
    rgb = np.stack([phantom] * 3, axis=2)
    assert prepare(rgb, "cxr_xrv").shape[0] == 1


def test_unknown_spec():
    with pytest.raises(KeyError):
        get_spec("nope")


def test_center_crop_matches_torchxrayvision_rule():
    from medproof.intake.preprocess import center_crop_box

    assert center_crop_box(300, 500) == (0, 100, 300)
    assert center_crop_box(500, 300) == (100, 0, 300)
    assert center_crop_box(301, 500) == (0, 100, 301)  # same integer rule as xrv.XRayCenterCrop


def test_cxr_spec_crops_instead_of_stretching():
    wide = np.zeros((300, 500), np.float32)
    wide[:, 100:400] = 1.0  # exactly the central square
    out = prepare(wide, "cxr_xrv")
    assert out.min() > 1000  # every pixel the model sees is the bright centre


def test_router_spec_is_448_rgb_in_minus1_to_1(phantom):
    out = prepare(phantom, "router_medsiglip")
    assert out.shape == (3, 448, 448) and out.dtype == np.float32
    assert out.min() >= -1.0001 and out.max() <= 1.0001
    assert np.array_equal(out[0], out[1])  # grayscale replicated to three channels
    assert get_spec("router_medsiglip").mean == (0.5, 0.5, 0.5)
