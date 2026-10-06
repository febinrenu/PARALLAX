import hashlib
import io

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from PIL import Image

from medproof.intake.decode import DecodeError, is_dicom, load_image
from tests.conftest import jpeg_bytes, make_dicom, png_bytes, to_u8


def test_dicom_detected(dicom_mono2):
    assert is_dicom(dicom_mono2)


def test_mono2_decodes_with_metadata(dicom_mono2):
    img = load_image(dicom_mono2)
    assert img.source_format == "dicom"
    assert img.photometric == "MONOCHROME2"
    assert img.display.dtype == np.uint8 and img.display.shape == (512, 512)
    assert img.analysis.dtype == np.float32
    assert 0.0 <= img.analysis.min() and img.analysis.max() <= 1.0
    assert img.pixel_spacing == pytest.approx((0.143, 0.143))
    assert img.bits_stored == 12


def test_monochrome1_is_inverted_to_match_monochrome2(dicom_mono1, dicom_mono2):
    """The same anatomy stored as MONOCHROME1 must display identically to MONOCHROME2."""
    a, b = load_image(dicom_mono2), load_image(dicom_mono1)
    assert b.photometric == "MONOCHROME1"
    assert np.abs(a.display.astype(int) - b.display.astype(int)).max() <= 1
    assert np.abs(a.analysis - b.analysis).max() < 2e-3
    # Raw stored values really are opposite, so a missing inversion would fail the above.
    import pydicom

    raw1 = pydicom.dcmread(io.BytesIO(dicom_mono1)).pixel_array.astype(int)
    raw2 = pydicom.dcmread(io.BytesIO(dicom_mono2)).pixel_array.astype(int)
    assert np.abs((raw1 + raw2) - 4095).max() <= 1


def test_monochrome1_bright_marker_stays_bright(phantom, dicom_mono1):
    img = load_image(dicom_mono1)
    marker = img.display[int(512 * 0.12) : int(512 * 0.16), int(512 * 0.14) : int(512 * 0.20)]
    lung = img.display[int(512 * 0.45) : int(512 * 0.50), int(512 * 0.30) : int(512 * 0.38)]
    assert marker.mean() > lung.mean() + 100


def test_rescale_slope_and_intercept_are_applied(phantom):
    plain = load_image(make_dicom(phantom, window=None))
    scaled = load_image(make_dicom(phantom, slope=2.0, intercept=-100.0, window=None))
    # A linear modality LUT must not change the normalised analysis copy or the percentile window.
    assert np.abs(plain.analysis - scaled.analysis).max() < 2e-3
    assert np.abs(plain.display.astype(int) - scaled.display.astype(int)).max() <= 2


def test_window_tags_used_when_present_and_percentile_fallback_otherwise(phantom):
    with_tags = load_image(make_dicom(phantom))
    no_tags = load_image(make_dicom(phantom, window=None))
    assert with_tags.window == pytest.approx((1023.5, 4095.0))
    assert not any("window" in w for w in with_tags.warnings)
    assert any("window" in w for w in no_tags.warnings)
    assert no_tags.display.max() == 255  # percentile window stretches to full range


def test_multiframe_uses_first_frame_and_warns(phantom):
    img = load_image(make_dicom(phantom, frames=3))
    assert img.display.shape == (512, 512)
    assert any("multi-frame" in w for w in img.warnings)


def test_sha256_is_of_raw_bytes(dicom_mono2):
    assert load_image(dicom_mono2).sha256 == hashlib.sha256(dicom_mono2).hexdigest()


def test_png_8bit_roundtrip(phantom):
    u8 = to_u8(phantom)
    img = load_image(png_bytes(u8))
    assert img.source_format == "png" and not img.is_color
    np.testing.assert_array_equal(img.display, u8)
    assert img.analysis.max() <= 1.0


def test_png_16bit_keeps_full_range(phantom):
    u16 = np.rint(phantom * 65535).astype(np.uint16)
    img = load_image(png_bytes(u16))
    assert img.bits_stored == 16
    assert img.display.dtype == np.uint8
    expected = (phantom - phantom.min()) / (phantom.max() - phantom.min())  # analysis is min-max scaled
    assert np.abs(img.analysis - expected).max() < 2e-3


def test_jpeg_exif_orientation_is_applied(phantom):
    wide = to_u8(phantom)[:, :400]  # 512 x 400
    raw = jpeg_bytes(wide, orientation=6)  # rotate 90 deg clockwise on display
    img = load_image(raw)
    assert img.source_format == "jpeg"
    assert img.shape == (400, 512)


def test_rgb_with_identical_channels_is_treated_as_gray(phantom):
    g = to_u8(phantom)
    img = load_image(png_bytes(np.stack([g, g, g], axis=2)))
    assert not img.is_color and img.display.ndim == 2


def test_colour_image_stays_colour(phantom):
    g = to_u8(phantom)
    rgb = np.stack([g, np.roll(g, 9, axis=0), 255 - g], axis=2)
    img = load_image(png_bytes(rgb))
    assert img.is_color and img.display.shape == (512, 512, 3) and img.analysis.shape == (512, 512, 3)


@pytest.mark.parametrize("raw", [b"", b"not an image at all", b"\x89PNG\r\n\x1a\n" + b"\x00" * 20])
def test_bad_input_raises_decode_error(raw):
    with pytest.raises(DecodeError):
        load_image(raw)


def test_truncated_dicom_raises_decode_error(dicom_mono2):
    with pytest.raises(DecodeError):
        load_image(dicom_mono2[:300])


def test_missing_file_raises_decode_error(tmp_path):
    with pytest.raises(DecodeError):
        load_image(tmp_path / "nope.png")


def test_path_input(tmp_path, phantom):
    p = tmp_path / "x.png"
    p.write_bytes(png_bytes(to_u8(phantom)))
    assert load_image(p).shape == (512, 512)


@settings(max_examples=25, deadline=None)
@given(
    st.integers(8, 48),
    st.integers(8, 48),
    st.integers(0, 2**31 - 1),
)
def test_random_png_roundtrip_property(h, w, seed):
    arr = np.random.default_rng(seed).integers(0, 256, (h, w), dtype=np.uint8)
    img = load_image(png_bytes(arr))
    np.testing.assert_array_equal(img.display, arr)
    assert img.analysis.dtype == np.float32
    assert 0.0 <= img.analysis.min() and img.analysis.max() <= 1.0


@settings(max_examples=20, deadline=None)
@given(st.integers(0, 2**31 - 1))
def test_mono1_analysis_is_involution_of_mono2_property(seed):
    rng = np.random.default_rng(seed)
    base = np.clip(rng.random((24, 24)).astype(np.float32), 0, 1)
    base[0, 0], base[0, 1] = 0.0, 1.0  # pin the range so min-max normalisation is identical
    a = load_image(make_dicom(base, photometric="MONOCHROME2"))
    b = load_image(make_dicom(base, photometric="MONOCHROME1"))
    assert np.abs(a.analysis - b.analysis).max() < 2e-3
