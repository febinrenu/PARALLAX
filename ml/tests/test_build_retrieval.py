"""build_retrieval must open RSNA's DICOM files and, for RSNA, index the calibration split."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pydicom")
pytest.importorskip("pandas")

from PIL import Image  # noqa: E402

from ml.eval_p3.build_retrieval import INDEX_SPLIT, open_image  # noqa: E402


def _dicom_bytes(path):
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.1"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    arr = (np.linspace(0, 4000, 64 * 64).reshape(64, 64)).astype(np.uint16)
    ds.Rows, ds.Columns = 64, 64
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 12, 11, 0
    ds.PixelData = arr.tobytes()
    ds.is_little_endian, ds.is_implicit_VR = True, False
    pydicom.dcmwrite(str(path), ds)


def test_open_image_reads_a_dicom_as_a_displayable_rgb_image(tmp_path):
    path = tmp_path / "x.dcm"
    _dicom_bytes(path)
    im = open_image(path)
    assert isinstance(im, Image.Image) and im.mode == "RGB" and im.size == (64, 64)
    assert len(set(np.asarray(im).ravel().tolist())) > 10  # a gradient, not a flat or blank image


def test_open_image_reads_ordinary_images_unchanged(tmp_path):
    path = tmp_path / "x.png"
    Image.fromarray(np.full((8, 8), 120, np.uint8)).save(path)
    assert open_image(path).size == (8, 8)


def test_rsna_is_indexed_from_the_calibration_split_and_the_others_from_train():
    assert INDEX_SPLIT["rsna"] == "cal"
    assert INDEX_SPLIT.get("ham10000", "train") == "train"


def test_thumbnails_are_small_pngs_named_for_the_api(tmp_path):
    from ml.eval_p3.build_retrieval import safe_thumbnail_name, thumbnail

    src = tmp_path / "x.png"
    Image.fromarray(np.tile(np.arange(256, dtype=np.uint8), (200, 1))).save(src)
    dst = tmp_path / "t" / "IMG1.png"
    thumbnail(src, dst)
    im = Image.open(dst)
    assert im.format == "PNG" and max(im.size) <= 128 and dst.stat().st_size < 20_000
    assert safe_thumbnail_name("IMG0002239") and safe_thumbnail_name("Tr_glioma_Tr-gl_323")
    assert not safe_thumbnail_name("../x") and not safe_thumbnail_name("a b") and not safe_thumbnail_name("é")


def test_only_licence_cleared_datasets_get_thumbnails_written(tmp_path):
    from ml.eval_p3.build_retrieval import thumbnail_datasets

    assert thumbnail_datasets() == {"fracatlas", "brain_mri"}
