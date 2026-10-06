"""Synthetic fixtures shared by intake and verify tests. Nothing here is a real patient image."""

from __future__ import annotations

import io

import cv2
import numpy as np
import pydicom
import pytest
from PIL import Image
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

PHI_SENTINEL = "PHI_SENTINEL"


def make_phantom(size: int = 512, seed: int = 7) -> np.ndarray:
    """Chest-like phantom, float32 in [0, 1], mirror-symmetric except for one small marker."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
    img = 0.06 + 0.04 * yy  # table / background gradient
    body = (((xx - 0.5) / 0.40) ** 2 + ((yy - 0.52) / 0.46) ** 2) <= 1.0
    img = np.where(body, 0.58, img)
    for cx in (0.34, 0.66):  # two lung fields
        lung = (((xx - cx) / 0.13) ** 2 + ((yy - 0.46) / 0.27) ** 2) <= 1.0
        img = np.where(lung, 0.24, img)
    img = np.where(body & (np.abs(xx - 0.5) < 0.025), 0.82, img)  # spine
    ribs = 0.05 * np.sin(yy * 2 * np.pi * 9) * body
    tex = cv2.GaussianBlur(rng.normal(0, 1, (size, size)).astype(np.float32), (0, 0), 2.0) * 0.06
    img = img + ribs * (1 - np.abs(xx - 0.5)) + tex * body
    img[int(size * 0.12) : int(size * 0.16), int(size * 0.14) : int(size * 0.20)] = 0.95  # marker
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def make_dicom(
    phantom: np.ndarray,
    *,
    photometric: str = "MONOCHROME2",
    slope: float = 1.0,
    intercept: float = -1024.0,
    window: tuple[float, float] | None = (1023.5, 4095.0),
    frames: int = 1,
    phi: bool = False,
) -> bytes:
    """Build an explicit-VR 12-bit DICOM file whose Hounsfield-style values follow phantom."""
    stored = np.rint(phantom * 4095.0)
    if photometric == "MONOCHROME1":
        stored = 4095.0 - stored
    stored = np.rint(stored / slope).astype(np.uint16)

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.1.1"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.Modality = "DX"
    ds.PhotometricInterpretation = photometric
    ds.SamplesPerPixel = 1
    ds.Rows, ds.Columns = stored.shape
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 12, 11, 0
    ds.RescaleSlope, ds.RescaleIntercept = slope, intercept
    ds.PixelSpacing = [0.143, 0.143]
    if window is not None:
        ds.WindowCenter, ds.WindowWidth = window
    if frames > 1:
        ds.NumberOfFrames = frames
        ds.PixelData = np.stack([stored] * frames).tobytes()
    else:
        ds.PixelData = stored.tobytes()
    if phi:
        add_phi(ds)
    buf = io.BytesIO()
    ds.save_as(buf)
    return buf.getvalue()


def add_phi(ds) -> None:
    s = PHI_SENTINEL
    ds.PatientName = f"{s}_NAME^JANE"
    ds.PatientID = f"{s}_ID123"
    ds.PatientBirthDate = "19700102"
    ds.PatientAge = "055Y"
    ds.PatientSex = "F"
    ds.OtherPatientIDs = f"{s}_OTHERID"
    ds.PatientAddress = f"{s}_ADDRESS"
    ds.InstitutionName = f"{s}_HOSPITAL"
    ds.InstitutionAddress = f"{s}_STREET"
    ds.ReferringPhysicianName = f"{s}_REFERRER^DR"
    ds.PerformingPhysicianName = f"{s}_PERFORMER^DR"
    ds.OperatorsName = f"{s}_OPERATOR"
    ds.AccessionNumber = f"{s}_ACC"
    ds.StationName = f"{s}_STATION"
    ds.StudyID = f"{s}_STUDYID"
    ds.StudyDescription = f"{s}_STUDYDESC chest for Jane"
    ds.PatientComments = f"{s}_COMMENT"
    ds.StudyDate, ds.SeriesDate, ds.AcquisitionDate = "20240315", "20240315", "20240316"
    ds.StudyTime = "101500"
    ds.AcquisitionDateTime = "20240316101500"
    ds.BurnedInAnnotation = "NO"
    ds.add_new(0x00091001, "LO", f"{s}_PRIVATE")
    from pydicom.dataset import Dataset
    from pydicom.sequence import Sequence

    item = Dataset()
    item.ReferringPhysicianName = f"{s}_NESTED^DR"
    item.add_new(0x00111001, "LO", f"{s}_NESTED_PRIVATE")
    ds.ReferringPhysicianIdentificationSequence = Sequence([item])


def png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def jpeg_bytes(arr: np.ndarray, orientation: int | None = None, quality: int = 92) -> bytes:
    im = Image.fromarray(arr)
    buf = io.BytesIO()
    if orientation is not None:
        exif = Image.Exif()
        exif[0x0112] = orientation
        im.save(buf, format="JPEG", quality=quality, exif=exif)
    else:
        im.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def to_u8(img: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(img, 0, 1) * 255).astype(np.uint8)


@pytest.fixture(scope="session")
def phantom() -> np.ndarray:
    return make_phantom()


@pytest.fixture(scope="session")
def dicom_mono2(phantom) -> bytes:
    return make_dicom(phantom, photometric="MONOCHROME2")


@pytest.fixture(scope="session")
def dicom_mono1(phantom) -> bytes:
    return make_dicom(phantom, photometric="MONOCHROME1")


@pytest.fixture(scope="session")
def dicom_phi(phantom) -> bytes:
    return make_dicom(phantom, phi=True)


def _load_pydicom(raw: bytes):
    return pydicom.dcmread(io.BytesIO(raw))
