import io

import pydicom

from medproof.intake.decode import load_image
from medproof.intake.phi import scrub_bytes
from tests.conftest import PHI_SENTINEL


def test_no_phi_survives_in_serialized_file(dicom_phi):
    assert PHI_SENTINEL.encode() in dicom_phi  # the fixture really carries PHI
    clean, receipt = scrub_bytes(dicom_phi)
    assert PHI_SENTINEL.encode() not in clean
    for needle in (b"JANE", b"19700102", b"Jane"):
        assert needle not in clean
    assert receipt.tags_scrubbed > 0


def test_nested_and_private_tags_are_removed(dicom_phi):
    clean, receipt = scrub_bytes(dicom_phi)
    ds = pydicom.dcmread(io.BytesIO(clean))
    assert not any(e.tag.is_private for e in ds)
    seq = ds.get("ReferringPhysicianIdentificationSequence")
    for item in seq or []:
        assert not any(e.tag.is_private for e in item)
        assert item.get("ReferringPhysicianName", "") == ""
    assert receipt.private_tags_removed >= 2


def test_identifying_tags_are_blank_and_descriptions_removed(dicom_phi):
    clean, _ = scrub_bytes(dicom_phi)
    ds = pydicom.dcmread(io.BytesIO(clean))
    for kw in ("PatientName", "PatientID", "PatientBirthDate", "InstitutionName",
               "ReferringPhysicianName", "AccessionNumber", "StudyID"):
        assert str(ds.get(kw, "")) == "", kw
    assert "StudyDescription" not in ds and "PatientComments" not in ds


def test_age_and_sex_are_kept_for_clinical_checks(dicom_phi):
    clean, _ = scrub_bytes(dicom_phi)
    ds = pydicom.dcmread(io.BytesIO(clean))
    assert ds.PatientAge == "055Y" and ds.PatientSex == "F"


def test_dates_shift_by_one_consistent_offset(dicom_phi):
    clean, receipt = scrub_bytes(dicom_phi, days_shift=100)
    ds = pydicom.dcmread(io.BytesIO(clean))
    assert ds.StudyDate == "20231206"  # 2024-03-15 minus 100 days
    assert ds.AcquisitionDate == "20231207"
    assert ds.AcquisitionDateTime == "20231207101500"
    assert ds.StudyTime == "101500"  # times are kept
    assert "StudyDate" in receipt.dates_shifted
    assert str(100) not in str(receipt.to_dict())  # offset is not disclosed


def test_uids_are_remapped_consistently(dicom_phi):
    before = pydicom.dcmread(io.BytesIO(dicom_phi))
    clean, receipt = scrub_bytes(dicom_phi)
    after = pydicom.dcmread(io.BytesIO(clean))
    assert after.StudyInstanceUID != before.StudyInstanceUID
    assert after.SOPInstanceUID != before.SOPInstanceUID
    assert after.file_meta.MediaStorageSOPInstanceUID == after.SOPInstanceUID
    assert {"StudyInstanceUID", "SOPInstanceUID"} <= set(receipt.uids_remapped)


def test_pixels_are_untouched_and_still_decode(dicom_phi):
    clean, _ = scrub_bytes(dicom_phi)
    a, b = load_image(dicom_phi), load_image(clean)
    assert (a.display == b.display).all()


def test_burned_in_annotation_is_reported(phantom):
    from tests.conftest import make_dicom

    raw = make_dicom(phantom, phi=True)
    ds = pydicom.dcmread(io.BytesIO(raw))
    ds.BurnedInAnnotation = "YES"
    buf = io.BytesIO()
    ds.save_as(buf)
    _, receipt = scrub_bytes(buf.getvalue())
    assert receipt.burned_in_annotation


def test_scrubbing_twice_is_safe(dicom_phi):
    once, _ = scrub_bytes(dicom_phi)
    twice, receipt = scrub_bytes(once)
    assert PHI_SENTINEL.encode() not in twice
    assert receipt.blanked == []  # nothing left to blank the second time
