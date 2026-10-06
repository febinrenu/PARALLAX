"""Scrub patient-identifying tags from a DICOM dataset on upload.

Follows a subset of the DICOM PS3.15 basic application-level confidentiality profile:
identifying tags are blanked or removed, dates are shifted by one random per-study offset,
UIDs are remapped consistently, and private tags are dropped. It does not touch pixel data,
so burned-in annotations are reported as a warning. This is a privacy default for a demo
tool, not a certified de-identification pipeline.

PatientAge and PatientSex are kept on purpose: the clinical-consistency checks need them.
"""

from __future__ import annotations

import io
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pydicom
from pydicom.dataset import Dataset
from pydicom.uid import generate_uid

# Blanked (element kept, value emptied) so the file stays a valid DICOM object.
BLANK = (
    "PatientName", "PatientID", "PatientBirthDate", "PatientBirthTime", "PatientAddress",
    "OtherPatientIDs", "OtherPatientNames", "PatientMotherBirthName", "PatientTelephoneNumbers",
    "InstitutionName", "InstitutionAddress", "InstitutionalDepartmentName",
    "ReferringPhysicianName", "PerformingPhysicianName", "OperatorsName",
    "PhysiciansOfRecord", "NameOfPhysiciansReadingStudy", "RequestingPhysician",
    "AccessionNumber", "StationName", "DeviceSerialNumber", "StudyID",
    "PatientInsurancePlanCodeSequence", "MedicalRecordLocator", "CountryOfResidence",
    "RegionOfResidence", "EthnicGroup", "Occupation", "PatientReligiousPreference",
)  # fmt: skip

# Removed outright: free text that routinely carries names, places and dates.
DELETE = (
    "PatientComments", "ImageComments", "StudyDescription", "SeriesDescription",
    "AdditionalPatientHistory", "AdmittingDiagnosesDescription", "RequestAttributesSequence",
    "RequestedProcedureDescription", "ReasonForTheRequestedProcedure",
    "PerformedProcedureStepDescription", "ProtocolName", "DerivationDescription",
)  # fmt: skip

UID_TAGS = ("StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID", "FrameOfReferenceUID")


@dataclass
class ScrubReceipt:
    blanked: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    dates_shifted: list[str] = field(default_factory=list)
    uids_remapped: list[str] = field(default_factory=list)
    private_tags_removed: int = 0
    burned_in_annotation: bool = False
    profile: str = "PS3.15 basic profile subset (not a certified de-identifier)"

    @property
    def tags_scrubbed(self) -> int:
        return (
            len(self.blanked)
            + len(self.removed)
            + len(self.dates_shifted)
            + len(self.uids_remapped)
            + self.private_tags_removed
        )

    def to_dict(self) -> dict:
        return {
            "tags_scrubbed": self.tags_scrubbed,
            "blanked": self.blanked,
            "removed": self.removed,
            "dates_shifted": self.dates_shifted,
            "uids_remapped": self.uids_remapped,
            "private_tags_removed": self.private_tags_removed,
            "burned_in_annotation": self.burned_in_annotation,
            "profile": self.profile,
        }


def _shift_date(value: str, days: int) -> str:
    d = datetime.strptime(value[:8], "%Y%m%d") - timedelta(days=days)
    return d.strftime("%Y%m%d") + value[8:]


def scrub_dataset(ds: Dataset, *, days_shift: int | None = None) -> tuple[Dataset, ScrubReceipt]:
    """Scrub `ds` in place and return it with a receipt of what changed."""
    receipt = ScrubReceipt()
    shift = days_shift if days_shift is not None else secrets.randbelow(3000) + 30
    salt = secrets.token_hex(8)
    uid_map: dict[str, str] = {}
    blank, delete, uid_tags = set(BLANK), set(DELETE), set(UID_TAGS)

    def remap(old: str) -> str:
        if old not in uid_map:
            uid_map[old] = str(generate_uid(entropy_srcs=[salt, old]))
        return uid_map[old]

    def visit(dataset: Dataset) -> None:
        for tag in list(dataset.keys()):
            elem = dataset[tag]
            if tag.is_private:
                continue  # dropped wholesale by remove_private_tags() below
            if elem.VR == "SQ":
                if elem.keyword in delete:
                    del dataset[tag]
                    receipt.removed.append(elem.keyword)
                    continue
                for item in elem.value:
                    visit(item)
                continue
            kw = elem.keyword
            if kw in delete:
                del dataset[tag]
                receipt.removed.append(kw)
            elif kw in blank or elem.VR == "PN":
                if elem.value not in ("", None, b""):
                    receipt.blanked.append(kw or str(tag))
                elem.value = ""
            elif elem.VR in ("DA", "DT") and elem.value:
                try:
                    elem.value = _shift_date(str(elem.value), shift)
                    receipt.dates_shifted.append(kw or str(tag))
                except ValueError:
                    elem.value = ""
                    receipt.blanked.append(kw or str(tag))
            elif kw in uid_tags and elem.value:
                elem.value = remap(str(elem.value))
                receipt.uids_remapped.append(kw)

    visit(ds)
    before = _count_private(ds)
    ds.remove_private_tags()
    receipt.private_tags_removed = before

    meta = getattr(ds, "file_meta", None)
    if meta is not None and "MediaStorageSOPInstanceUID" in meta and "SOPInstanceUID" in ds:
        meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    receipt.burned_in_annotation = str(getattr(ds, "BurnedInAnnotation", "")).upper() == "YES"
    return ds, receipt


def _count_private(ds: Dataset) -> int:
    n = 0
    for elem in ds:
        if elem.tag.is_private:
            n += 1
        elif elem.VR == "SQ":
            for item in elem.value:
                n += _count_private(item)
    return n


def scrub_bytes(raw: bytes, *, days_shift: int | None = None) -> tuple[bytes, ScrubReceipt]:
    """Scrub a DICOM file given as bytes. Returns the cleaned file bytes and the receipt."""
    ds = pydicom.dcmread(io.BytesIO(raw), force=True)
    ds, receipt = scrub_dataset(ds, days_shift=days_shift)
    out = io.BytesIO()
    ds.save_as(out)
    return out.getvalue(), receipt
