from types import SimpleNamespace

from medproof.core.schemas import StageResult
from medproof.intake.run import run, run_bytes
from tests.conftest import png_bytes, to_u8


def test_bad_bytes_give_failed_stage_result_not_exception():
    result, img = run_bytes(b"garbage")
    assert isinstance(result, StageResult)
    assert result.ok is False and img is None and result.warnings


def test_empty_upload_is_a_failed_stage():
    assert run_bytes(b"")[0].ok is False


def test_dicom_stage_reports_scrub_receipt_and_quality(dicom_phi):
    result, img = run_bytes(dicom_phi, "cxr")
    assert result.ok and img is not None and result.stage == "intake"
    assert result.payload["phi_scrub"]["tags_scrubbed"] > 0
    assert result.payload["image"]["sha256"] == img.sha256
    assert "quality" in result.payload
    assert "PHI_SENTINEL" not in str(result.payload)


def test_png_stage_has_no_scrub_block(phantom):
    result, _ = run_bytes(png_bytes(to_u8(phantom)), "cxr")
    assert result.ok and "phi_scrub" not in result.payload


def test_run_accepts_a_context_object(phantom):
    ctx = SimpleNamespace(raw_bytes=png_bytes(to_u8(phantom)), modality_hint="cxr")
    assert run(ctx).ok
    assert run(SimpleNamespace()).ok is False
