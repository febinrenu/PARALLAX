import numpy as np
import pytest

from medproof.core.schemas import Finding, ImageEvidence
from medproof.readers.base import ReaderFinding, ReaderOutput
from medproof.readers.cxr_config import CxrConfig
from medproof.readers.evidence import to_findings
from tests.conftest import png_bytes  # noqa: F401  (ensures fixtures module imports)

HW = (300, 500)


def _heat():
    h = np.zeros(HW, np.float32)
    h[60:120, 350:450] = 1.0
    return h


def _out(**kw):
    f1 = ReaderFinding(
        "Pneumonia", 0.9, 0.9, heatmap=_heat(), boxes_xyxy=[(350.0, 60.0, 450.0, 120.0)],
        box_scores=[1.0], method="gradcam++", region_name="right lower zone",
    )
    f2 = ReaderFinding(
        "Effusion", 0.7, 0.7, heatmap=_heat(), boxes_xyxy=[(10.0, 10.0, 60.0, 60.0)],
        box_scores=[1.0], method="gradcam++", region_name=None, flags=["anatomy_unavailable"],
    )
    base = dict(
        modality="cxr", model_id="xrv-densenet121-all@abcd1234", labels=["Pneumonia", "Effusion", "Mass"],
        logits=np.array([0.9, 0.7, 0.1], np.float32), probs=np.array([0.9, 0.7, 0.1], np.float32),
        findings=[f1, f2], image_hw=HW,
    )
    base.update(kw)
    return ReaderOutput(**base)


def test_findings_validate_against_the_contract(tmp_path):
    findings, _ = to_findings(_out(), CxrConfig(), artifact_dir=tmp_path)
    assert len(findings) == 2 and all(isinstance(f, Finding) for f in findings)
    for f in findings:
        Finding.model_validate(f.model_dump())
        assert f.modality == "cxr" and f.status == "uncertain"
        assert f.prob_calibrated == f.prob_raw and "uncalibrated" in f.flags
        assert f.conformal_set == []


def test_ids_are_sequential_and_respect_start_offsets(tmp_path):
    findings, _ = to_findings(_out(), CxrConfig(), finding_start=4, evidence_start=7, artifact_dir=tmp_path)
    assert [f.finding_id for f in findings] == ["f4", "f5"]
    assert [f.image_evidence[0].evidence_id for f in findings] == ["ie_7", "ie_8"]


def test_each_finding_has_one_evidence_with_model_method_box_and_region(tmp_path):
    findings, _ = to_findings(_out(), CxrConfig(), artifact_dir=tmp_path)
    ev = findings[0].image_evidence
    assert len(ev) == 1 and isinstance(ev[0], ImageEvidence)
    e = ev[0]
    assert e.kind == "heatmap" and e.method == "gradcam++"
    assert e.source_model == "xrv-densenet121-all@abcd1234"
    assert e.bbox_xyxy == (350.0, 60.0, 450.0, 120.0) and e.region_name == "right lower zone"
    assert e.faithful is None and e.faithfulness_drop is None  # filled by the faithfulness test later
    assert findings[1].image_evidence[0].region_name is None


def test_bbox_is_inside_the_image(tmp_path):
    for f in to_findings(_out(), CxrConfig(), artifact_dir=tmp_path)[0]:
        x0, y0, x1, y1 = f.image_evidence[0].bbox_xyxy
        assert 0 <= x0 < x1 <= HW[1] and 0 <= y0 < y1 <= HW[0]


def test_heatmap_png_is_written_on_the_image_grid(tmp_path):
    from PIL import Image

    findings, _ = to_findings(_out(), CxrConfig(), artifact_dir=tmp_path)
    ref = findings[0].image_evidence[0].heatmap_ref
    assert ref and (tmp_path / ref.split("/")[-1]).exists()
    im = Image.open(tmp_path / ref.split("/")[-1])
    assert im.size == (HW[1], HW[0]) and im.mode == "L"
    assert np.asarray(im)[90, 400] == 255 and np.asarray(im)[0, 0] == 0


def test_no_artifact_dir_means_no_heatmap_ref():
    findings, _ = to_findings(_out(), CxrConfig())
    assert findings[0].image_evidence[0].heatmap_ref is None


@pytest.mark.parametrize("prob,tier", [(0.95, "high"), (0.7, "moderate"), (0.55, "low")])
def test_tier_from_probability(prob, tier, tmp_path):
    out = _out()
    out.findings[0].prob_raw = prob
    assert to_findings(out, CxrConfig(), artifact_dir=tmp_path)[0][0].tier == tier


def test_study_flags_propagate_to_findings_and_warnings(tmp_path):
    out = _out(flags=["laterality_uncertain"], warnings=["heart on image left"])
    findings, warnings = to_findings(out, CxrConfig(), artifact_dir=tmp_path)
    assert all("laterality_uncertain" in f.flags for f in findings)
    assert "anatomy_unavailable" in findings[1].flags
    assert "heart on image left" in warnings


def test_finding_without_any_location_is_dropped_not_invented(tmp_path):
    out = _out()
    out.findings[1].boxes_xyxy = []
    out.findings[1].heatmap = None
    findings, warnings = to_findings(out, CxrConfig(), artifact_dir=tmp_path)
    assert [f.label for f in findings] == ["Pneumonia"]
    assert any("Effusion" in w for w in warnings)


def test_box_only_finding_uses_bbox_kind():
    out = _out()
    for f in out.findings:
        f.heatmap = None
    findings, _ = to_findings(out, CxrConfig())
    assert findings[0].image_evidence[0].kind == "bbox" and findings[0].image_evidence[0].heatmap_ref is None


def test_segmentation_mask_adds_a_second_evidence_with_its_own_id(tmp_path):
    from PIL import Image

    out = _out()
    m = np.zeros(HW, bool)
    m[70:110, 360:440] = True
    out.findings[0].mask, out.findings[0].mask_source = m, "unet"
    findings, _ = to_findings(out, CxrConfig(), artifact_dir=tmp_path)
    ev = findings[0].image_evidence
    assert [e.kind for e in ev] == ["heatmap", "mask"] and [e.evidence_id for e in ev] == ["ie_1", "ie_2"]
    assert ev[1].method == "unet" and ev[1].bbox_xyxy == ev[0].bbox_xyxy
    im = np.asarray(Image.open(tmp_path / "mask_f1.png"))
    assert im.shape == HW and im[90, 400] == 255 and im[0, 0] == 0
    assert findings[1].image_evidence[0].evidence_id == "ie_3"  # later findings continue the counter
    for f in findings:
        Finding.model_validate(f.model_dump())


def test_a_cam_region_mask_alone_does_not_become_mask_evidence(tmp_path):
    out = _out()
    out.findings[0].mask = np.ones(HW, bool)  # CAM region only: no mask_source
    findings, _ = to_findings(out, CxrConfig(), artifact_dir=tmp_path)
    assert len(findings[0].image_evidence) == 1
