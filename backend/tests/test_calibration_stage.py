"""P2's serving-time calibration as a pipeline stage: multi-class (brain, skin) and binary (bone)
calibrators from the model bundles; chest stays uncalibrated because the product weights saw RSNA."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from medproof import pipeline
from medproof.calibrate.abstain import TierRule
from medproof.calibrate.binary import BinaryCalibrator
from medproof.calibrate.calibrator import Calibrator
from medproof.core.config import PipelineConfig
from medproof.core.schemas import Finding, ImageEvidence, StageResult
from medproof.core.status_rule import compute_status

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]


def _finding(fid: str, label: str, prob: float, modality: str = "brain_mri") -> Finding:
    return Finding(
        finding_id=fid, modality=modality, label=label, prob_raw=prob, prob_calibrated=prob, conformal_set=[],
        coverage_target=0.9, tier="moderate", status="uncertain", flags=["uncalibrated"],
        image_evidence=[ImageEvidence(evidence_id=f"e{fid}", kind="heatmap", source_model="m", method="gradcam", faithful=True)],
    )


def _bundle(tmp_path, modality: str) -> PipelineConfig:
    root = tmp_path / "models"
    if modality == "bone_xray":
        d = root / "bone_det"
        d.mkdir(parents=True)
        BinaryCalibrator("fracture", a=1.0, b=0.0, alpha=0.1, thr_fnr=0.2, abstain_low=0.4, abstain_high=0.6).save(d / "calibration.json")
    else:
        d = root / pipeline.READER_BUNDLES[modality]
        d.mkdir(parents=True)
        rule = TierRule(t_low=0.3, t_moderate=0.6, t_high=0.85, max_set_size=2)
        Calibrator(CLASSES, temperature=2.0, alpha=0.1, qhat=0.95, tier_rule=rule, method="aps").save(d / "calibration.json")
    return PipelineConfig(models_root=root)


def _ctx(modality: str, findings: list[Finding], logits=None, config=None):
    out = SimpleNamespace(labels=CLASSES, logits=np.asarray(logits, np.float32) if logits is not None else None)
    reader = StageResult(stage="reader", ok=True, ms=0, payload={"findings": [f.model_dump() for f in findings]})
    return SimpleNamespace(modality_hint=modality, reader_output=out, stage_results=[reader], config=config, decoded=object())


@pytest.fixture(autouse=True)
def _fresh_calibrators():
    pipeline._CALIBRATORS.clear()
    yield
    pipeline._CALIBRATORS.clear()


def test_calibration_runs_after_verification_and_before_the_report():
    names = [s.name for s in pipeline.PIPELINE]
    assert names.index("stability") < names.index("calibration")
    if "report" in names:
        assert names.index("calibration") < names.index("report")  # the report reads the calibrated tier


def test_multiclass_findings_get_calibrated_probability_set_and_tier(tmp_path):
    cfg = _bundle(tmp_path, "brain_mri")
    logits = [4.0, 1.0, 0.0, -1.0]
    result = pipeline._calibration_step(_ctx("brain_mri", [_finding("f1", "glioma", 0.93)], logits, cfg))
    assert result.ok
    f = Finding.model_validate(result.payload["findings"][0])
    expected = Calibrator.load(cfg.models_root / "brain_cls" / "calibration.json").calibrate(np.array(logits))
    assert f.prob_calibrated == pytest.approx(float(expected.probs[0][0]), abs=1e-6)
    assert f.conformal_set == expected.conformal_sets[0]
    assert f.tier == expected.tiers[0]
    assert f.coverage_target == pytest.approx(0.9)
    assert "uncalibrated" not in f.flags


def test_binary_findings_use_platt_scaling_and_the_abstention_band(tmp_path):
    cfg = _bundle(tmp_path, "bone_xray")
    result = pipeline._calibration_step(_ctx("bone_xray", [_finding("f1", "fracture", 0.5, "bone_xray")], config=cfg))
    f = Finding.model_validate(result.payload["findings"][0])
    assert f.prob_calibrated == pytest.approx(0.5, abs=1e-6)  # a=1, b=0 leaves the score unchanged
    assert f.tier == "abstain"  # inside the 0.4 to 0.6 band
    assert "uncalibrated" not in f.flags


def test_chest_stays_uncalibrated_and_says_why(tmp_path):
    result = pipeline._calibration_step(_ctx("cxr", [_finding("f1", "Pneumonia", 0.7, "cxr")], config=PipelineConfig(models_root=tmp_path)))
    assert not result.ok
    assert "RSNA" in result.warnings[0]
    assert "findings" not in result.payload  # nothing changed, so nothing to merge


def test_a_missing_calibration_file_degrades_with_the_reason(tmp_path):
    result = pipeline._calibration_step(_ctx("skin_dermoscopy", [_finding("f1", "nv", 0.8, "skin_dermoscopy")], [1, 0, 0, 0], PipelineConfig(models_root=tmp_path)))
    assert not result.ok and "calibration" in result.warnings[0]


def test_class_order_mismatch_is_refused(tmp_path):
    cfg = _bundle(tmp_path, "brain_mri")
    ctx = _ctx("brain_mri", [_finding("f1", "glioma", 0.9)], [1, 0, 0, 0], cfg)
    ctx.reader_output.labels = list(reversed(CLASSES))
    result = pipeline._calibration_step(ctx)
    assert not result.ok and "class order" in result.warnings[0]


def test_abstain_tier_is_never_verified():
    f = _finding("f1", "glioma", 0.9).model_copy(update={"tier": "abstain", "conformal_set": ["glioma"]})
    assert compute_status(f) == "uncertain"
