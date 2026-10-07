"""U-Net / MedSAM mask attachment and the bone detector wrapper, with stand-in models."""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from medproof.core.schemas import Finding  # noqa: E402
from medproof.readers import segmenter as SG  # noqa: E402
from medproof.readers.base import ReaderFinding, ReaderOutput  # noqa: E402
from medproof.readers.bone import BoneReader  # noqa: E402
from medproof.readers.cxr_config import CxrConfig  # noqa: E402
from medproof.readers.evidence import to_findings  # noqa: E402
from medproof.segment.medsam import MedSAM  # noqa: E402
from tests.conftest import make_phantom  # noqa: E402

HW = (96, 128)


def _out(labels=("glioma", "notumor"), boxes=((20.0, 20.0, 60.0, 60.0), None)):
    findings = []
    for lab, box in zip(labels, boxes):
        findings.append(ReaderFinding(lab, 1.0, 0.8, heatmap=np.ones(HW, np.float32) if box else None,
                                      boxes_xyxy=[box] if box else [], method="gradcam++"))
    return ReaderOutput(modality="brain_mri", model_id="brain_cls@1", labels=list(labels), logits=np.zeros(len(labels), np.float32),
                        probs=np.full(len(labels), 0.8, np.float32), findings=findings, image_hw=HW)


class TinyUNet(nn.Module):
    """Predicts a bright square (logit +10 inside, -10 outside) on its own 64x64 grid."""

    def forward(self, x):
        assert x.shape[1:] == (3, 64, 64)
        out = torch.full((x.shape[0], 1, 64, 64), -10.0)
        out[:, :, 16:48, 24:56] = 10.0
        return out


def test_unet_segmenter_returns_a_mask_on_the_original_grid(tmp_path):
    seg = SG.UNetSegmenter(TinyUNet(), spec=SG.spec_from_meta({"name": "s", "size": 64, "channels": 3, "mean": [0, 0, 0], "std": [1, 1, 1], "resize": "stretch", "value_range": [0, 1]}))
    img = make_phantom(HW[0])[:, : HW[1]] if False else np.random.default_rng(0).random(HW).astype(np.float32)
    m = seg.segment(img)
    assert m.shape == HW and m.dtype == bool
    ys, xs = np.nonzero(m)
    assert abs(xs.min() / HW[1] - 24 / 64) < 0.05 and abs(xs.max() / HW[1] - 55 / 64) < 0.05
    assert abs(ys.min() / HW[0] - 16 / 64) < 0.05


def test_attach_unet_masks_only_to_positive_findings_with_a_big_enough_mask():
    seg = SG.UNetSegmenter(TinyUNet(), spec=SG.spec_from_meta({"name": "s", "size": 64, "channels": 3, "mean": [0, 0, 0], "std": [1, 1, 1], "resize": "stretch", "value_range": [0, 1]}))
    out = _out()
    SG.attach_unet_masks(out, np.zeros(HW, np.float32), seg, negative=("notumor",))
    pos, neg = out.findings
    assert pos.mask_source == "unet" and pos.mask.any() and pos.boxes_xyxy[0] != (20.0, 20.0, 60.0, 60.0)
    assert neg.mask_source == ""  # the negative class gets no tumour mask
    small = SG.UNetSegmenter(TinyUNet(), spec=seg.spec, min_area_frac=0.9)
    out2 = _out()
    SG.attach_unet_masks(out2, np.zeros(HW, np.float32), small)
    assert out2.findings[0].mask_source == "" and "mask_too_small" in out2.findings[0].flags


def test_attach_medsam_masks_use_the_finding_box_and_flag_empty_masks():
    def pred(rgb, box):
        m = np.zeros(rgb.shape[:2], np.float32)
        x0, y0, x1, y1 = (int(v) for v in box)
        m[y0 + 5 : y1 - 5, x0 + 5 : x1 - 5] = 0.9
        return m

    out = _out(labels=("mel", "nv"), boxes=((20.0, 20.0, 60.0, 60.0), (70.0, 30.0, 100.0, 70.0)))
    img = np.random.default_rng(1).random(HW).astype(np.float32)
    SG.attach_medsam_masks(out, img, MedSAM(pred))
    for rf in out.findings:
        assert rf.mask_source == "medsam" and rf.mask.shape == HW and rf.mask.any()
        assert rf.mask.sum() < (rf.boxes_xyxy[0][2] - rf.boxes_xyxy[0][0]) * (rf.boxes_xyxy[0][3] - rf.boxes_xyxy[0][1])
    empty = _out(labels=("mel",), boxes=((20.0, 20.0, 60.0, 60.0),))
    SG.attach_medsam_masks(empty, img, MedSAM(lambda rgb, b: np.zeros(rgb.shape[:2], np.float32)))
    assert empty.findings[0].mask_source == "" and "mask_empty" in empty.findings[0].flags


def test_mask_evidence_flows_into_contract_findings(tmp_path):
    out = _out(labels=("mel",), boxes=((20.0, 20.0, 60.0, 60.0),))
    SG.attach_medsam_masks(out, np.zeros(HW, np.float32), MedSAM(lambda rgb, b: np.pad(np.ones((30, 30), np.float32), ((25, 41), (25, 73)))))
    findings, _ = to_findings(out, CxrConfig(), artifact_dir=tmp_path)
    assert [e.kind for e in findings[0].image_evidence] == ["heatmap", "mask"]
    Finding.model_validate(findings[0].model_dump())


def test_unet_from_dir_checks_weights(tmp_path):
    d = tmp_path / "seg"
    d.mkdir()
    spec = {"name": "brain_seg", "size": 64, "channels": 3, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "resize": "stretch", "value_range": [0.0, 1.0]}
    (d / "model_meta.json").write_text(json.dumps({"model_id": "brain_seg@x", "task": "segmentation", "arch": "smp.Unet(resnet34)", "classes": ["background", "tumour"],
                                                    "preproc_spec": spec, "weights_file": "weights.pt", "weights_sha256": None, "threshold": 0.4}))
    with pytest.raises(SG.ModelBundleError):
        SG.UNetSegmenter.from_dir(d, model_factory=lambda enc: TinyUNet())  # no weights file
    torch.save(TinyUNet().state_dict(), d / "weights.pt")
    s = SG.UNetSegmenter.from_dir(d, model_factory=lambda enc: TinyUNet())
    assert s.threshold == 0.4 and s.spec.size == 64 and s.model_id == "brain_seg@x"
    assert SG.encoder_from_arch("smp.Unet(resnet34)") == "resnet34"


# ---- bone ----------------------------------------------------------------------------------
class Detector:
    def __init__(self, dets):
        self.dets, self.calls = dets, 0

    def __call__(self, image):
        self.calls += 1
        return self.dets


def test_bone_image_score_is_the_max_box_confidence_and_boxes_are_findings():
    det = Detector([((10.0, 10.0, 50.0, 40.0), 0.30), ((60.0, 20.0, 90.0, 70.0), 0.72)])
    r = BoneReader(det, model_id="bone_det@1", cfg=CxrConfig(positive_threshold=0.25))
    out = r.predict(np.zeros(HW, np.float32))
    assert out.modality == "bone_xray" and out.labels == ["fracture"] and out.probs[0] == pytest.approx(0.72)
    f = out.findings[0]
    assert f.label == "fracture" and f.prob_raw == pytest.approx(0.72) and f.heatmap is None
    assert f.boxes_xyxy[0] == (60.0, 20.0, 90.0, 70.0) and len(f.boxes_xyxy) == 2  # strongest box first
    assert f.method == "yolo"
    assert r.score(np.zeros(HW, np.float32))[0] == pytest.approx(0.72)


def test_bone_below_threshold_or_no_boxes_gives_no_finding():
    r = BoneReader(Detector([((10.0, 10.0, 50.0, 40.0), 0.10)]), model_id="m", cfg=CxrConfig(positive_threshold=0.25))
    assert r.predict(np.zeros(HW, np.float32)).findings == []
    assert BoneReader(Detector([]), model_id="m").predict(np.zeros(HW, np.float32)).probs[0] == 0.0


def test_bone_boxes_are_clipped_and_body_part_is_carried():
    r = BoneReader(Detector([((-5.0, -5.0, 500.0, 500.0), 0.9)]), model_id="m", cfg=CxrConfig(positive_threshold=0.25))
    out = r.predict(np.zeros(HW, np.float32), body_part="hand")
    assert out.findings[0].boxes_xyxy[0] == (0.0, 0.0, 128.0, 96.0)
    assert out.findings[0].region_name == "hand"
    findings, _ = to_findings(out, r.cfg)
    assert findings[0].image_evidence[0].kind == "bbox" and findings[0].image_evidence[0].region_name == "hand"
    Finding.model_validate(findings[0].model_dump())


def test_bone_stage_never_raises():
    from types import SimpleNamespace

    from medproof.readers import bone as B

    r = BoneReader(Detector([((10.0, 10.0, 50.0, 40.0), 0.8)]), model_id="m", cfg=CxrConfig(positive_threshold=0.25))
    res = B.run(SimpleNamespace(decoded=SimpleNamespace(analysis=np.zeros(HW, np.float32), display=np.zeros(HW, np.uint8), shape=HW), body_part="leg"), reader=r)
    assert res.ok and res.payload["findings"][0]["image_evidence"][0]["kind"] == "bbox"
    assert B.run(SimpleNamespace(), reader=r).ok is False
