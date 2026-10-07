"""The reader stages attach U-Net and MedSAM masks before building findings, and degrade when a mask model fails."""

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from medproof.readers import bone as B  # noqa: E402
from medproof.readers import classifier as C  # noqa: E402
from medproof.readers.base import ReaderFinding, ReaderOutput  # noqa: E402
from medproof.readers.cxr_config import CxrConfig  # noqa: E402
from medproof.segment.medsam import MedSAM  # noqa: E402

HW = (96, 128)
IMG = np.random.default_rng(0).random(HW).astype(np.float32)


def _output(modality="brain_mri", label="glioma", box=(20.0, 20.0, 60.0, 60.0), heat=True):
    f = ReaderFinding(label, 1.0, 0.8, heatmap=np.ones(HW, np.float32) if heat else None, boxes_xyxy=[box], method="gradcam++")
    return ReaderOutput(modality=modality, model_id="m@1", labels=[label], logits=np.zeros(1, np.float32),
                        probs=np.array([0.8], np.float32), findings=[f], image_hw=HW)


class FakeReader:
    def __init__(self, out):
        self.out, self.cfg = out, CxrConfig()

    def predict(self, img, *a):
        return self.out


class FakeUNet:
    min_area_frac = 0.001

    def segment(self, img):
        m = np.zeros(HW, bool)
        m[30:70, 50:100] = True
        return m


def _sam(empty=False):
    def pred(rgb, box):
        m = np.zeros(rgb.shape[:2], np.float32)
        if not empty:
            x0, y0, x1, y1 = (int(v) for v in box)
            m[y0 + 3 : y1 - 3, x0 + 3 : x1 - 3] = 0.9
        return m

    return MedSAM(pred)


def _ctx(tmp_path):
    return SimpleNamespace(decoded=SimpleNamespace(analysis=IMG, display=(IMG * 255).astype(np.uint8), shape=HW), artifact_dir=tmp_path)


def _kinds(res):
    return [e["kind"] for e in res.payload["findings"][0]["image_evidence"]]


def test_without_mask_models_nothing_changes(tmp_path):
    res = C.run(_ctx(tmp_path), reader=FakeReader(_output()))
    assert res.ok and _kinds(res) == ["heatmap"]


def test_unet_mask_reaches_the_finding_and_is_written_under_the_artifact_dir(tmp_path):
    res = C.run(_ctx(tmp_path), reader=FakeReader(_output()), unet=FakeUNet())
    assert res.ok and _kinds(res) == ["heatmap", "mask"]
    mask_ev = res.payload["findings"][0]["image_evidence"][1]
    assert mask_ev["method"] == "unet" and list(tmp_path.glob("*mask*"))


def test_medsam_mask_reaches_a_skin_finding(tmp_path):
    res = C.run(_ctx(tmp_path), reader=FakeReader(_output("skin_dermoscopy", "mel")), medsam=_sam())
    assert res.ok and _kinds(res) == ["heatmap", "mask"]
    assert res.payload["findings"][0]["image_evidence"][1]["method"] == "medsam"


def test_a_failing_mask_model_keeps_the_finding_and_warns(tmp_path):
    class Boom:
        def segment(self, *a):
            raise RuntimeError("gone")

    res = C.run(_ctx(tmp_path), reader=FakeReader(_output()), unet=Boom(), medsam=Boom())
    assert res.ok and _kinds(res) == ["heatmap"]
    assert any("mask" in w.lower() for w in res.warnings)
    assert "gone" not in " ".join(res.warnings)  # no internals leak into user-visible warnings


def test_an_empty_medsam_mask_is_flagged_not_attached(tmp_path):
    res = C.run(_ctx(tmp_path), reader=FakeReader(_output("skin_dermoscopy", "mel")), medsam=_sam(empty=True))
    assert _kinds(res) == ["heatmap"] and "mask_empty" in res.payload["findings"][0]["flags"]


def test_bone_boxes_can_be_refined_by_medsam(tmp_path):
    out = _output("bone_xray", "fracture", heat=False)
    out.findings[0].method = "yolo"
    ctx = _ctx(tmp_path)
    ctx.body_part = "hand"
    res = B.run(ctx, reader=FakeReader(out), medsam=_sam())
    assert res.ok and _kinds(res) == ["bbox", "mask"]


def test_a_cam_region_mask_does_not_block_the_medsam_mask(tmp_path):
    out = _output("skin_dermoscopy", "mel")
    out.findings[0].mask = np.ones(HW, bool)  # what the classifier reader leaves from the heatmap's largest region
    res = C.run(_ctx(tmp_path), reader=FakeReader(out), medsam=_sam())
    assert _kinds(res) == ["heatmap", "mask"] and res.payload["findings"][0]["image_evidence"][1]["method"] == "medsam"


def test_medsam_does_not_overwrite_a_unet_mask(tmp_path):
    res = C.run(_ctx(tmp_path), reader=FakeReader(_output()), unet=FakeUNet(), medsam=_sam())
    assert [e["method"] for e in res.payload["findings"][0]["image_evidence"] if e["kind"] == "mask"] == ["unet"]
