"""A heatmap that does not sit on the segmented tumour is flagged and cannot count as faithful evidence."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from medproof.readers import segmenter as SG  # noqa: E402
from medproof.readers.base import ReaderFinding, ReaderOutput  # noqa: E402
from medproof.readers.cxr_config import CxrConfig  # noqa: E402
from medproof.readers.evidence import to_findings  # noqa: E402
from medproof.verify import faithfulness as F  # noqa: E402

HW = (96, 128)


class UNet:
    min_area_frac = 0.001

    def segment(self, img):
        m = np.zeros(HW, bool)
        m[30:70, 60:100] = True
        return m


def _out(heat_box):
    heat = np.zeros(HW, np.float32)
    y0, y1, x0, x1 = heat_box
    heat[y0:y1, x0:x1] = 1.0
    f = ReaderFinding("glioma", 1.0, 0.9, heatmap=heat, boxes_xyxy=[(float(x0), float(y0), float(x1), float(y1))], method="gradcam++")
    return ReaderOutput(modality="brain_mri", model_id="m", labels=["glioma"], logits=np.zeros(1, np.float32),
                        probs=np.array([0.9], np.float32), findings=[f], image_hw=HW)


def test_share_of_top_heat_inside_the_mask():
    heat = np.zeros(HW, np.float32)
    heat[30:50, 60:80] = 1.0
    m = np.zeros(HW, bool)
    m[30:70, 60:100] = True
    assert SG.heat_share_in_mask(heat, m) == pytest.approx(1.0)
    left = np.zeros(HW, np.float32)
    left[:, :20] = 1.0
    assert SG.heat_share_in_mask(left, m) == pytest.approx(0.0)
    assert SG.heat_share_in_mask(np.zeros(HW, np.float32), m) is None  # a flat map says nothing


def test_heatmap_on_the_tumour_is_not_flagged():
    out = _out((35, 55, 65, 90))
    SG.attach_unet_masks(out, np.zeros(HW, np.float32), UNet())
    assert "heatmap_off_target" not in out.findings[0].flags


def test_heatmap_far_from_the_tumour_is_flagged():
    out = _out((60, 90, 0, 20))  # the lower-left corner, like a skull-edge shortcut
    SG.attach_unet_masks(out, np.zeros(HW, np.float32), UNet())
    assert "heatmap_off_target" in out.findings[0].flags


def test_off_target_heatmap_cannot_be_marked_faithful():
    out = _out((60, 90, 0, 20))
    SG.attach_unet_masks(out, np.zeros(HW, np.float32), UNet())
    findings, _ = to_findings(out, CxrConfig())
    assert "heatmap_off_target" in findings[0].flags

    class Reader:
        score = staticmethod(lambda img: np.array([float(img[60:90, 0:20].mean())], np.float32))  # truly depends on the corner

    img = np.full(HW, 0.3, np.float32)
    img[60:90, 0:20] = 0.9
    res = F.assess_output(Reader(), img, out, F.FaithfulnessConfig())
    assert res["glioma"].faithful is True  # the model does use that corner
    updated = F.apply_to_findings(findings, res)
    ev = updated[0].image_evidence[0]
    assert ev.faithful is False and "unfaithful" in updated[0].flags and "heatmap_off_target" in updated[0].flags
    assert ev.faithfulness_drop is not None  # the measured drop is kept for the audit view


def test_on_target_faithful_heatmap_stays_faithful():
    out = _out((35, 55, 65, 90))
    SG.attach_unet_masks(out, np.zeros(HW, np.float32), UNet())
    findings, _ = to_findings(out, CxrConfig())

    class Reader:
        score = staticmethod(lambda img: np.array([float(img[35:55, 65:90].mean())], np.float32))

    img = np.full(HW, 0.3, np.float32)
    img[35:55, 65:90] = 0.9
    updated = F.apply_to_findings(findings, F.assess_output(Reader(), img, out, F.FaithfulnessConfig()))
    assert updated[0].image_evidence[0].faithful is True and "unfaithful" not in updated[0].flags
