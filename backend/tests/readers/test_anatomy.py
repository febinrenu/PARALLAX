import numpy as np
import pytest

torch = pytest.importorskip("torch")

from medproof.readers.anatomy import AnatomySegmenter  # noqa: E402
from medproof.readers.zones import laterality_ok  # noqa: E402

TARGETS = [
    "Left Clavicle", "Right Clavicle", "Left Scapula", "Right Scapula", "Left Lung", "Right Lung",
    "Left Hilus Pulmonis", "Right Hilus Pulmonis", "Heart", "Aorta", "Facies Diaphragmatica",
    "Mediastinum", "Weasand", "Spine",
]  # fmt: skip


class StubPsp(torch.nn.Module):
    """Returns fixed logits on the 512 grid: right lung image-left, left lung image-right, heart low centre."""

    targets = TARGETS

    def forward(self, x):
        assert x.shape == (1, 1, 512, 512)
        out = torch.full((1, 14, 512, 512), -10.0)
        out[0, TARGETS.index("Right Lung"), 60:460, 40:230] = 10.0
        out[0, TARGETS.index("Left Lung"), 60:460, 282:472] = 10.0
        out[0, TARGETS.index("Heart"), 250:430, 200:330] = 10.0
        return out


def test_masks_are_on_the_full_grid_and_labelled_by_patient_side():
    seg = AnatomySegmenter(StubPsp())
    a = seg(np.full((600, 800), 0.5, np.float32))  # non-square: central 600x600 is analysed
    for m in (a.right_lung, a.left_lung, a.heart):
        assert m.shape == (600, 800) and m.dtype == bool and m.any()
    assert a.right_lung[300, 100 + 100] and not a.right_lung[300, 700]  # image-left half, offset by the crop
    assert a.left_lung[300, 100 + 400]
    assert a.right_lung[:, :100].sum() == 0 and a.left_lung[:, 700:].sum() == 0  # outside the crop
    assert laterality_ok(a) is True


def test_try_load_returns_none_when_weights_cannot_load(monkeypatch):
    import torchxrayvision as xrv

    def boom(*a, **k):
        raise OSError("offline")

    monkeypatch.setattr(xrv.baseline_models.chestx_det, "PSPNet", boom)
    assert AnatomySegmenter.try_load() is None
