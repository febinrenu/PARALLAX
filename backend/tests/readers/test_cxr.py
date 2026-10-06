from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn, F = torch.nn, torch.nn.functional

from medproof.core.schemas import StageResult  # noqa: E402
from medproof.intake.decode import load_image  # noqa: E402
from medproof.readers.cxr import CxrReader, largest_region, run  # noqa: E402
from medproof.readers.cxr_config import CxrConfig  # noqa: E402
from medproof.readers.evidence import to_findings  # noqa: E402
from medproof.readers.zones import Anatomy  # noqa: E402
from tests.conftest import png_bytes, to_u8  # noqa: E402

HW = (300, 500)  # height, width: deliberately not square
BLOB = (slice(60, 120), slice(250, 350))  # inside the central 300x300 square (x 100..400)
LABELS = ["Pneumonia", "Effusion", "Mass"]


class _Pool(nn.Module):
    def forward(self, x):
        return F.adaptive_avg_pool2d(x, 7)


class StubXrv(nn.Module):
    """Probabilities rise with the amount of bright area, like a localised detector would."""

    pathologies = LABELS

    def __init__(self, thresholds, gain=60.0):
        super().__init__()
        self.features = _Pool()
        self.t = thresholds
        self.gain = gain

    def stat(self, x):
        return F.relu(self.features(x)).mean(dim=(1, 2, 3)) / 1024.0

    def forward(self, x):
        s = self.stat(x)[:, None]
        t = torch.tensor(self.t, dtype=torch.float32)[None]
        return torch.sigmoid(self.gain * (s - t))


def _image(with_blob=True):
    a = np.zeros(HW, np.float32)
    if with_blob:
        a[BLOB] = 1.0
    return a


def _anatomy(flip=False):
    def box(y0, y1, x0, x1):
        m = np.zeros(HW, bool)
        m[y0:y1, x0:x1] = True
        return m

    rl, ll, ht = box(20, 280, 40, 230), box(20, 280, 270, 460), box(150, 260, 215, 290)
    if flip:
        rl, ll, ht = rl[:, ::-1], ll[:, ::-1], ht[:, ::-1]
    return Anatomy(right_lung=rl, left_lung=ll, heart=ht)


def make_reader(anatomy_fn="default", sharpness=30.0, **cfg_kw):
    from medproof.intake.preprocess import prepare

    probe = StubXrv([0.0, 0.0, 0.0])
    stat = float(probe.stat(torch.from_numpy(prepare(_image(), "cxr_xrv"))[None]))
    gain = sharpness / stat  # an empty image scores far below every threshold
    model = StubXrv([stat - 3 / gain, stat - 0.6 / gain, stat + 4 / gain], gain)  # probs ~0.95, ~0.65, ~0.02
    fn = (lambda img: _anatomy()) if anatomy_fn == "default" else anatomy_fn
    return CxrReader(model, anatomy_fn=fn, cfg=CxrConfig(**cfg_kw), model_id="xrv-densenet121-all@abcd1234")


def test_only_labels_above_threshold_become_findings_sorted_by_probability():
    out = make_reader().predict(_image())
    assert [f.label for f in out.findings] == ["Pneumonia", "Effusion"]
    assert out.findings[0].prob_raw > out.findings[1].prob_raw >= 0.5
    assert out.labels == LABELS and out.probs.shape == (3,) and out.modality == "cxr"
    assert out.image_hw == HW


def test_location_follows_the_bright_region_in_original_pixels():
    f = make_reader().predict(_image()).findings[0]
    x0, y0, x1, y1 = f.boxes_xyxy[0]
    assert 0 <= x0 < x1 <= HW[1] and 0 <= y0 < y1 <= HW[0]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    assert 250 <= cx <= 350 and 60 <= cy <= 120
    inter = max(0, min(x1, 350) - max(x0, 250)) * max(0, min(y1, 120) - max(y0, 60))
    union = (x1 - x0) * (y1 - y0) + 100 * 60 - inter
    assert inter / union > 0.3
    assert f.heatmap.shape == HW and f.heatmap.dtype == np.float32 and 0.95 <= f.heatmap.max() <= 1.0
    assert f.mask.shape == HW and f.mask[90, 300]


def test_region_is_named_with_patient_side_laterality():
    f = make_reader().predict(_image()).findings[0]
    assert f.region_name == "left upper zone"  # image-right lung, top third


def test_cam_method_is_recorded_and_configurable():
    assert make_reader().predict(_image()).findings[0].method == "gradcam++"
    assert make_reader(cam_method="hirescam").predict(_image()).findings[0].method == "hirescam"


def test_max_findings_caps_the_number_of_cams():
    assert len(make_reader(max_findings=1).predict(_image()).findings) == 1


def test_no_positive_labels_gives_no_findings():
    out = make_reader().predict(_image(with_blob=False))
    assert out.findings == [] and out.probs.max() < 0.5


def test_anatomy_failure_keeps_findings_without_region_names():
    def boom(_img):
        raise RuntimeError("weights missing")

    out = make_reader(anatomy_fn=boom).predict(_image())
    assert out.findings and all(f.region_name is None for f in out.findings)
    assert "anatomy_unavailable" in out.flags
    assert any("anatomy segmentation failed" in w for w in out.warnings)


def test_no_segmenter_is_flagged_not_silent():
    out = make_reader(anatomy_fn=None).predict(_image())
    assert "anatomy_unavailable" in out.flags and out.findings[0].region_name is None


def test_flipped_image_flags_laterality_uncertain():
    out = make_reader(anatomy_fn=lambda img: _anatomy(flip=True)).predict(_image())
    assert "laterality_uncertain" in out.flags
    assert any("flipped" in w for w in out.warnings)
    findings, _ = to_findings(out, CxrConfig())
    assert all("laterality_uncertain" in f.flags for f in findings)


def test_score_matches_predict_and_drops_when_the_region_is_removed():
    r = make_reader()
    img = _image()
    np.testing.assert_allclose(r.score(img), r.predict(img).probs, atol=1e-6)
    deleted = img.copy()
    deleted[BLOB] = 0.0
    assert r.score(deleted)[0] < r.score(img)[0] - 0.5


def test_decoded_image_input_and_contract_objects():
    img = load_image(png_bytes(to_u8(_image())))
    out = make_reader().predict(img)
    findings, _ = to_findings(out, CxrConfig(), artifact_dir=None)
    assert [f.finding_id for f in findings] == ["f1", "f2"]
    assert findings[0].image_evidence[0].source_model == "xrv-densenet121-all@abcd1234"
    assert findings[0].image_evidence[0].region_name == "left upper zone"


def test_largest_region_contains_the_peak_and_flat_maps_have_none():
    cam = np.zeros((50, 50), np.float32)
    cam[10:20, 10:20] = 0.6
    cam[30:34, 30:34] = 1.0  # peak is in the smaller component
    mask, box = largest_region(cam, 0.5)
    assert mask[31, 31] and not mask[15, 15] and box == (30.0, 30.0, 34.0, 34.0)
    assert largest_region(np.zeros((8, 8), np.float32), 0.5) is None


def test_run_returns_stage_result_with_valid_findings(tmp_path):
    ctx = SimpleNamespace(decoded=load_image(png_bytes(to_u8(_image()))), artifact_dir=tmp_path, evidence_start=5)
    res = run(ctx, reader=make_reader())
    assert isinstance(res, StageResult) and res.stage == "reader" and res.ok
    assert [f["image_evidence"][0]["evidence_id"] for f in res.payload["findings"]] == ["ie_5", "ie_6"]
    assert any(p.suffix == ".png" for p in tmp_path.iterdir())


def test_run_never_raises():
    assert run(SimpleNamespace()).ok is False

    class Broken:
        cfg = CxrConfig()

        def predict(self, img):
            raise RuntimeError("no weights")

    res = run(SimpleNamespace(decoded=load_image(png_bytes(to_u8(_image())))), reader=Broken())
    assert res.ok is False and res.warnings


def test_model_reads_only_the_central_square_and_says_so():
    out = make_reader().predict(_image())
    assert "partial_view" in out.flags and any("central square" in w for w in out.warnings)
    heat = out.findings[0].heatmap
    assert heat[:, :100].max() == 0.0 and heat[:, 400:].max() == 0.0  # outside x 100..400 was not seen


def test_content_outside_the_central_square_is_invisible_to_the_model():
    r = make_reader()
    img = _image(with_blob=False)
    img[60:120, 420:490] = 1.0  # right edge, outside the crop
    assert r.score(img)[0] < 0.01


def test_square_image_has_no_partial_view_flag():
    from medproof.intake.preprocess import center_crop_box

    assert center_crop_box(300, 300) == (0, 0, 300)
    sq = np.zeros((300, 300), np.float32)
    sq[60:120, 150:250] = 1.0
    out = make_reader(anatomy_fn=None).predict(sq)
    assert "partial_view" not in out.flags


def test_reader_sets_the_torch_thread_count_from_config():
    import torch

    make_reader(threads=1)
    assert torch.get_num_threads() == 1
