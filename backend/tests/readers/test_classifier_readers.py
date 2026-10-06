"""Brain and skin reader wrappers, tested with tiny stand-in networks and a real weights file round trip."""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn, F = torch.nn, torch.nn.functional

from medproof.core.schemas import Finding  # noqa: E402
from medproof.intake.decode import load_image  # noqa: E402
from medproof.readers import classifier as C  # noqa: E402
from medproof.readers.evidence import to_findings  # noqa: E402
from tests.conftest import make_phantom, png_bytes, to_u8  # noqa: E402

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]


class TinyNet(nn.Module):
    """3-channel classifier whose class 0 reacts to brightness in the upper-left quadrant."""

    def __init__(self, n=4):
        super().__init__()
        self.stem = nn.Conv2d(3, 4, 3, padding=1)
        self.head = nn.Linear(4, n)

    def forward(self, x):
        f = torch.relu(self.stem(x))
        return self.head(F.adaptive_avg_pool2d(f, 1).flatten(1))


def _factory(arch, n_classes):
    return TinyNet(n_classes)


def _layer(model):
    return model.stem


def _write_bundle(tmp_path, classes=CLASSES, spec="brain_effnet", task="classification", arch="tiny", weights=True):
    torch.manual_seed(0)
    net = TinyNet(len(classes))
    d = tmp_path / "brain_cls"
    d.mkdir()
    if weights:
        torch.save(net.state_dict(), d / "weights.pt")
    meta = {"model_id": "brain_cls@abc12345", "task": task, "arch": arch, "classes": classes, "preproc_spec": spec,
            "weights_file": "weights.pt" if weights else None, "weights_sha256": None}
    (d / "model_meta.json").write_text(json.dumps(meta))
    return d, net


def _img():
    return make_phantom(160)


def test_bundle_loads_meta_and_weights_from_a_model_directory(tmp_path):
    d, net = _write_bundle(tmp_path)
    r = C.ImageClassifierReader.from_dir(d, modality="brain_mri", model_factory=_factory, layer_getter=_layer)
    assert r.labels == CLASSES and r.model_id.startswith("brain_cls@abc12345")
    for a, b in zip(net.state_dict().values(), r.model.state_dict().values()):
        assert torch.equal(a, b)


def test_missing_or_corrupt_weights_raise_a_clear_error(tmp_path):
    d, _ = _write_bundle(tmp_path, weights=False)
    with pytest.raises(C.ModelBundleError):
        C.ImageClassifierReader.from_dir(d, modality="brain_mri", model_factory=_factory, layer_getter=_layer)
    d2 = tmp_path / "bad"
    d2.mkdir()
    (d2 / "model_meta.json").write_text(json.dumps({"model_id": "x", "classes": CLASSES, "preproc_spec": "brain_effnet", "weights_file": "weights.pt", "task": "classification", "arch": "tiny"}))
    (d2 / "weights.pt").write_bytes(b"not a checkpoint")
    with pytest.raises(C.ModelBundleError):
        C.ImageClassifierReader.from_dir(d2, modality="brain_mri", model_factory=_factory, layer_getter=_layer)


def test_weights_hash_mismatch_is_rejected(tmp_path):
    d, _ = _write_bundle(tmp_path)
    meta = json.loads((d / "model_meta.json").read_text())
    meta["weights_sha256"] = "0" * 64
    (d / "model_meta.json").write_text(json.dumps(meta))
    with pytest.raises(C.ModelBundleError, match="sha256"):
        C.ImageClassifierReader.from_dir(d, modality="brain_mri", model_factory=_factory, layer_getter=_layer)


def test_spec_can_be_a_name_or_a_dict(tmp_path):
    spec = {"name": "brain_seg", "size": 64, "channels": 3, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "resize": "stretch", "value_range": [0.0, 1.0]}
    assert C.spec_from_meta("skin_cls").size == 224
    s = C.spec_from_meta(spec)
    assert s.size == 64 and s.channels == 3 and s.mean == (0.485, 0.456, 0.406) and s.value_range == (0.0, 1.0)


def test_predict_gives_softmax_probabilities_and_the_top_class_as_a_finding(tmp_path):
    d, _ = _write_bundle(tmp_path)
    r = C.ImageClassifierReader.from_dir(d, modality="brain_mri", model_factory=_factory, layer_getter=_layer, positive_threshold=0.0, negative_classes=())
    out = r.predict(_img())
    assert out.modality == "brain_mri" and out.output_kind == "probability" and out.probs.shape == (4,)
    assert out.probs.sum() == pytest.approx(1.0, abs=1e-5) and out.image_hw == (160, 160)
    top = CLASSES[int(out.probs.argmax())]
    assert out.findings and out.findings[0].label == top
    f = out.findings[0]
    assert f.heatmap.shape == (160, 160) and f.method and f.boxes_xyxy


def test_negative_classes_are_not_reported_as_findings(tmp_path):
    d, _ = _write_bundle(tmp_path)
    r = C.ImageClassifierReader.from_dir(d, modality="brain_mri", model_factory=_factory, layer_getter=_layer, positive_threshold=0.0, negative_classes=("notumor",))
    out = r.predict(_img())
    assert "notumor" not in [f.label for f in out.findings]
    r2 = C.ImageClassifierReader.from_dir(d, modality="brain_mri", model_factory=_factory, layer_getter=_layer, positive_threshold=0.0, negative_classes=tuple(CLASSES))
    assert r2.predict(_img()).findings == []


def test_threshold_and_score_match_predict(tmp_path):
    d, _ = _write_bundle(tmp_path)
    r = C.ImageClassifierReader.from_dir(d, modality="brain_mri", model_factory=_factory, layer_getter=_layer, positive_threshold=0.99)
    img = _img()
    assert r.predict(img).findings == []
    np.testing.assert_allclose(r.score(img), r.predict(img).probs, atol=1e-6)


def test_contract_objects_and_faithfulness_compatibility(tmp_path):
    from medproof.verify import faithfulness as FA
    from medproof.verify import stability as ST

    d, _ = _write_bundle(tmp_path)
    r = C.ImageClassifierReader.from_dir(d, modality="brain_mri", model_factory=_factory, layer_getter=_layer, positive_threshold=0.0)
    img = _img()
    out = r.predict(img)
    findings, _ = to_findings(out, r.cfg)
    for f in findings:
        Finding.model_validate(f.model_dump())
        assert f.modality == "brain_mri" and f.image_evidence[0].source_model == r.model_id
    res = FA.assess_output(r, img, out, FA.FaithfulnessConfig(n_random=9), seed=0)
    assert {f.label for f in findings} <= set(res)  # a finding without a heatmap is withheld from the contract but still assessed
    st = ST.assess_output(r, img, out, ST.StabilityConfig())
    assert all(v.tests == 8 for v in st.values())


def test_run_stage_never_raises_and_reports_findings(tmp_path):
    from types import SimpleNamespace

    d, _ = _write_bundle(tmp_path)
    r = C.ImageClassifierReader.from_dir(d, modality="brain_mri", model_factory=_factory, layer_getter=_layer, positive_threshold=0.0)
    ctx = SimpleNamespace(decoded=load_image(png_bytes(to_u8(_img()))), artifact_dir=tmp_path / "art")
    res = C.run(ctx, reader=r)
    assert res.ok and res.stage == "reader" and res.payload["findings"]
    assert any(p.suffix == ".png" for p in (tmp_path / "art").iterdir())
    assert C.run(SimpleNamespace(), reader=r).ok is False

    class Boom:
        cfg = r.cfg

        def predict(self, img):
            raise RuntimeError("no weights")

    assert C.run(ctx, reader=Boom()).ok is False


def test_default_layer_getters_and_negative_class_names_cover_the_planned_models():
    assert "notumor" in C.BRAIN_NEGATIVE and C.layer_for_arch("efficientnet_b0.ra_in1k") == "bn2"
    assert C.layer_for_arch("convnext_tiny.fb_in22k_ft_in1k") == "stages.3"
    with pytest.raises(ValueError):
        C.layer_for_arch("unknown_arch")
