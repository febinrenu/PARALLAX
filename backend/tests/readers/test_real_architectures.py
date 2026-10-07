"""Real timm / smp architectures and the data owner's committed metadata, with randomly initialised weights.

These check that our loaders, layer paths, preprocessing specs and CAM hooks work on the real network code.
They say nothing about model quality: that needs the trained weights, which are not in the repository.
"""

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
timm = pytest.importorskip("timm")
smp = pytest.importorskip("segmentation_models_pytorch")

from medproof.readers import classifier as C  # noqa: E402
from medproof.readers import segmenter as SG  # noqa: E402
from tests.conftest import make_phantom  # noqa: E402

ARTIFACTS = Path(__file__).resolve().parents[3] / "ml" / "artifacts"
METAS = {p.parent.name: json.loads(p.read_text(encoding="utf-8")) for p in sorted(ARTIFACTS.glob("*/model_meta.json"))}
needs_meta = pytest.mark.skipif(not METAS, reason="ml/artifacts model_meta.json files not present")


def _bundle(tmp_path, name, meta, state):
    d = tmp_path / name
    d.mkdir()
    torch.save(state, d / "weights.pt")
    m = dict(meta, weights_file="weights.pt", weights_sha256=None)
    (d / "model_meta.json").write_text(json.dumps(m))
    return d


@needs_meta
@pytest.mark.parametrize("name", ["brain_cls", "skin_cls"])
def test_classifier_reader_runs_on_the_real_architecture_and_real_metadata(tmp_path, name):
    if name not in METAS:
        pytest.skip(f"{name} metadata not present")
    meta = METAS[name]
    model = timm.create_model(meta["arch"], pretrained=False, num_classes=len(meta["classes"]))
    d = _bundle(tmp_path, name, meta, model.state_dict())
    modality = "brain_mri" if name == "brain_cls" else "skin_dermoscopy"
    r = C.ImageClassifierReader.from_dir(d, modality=modality, positive_threshold=0.0, negative_classes=())
    assert r.labels == meta["classes"] and r.model_id == meta["model_id"]
    img = np.stack([make_phantom(192)] * 3, axis=2) if modality == "skin_dermoscopy" else make_phantom(192)
    out = r.predict(img)
    assert out.probs.shape == (len(meta["classes"]),) and out.probs.sum() == pytest.approx(1.0, abs=1e-4)
    assert out.findings and out.findings[0].heatmap.shape == (192, 192)
    # the default CAM layer for this architecture exists and gives a spatial map (not a pooled vector)
    layer = dict(r.model.named_modules())[C.layer_for_arch(meta["arch"])]
    with C.CamExtractor(r.model, layer) as cx:
        cam = cx.maps(r._tensor(img), [0], "gradcam++")[0]
    assert cam.ndim == 2 and min(cam.shape) >= 5


@needs_meta
def test_unet_segmenter_loads_real_architecture_with_real_metadata(tmp_path):
    if "brain_seg" not in METAS:
        pytest.skip("brain_seg metadata not present")
    meta = METAS["brain_seg"]
    model = smp.Unet(SG.encoder_from_arch(meta["arch"]), encoder_weights=None, in_channels=3, classes=1)
    d = _bundle(tmp_path, "brain_seg", meta, model.state_dict())
    seg = SG.UNetSegmenter.from_dir(d)
    assert seg.spec.size == meta["preproc_spec"]["size"] and seg.threshold == meta.get("threshold", 0.5)
    img = make_phantom(160)
    p = seg.probability(img)
    assert p.shape == (160, 160) and 0.0 <= p.min() and p.max() <= 1.0 and seg.segment(img).dtype == bool


@needs_meta
def test_every_committed_metadata_file_is_understood_by_the_loaders():
    for name, m in METAS.items():
        if m.get("task") == "classification" and "arch" in m:
            assert C.layer_for_arch(m["arch"]) and C.spec_from_meta(m["preproc_spec"]).size > 0
        if m.get("task") == "segmentation":
            assert SG.encoder_from_arch(m["arch"]) and C.spec_from_meta(m["preproc_spec"]).channels == 3
        if m.get("task") == "detection":
            assert m.get("weights_file") and int(m.get("imgsz", 640)) > 0 and m["classes"] == ["fracture"]


@needs_meta
def test_missing_trained_weights_fail_with_a_clear_message():
    """The repository holds metadata but not weights; the error must say what to do, not crash obscurely."""
    for name in ("brain_cls", "skin_cls"):
        if name in METAS and not (ARTIFACTS / name / "weights.pt").is_file():
            with pytest.raises(C.ModelBundleError, match="pull the training output"):
                C.ImageClassifierReader.from_dir(ARTIFACTS / name, modality="brain_mri")


def test_bone_reader_loads_a_real_yolo_checkpoint_and_parses_its_predictions(tmp_path):
    """Random-weight YOLO26n written with the library's own save: checks the loader and prediction parsing
    against the installed ultralytics version. Random weights give no meaningful boxes."""
    ul = pytest.importorskip("ultralytics")
    from medproof.readers.bone import BoneReader

    d = tmp_path / "bone_det"
    d.mkdir()
    ul.YOLO("yolo26n.yaml").save(str(d / "weights.pt"))
    (d / "model_meta.json").write_text(json.dumps({"model_id": "bone_det@random", "task": "detection", "arch": "yolo26n", "classes": ["fracture"],
                                                    "weights_file": "weights.pt", "weights_sha256": None, "imgsz": 320}))
    r = BoneReader.from_dir(d)
    out = r.predict(make_phantom(200)[:, :180], body_part="leg")
    assert out.modality == "bone_xray" and out.probs.shape == (1,) and 0.0 <= out.probs[0] <= 1.0
    assert r.score(make_phantom(200))[0] == pytest.approx(r.score(make_phantom(200))[0])
    (d / "weights.pt").write_bytes(b"corrupt")
    with pytest.raises(C.ModelBundleError):
        BoneReader.from_dir(d)
