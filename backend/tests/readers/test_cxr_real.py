"""Tests against the real pretrained weights. Skipped (not passed) when the weights are not cached."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

xrv = pytest.importorskip("torchxrayvision")
torch = pytest.importorskip("torch")

from medproof.core.schemas import Finding  # noqa: E402
from medproof.intake.decode import load_image  # noqa: E402
from medproof.readers.anatomy import AnatomySegmenter  # noqa: E402
from medproof.readers.cxr import CxrReader, run  # noqa: E402
from medproof.readers.cxr_config import CxrConfig  # noqa: E402
from tests.conftest import make_phantom, png_bytes, to_u8  # noqa: E402

_CACHE = Path(xrv.utils.get_cache_dir()).expanduser()


def _has(name: str) -> bool:
    f = _CACHE / name
    return f.is_file() and f.stat().st_size > 1_000_000


DENSE = "nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt"
PSP = "pspnet_chestxray_best_model_4.pth"
needs_dense = pytest.mark.skipif(not _has(DENSE), reason="densenet121-res224-all weights not cached")
needs_psp = pytest.mark.skipif(not _has(PSP), reason="PSPNet weights not cached")


@pytest.fixture(scope="module")
def reader():
    return CxrReader.load(CxrConfig(weights="densenet121-res224-all"))


@needs_dense
def test_labels_and_output_kind_match_the_library(reader):
    assert reader.labels == list(xrv.datasets.default_pathologies)
    assert len(reader.labels) == 18 and "Pneumonia" in reader.labels
    assert reader.output_kind == "probability"  # op_threshs set: sigmoid then operating-point scaling
    assert reader.model_id.startswith("xrv-densenet121-all@") and not reader.model_id.endswith("@unknown")


@needs_dense
def test_forward_on_a_phantom_gives_18_probabilities(reader, phantom):
    p = reader.score(phantom)
    assert p.shape == (18,) and np.isfinite(p).all() and p.min() >= 0.0 and p.max() <= 1.0


@needs_dense
def test_preprocessing_matches_the_library_convention(reader, phantom):
    """Our prepare() output equals xrv's own normalise + center crop + resize path (up to resampling)."""
    from medproof.intake.preprocess import prepare

    wide = np.pad(phantom, ((0, 0), (64, 64)))  # 512 x 640, so the crop matters
    ours = prepare(wide, "cxr_xrv")
    img = xrv.utils.normalize((wide * 255).astype(np.uint8), 255)[None]
    ref = xrv.datasets.XRayResizer(224, engine="cv2")(xrv.datasets.XRayCenterCrop()(img))
    ref_p = reader.model(torch.from_numpy(ref)[None]).detach().numpy()[0]
    our_p = reader.model(torch.from_numpy(ours)[None]).detach().numpy()[0]
    assert np.abs(ours - ref).mean() < 25  # on a +-1024 scale
    assert np.abs(our_p - ref_p).max() < 0.08


@needs_dense
def test_end_to_end_stage_on_a_phantom_returns_valid_findings(reader, phantom, tmp_path):
    ctx = SimpleNamespace(decoded=load_image(png_bytes(to_u8(phantom))), artifact_dir=tmp_path)
    res = run(ctx, reader=reader)
    assert res.ok and res.stage == "reader" and len(res.payload["probs"]) == 18
    for f in res.payload["findings"]:
        Finding.model_validate(f)
        assert f["image_evidence"][0]["source_model"] == reader.model_id


@needs_psp
def test_real_segmenter_returns_14_channels_and_masks_on_the_grid(phantom):
    seg = AnatomySegmenter.try_load()
    assert seg is not None
    x = torch.from_numpy(np.zeros((1, 1, 512, 512), np.float32) - 1024)
    assert seg.model(x).shape == (1, 14, 512, 512)
    a = seg(phantom)
    assert a.heart.shape == phantom.shape and a.right_lung.dtype == bool
