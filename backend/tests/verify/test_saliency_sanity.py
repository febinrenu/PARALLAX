import copy
import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from medproof.readers.cam import METHODS, CamExtractor, resize_cam  # noqa: E402
from medproof.verify import saliency_sanity as S  # noqa: E402


class Net(nn.Module):
    """Small CNN: features ends before the ReLU like the DenseNet backbone; classifier reads pooled features."""

    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 8, 3, padding=1), nn.BatchNorm2d(8)
        )
        self.classifier = nn.Linear(8, 3)

    def forward(self, x):
        f = torch.relu(self.features(x))
        return self.classifier(torch.nn.functional.adaptive_avg_pool2d(f, 1).flatten(1))


def _x():
    g = torch.Generator().manual_seed(1)
    x = torch.rand(1, 1, 32, 32, generator=g)
    x[..., 8:16, 8:16] += 1.5
    return x


def _hash(model):
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


ORDER = ["classifier", "features.3", "features.2", "features.0"]


def test_similarity_to_the_original_drops_when_weights_are_randomised():
    r = S.run_sanity(Net().eval(), _x(), 0, order=ORDER, seed=0)
    assert r["stages"] == ORDER
    for m in METHODS:
        sp = r["methods"][m]["spearman"]
        assert len(sp) == len(ORDER) and all(-1.0 <= v <= 1.0 for v in sp)
        assert abs(sp[-1]) < 0.9  # a fully randomised network should not reproduce the explanation


def test_the_callers_model_is_not_modified():
    model = Net().eval()
    before = _hash(model)
    S.run_sanity(model, _x(), 0, order=ORDER, seed=0)
    assert _hash(model) == before


def test_deterministic_for_a_seed_and_different_for_another():
    a = S.run_sanity(Net().eval(), _x(), 0, order=ORDER, seed=3)
    b = S.run_sanity(Net().eval(), _x(), 0, order=ORDER, seed=3)
    c = S.run_sanity(Net().eval(), _x(), 0, order=ORDER, seed=4)
    assert a == b and a["methods"] != c["methods"]


def test_a_method_that_ignores_the_weights_is_flagged_and_never_chosen():
    def edge_map(model, layer, x, idx):  # depends on the input only, so randomising weights cannot change it
        g = x[0, 0].numpy()
        e = np.abs(cv2.Laplacian(g, cv2.CV_32F))
        return (e / max(float(e.max()), 1e-8)).astype(np.float32)

    methods = {"edge": edge_map, **{m: S.cam_method(m) for m in METHODS}}
    r = S.run_sanity(Net().eval(), _x(), 0, order=ORDER, seed=0, methods=methods)
    assert r["methods"]["edge"]["passes"] is False
    assert abs(r["methods"]["edge"]["spearman"][-1]) > 0.99
    assert r["chosen"] != "edge" and r["methods"][r["chosen"]]["passes"] in (True, False)


def test_chosen_method_has_the_lowest_mean_similarity_among_eligible_methods():
    r = S.run_sanity(Net().eval(), _x(), 1, order=ORDER, seed=0)
    ok = {m: v["mean_abs_spearman"] for m, v in r["methods"].items() if v["degenerate_fraction"] <= S.MAX_DEGENERATE}
    assert ok and r["methods"][r["chosen"]]["mean_abs_spearman"] == pytest.approx(min(ok.values()))


def test_a_method_whose_maps_collapse_to_zero_is_reported_and_not_chosen():
    calls = {"n": 0}

    def vanishes(model, layer, x, idx):
        calls["n"] += 1
        m = np.zeros((8, 8), np.float32)
        if calls["n"] == 1:  # fine on the original model, empty after every randomisation
            m[2:5, 2:5] = 1.0
        return m

    methods = {"vanishes": vanishes, "gradcam++": S.cam_method("gradcam++")}
    r = S.run_sanity(Net().eval(), _x(), 0, order=ORDER, seed=0, methods=methods)
    v = r["methods"]["vanishes"]
    assert v["degenerate_fraction"] == 1.0 and v["passes"] is False
    assert r["chosen"] == "gradcam++"


def test_metrics_spearman_and_ssim_basic_properties():
    rng = np.random.default_rng(0)
    a = rng.random((32, 32)).astype(np.float32)
    assert S.spearman(a, a) == pytest.approx(1.0) and S.ssim(a, a) == pytest.approx(1.0)
    assert S.spearman(a, -a) == pytest.approx(-1.0)
    assert S.spearman(a, np.zeros_like(a)) == 0.0  # a constant map carries no rank information
    assert S.ssim(a, rng.random((32, 32)).astype(np.float32)) < 0.2


def test_default_order_lists_parameter_layers_top_down():
    order = S.default_order(Net())
    assert order[0] == "classifier" and order[-1] == "features.0"
    assert set(order) == {"classifier", "features.0", "features.2", "features.3"}


def test_result_serialises_to_small_json(tmp_path):
    r = S.run_sanity(Net().eval(), _x(), 0, order=ORDER, seed=0)
    p = tmp_path / "r.json"
    S.write_result(r, p)
    import json

    back = json.loads(p.read_text())
    assert back["chosen"] == r["chosen"] and back["seed"] == 0
    assert p.stat().st_size < 20_000


_WEIGHTS = Path.home() / ".torchxrayvision" / "models_data" / "nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt"


@pytest.mark.skipif(not _WEIGHTS.is_file(), reason="densenet121-res224-all weights not cached")
def test_real_densenet_on_a_phantom_randomisation_changes_the_maps():
    import torchxrayvision as xrv

    from medproof.intake.preprocess import prepare
    from tests.conftest import make_phantom

    model = xrv.models.DenseNet(weights="densenet121-res224-all").eval()
    x = torch.from_numpy(prepare(make_phantom(256), "cxr_xrv"))[None]
    order = S.densenet_order()
    r = S.run_sanity(model, x, 8, order=order, seed=0)
    assert r["stages"] == order and r["chosen"] in METHODS
    assert all(abs(v["spearman"][-1]) < 0.95 for v in r["methods"].values())
