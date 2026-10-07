"""Saliency sanity check: model-parameter randomisation (Adebayo et al., 2018).

A trustworthy explanation must depend on what the model learned. We randomise the network from
the top layer downwards, cumulatively, recompute each CAM variant after every stage, and measure
how similar it stays to the original map (Spearman rank correlation and SSIM). A method whose
maps barely change is explaining the input, not the model. The variant that changes most is chosen.

This is an offline check run once per model, not part of a study run.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.stats import spearmanr

from medproof.readers.cam import METHODS, CamExtractor, resize_cam

PASS_MAX_ABS_SPEARMAN = 0.5  # after full randomisation a sound method should be below this
GRID = 64
MAX_DEGENERATE = 0.5  # share of stages whose map may be empty before a method is not trusted

MapFn = Callable[[torch.nn.Module, torch.nn.Module, torch.Tensor, int], np.ndarray]


def cam_method(name: str) -> MapFn:
    def fn(model, layer, x, idx):
        with CamExtractor(model, layer) as cx:
            return cx.maps(x, [idx], name)[0]

    return fn


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0  # a constant map has no ranking to compare
    v = spearmanr(a, b).statistic
    return 0.0 if not np.isfinite(v) else float(v)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Mean structural similarity on [0, 1] maps with an 11x11 Gaussian window."""
    a, b = a.astype(np.float64), b.astype(np.float64)
    c1, c2 = 0.01**2, 0.03**2
    blur = lambda z: cv2.GaussianBlur(z, (11, 11), 1.5)  # noqa: E731
    mu_a, mu_b = blur(a), blur(b)
    var_a, var_b = blur(a * a) - mu_a**2, blur(b * b) - mu_b**2
    cov = blur(a * b) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2))
    return float(s.mean())


def default_order(model: torch.nn.Module) -> list[str]:
    """Names of the modules that own parameters, last to first (top of the network down)."""
    names = [n for n, m in model.named_modules() if n and list(m.parameters(recurse=False))]
    return list(reversed(names))


def densenet_order() -> list[str]:
    """Top-down stages for the torchxrayvision DenseNet121."""
    return ["classifier", "features.norm5", "features.denseblock4", "features.transition3", "features.denseblock3",
            "features.transition2", "features.denseblock2", "features.transition1", "features.denseblock1",
            "features.norm0", "features.conv0"]


def _randomise(module: torch.nn.Module, gen: torch.Generator) -> None:
    for m in module.modules():
        for p in m.parameters(recurse=False):
            with torch.no_grad():
                if p.dim() > 1:
                    std = float(np.sqrt(2.0 / max(1, p[0].numel())))
                    p.copy_(torch.randn(p.shape, generator=gen) * std)
                else:
                    p.copy_(torch.randn(p.shape, generator=gen) * 0.1 + (1.0 if "BatchNorm" in type(m).__name__ else 0.0))
        if hasattr(m, "reset_running_stats"):
            m.reset_running_stats()


def run_sanity(
    model: torch.nn.Module,
    x: torch.Tensor,
    class_idx: int,
    *,
    layer: str = "features",
    methods: dict[str, MapFn] | None = None,
    order: list[str] | None = None,
    seed: int = 0,
) -> dict:
    """Cascading randomisation. The caller's model is never modified."""
    methods = methods or {m: cam_method(m) for m in METHODS}
    order = order or default_order(model)
    work = copy.deepcopy(model).eval()
    gen = torch.Generator().manual_seed(seed)
    get_layer = lambda: dict(work.named_modules())[layer]  # noqa: E731

    def maps() -> dict[str, np.ndarray]:
        return {n: resize_cam(np.asarray(f(work, get_layer(), x, class_idx), np.float32), (GRID, GRID)) for n, f in methods.items()}

    original = maps()
    curves = {n: {"spearman": [], "ssim": [], "empty": []} for n in methods}
    named = dict(work.named_modules())
    for stage in order:
        _randomise(named[stage], gen)
        for n, mp in maps().items():
            curves[n]["spearman"].append(round(spearman(original[n], mp), 4))
            curves[n]["ssim"].append(round(ssim(original[n], mp), 4))
            curves[n]["empty"].append(bool(mp.max() <= 1e-8))  # an all-zero map says nothing; it is not "different"
    for n, c in curves.items():
        c["mean_abs_spearman"] = round(float(np.mean(np.abs(c["spearman"]))), 4)
        c["mean_ssim"] = round(float(np.mean(c["ssim"])), 4)
        c["final_abs_spearman"] = round(abs(c["spearman"][-1]), 4)
        c["degenerate_fraction"] = round(float(np.mean(c["empty"])), 4)
        c["passes"] = bool(c["final_abs_spearman"] < PASS_MAX_ABS_SPEARMAN and c["degenerate_fraction"] <= MAX_DEGENERATE)
    # A method whose maps mostly collapse to zero looks "insensitive" only because it shows nothing.
    eligible = [n for n in curves if curves[n]["degenerate_fraction"] <= MAX_DEGENERATE] or list(curves)
    chosen = min(eligible, key=lambda n: (curves[n]["mean_abs_spearman"], curves[n]["mean_ssim"]))
    return {"stages": list(order), "seed": seed, "class_idx": int(class_idx), "methods": curves, "chosen": chosen,
            "pass_threshold": PASS_MAX_ABS_SPEARMAN}


def write_result(result: dict, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=1), encoding="utf-8")
