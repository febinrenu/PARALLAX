"""Grad-CAM family on any torch model and target layer.

Three variants: "gradcam", "gradcam++" and "hirescam". The extractor does one forward pass
and one gradient per requested class. Maps come back at the layer's own resolution,
float32, max-normalised to [0, 1] (all zeros when the signal is flat). Use `resize_cam`
to move a map onto the original image grid.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

METHODS = ("gradcam", "gradcam++", "hirescam")
_EPS = 1e-8


def _gradcam(acts: torch.Tensor, grads: torch.Tensor) -> torch.Tensor:
    weights = grads.mean(dim=(2, 3), keepdim=True)
    return (weights * acts).sum(dim=1)


def _gradcampp(acts: torch.Tensor, grads: torch.Tensor) -> torch.Tensor:
    g2, g3 = grads**2, grads**3
    denom = 2.0 * g2 + acts.sum(dim=(2, 3), keepdim=True) * g3
    denom = torch.where(denom != 0, denom, torch.ones_like(denom))
    alpha = g2 / denom
    weights = (alpha * torch.relu(grads)).sum(dim=(2, 3), keepdim=True)
    return (weights * acts).sum(dim=1)


def _hirescam(acts: torch.Tensor, grads: torch.Tensor) -> torch.Tensor:
    return (grads * acts).sum(dim=1)


_FUNCS = {"gradcam": _gradcam, "gradcam++": _gradcampp, "hirescam": _hirescam}


class CamExtractor:
    """Context manager that hooks `layer` of `model`. The model should be in eval mode."""

    def __init__(self, model: torch.nn.Module, layer: torch.nn.Module, relu_acts: bool = True):
        # relu_acts: DenseNet-style backbones hook the layer before the final ReLU, so the
        # captured map can be negative. Grad-CAM++ assumes non-negative activations, and the
        # classifier only ever sees the rectified map, so rectify here too.
        self.model, self.layer, self.relu_acts = model, layer, relu_acts
        self._acts: torch.Tensor | None = None
        self._handle = None

    def __enter__(self) -> CamExtractor:
        self._handle = self.layer.register_forward_hook(self._save)
        return self

    def __exit__(self, *exc) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def _save(self, _module, _inp, out) -> None:
        self._acts = out

    def maps(self, x: torch.Tensor, class_indices: list[int], method: str = "gradcam++") -> list[np.ndarray]:
        if method not in _FUNCS:
            raise ValueError(f"unknown CAM method {method!r}; choose from {METHODS}")
        with torch.enable_grad():
            x = x.detach().clone().requires_grad_(True)
            logits = self.model(x)
            acts = self._acts
            if acts is None or not acts.requires_grad:
                raise RuntimeError("target layer did not take part in the forward pass")
            out: list[np.ndarray] = []
            for i in class_indices:
                score = logits[0, i]
                (grads,) = torch.autograd.grad(score, acts, retain_graph=True)
                used = torch.relu(acts.detach()) if self.relu_acts else acts.detach()
                cam = torch.relu(_FUNCS[method](used, grads))[0]
                out.append(_normalise(cam.cpu().numpy().astype(np.float32)))
        return out


def _normalise(cam: np.ndarray) -> np.ndarray:
    m = float(cam.max())
    if not np.isfinite(m) or m <= _EPS:
        return np.zeros_like(cam, dtype=np.float32)
    return (cam / m).astype(np.float32)


def resize_cam(cam: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """Bilinear resize onto an image grid of (height, width), clipped to [0, 1]."""
    h, w = hw
    up = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(up, 0.0, 1.0).astype(np.float32)
