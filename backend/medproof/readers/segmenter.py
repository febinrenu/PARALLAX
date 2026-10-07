"""Segmentation masks for findings: brain U-Net, and MedSAM masks prompted by a finding's box.

Both attach a mask and its source ("unet" or "medsam") to the reader's findings; evidence.to_findings
then turns each into a second, mask-kind ImageEvidence. Masks and boxes live on the original image grid.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from medproof.intake.decode import DecodedImage
from medproof.intake.preprocess import PreprocSpec, prepare
from medproof.readers.base import ReaderOutput
from medproof.readers.classifier import ModelBundleError, _sha256, spec_from_meta
from medproof.segment.medsam import box_from_mask

__all__ = ["ModelBundleError", "UNetSegmenter", "attach_masks", "attach_medsam_masks", "attach_unet_masks", "encoder_from_arch", "spec_from_meta"]


def encoder_from_arch(arch: str) -> str:
    """"smp.Unet(resnet34)" -> "resnet34"."""
    return arch[arch.index("(") + 1 : arch.rindex(")")]


def _default_factory(encoder: str):
    import segmentation_models_pytorch as smp

    return smp.Unet(encoder, encoder_weights=None, in_channels=3, classes=1)


class UNetSegmenter:
    def __init__(self, model, spec: PreprocSpec, *, threshold: float = 0.5, min_area_frac: float = 0.001, model_id: str = "unet", device: str = "cpu"):
        self.model = model.eval()
        self.spec, self.threshold, self.min_area_frac = spec, threshold, min_area_frac
        self.model_id, self.device = model_id, device

    def probability(self, img: DecodedImage | np.ndarray) -> np.ndarray:
        import torch

        hw = img.shape if isinstance(img, DecodedImage) else tuple(np.asarray(img).shape[:2])
        x = torch.from_numpy(prepare(img, self.spec))[None].to(self.device)
        with torch.no_grad():
            p = torch.sigmoid(self.model(x))[0, 0].cpu().numpy().astype(np.float32)
        return cv2.resize(p, (int(hw[1]), int(hw[0])), interpolation=cv2.INTER_LINEAR)

    def segment(self, img: DecodedImage | np.ndarray) -> np.ndarray:
        return self.probability(img) >= self.threshold

    @classmethod
    def from_dir(cls, model_dir: str | Path, model_factory: Callable | None = None, device: str = "cpu") -> UNetSegmenter:
        import torch

        d = Path(model_dir)
        try:
            meta = json.loads((d / "model_meta.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ModelBundleError(f"{d}: model_meta.json missing or unreadable") from exc
        wf = meta.get("weights_file")
        if not wf or not (d / wf).is_file():
            raise ModelBundleError(f"{d}: weights file {wf!r} not found; pull the training output first")
        if meta.get("weights_sha256") and _sha256(d / wf) != meta["weights_sha256"]:
            raise ModelBundleError(f"{d / wf}: sha256 does not match model_meta.json; the file is corrupt")
        model = (model_factory or _default_factory)(encoder_from_arch(meta["arch"]))
        try:
            model.load_state_dict(torch.load(d / wf, map_location="cpu", weights_only=True))
        except Exception as exc:
            raise ModelBundleError(f"{d / wf}: cannot load weights ({type(exc).__name__})") from exc
        return cls(model, spec_from_meta(meta["preproc_spec"]), threshold=float(meta.get("threshold", 0.5)), model_id=meta["model_id"], device=device)


OFF_TARGET_SHARE = 0.20  # provisional: below this, the heatmap's salient pixels are mostly outside the tumour mask


def heat_share_in_mask(heat: np.ndarray, mask: np.ndarray, rel: float = 0.5) -> float | None:
    """Share of the heatmap's salient pixels (at least `rel` of its peak) that lie inside `mask`; None for a flat map."""
    hi = float(heat.max())
    if hi - float(heat.min()) <= 1e-6:
        return None
    salient = heat >= rel * hi
    return float(np.logical_and(salient, mask).sum() / max(1, salient.sum()))


def attach_unet_masks(out: ReaderOutput, img, seg: UNetSegmenter, negative: tuple = ()) -> None:
    """Give every positive finding the U-Net mask (one forward pass) and tighten its box to the mask."""
    skip = {n.lower() for n in negative}
    targets = [f for f in out.findings if f.label.lower() not in skip]
    if not targets:
        return
    mask = seg.segment(img)
    if mask.mean() < seg.min_area_frac or not mask.any():
        for f in targets:
            f.flags.append("mask_too_small")
        return
    box = box_from_mask(mask)
    for f in targets:
        if f.heatmap is not None and f.heatmap.shape == mask.shape:
            share = heat_share_in_mask(f.heatmap, mask)
            if share is not None and share < OFF_TARGET_SHARE and "heatmap_off_target" not in f.flags:
                f.flags.append("heatmap_off_target")  # the model looked somewhere other than the segmented tumour
        f.mask, f.mask_source = mask, "unet"
        f.boxes_xyxy, f.box_scores = [box], [1.0]


def attach_medsam_masks(out: ReaderOutput, img, medsam) -> None:
    """Prompt MedSAM with each finding's box and attach the resulting lesion mask."""
    image = getattr(img, "display", img)
    for f in out.findings:
        if not f.boxes_xyxy:
            continue
        res = medsam.segment(image, f.boxes_xyxy[0])
        if res.mask.any():
            f.mask, f.mask_source = res.mask, "medsam"
        else:
            f.flags.append("mask_empty")


def attach_masks(out: ReaderOutput, img, unet=None, medsam=None) -> list[str]:
    """Attach whichever mask models are given: U-Net first, then MedSAM for findings still without a mask.

    A failing mask model never loses the finding: it is reported as a generic warning and the finding keeps its
    heatmap or box. Returns the warnings.
    """
    warnings: list[str] = []
    if unet is not None and out.findings:
        try:
            attach_unet_masks(out, img, unet)
        except Exception as exc:
            warnings.append(f"tumour mask unavailable ({type(exc).__name__})")
    if medsam is not None:
        pending = [f for f in out.findings if not f.mask_source]  # a CAM-region mask has no source and is replaced
        if pending:
            try:
                attach_medsam_masks(replace(out, findings=pending), img, medsam)
            except Exception as exc:
                warnings.append(f"lesion mask unavailable ({type(exc).__name__})")
    return warnings
