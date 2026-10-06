"""Chest X-ray specialist reader: TorchXRayVision DenseNet121 plus Grad-CAM localisation.

predict() returns a ReaderOutput whose findings carry a heatmap, box and patient-side region
name on the original image grid. evidence.to_findings() turns that into contract objects.
The model and the anatomy segmenter are injectable so the logic is testable without weights.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from medproof.core.schemas import StageResult
from medproof.intake.decode import DecodedImage
from medproof.intake.preprocess import center_crop_box, get_spec, prepare
from medproof.readers.base import ReaderFinding, ReaderOutput
from medproof.readers.cam import CamExtractor, resize_cam
from medproof.readers.cxr_config import CxrConfig
from medproof.readers.evidence import to_findings
from medproof.readers.zones import Anatomy, laterality_ok, name_region

AnatomyFn = Callable[[DecodedImage | np.ndarray], Anatomy | None]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def largest_region(cam: np.ndarray, peak_frac: float) -> tuple[np.ndarray, tuple[float, float, float, float]] | None:
    """Connected area of `cam` at or above peak_frac of its maximum, containing the peak.

    Returns (bool mask, bbox xyxy with exclusive max edge) or None for a flat map.
    """
    peak = float(cam.max())
    if peak <= 0.0:
        return None
    binary = (cam >= peak * peak_frac).astype(np.uint8)
    n, labels = cv2.connectedComponents(binary, connectivity=8)
    py, px = np.unravel_index(int(cam.argmax()), cam.shape)
    lab = labels[py, px]
    if n <= 1 or lab == 0:
        return None
    mask = labels == lab
    ys, xs = np.nonzero(mask)
    return mask, (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


class CxrReader:
    def __init__(
        self,
        model: Any,
        *,
        anatomy_fn: AnatomyFn | None = None,
        cfg: CxrConfig | None = None,
        model_id: str | None = None,
        output_kind: str = "probability",
    ):
        self.cfg = cfg or CxrConfig()
        import torch

        torch.set_num_threads(max(1, self.cfg.threads))
        self.model = model.eval()
        self.anatomy_fn = anatomy_fn
        self.model_id = model_id or f"{self.cfg.model_name}@unknown"
        self.output_kind = output_kind
        self.labels = [str(p) for p in model.pathologies]
        self._spec = get_spec(self.cfg.spec)

    # -- inference ---------------------------------------------------------------------------
    def _tensor(self, img: DecodedImage | np.ndarray):
        import torch

        arr = prepare(img, self._spec)
        return torch.from_numpy(arr)[None].to(self.cfg.device)

    def _forward(self, x) -> np.ndarray:
        import torch

        with torch.no_grad():
            return self.model(x)[0].detach().cpu().numpy().astype(np.float32)

    def _probs(self, raw: np.ndarray) -> np.ndarray:
        return raw if self.output_kind == "probability" else _sigmoid(raw)

    def raw_logits(self, img: DecodedImage | np.ndarray) -> np.ndarray:
        """Pre-sigmoid logits of every label (classifier on the pooled features), used by the energy score.

        The library's forward() applies sigmoid and operating-point scaling, which hides the logit scale.
        """
        import torch

        with torch.no_grad():
            z = self.model.classifier(self.model.features2(self._tensor(img)))
        return z[0].detach().cpu().numpy().astype(np.float32)

    def score(self, img: DecodedImage | np.ndarray) -> np.ndarray:
        """Probabilities for every label, without localisation. Used by faithfulness and stability."""
        return self._probs(self._forward(self._tensor(img)))

    def predict(self, img: DecodedImage | np.ndarray) -> ReaderOutput:
        hw = img.shape if isinstance(img, DecodedImage) else tuple(np.asarray(img).shape[:2])
        x = self._tensor(img)
        # The model sees only the central square (its training convention); keep that window so
        # maps and boxes land back on the full original grid.
        y0, x0, side = (
            center_crop_box(int(hw[0]), int(hw[1])) if self._spec.resize == "center_crop"
            else (0, 0, max(int(hw[0]), int(hw[1])))
        )
        raw = self._forward(x)
        probs = self._probs(raw)
        out = ReaderOutput(
            modality="cxr", model_id=self.model_id, labels=list(self.labels), logits=raw, probs=probs,
            output_kind=self.output_kind, image_hw=(int(hw[0]), int(hw[1])),
        )
        lost = 1.0 - (side * side) / float(hw[0] * hw[1]) if self._spec.resize == "center_crop" else 0.0
        if lost > 0.10:
            out.warnings.append(
                f"image is not square: the model reads the central square only ({lost:.0%} of the image, "
                "at the sides or top and bottom, was not analysed)"
            )
            out.flags.append("partial_view")
        positives = [i for i in np.argsort(-probs) if self.labels[i] and probs[i] >= self.cfg.positive_threshold]
        positives = positives[: self.cfg.max_findings]
        if not positives:
            return out

        anatomy = None
        if self.anatomy_fn is not None:
            try:
                anatomy = self.anatomy_fn(img)
            except Exception as exc:  # a failed segmenter must not lose the findings
                out.warnings.append(f"anatomy segmentation failed ({type(exc).__name__}): regions not named")
        if anatomy is None:
            out.flags.append("anatomy_unavailable")
        elif laterality_ok(anatomy) is False:
            out.flags.append("laterality_uncertain")
            out.warnings.append(
                "heart lies on the image-left of the lungs: image may be flipped or dextrocardia; "
                "left/right region names are unverified"
            )

        with CamExtractor(self.model, self.model.features) as cx:
            cams = cx.maps(x, [int(i) for i in positives], self.cfg.cam_method)
        for i, cam_small in zip(positives, cams):
            cam = np.zeros(out.image_hw, np.float32)
            if self._spec.resize == "center_crop":
                cam[y0 : y0 + side, x0 : x0 + side] = resize_cam(cam_small, (side, side))
            else:
                cam = resize_cam(cam_small, out.image_hw)
            rf = ReaderFinding(
                label=self.labels[i], logit=float(raw[i]), prob_raw=float(probs[i]),
                heatmap=cam, method=self.cfg.cam_method,
            )
            region = largest_region(cam, self.cfg.cam_peak_frac)
            if region is None:
                rf.heatmap = None  # a flat map is not evidence of where the finding is
                rf.flags.append("no_localization")
            else:
                rf.mask, box = region
                rf.boxes_xyxy, rf.box_scores = [box], [float(cam.max())]
                if anatomy is not None:
                    rf.region_name = name_region(rf.mask, anatomy).name
            out.findings.append(rf)
        return out

    # -- construction ------------------------------------------------------------------------
    @classmethod
    def load(cls, cfg: CxrConfig | None = None, anatomy_fn: AnatomyFn | None = None) -> CxrReader:
        """Load TorchXRayVision weights (downloaded on first use) and wrap them."""
        import torchxrayvision as xrv

        cfg = cfg or CxrConfig()
        model = xrv.models.DenseNet(weights=cfg.weights)
        model.to(cfg.device)
        sha8 = _weights_sha8(model)
        # Verified in torchxrayvision 1.5.5: with op_threshs set (all pretrained sets) forward()
        # returns sigmoid then operating-point normalisation, i.e. probabilities where 0.5 is the
        # label's operating point. Without either, outputs are raw logits.
        has_ops = getattr(model, "op_threshs", None) is not None
        kind = "probability" if has_ops or getattr(model, "apply_sigmoid", False) else "logit"
        return cls(model, anatomy_fn=anatomy_fn, cfg=cfg, model_id=f"{cfg.model_name}@{sha8}", output_kind=kind)


def _weights_sha8(model: Any) -> str:
    path = getattr(model, "weights_filename_local", None)
    if path and Path(path).is_file():
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()[:8]
    return "unknown"


# -- stage entry point -----------------------------------------------------------------------
_READER: CxrReader | None = None


def get_reader() -> CxrReader:
    global _READER
    if _READER is None:
        from medproof.readers.anatomy import AnatomySegmenter

        _READER = CxrReader.load(anatomy_fn=AnatomySegmenter.try_load())
    return _READER


def run(ctx, reader: CxrReader | None = None) -> StageResult:
    """Reader stage. `ctx` needs `.decoded` (DecodedImage); `.artifact_dir` and id offsets are optional."""
    t0 = time.perf_counter()

    def done(ok: bool, payload: dict, warnings: list[str]) -> StageResult:
        return StageResult(stage="reader", ok=ok, ms=int((time.perf_counter() - t0) * 1000), payload=payload, warnings=warnings)

    img = getattr(ctx, "decoded", None)
    if img is None:
        return done(False, {"error": "no decoded image"}, ["no decoded image"])
    try:
        rdr = reader or get_reader()
        out = rdr.predict(img)
        findings, warnings = to_findings(
            out, rdr.cfg,
            finding_start=getattr(ctx, "finding_start", 1),
            evidence_start=getattr(ctx, "evidence_start", 1),
            artifact_dir=getattr(ctx, "artifact_dir", None),
        )
    except Exception as exc:  # any model, weight or runtime failure degrades to a failed stage
        msg = f"CXR reader unavailable ({type(exc).__name__})"
        return done(False, {"error": msg}, [msg])
    payload = {
        "model_id": out.model_id,
        "labels": out.labels,
        "probs": [round(float(p), 5) for p in (out.probs if out.probs is not None else [])],
        "flags": out.flags,
        "findings": [f.model_dump() for f in findings],
    }
    return done(True, payload, warnings)
