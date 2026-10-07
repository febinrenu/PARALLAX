"""Bone X-ray fracture reader: wraps a box detector (YOLO weights trained by the data owner).

The image-level score is the highest box confidence, which is also what the detector is validated
on (image-level AUROC). Findings carry boxes only; there is no heatmap, so evidence has kind "bbox".
The body part comes from the router and is carried as the region name.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np

from medproof.core.schemas import StageResult
from medproof.readers.base import ReaderFinding, ReaderOutput
from medproof.readers.classifier import ModelBundleError, _sha256
from medproof.readers.cxr_config import CxrConfig
from medproof.readers.evidence import to_findings

Detector = Callable[[np.ndarray], list]  # uint8 HxWx3 image -> [((x0, y0, x1, y1), confidence), ...]
LABEL = "fracture"
DEFAULT_THRESHOLD = 0.25  # the detector's usual confidence cut-off; the plan's calibration replaces it


def _image_u8(img) -> np.ndarray:
    a = getattr(img, "display", None)
    if a is None:
        a = np.asarray(img)
        if a.dtype != np.uint8:
            a = np.rint(np.clip(a.astype(np.float32), 0, 1) * 255).astype(np.uint8)
    a = np.asarray(a)
    return np.repeat(a[..., None], 3, axis=2) if a.ndim == 2 else a


class BoneReader:
    def __init__(self, detector: Detector, *, model_id: str, cfg: CxrConfig | None = None):
        base = cfg or CxrConfig()
        self.cfg = replace(base, positive_threshold=base.positive_threshold if cfg else DEFAULT_THRESHOLD)
        self.detector, self.model_id = detector, model_id

    def _detections(self, img):
        rgb = _image_u8(img)
        h, w = rgb.shape[:2]
        dets = []
        for box, score in self.detector(rgb):
            x0, y0, x1, y1 = (float(v) for v in box)
            clipped = (max(0.0, x0), max(0.0, y0), min(float(w), x1), min(float(h), y1))
            if clipped[2] > clipped[0] and clipped[3] > clipped[1]:
                dets.append((clipped, float(score)))
        return sorted(dets, key=lambda d: -d[1]), (h, w)

    def score(self, img) -> np.ndarray:
        dets, _ = self._detections(img)
        return np.array([dets[0][1] if dets else 0.0], np.float32)

    def predict(self, img, body_part: str | None = None) -> ReaderOutput:
        dets, hw = self._detections(img)
        top = dets[0][1] if dets else 0.0
        out = ReaderOutput(modality="bone_xray", model_id=self.model_id, labels=[LABEL], logits=np.array([top], np.float32),
                           probs=np.array([top], np.float32), output_kind="probability", image_hw=hw)
        keep = [d for d in dets if d[1] >= self.cfg.positive_threshold]
        if keep:
            out.findings.append(ReaderFinding(
                label=LABEL, logit=top, prob_raw=top, boxes_xyxy=[d[0] for d in keep], box_scores=[d[1] for d in keep],
                method="yolo", region_name=body_part,
            ))
        return out

    @classmethod
    def from_dir(cls, model_dir: str | Path, detector_factory: Callable | None = None, cfg: CxrConfig | None = None) -> BoneReader:
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
        imgsz = int(meta.get("imgsz", 640))
        try:
            detector = (detector_factory or _ultralytics_detector)(d / wf, imgsz)
        except Exception as exc:
            raise ModelBundleError(f"{d / wf}: cannot load detector ({type(exc).__name__})") from exc
        return cls(detector, model_id=meta["model_id"], cfg=cfg)


def _ultralytics_detector(weights: Path, imgsz: int) -> Detector:
    from ultralytics import YOLO

    model = YOLO(str(weights))

    def detect(rgb: np.ndarray) -> list:
        res = model.predict(rgb, imgsz=imgsz, conf=0.001, verbose=False)[0]
        return [(tuple(b), float(c)) for b, c in zip(res.boxes.xyxy.cpu().numpy().tolist(), res.boxes.conf.cpu().numpy().tolist())]

    return detect


def run(ctx, reader: BoneReader | None = None) -> StageResult:
    """Reader stage. `ctx` needs `.decoded`; `.body_part`, `.artifact_dir` and id offsets are optional."""
    t0 = time.perf_counter()

    def done(ok, payload, warnings):
        return StageResult(stage="reader", ok=ok, ms=int((time.perf_counter() - t0) * 1000), payload=payload, warnings=warnings)

    img = getattr(ctx, "decoded", None)
    if img is None or reader is None:
        return done(False, {"error": "no decoded image or reader"}, ["reader skipped: missing inputs"])
    try:
        out = reader.predict(img, getattr(ctx, "body_part", None))
        findings, warnings = to_findings(out, reader.cfg, finding_start=getattr(ctx, "finding_start", 1),
                                         evidence_start=getattr(ctx, "evidence_start", 1), artifact_dir=getattr(ctx, "artifact_dir", None))
    except Exception as exc:
        msg = f"bone reader unavailable ({type(exc).__name__})"
        return done(False, {"error": msg}, [msg])
    from medproof.readers.cxr import _share

    _share(ctx, reader, out, findings)
    return done(True, {"model_id": out.model_id, "probs": [round(float(p), 5) for p in out.probs], "findings": [f.model_dump() for f in findings]}, warnings)
