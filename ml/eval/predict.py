"""Load the trained models from their artifacts and run batches through them (used by the corruption benchmark and trust-signal runs).

Weights live in ml/artifacts/<model>/weights.pt (never committed). Everything here uses the same preprocessing path as serving:
the product decoder, then `prepare(spec)`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
for p in (REPO, REPO / "backend"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

ART = REPO / "ml" / "artifacts"


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def meta(name: str) -> dict:
    return json.loads((ART / name / "model_meta.json").read_text())


def load_timm(name: str):
    import timm

    m = meta(name)
    model = timm.create_model(m["arch"], pretrained=False, num_classes=len(m["classes"]))
    model.load_state_dict(torch.load(ART / name / m["weights_file"], map_location="cpu"))
    return model.eval().to(device())


def load_unet(name: str):
    import segmentation_models_pytorch as smp

    m = meta(name)
    model = smp.Unet(m["arch"].split("(")[1].rstrip(")"), encoder_weights=None, in_channels=3, classes=1)
    model.load_state_dict(torch.load(ART / name / m["weights_file"], map_location="cpu"))
    return model.eval().to(device())


def load_yolo(name: str):
    from ultralytics import YOLO

    return YOLO(str(ART / name / meta(name)["weights_file"]))


def load_cxr(tag: str):
    import torchxrayvision as xrv

    from ml.eval.cxr import WEIGHTS

    return xrv.models.DenseNet(weights=WEIGHTS[tag][0]).eval().to(device())


@torch.no_grad()
def logits(model, X: torch.Tensor, bs: int = 128) -> np.ndarray:
    dev = next(model.parameters()).device
    out = []
    for i in range(0, len(X), bs):
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            out.append(model(X[i : i + bs].to(dev).float()).float().cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def seg_probs(model, X: torch.Tensor, bs: int = 32) -> np.ndarray:
    dev = next(model.parameters()).device
    out = []
    for i in range(0, len(X), bs):
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            out.append(torch.sigmoid(model(X[i : i + bs].to(dev).float())).float().cpu())
    return torch.cat(out)[:, 0].numpy()


def yolo_scores(model, images: list[np.ndarray], imgsz: int = 640, bs: int = 16) -> np.ndarray:
    """Image-level fracture score = highest box confidence. Images are float [0, 1] HxW or HxWxC arrays."""
    import cv2

    out = []
    for i in range(0, len(images), bs):
        arrs = []
        for a in images[i : i + bs]:
            u8 = np.rint(np.clip(a, 0, 1) * 255).astype(np.uint8)
            arrs.append(cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR) if u8.ndim == 2 else cv2.cvtColor(u8, cv2.COLOR_RGB2BGR))
        for r in model.predict(arrs, imgsz=imgsz, conf=0.001, iou=0.6, max_det=100, verbose=False):
            out.append(float(r.boxes.conf.max()) if len(r.boxes) else 0.0)
    return np.array(out, dtype=np.float32)
