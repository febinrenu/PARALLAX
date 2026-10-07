"""Image-classifier readers for brain MRI and skin lesions (weights trained by the data owner).

A model directory holds `model_meta.json` plus the weights file named in it, as written by the
training notebooks and listed in `ml/artifacts/registry.json`. The reader rebuilds the network from
`arch` and `classes`, checks the weights hash, and uses the preprocessing spec named in the metadata,
so inference sees the same inputs as training. Localisation is Grad-CAM on the last feature map.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np

from medproof.core.schemas import StageResult
from medproof.intake.decode import DecodedImage
from medproof.intake.preprocess import PreprocSpec, get_spec, prepare
from medproof.readers.base import ReaderFinding, ReaderOutput
from medproof.readers.cam import CamExtractor, resize_cam
from medproof.readers.cxr import largest_region
from medproof.readers.cxr_config import CxrConfig
from medproof.readers.evidence import to_findings

BRAIN_NEGATIVE = ("notumor", "no_tumor", "no tumor", "normal")
SKIN_NEGATIVE: tuple = ()


class ModelBundleError(RuntimeError):
    """The model directory is missing, inconsistent or corrupt. The message names the problem."""


def spec_from_meta(spec) -> PreprocSpec:
    if isinstance(spec, str):
        return get_spec(spec)
    return PreprocSpec(
        name=spec["name"], size=int(spec["size"]), channels=int(spec["channels"]), mean=tuple(spec["mean"]), std=tuple(spec["std"]),
        resize=spec.get("resize", "stretch"), value_range=tuple(spec.get("value_range", (0.0, 1.0))),
    )


def layer_for_arch(arch: str) -> str:
    """Dotted path of the last feature map for the architectures the plan uses."""
    a = arch.lower()
    if a.startswith("efficientnet"):
        return "bn2"
    if a.startswith("convnext"):
        return "stages.3"
    raise ValueError(f"no default CAM layer for architecture {arch!r}; pass layer_getter")


def _default_factory(arch: str, n_classes: int):
    import timm

    return timm.create_model(arch, pretrained=False, num_classes=n_classes)


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


class ImageClassifierReader:
    def __init__(
        self,
        model,
        labels: list[str],
        spec: PreprocSpec,
        *,
        modality: str,
        model_id: str,
        layer_getter: Callable,
        cfg: CxrConfig | None = None,
        positive_threshold: float | None = None,
        negative_classes: tuple = (),
    ):
        import torch

        base = cfg or CxrConfig()
        self.cfg = replace(base, positive_threshold=base.positive_threshold if positive_threshold is None else positive_threshold)
        torch.set_num_threads(max(1, self.cfg.threads))
        self.model = model.eval()
        self.labels, self._spec = list(labels), spec
        self.modality, self.model_id = modality, model_id
        self._layer_getter = layer_getter
        self.negative = {n.lower() for n in negative_classes}

    # -- inference ---------------------------------------------------------------------------
    def _tensor(self, img):
        import torch

        return torch.from_numpy(prepare(img, self._spec))[None].to(self.cfg.device)

    def _forward(self, x) -> np.ndarray:
        import torch

        with torch.no_grad():
            return self.model(x)[0].detach().cpu().numpy().astype(np.float32)

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        e = np.exp(z - z.max())
        return (e / e.sum()).astype(np.float32)

    def score(self, img) -> np.ndarray:
        """Class probabilities without localisation (used by faithfulness and stability)."""
        return self._softmax(self._forward(self._tensor(img)))

    def predict(self, img: DecodedImage | np.ndarray) -> ReaderOutput:
        hw = img.shape if isinstance(img, DecodedImage) else tuple(np.asarray(img).shape[:2])
        x = self._tensor(img)
        raw = self._forward(x)
        probs = self._softmax(raw)
        out = ReaderOutput(modality=self.modality, model_id=self.model_id, labels=list(self.labels), logits=raw, probs=probs,
                           output_kind="probability", image_hw=(int(hw[0]), int(hw[1])))
        chosen = [i for i in np.argsort(-probs) if self.labels[i].lower() not in self.negative and probs[i] >= self.cfg.positive_threshold]
        chosen = chosen[: self.cfg.max_findings]
        if not chosen:
            return out
        with CamExtractor(self.model, self._layer_getter(self.model)) as cx:
            cams = cx.maps(x, [int(i) for i in chosen], self.cfg.cam_method)
        for i, small in zip(chosen, cams):
            cam = resize_cam(small, out.image_hw)
            rf = ReaderFinding(label=self.labels[i], logit=float(raw[i]), prob_raw=float(probs[i]), heatmap=cam, method=self.cfg.cam_method)
            region = largest_region(cam, self.cfg.cam_peak_frac)
            if region is None:
                rf.heatmap = None
                rf.flags.append("no_localization")
            else:
                rf.mask, box = region
                rf.boxes_xyxy, rf.box_scores = [box], [float(cam.max())]
            out.findings.append(rf)
        return out

    # -- construction ------------------------------------------------------------------------
    @classmethod
    def from_dir(
        cls,
        model_dir: str | Path,
        *,
        modality: str,
        model_factory: Callable | None = None,
        layer_getter: Callable | None = None,
        cfg: CxrConfig | None = None,
        positive_threshold: float | None = None,
        negative_classes: tuple | None = None,
    ) -> ImageClassifierReader:
        import torch

        d = Path(model_dir)
        try:
            meta = json.loads((d / "model_meta.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ModelBundleError(f"{d}: model_meta.json missing or unreadable") from exc
        wf = meta.get("weights_file")
        if not wf or not (d / wf).is_file():
            raise ModelBundleError(f"{d}: weights file {wf!r} not found; pull the training output first")
        want = meta.get("weights_sha256")
        if want and _sha256(d / wf) != want:
            raise ModelBundleError(f"{d / wf}: sha256 does not match model_meta.json; the file is corrupt")
        classes = list(meta["classes"])
        model = (model_factory or _default_factory)(meta["arch"], len(classes))
        try:
            state = torch.load(d / wf, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
        except Exception as exc:  # truncated file, wrong architecture, or an unsafe pickle
            raise ModelBundleError(f"{d / wf}: cannot load weights ({type(exc).__name__})") from exc
        getter = layer_getter or (lambda m, path=layer_for_arch(meta["arch"]): dict(m.named_modules())[path])
        neg = negative_classes if negative_classes is not None else (BRAIN_NEGATIVE if modality == "brain_mri" else SKIN_NEGATIVE)
        return cls(model, classes, spec_from_meta(meta["preproc_spec"]), modality=modality, model_id=meta["model_id"],
                   layer_getter=getter, cfg=cfg, positive_threshold=positive_threshold, negative_classes=neg)


def run(ctx, reader=None) -> StageResult:
    """Reader stage for classifier readers. `ctx` needs `.decoded`; `.artifact_dir` and id offsets are optional."""
    t0 = time.perf_counter()

    def done(ok, payload, warnings):
        return StageResult(stage="reader", ok=ok, ms=int((time.perf_counter() - t0) * 1000), payload=payload, warnings=warnings)

    img = getattr(ctx, "decoded", None)
    if img is None or reader is None:
        return done(False, {"error": "no decoded image or reader"}, ["reader skipped: missing inputs"])
    try:
        out = reader.predict(img)
        findings, warnings = to_findings(out, reader.cfg, finding_start=getattr(ctx, "finding_start", 1),
                                         evidence_start=getattr(ctx, "evidence_start", 1), artifact_dir=getattr(ctx, "artifact_dir", None))
    except Exception as exc:
        msg = f"reader unavailable ({type(exc).__name__})"
        return done(False, {"error": msg}, [msg])
    from medproof.readers.cxr import _share

    _share(ctx, reader, out, findings)
    payload = {"model_id": out.model_id, "labels": out.labels, "probs": [round(float(p), 5) for p in out.probs], "findings": [f.model_dump() for f in findings]}
    return done(True, payload, warnings)
