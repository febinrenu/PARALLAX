"""Modality router: embedding -> trained probe, with a zero-shot text-prompt fallback.

1. A trained probe (logistic regression on image embeddings, temperature-scaled) gives the
   modality and a calibrated confidence.
2. If there is no probe for this embedder, or its confidence is below `min_confidence`, the
   router scores the image against text prompts instead (method "zero_shot").
3. If even that is below `zero_shot_floor`, the modality is "other" with flag
   `router_uncertain`, so downstream stages treat the image as unsupported.
For bone X-rays a second probe head names the body part when it was trained.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from medproof.core.schemas import StageResult
from medproof.intake.decode import DecodedImage
from medproof.intake.embedder import Embedder, EmbeddingCache
from medproof.intake.router_config import CLASSES, RouterConfig
from medproof.intake.ood import OODModel
from medproof.intake.router_train import Probe

LAST_LOAD_ERROR = ""


@dataclass
class RouterOutput:
    modality: str
    body_part: str | None
    confidence: float
    probs: dict[str, float]
    method: str  # "probe" | "zero_shot"
    flags: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    ood: dict | None = None  # {distance, threshold, score, is_ood}; score above 1 means out of distribution

    def to_dict(self) -> dict:
        return {
            "modality": self.modality,
            "body_part": self.body_part,
            "confidence": round(float(self.confidence), 5),
            "probs": {k: round(float(v), 5) for k, v in self.probs.items()},
            "method": self.method,
            "flags": list(self.flags),
            "ood": None if self.ood is None else {k: (round(float(v), 5) if not isinstance(v, bool) else v) for k, v in self.ood.items()},
        }


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


class Router:
    def __init__(
        self,
        embedder: Embedder,
        *,
        config: RouterConfig | None = None,
        cache: EmbeddingCache | None = None,
        probe: Probe | None = None,
        ood: OODModel | None = None,
    ):
        self.embedder = embedder
        self.cfg = config or RouterConfig()
        self.cache = cache
        self._load_warnings: list[str] = []
        self.probe = probe if probe is not None else self._load_probe()
        self.ood = ood if ood is not None else self._load_ood()
        self._prototypes: np.ndarray | None = None

    # -- setup ------------------------------------------------------------------------------
    def _load_probe(self) -> Probe | None:
        path = Path(self.cfg.probe_path)
        if not path.is_file():
            self._load_warnings.append(f"no trained probe at {path}: routing by zero-shot prompts only")
            return None
        try:
            probe = Probe.load(path)
        except (OSError, ValueError, KeyError) as exc:
            self._load_warnings.append(f"probe unreadable ({type(exc).__name__}): routing by zero-shot prompts only")
            return None
        if probe.embedder_id != self.embedder.model_id:
            self._load_warnings.append(
                f"probe was trained with embedder {probe.embedder_id!r}, not {self.embedder.model_id!r}: "
                "ignoring it and routing by zero-shot prompts"
            )
            return None
        if probe.dim != self.embedder.dim:
            self._load_warnings.append("probe and embedder dimensions differ: ignoring the probe")
            return None
        return probe

    def _load_ood(self) -> OODModel | None:
        path = Path(self.cfg.ood_path)
        if not path.is_file():
            return None
        try:
            model = OODModel.load(path)
        except (OSError, ValueError, KeyError) as exc:
            self._load_warnings.append(f"OOD model unreadable ({type(exc).__name__}): out-of-distribution check off")
            return None
        if model.embedder_id != self.embedder.model_id or model.dim != self.embedder.dim:
            self._load_warnings.append("OOD model belongs to another embedder: out-of-distribution check off")
            return None
        return model

    def _protos(self) -> np.ndarray:
        if self._prototypes is None:
            rows = []
            for cls in CLASSES:
                v = self.embedder.embed_texts(list(self.cfg.prompts[cls])).mean(axis=0)
                rows.append(v / max(float(np.linalg.norm(v)), 1e-8))
            self._prototypes = np.stack(rows).astype(np.float32)
        return self._prototypes

    # -- inference --------------------------------------------------------------------------
    def _embed(self, img: DecodedImage | np.ndarray) -> np.ndarray:
        if isinstance(img, DecodedImage):
            if self.cache is not None:
                hit = self.cache.get(self.embedder.model_id, img.sha256)
                if hit is not None:
                    return hit
            v = self.embedder.embed_images([img.analysis])[0]
            if self.cache is not None:
                self.cache.put(self.embedder.model_id, img.sha256, v)
            return v
        return self.embedder.embed_images([np.asarray(img, np.float32)])[0]

    def predict(self, img: DecodedImage | np.ndarray) -> RouterOutput:
        emb = self._embed(img)
        out = self._route(emb)
        if self.ood is not None:
            out.ood = self.ood.score(emb, out.modality)
            if out.ood is not None and out.ood["is_ood"]:
                out.flags.append("ood")
                out.warnings.append(f"image is unusual for a {out.modality} (score {out.ood['score']:.2f}): treat results with extra caution")
        return out

    def _route(self, emb: np.ndarray) -> RouterOutput:
        warnings = list(self._load_warnings)
        if self.probe is not None:
            p = self.probe.predict_proba(emb)[0]
            probs = dict(zip(self.probe.classes, map(float, p)))
            top = max(probs, key=probs.get)
            if probs[top] >= self.cfg.min_confidence:
                return RouterOutput(top, self._body_part(top, emb), probs[top], probs, "probe", [], warnings)
            warnings.append(f"probe confidence {probs[top]:.2f} below {self.cfg.min_confidence:.2f}: used zero-shot prompts")
        zs = _softmax(self.embedder.logit_scale * (self._protos() @ emb))
        probs = dict(zip(CLASSES, map(float, zs)))
        top = max(probs, key=probs.get)
        flags: list[str] = []
        if probs[top] < self.cfg.zero_shot_floor:
            flags.append("router_uncertain")
            warnings.append(f"zero-shot confidence {probs[top]:.2f} below {self.cfg.zero_shot_floor:.2f}: treated as unsupported")
            top = "other"
        return RouterOutput(top, self._body_part(top, emb), probs[max(probs, key=probs.get)] if flags else probs[top], probs, "zero_shot", flags, warnings)

    def _body_part(self, modality: str, emb: np.ndarray) -> str | None:
        if modality != "bone_xray" or self.probe is None:
            return None
        bp = self.probe.predict_body_part(emb)
        if bp is None:
            return None
        return self.probe.bp_classes[int(bp[0].argmax())]


# -- stage entry point ---------------------------------------------------------------------
_DEFAULT: Router | None = None
_DEFAULT_TRIED = False


def get_default_router() -> Router | None:
    """The shared router, or None. Auto-enabled only when a trained probe exists (or MEDPROOF_ROUTER=1),
    so nothing tries to download a model by accident. Failure is remembered in LAST_LOAD_ERROR."""
    global _DEFAULT, _DEFAULT_TRIED, LAST_LOAD_ERROR
    if _DEFAULT_TRIED:
        return _DEFAULT
    cfg = RouterConfig()
    if os.environ.get("MEDPROOF_ROUTER") == "0" or not (Path(cfg.probe_path).is_file() or os.environ.get("MEDPROOF_ROUTER") == "1"):
        return None
    _DEFAULT_TRIED = True
    from medproof.intake.embedder import load_default

    emb, reason = load_default(cfg)
    if emb is None:
        LAST_LOAD_ERROR = reason
        return None
    _DEFAULT = Router(emb, config=cfg, cache=EmbeddingCache(cfg.cache_dir))
    return _DEFAULT


def run(ctx, router: Router | None = None) -> StageResult:
    """Router stage. `ctx` needs `.decoded`; an explicit `router` or `ctx.router` overrides the default."""
    t0 = time.perf_counter()

    def done(ok: bool, payload: dict, warnings: list[str]) -> StageResult:
        return StageResult(stage="router", ok=ok, ms=int((time.perf_counter() - t0) * 1000), payload=payload, warnings=warnings)

    img = getattr(ctx, "decoded", None)
    if img is None:
        return done(False, {"error": "no decoded image"}, ["no decoded image"])
    rt = router or getattr(ctx, "router", None) or get_default_router()
    if rt is None:
        msg = "router unavailable" + (f" ({LAST_LOAD_ERROR})" if LAST_LOAD_ERROR else "")
        return done(False, {"error": "router_unavailable"}, [msg])
    try:
        out = rt.predict(img)
    except Exception as exc:  # any model or runtime failure degrades to a failed stage
        msg = f"router failed ({type(exc).__name__})"
        return done(False, {"error": "router_failed"}, [msg])
    return done(True, out.to_dict(), out.warnings)
