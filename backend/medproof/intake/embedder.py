"""Image and text embeddings for routing. MedSigLIP is loaded only through load_default().

The Embedder protocol is what the router depends on, so tests can supply a stand-in without
model access. Embeddings are L2-normalised float32, and cached on disk by (model id, sha256).
"""

from __future__ import annotations

import os
import re
from dataclasses import replace
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from medproof.intake.preprocess import get_spec, prepare
from medproof.intake.router_config import RouterConfig


class Embedder(Protocol):
    model_id: str
    dim: int
    logit_scale: float  # multiplies cosine similarity for zero-shot softmax

    def embed_images(self, images: Sequence[np.ndarray]) -> np.ndarray: ...

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray: ...


def _features(out):
    """transformers 4.x returns a tensor from get_*_features; 5.x returns an output object whose
    pooler_output is the embedding (verified equal, after L2 normalisation, to the model's own
    image_embeds / text_embeds)."""
    return out if hasattr(out, "detach") else out.pooler_output


def l2_normalise(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return (x / np.maximum(n, 1e-8)).astype(np.float32)


class EmbeddingCache:
    """One .npy per (model id, image sha256) under a directory."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _path(self, model_id: str, key: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", model_id)
        return self.root / safe / f"{key}.npy"

    def get(self, model_id: str, key: str) -> np.ndarray | None:
        p = self._path(model_id, key)
        try:
            return np.load(p, allow_pickle=False) if p.is_file() else None
        except (OSError, ValueError):
            return None  # a damaged entry is a miss

    def put(self, model_id: str, key: str, vec: np.ndarray) -> None:
        p = self._path(model_id, key)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, np.asarray(vec, dtype=np.float32), allow_pickle=False)


def _hf_token() -> str | None:
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    # .env in the working directory or any parent (the repo root when run from backend/)
    for folder in [Path.cwd(), *Path.cwd().parents]:
        env = folder / ".env"
        if env.is_file():
            for line in env.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.startswith("HF_TOKEN="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'") or None
            break
    return None


class MedSigLIPEmbedder:
    """google/medsiglip-448 through transformers: image tower and text tower, fp32."""

    def __init__(self, model, processor, cfg: RouterConfig):
        import torch

        self.model = model.eval()
        self.processor = processor
        self.cfg = cfg
        self.model_id = cfg.model_id
        self._torch = torch
        # Input size comes from the loaded model; the spec supplies channels and normalisation.
        size = getattr(getattr(model.config, "vision_config", None), "image_size", None)
        spec = get_spec(cfg.spec)
        self._spec = replace(spec, size=int(size)) if size else spec
        text_cfg = getattr(model.config, "text_config", None)
        self._max_len = min(64, int(getattr(text_cfg, "max_position_embeddings", 64)))
        with torch.no_grad():
            probe = self.embed_images([np.zeros((self._spec.size, self._spec.size), np.float32)])
        self.dim = int(probe.shape[1])
        scale = getattr(model, "logit_scale", None)
        self.logit_scale = float(scale.exp().item()) if scale is not None else 100.0

    def embed_images(self, images: Sequence[np.ndarray]) -> np.ndarray:
        torch = self._torch
        batch = np.stack([prepare(np.asarray(i, np.float32), self._spec) for i in images])
        with torch.no_grad():
            feats = self.model.get_image_features(pixel_values=torch.from_numpy(batch).to(self.cfg.device))
        return l2_normalise(_features(feats).detach().cpu().numpy())

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        torch = self._torch
        # SigLIP-family text towers were trained with fixed-length padding.
        tok = self.processor(text=list(texts), padding="max_length", max_length=self._max_len, truncation=True, return_tensors="pt")
        with torch.no_grad():
            feats = self.model.get_text_features(
                input_ids=tok["input_ids"].to(self.cfg.device),
                **({"attention_mask": tok["attention_mask"].to(self.cfg.device)} if "attention_mask" in tok else {}),
            )
        return l2_normalise(_features(feats).detach().cpu().numpy())


def load_default(cfg: RouterConfig | None = None) -> tuple[MedSigLIPEmbedder | None, str]:
    """Load MedSigLIP. Returns (embedder, "") or (None, reason). Never raises."""
    cfg = cfg or RouterConfig()
    try:
        from transformers import AutoModel, AutoProcessor
    except ImportError:
        return None, "transformers is not installed"
    token = _hf_token()
    try:
        kwargs = {"token": token} if token else {}
        model = AutoModel.from_pretrained(cfg.model_id, **kwargs)
        processor = AutoProcessor.from_pretrained(cfg.model_id, **kwargs)
        model.to(cfg.device)
        return MedSigLIPEmbedder(model, processor, cfg), ""
    except Exception as exc:  # gated repo, no token, offline, or an unexpected model layout
        msg = str(exc).splitlines()[0][:160] if str(exc) else type(exc).__name__
        return None, f"{type(exc).__name__}: {msg}"
