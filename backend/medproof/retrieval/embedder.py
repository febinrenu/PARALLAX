"""Image embedders for retrieval. The index records which embedder built it and refuses another.

BiomedCLIP (open_clip, ungated, MIT) is the working default. MedSigLIP (`google/medsiglip-448`) is the
plan's preferred model and shares the same interface; it is gated and needs accepted terms.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Protocol

import numpy as np
from PIL import Image


class Embedder(Protocol):
    name: str

    def embed(self, images: Sequence[Image.Image]) -> np.ndarray: ...


def embed_images(
    embedder: Embedder, items: Iterable[tuple[str, Callable[[], Image.Image]]], batch_size: int = 32
) -> tuple[list[str], np.ndarray, list[tuple[str, str]]]:
    """Embed lazily loaded images. Returns (ids, vectors, skipped); an unreadable image is skipped, not fatal."""
    ids: list[str] = []
    chunks: list[np.ndarray] = []
    skipped: list[tuple[str, str]] = []
    pend_ids: list[str] = []
    pend_imgs: list[Image.Image] = []

    def flush() -> None:
        if pend_imgs:
            chunks.append(np.asarray(embedder.embed(pend_imgs), dtype="float32"))
            ids.extend(pend_ids)
            pend_ids.clear()
            pend_imgs.clear()

    for image_id, load in items:
        try:
            img = load().convert("RGB")  # a truncated file opens fine and only fails when decoded here
        except (OSError, ValueError) as exc:
            skipped.append((image_id, f"{type(exc).__name__}: {exc}"))
            continue
        pend_ids.append(image_id)
        pend_imgs.append(img)
        if len(pend_imgs) >= batch_size:
            flush()
    flush()
    vectors = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 0), dtype="float32")
    return ids, vectors, skipped


class OpenClipEmbedder:
    name = "biomedclip"
    MODEL = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

    def __init__(self, device: str | None = None) -> None:
        import open_clip
        import torch

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model, _, preprocess = open_clip.create_model_and_transforms(self.MODEL)
        self.model = model.to(self.device).eval()
        self.preprocess = preprocess

    def embed(self, images: Sequence[Image.Image]) -> np.ndarray:
        batch = self._torch.stack([self.preprocess(im) for im in images]).to(self.device)
        with self._torch.inference_mode():
            feats = self.model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        out: np.ndarray = feats.float().cpu().numpy()
        return out


class MedSigLipEmbedder:
    """google/medsiglip-448 through transformers (image tower only). Gated: needs accepted terms and HF_TOKEN."""

    name = "medsiglip-448"
    MODEL = "google/medsiglip-448"

    def __init__(self, device: str | None = None) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(self.MODEL)  # images only: no text tokenizer needed
        self.model = AutoModel.from_pretrained(self.MODEL).to(self.device).eval()

    def embed(self, images: Sequence[Image.Image]) -> np.ndarray:
        inputs = self.processor(images=list(images), return_tensors="pt").to(self.device)
        with self._torch.inference_mode():
            feats = self.model.get_image_features(**inputs)
            feats = feats if hasattr(feats, "detach") else feats.pooler_output  # transformers 5 returns an output object
            feats = feats / feats.norm(dim=-1, keepdim=True)
        out: np.ndarray = feats.float().cpu().numpy()
        return out
