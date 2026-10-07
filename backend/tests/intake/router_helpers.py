"""Test doubles for router logic tests.

TestDoubleEmbedder is NOT a model. It maps an image to a small deterministic vector made of
simple image statistics, so the probe, confidence, zero-shot and cache logic can be tested
without model access. No test using it says anything about real modality accuracy.
"""

from __future__ import annotations

import numpy as np

from medproof.intake.router_config import CLASSES

DIM = 16


def _stats(img: np.ndarray) -> np.ndarray:
    g = img.mean(axis=2) if img.ndim == 3 else img
    h, w = g.shape
    cells = [g[: h // 2, : w // 2], g[: h // 2, w // 2 :], g[h // 2 :, : w // 2], g[h // 2 :, w // 2 :]]
    f = [float(c.mean()) for c in cells] + [float(c.std()) for c in cells]
    f += [float(g.mean()), float(g.std()), float(np.abs(np.diff(g, axis=0)).mean()), float(np.abs(np.diff(g, axis=1)).mean())]
    f += [float(img.ndim == 3), float(g.max()), float(g.min()), 1.0]
    return np.asarray(f, np.float32)


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return (v / np.maximum(n, 1e-8)).astype(np.float32)


class TestDoubleEmbedder:
    __test__ = False  # not a pytest class
    model_id = "test-double-embedder"
    dim = DIM
    logit_scale = 20.0

    def __init__(self, text_vectors: dict[str, np.ndarray] | None = None):
        self.image_calls = 0
        self.text_calls = 0
        self._text = text_vectors or {}

    def embed_images(self, images):
        self.image_calls += len(images)
        return _unit(np.stack([_stats(np.asarray(i, np.float32)) for i in images]))

    def embed_texts(self, texts):
        self.text_calls += len(texts)
        out = []
        for t in texts:
            if t not in self._text:
                raise KeyError(f"no test vector for prompt {t!r}")
            out.append(self._text[t])
        return _unit(np.stack(out))


def prototype_vectors(config) -> dict[str, np.ndarray]:
    """One orthogonal direction per class, used as fixed 'text' vectors for zero-shot logic tests."""
    vecs = {}
    for k, cls in enumerate(CLASSES):
        v = np.zeros(DIM, np.float32)
        v[k] = 1.0
        for p in config.prompts[cls]:
            vecs[p] = v
    return vecs


def clustered(n_per_class=40, groups_per_class=8, dim=DIM, seed=0, noise=0.15):
    """Synthetic separable embeddings: X, y (class names), groups. Logic tests only."""
    rng = np.random.default_rng(seed)
    X, y, g = [], [], []
    for k, cls in enumerate(CLASSES):
        centre = np.zeros(dim, np.float32)
        centre[k] = 1.0
        for i in range(n_per_class):
            X.append(_unit(centre + rng.normal(0, noise, dim).astype(np.float32)))
            y.append(cls)
            g.append(f"{cls}-g{i % groups_per_class}")
    return np.stack(X), np.asarray(y), np.asarray(g)
