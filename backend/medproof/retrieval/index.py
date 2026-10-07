"""Precedent retrieval (D8): the nearest confirmed public cases for a finding, with their true labels.

An exact cosine-similarity index (FAISS inner product on L2-normalised vectors) per dataset. It is
built only from training images, so a precedent can never be a test image, and a query excludes its
own id and its own lesion or patient group so near-duplicates do not inflate similarity.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from medproof.core.schemas import Precedent


@dataclass(frozen=True)
class Hit:
    id: str
    label: str
    score: float  # cosine similarity


def _normalise(v: np.ndarray) -> np.ndarray:
    v = np.ascontiguousarray(v, dtype="float32")
    norms = np.linalg.norm(v, axis=-1, keepdims=True)
    out: np.ndarray = v / np.maximum(norms, 1e-12)
    return out


class PrecedentIndex:
    def __init__(
        self, dataset: str, embedder: str, ids: list[str], labels: list[str], groups: list[str], vectors: np.ndarray
    ) -> None:
        self.dataset, self.embedder = dataset, embedder
        self.ids, self.labels, self.groups, self.vectors = ids, labels, groups, vectors
        dim = vectors.shape[1] if vectors.ndim == 2 and vectors.shape[0] else (vectors.shape[1] if vectors.ndim == 2 else 0)
        self.dim = dim
        self._faiss = faiss.IndexFlatIP(dim) if dim else None
        if self._faiss is not None and len(ids):
            self._faiss.add(vectors)

    @classmethod
    def build(
        cls,
        dataset: str,
        embedder: str,
        ids: Sequence[str],
        labels: Sequence[str],
        vectors: np.ndarray,
        groups: Sequence[str] | None = None,
    ) -> PrecedentIndex:
        if not (len(ids) == len(labels) == len(vectors)):
            raise ValueError("ids, labels and vectors must have the same length")
        if groups is not None and len(groups) != len(ids):
            raise ValueError("groups must have one entry per id")
        return cls(dataset, embedder, list(ids), list(labels), list(groups) if groups is not None else list(ids), _normalise(vectors))

    def __len__(self) -> int:
        return len(self.ids)

    def search(
        self, vec: np.ndarray, k: int = 5, *, exclude_ids: Sequence[str] = (), exclude_groups: Sequence[str] = ()
    ) -> list[Hit]:
        if not len(self) or self._faiss is None:
            return []
        skip_ids, skip_groups = set(exclude_ids), set(exclude_groups)
        n_skipped = sum(1 for i, g in zip(self.ids, self.groups, strict=True) if i in skip_ids or g in skip_groups)
        want = min(len(self), k + n_skipped)
        scores, idx = self._faiss.search(_normalise(np.asarray(vec).reshape(1, -1)), want)
        hits: list[Hit] = []
        for s, j in zip(scores[0], idx[0], strict=True):
            if j < 0 or self.ids[j] in skip_ids or self.groups[j] in skip_groups:
                continue
            hits.append(Hit(self.ids[j], self.labels[j], float(s)))
            if len(hits) == k:
                break
        return hits

    def save(self, prefix: Path) -> None:
        prefix = Path(prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        np.save(prefix.with_suffix(".npy"), self.vectors)
        prefix.with_suffix(".json").write_text(
            json.dumps({"dataset": self.dataset, "embedder": self.embedder, "ids": self.ids, "labels": self.labels, "groups": self.groups}),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, prefix: Path, expect_embedder: str | None = None) -> PrecedentIndex:
        prefix = Path(prefix)
        meta = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
        if expect_embedder is not None and meta["embedder"] != expect_embedder:
            raise ValueError(f"index was built with embedder {meta['embedder']!r}, not {expect_embedder!r}")
        vectors = np.load(prefix.with_suffix(".npy"))
        return cls(meta["dataset"], meta["embedder"], meta["ids"], meta["labels"], meta["groups"], vectors)


def to_precedents(hits: Sequence[Hit], dataset: str, thumb_ref: Callable[[Hit], str]) -> list[Precedent]:
    return [
        Precedent(case_id=h.id, dataset=dataset, label=h.label, similarity=min(1.0, max(0.0, h.score)), thumb_ref=thumb_ref(h))
        for h in hits
    ]


def precision_at_k(neighbour_labels: Sequence[Sequence[str]], query_labels: Sequence[str]) -> float:
    """Mean over queries of the share of returned neighbours whose label equals the query's label."""
    if not len(query_labels):
        return 0.0
    per = [sum(1 for x in nb if x == q) / len(nb) if len(nb) else 0.0 for nb, q in zip(neighbour_labels, query_labels, strict=True)]
    return float(np.mean(per))
