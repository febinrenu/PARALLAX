from __future__ import annotations

import numpy as np
from PIL import Image

from medproof.retrieval.embedder import embed_images


class FakeEmbedder:
    name = "fake"

    def __init__(self):
        self.batches = []

    def embed(self, images):
        self.batches.append(len(images))
        # one vector per image: mean brightness in the first slot, so alignment is checkable
        return np.array([[float(np.asarray(im).mean()), 1.0, 0.0] for im in images], dtype="float32")


def loader(value):
    return lambda: Image.new("RGB", (8, 8), (value, value, value))


def broken():
    raise OSError("image file is truncated")


def test_vectors_stay_aligned_with_ids_and_are_batched():
    emb = FakeEmbedder()
    items = [(f"i{k}", loader(k * 10)) for k in range(7)]
    ids, vecs, skipped = embed_images(emb, items, batch_size=3)
    assert ids == [f"i{k}" for k in range(7)] and emb.batches == [3, 3, 1] and skipped == []
    assert [round(float(v[0])) for v in vecs] == [k * 10 for k in range(7)]


def test_unreadable_images_are_skipped_and_reported_not_fatal():
    emb = FakeEmbedder()
    items = [("a", loader(10)), ("bad", broken), ("c", loader(30))]
    ids, vecs, skipped = embed_images(emb, items, batch_size=2)
    assert ids == ["a", "c"] and len(vecs) == 2 and [s[0] for s in skipped] == ["bad"] and "truncated" in skipped[0][1]


def test_nothing_to_embed_gives_an_empty_result():
    ids, vecs, skipped = embed_images(FakeEmbedder(), [], batch_size=4)
    assert ids == [] and vecs.shape[0] == 0 and skipped == []


def test_a_truncated_file_that_only_fails_when_decoded_is_skipped_too():
    import io

    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (200, 10, 10)).save(buf, "PNG")
    cut = buf.getvalue()[: len(buf.getvalue()) // 2]  # opens fine, fails on load/convert

    def lazy():
        return Image.open(io.BytesIO(cut))

    emb = FakeEmbedder()
    ids, vecs, skipped = embed_images(emb, [("ok", loader(50)), ("cut", lazy)], batch_size=4)
    assert ids == ["ok"] and [s[0] for s in skipped] == ["cut"]
