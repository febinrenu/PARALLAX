from __future__ import annotations

import io

import numpy as np
from PIL import Image

from ml.data import phash as ph


def _scene(seed: int, size: int = 256) -> np.ndarray:
    """Smooth random scene with structure at several scales (a stand-in for a radiograph)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(size, size)).astype(np.float32)
    import cv2

    img = sum(cv2.GaussianBlur(x, (0, 0), s) * s for s in (3, 8, 20))
    img = (img - img.min()) / (img.max() - img.min())
    return (img * 255).astype(np.uint8)


def _jpeg(a: np.ndarray, q: int) -> np.ndarray:
    b = io.BytesIO()
    Image.fromarray(a).save(b, "JPEG", quality=q)
    return np.asarray(Image.open(io.BytesIO(b.getvalue())))


def test_identical_images_hash_equal():
    a = _scene(1)
    assert ph.phash_array(a) == ph.phash_array(a.copy())


def test_recompression_and_resize_stay_within_duplicate_threshold():
    a = _scene(2)
    base = ph.phash_array(a)
    assert ph.hamming(base, ph.phash_array(_jpeg(a, 60))) <= ph.DUP_MAX_DIST
    import cv2

    half = cv2.resize(a, (128, 128), interpolation=cv2.INTER_AREA)
    assert ph.hamming(base, ph.phash_array(half)) <= ph.DUP_MAX_DIST
    up = cv2.resize(a, (512, 512), interpolation=cv2.INTER_LINEAR)
    assert ph.hamming(base, ph.phash_array(up)) <= ph.DUP_MAX_DIST


def test_unrelated_images_are_far_apart():
    d = [ph.hamming(ph.phash_array(_scene(s)), ph.phash_array(_scene(s + 100))) for s in range(20)]
    assert np.median(d) > 15 and min(d) > ph.DUP_MAX_DIST


def test_brightness_and_contrast_shift_is_not_a_new_image():
    a = _scene(3).astype(np.float32)
    assert ph.hamming(ph.phash_array(a), ph.phash_array(np.clip(a * 0.8 + 20, 0, 255))) <= ph.DUP_MAX_DIST


def test_near_pairs_matches_naive_oracle():
    rng = np.random.default_rng(0)
    base = rng.integers(0, 2**63, size=300, dtype=np.uint64)
    h = base.copy()
    # plant near-duplicates with 0..6 flipped bits
    for k, i in enumerate(range(0, 60, 2)):
        flips = int(rng.integers(0, 7))
        mask = np.uint64(0)
        for b in rng.choice(64, size=flips, replace=False):
            mask |= np.uint64(1) << np.uint64(b)
        h[i + 1] = h[i] ^ mask
    fast = {(i, j, d) for i, j, d in ph.near_pairs(h, 4, block=64)}
    naive = {(i, j, ph.hamming(h[i], h[j])) for i in range(len(h)) for j in range(i + 1, len(h)) if ph.hamming(h[i], h[j]) <= 4}
    assert fast == naive and len(naive) > 5


def test_near_pairs_between_two_sets():
    a = np.array([1, 2**40 | 0xFFFF00, 0xF0F0F0F0F0F0F0F0], dtype=np.uint64)
    b = np.array([1 ^ 3, 0x0F0F0F0F0F0F0F0F], dtype=np.uint64)
    got = ph.near_pairs(a, 4, other=b)
    assert got == [(0, 0, 2)]


def test_components_union_find():
    lab = ph.components(6, [(0, 1), (1, 2), (4, 5)])
    assert lab[0] == lab[1] == lab[2] and lab[4] == lab[5]
    assert len({lab[0], lab[3], lab[4]}) == 3


def test_phash_file_reads_jpeg_and_png(tmp_path):
    a = _scene(4)
    Image.fromarray(a).save(tmp_path / "a.png")
    Image.fromarray(a).convert("RGB").save(tmp_path / "a.jpg", quality=90)
    assert ph.hamming(ph.phash_file(tmp_path / "a.png"), ph.phash_file(tmp_path / "a.jpg")) <= ph.DUP_MAX_DIST
    hs = ph.hash_paths([tmp_path / "a.png", tmp_path / "a.jpg"], workers=1)
    assert hs.dtype == np.uint64 and len(hs) == 2
