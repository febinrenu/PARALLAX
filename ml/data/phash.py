"""Perceptual hash and near-duplicate search for the leakage audit.

64-bit DCT pHash: grayscale -> 32x32 -> DCT -> top-left 8x8 block -> bit set where the coefficient is
above the block median. Same recipe as the common `imagehash.phash`, written with numpy + OpenCV so it
has no extra dependency. Two images are "duplicates" when their Hamming distance is <= DUP_MAX_DIST.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFile

DUP_MAX_DIST = 4
_WEIGHTS = (np.uint64(1) << np.arange(64, dtype=np.uint64)).astype(np.uint64)


def phash_array(gray: np.ndarray) -> int:
    """pHash of a 2-D array (any dtype, any range)."""
    g = np.asarray(gray, dtype=np.float32)
    if g.ndim == 3:
        g = g.mean(axis=2)
    small = cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA)
    low = cv2.dct(small)[:8, :8].ravel()
    bits = (low > np.median(low)).astype(np.uint64)
    return int((bits * _WEIGHTS).sum())


def _gray_of(path: str | Path) -> tuple[np.ndarray, bool]:
    """Grey pixels and whether the file was truncated. A few source files lose their last bytes; they still decode."""
    p = Path(path)
    if p.suffix.lower() in (".dcm", ".dicom"):
        from medproof.intake.decode import load_image

        return load_image(p.read_bytes()).analysis * 255.0, False
    truncated = False
    prev = ImageFile.LOAD_TRUNCATED_IMAGES
    try:
        for tolerant in (False, True):
            ImageFile.LOAD_TRUNCATED_IMAGES = tolerant
            try:
                with Image.open(p) as im:
                    im.draft("L", (128, 128))  # JPEG: decode at reduced size, plenty for a 32x32 hash
                    return np.asarray(im.convert("L")), truncated
            except OSError as e:
                if tolerant or "truncated" not in str(e):
                    raise
                truncated = True
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = prev
    raise AssertionError("unreachable")


def phash_file(path: str | Path) -> int:
    """pHash of an image file. DICOM goes through the product decoder so MONOCHROME1 is inverted first."""
    return phash_array(_gray_of(path)[0])


def phash_stats(path: str | Path) -> tuple[int, float, bool]:
    """(pHash, grey-level std on a 0..255 scale, truncated). Flat images all hash alike; callers use the std to exclude them."""
    g, trunc = _gray_of(path)
    return phash_array(g), float(np.asarray(g, dtype=np.float32).std()), trunc


def hash_paths_stats(paths: list[Path], workers: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    paths = [str(p) for p in paths]
    if workers <= 1 or len(paths) < 64:
        res = [phash_stats(p) for p in paths]
    else:
        with ProcessPoolExecutor(workers) as ex:
            res = list(ex.map(phash_stats, paths, chunksize=64))
    return np.array([r[0] for r in res], dtype=np.uint64), np.array([r[1] for r in res], dtype=np.float32), np.array([r[2] for r in res], dtype=bool)


def hash_paths(paths: list[Path], workers: int = 4) -> np.ndarray:
    """uint64 hash per path, in order. Unreadable files raise: a silent skip would hide a leak."""
    return hash_paths_stats(paths, workers)[0]


def popcount(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    b = x.astype(np.uint64).view(np.uint8).reshape(*x.shape, 8)
    return np.unpackbits(b, axis=-1).sum(axis=-1)


def hamming(a: int, b: int) -> int:
    return bin(int(a) ^ int(b)).count("1")


def near_pairs(hashes: np.ndarray, max_dist: int = DUP_MAX_DIST, other: np.ndarray | None = None, block: int = 512) -> list[tuple[int, int, int]]:
    """All index pairs (i, j, dist) with dist <= max_dist.

    With `other=None`, pairs inside `hashes` with i < j. With `other`, pairs (i in hashes, j in other).
    """
    h = np.asarray(hashes, dtype=np.uint64)
    o = h if other is None else np.asarray(other, dtype=np.uint64)
    out: list[tuple[int, int, int]] = []
    for s in range(0, len(h), block):
        d = popcount(h[s : s + block, None] ^ o[None, :])
        ii, jj = np.nonzero(d <= max_dist)
        for i, j in zip(ii, jj):
            gi = int(i) + s
            if other is None and j <= gi:
                continue
            out.append((gi, int(j), int(d[i, j])))
    return out


def components(n: int, edges: list[tuple[int, int]]) -> np.ndarray:
    """Connected-component label per node (labels are the smallest member index)."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return np.array([find(i) for i in range(n)])
