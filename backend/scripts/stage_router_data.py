"""Stage a small labelled image folder for training and checking the modality router.

Layout written under --out (default backend/data/router, git-ignored):
    cxr/            Montgomery County chest X-rays (NLM, public, research use)
    brain_mri/<patient>/   Brain Tumor Dataset, figshare 1512427 (CC BY 4.0), T1-contrast slices
    skin_dermoscopy/       ISIC 2018 Task 3 test images (CC-BY-NC 4.0)
    bone_xray/<part>/      FracAtlas, figshare 22363012 (CC BY 4.0), grouped by body part
    other/cifar/           CIFAR-10 test images (natural images)
Only small subsets are read; zip archives are range-read, never fully downloaded (CIFAR is a plain
170 MB tarball). Nothing here is committed; images stay out of git and out of the demo.

    python scripts/stage_router_data.py [--out DIR] [--n 120] [--only isic,fracatlas,...]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import tarfile
import time
import zipfile
from pathlib import Path

import numpy as np
import requests
from PIL import Image

UA = {"User-Agent": "Mozilla/5.0"}
SOURCES = {
    "isic": "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task3_Test_Input.zip",
    "fracatlas": "https://ndownloader.figshare.com/files/65518038",
    "brain": "https://ndownloader.figshare.com/files/3381290",
    "cifar": "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
    "montgomery": "https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/MontgomerySet/CXR_png/MCUCXR_{n:04d}_{s}.png",
}


class HTTPRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests, with a block cache."""

    BLOCK = 1 << 20

    def __init__(self, url: str):
        r = requests.get(url, headers={**UA, "Range": "bytes=0-0"}, stream=True, timeout=60, allow_redirects=True)
        r.close()
        if r.status_code != 206:
            raise OSError(f"{url} does not support range requests (status {r.status_code})")
        self.origin = url
        self.url = r.url
        self.size = int(r.headers["Content-Range"].split("/")[1])
        self.pos = 0
        self._blocks: dict[int, bytes] = {}

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def _block(self, i: int) -> bytes:
        if i not in self._blocks:
            lo = i * self.BLOCK
            hi = min(self.size - 1, lo + self.BLOCK - 1)
            for attempt in range(12):
                try:
                    r = requests.get(self.url, headers={**UA, "Range": f"bytes={lo}-{hi}"}, timeout=180)
                    if r.status_code == 206 and len(r.content) == hi - lo + 1:
                        self._blocks[i] = r.content
                        break
                except requests.RequestException:
                    pass
                if attempt % 3 == 2:  # signed redirect URLs expire: resolve a fresh one
                    try:
                        fresh = requests.get(self.origin, headers={**UA, "Range": "bytes=0-0"}, stream=True, timeout=60)
                        fresh.close()
                        self.url = fresh.url
                    except requests.RequestException:
                        pass
                time.sleep(min(30, 2 * (attempt + 1)))
            else:
                raise OSError(f"could not read bytes {lo}-{hi}")
        return self._blocks[i]

    def readinto(self, b):
        out, n = memoryview(b), 0
        while n < len(b) and self.pos < self.size:
            i, off = divmod(self.pos, self.BLOCK)
            chunk = self._block(i)[off : off + len(b) - n]
            out[n : n + len(chunk)] = chunk
            n += len(chunk)
            self.pos += len(chunk)
        return n


def _save(img: Image.Image, path: Path, manifest: list, source: str, max_side: int = 1024) -> None:
    img.thumbnail((max_side, max_side))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    manifest.append({"path": str(path), "source": source, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})


def _spread(items: list, n: int) -> list:
    if len(items) <= n:
        return items
    idx = np.linspace(0, len(items) - 1, n).round().astype(int)
    return [items[i] for i in sorted(set(idx))]


def stage_isic(out: Path, n: int, manifest: list) -> None:
    z = zipfile.ZipFile(HTTPRangeFile(SOURCES["isic"]))
    names = sorted(m for m in z.namelist() if m.lower().endswith(".jpg"))
    for name in _spread(names, n):
        with z.open(name) as f:
            _save(Image.open(io.BytesIO(f.read())).convert("RGB"), out / "skin_dermoscopy" / Path(name).name, manifest, "ISIC2018 Task3 test (CC-BY-NC 4.0)")


def stage_fracatlas(out: Path, n: int, manifest: list) -> None:
    import csv

    z = zipfile.ZipFile(HTTPRangeFile(SOURCES["fracatlas"]))
    csv_name = next(m for m in z.namelist() if m.endswith("dataset.csv"))
    rows = list(csv.DictReader(io.StringIO(z.read(csv_name).decode("utf-8", "ignore"))))
    parts = ("hand", "leg", "hip", "shoulder")
    by_part: dict[str, list[str]] = {p: [] for p in parts}
    for r in rows:
        on = [p for p in parts if str(r.get(p, "0")).strip() in {"1", "1.0"}]
        if len(on) == 1:
            by_part[on[0]].append(r["image_id"])
    index = {Path(m).name: m for m in z.namelist() if m.lower().endswith(".jpg")}
    for part, ids in by_part.items():
        for image_id in _spread(sorted(ids), max(1, n // 4)):
            if image_id in index:
                with z.open(index[image_id]) as f:
                    _save(Image.open(io.BytesIO(f.read())).convert("L"), out / "bone_xray" / part / image_id, manifest, "FracAtlas (CC BY 4.0)")


def stage_brain(out: Path, n: int, manifest: list) -> None:
    import h5py

    z = zipfile.ZipFile(HTTPRangeFile(SOURCES["brain"]))
    names = sorted((m for m in z.namelist() if m.endswith(".mat")), key=lambda s: int(re.findall(r"\d+", Path(s).stem)[-1]))
    for name in _spread(names, n):
        with h5py.File(io.BytesIO(z.read(name)), "r") as f:
            img = np.array(f["cjdata"]["image"], dtype=np.float32).T
            pid = "".join(chr(int(c)) for c in np.array(f["cjdata"]["PID"]).ravel())
        lo, hi = np.percentile(img, (0.5, 99.5))
        u8 = np.clip((img - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
        _save(Image.fromarray(u8), out / "brain_mri" / pid / f"{Path(name).stem}.png", manifest, "Brain Tumor Dataset, figshare 1512427 (CC BY 4.0)")


def _download(url: str) -> bytes:
    """Whole-file download that resumes with Range requests after a dropped connection."""
    buf = bytearray()
    size = None
    for attempt in range(40):
        try:
            h = {**UA, "Range": f"bytes={len(buf)}-"}
            with requests.get(url, headers=h, stream=True, timeout=120) as r:
                if r.status_code not in (200, 206):
                    raise OSError(f"status {r.status_code}")
                if r.status_code == 200:
                    buf.clear()
                for chunk in r.iter_content(1 << 20):
                    buf.extend(chunk)
                size = size or int(r.headers.get("Content-Range", "/0").split("/")[-1] or 0) or None
            if size is None or len(buf) >= size:
                return bytes(buf)
        except requests.RequestException:
            time.sleep(min(30, 2 * (attempt + 1)))
    raise OSError("download kept failing")


def stage_cifar(out: Path, n: int, manifest: list) -> None:
    import pickle

    raw = _download(SOURCES["cifar"])
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as t:
        batch = pickle.load(t.extractfile("cifar-10-batches-py/test_batch"), encoding="bytes")  # noqa: S301 (public file, local use)
    data = batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    for i in _spread(list(range(len(data))), n):
        _save(Image.fromarray(data[i]), out / "other" / "cifar" / f"cifar_{i:05d}.png", manifest, "CIFAR-10 test")


def stage_montgomery(out: Path, n: int, manifest: list) -> None:
    got = 0
    for k in range(1, 400):
        for s in (0, 1):
            if got >= n:
                return
            try:
                r = requests.get(SOURCES["montgomery"].format(n=k, s=s), headers=UA, timeout=120)
            except requests.RequestException:
                continue
            if r.status_code == 200 and r.content[:4] == b"\x89PNG":
                img = Image.open(io.BytesIO(r.content))
                img = img.convert("L") if img.mode not in ("I;16", "I") else img
                _save(img, out / "cxr" / f"MCUCXR_{k:04d}_{s}.png", manifest, "Montgomery County CXR set (NLM)")
                got += 1


STAGERS = {"isic": stage_isic, "fracatlas": stage_fracatlas, "brain": stage_brain, "cifar": stage_cifar, "montgomery": stage_montgomery}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/router")
    ap.add_argument("--n", type=int, default=120, help="images per source")
    ap.add_argument("--only", default=",".join(STAGERS))
    a = ap.parse_args(argv)
    out = Path(a.out)
    manifest: list = []
    status = {}
    for name in a.only.split(","):
        before = len(manifest)
        try:
            STAGERS[name](out, a.n, manifest)
            status[name] = f"{len(manifest) - before} images"
        except Exception as exc:  # report and continue so one dead source does not block the rest
            status[name] = f"FAILED: {type(exc).__name__}: {str(exc)[:120]}"
        print(name, status[name], flush=True)
    (out / "manifest.json").write_text(json.dumps({"status": status, "files": manifest}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
