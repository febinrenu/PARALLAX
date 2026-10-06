"""Decode every kept RSNA image once and cache the model input, plus DICOM demographics.

    python ml/eval/cxr_cache.py

Writes ml/data/cache/rsna_cxr224.npy (float16, N x 1 x 224 x 224, rows follow ml/data/splits/rsna.csv after dropping `dropped`)
and ml/data/cache/rsna_meta.csv (image_id, age, sex, view). Pixels go through the product's own decoder and the `cxr_xrv`
preprocessing spec, so a score computed from this cache equals what the reader would give the same file.
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for p in (REPO, REPO / "backend"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ml.data.common import CACHE_DIR, data_root, image_path, load_split  # noqa: E402

NPY = CACHE_DIR / "rsna_cxr224.npy"
META = CACHE_DIR / "rsna_meta.csv"


def kept_rows() -> pd.DataFrame:
    df = load_split("rsna")
    return df[df.split != "dropped"].reset_index(drop=True)


def _parse_age(age: str) -> float:
    """RSNA stores PatientAge as plain digits ("51"); standard DICOM uses "051Y". Anything else is unknown."""
    a = age.strip().upper()
    if a.endswith("Y"):
        a = a[:-1]
    return float(a) if a.isdigit() else float("nan")


def build_meta_only(workers: int = 8) -> pd.DataFrame:
    """Re-read only the DICOM headers (seconds, no pixels) to refresh rsna_meta.csv."""
    import pydicom

    df = kept_rows()
    root = data_root()

    def one(r):
        ds = pydicom.dcmread(image_path(r, root), stop_before_pixels=True)
        return _parse_age(str(getattr(ds, "PatientAge", "") or "")), str(getattr(ds, "PatientSex", "") or ""), str(getattr(ds, "ViewPosition", "") or "")

    with ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(one, [r for _, r in df.iterrows()]))
    m = pd.DataFrame(rows, columns=["age", "sex", "view"])
    m.insert(0, "image_id", df.image_id.to_numpy())
    m.to_csv(META, index=False)
    return m


def _one(path: Path):
    import pydicom

    from medproof.intake.decode import load_image
    from medproof.intake.preprocess import prepare

    raw = path.read_bytes()
    img = load_image(raw)
    ds = pydicom.dcmread(path, stop_before_pixels=True)
    age = str(getattr(ds, "PatientAge", "") or "").strip()
    age_n = _parse_age(age)
    return prepare(img, "cxr_xrv").astype(np.float16), (age_n, str(getattr(ds, "PatientSex", "") or ""), str(getattr(ds, "ViewPosition", "") or ""))


def build(workers: int = 6) -> pd.DataFrame:
    df = kept_rows()
    root = data_root()
    paths = [image_path(r, root) for _, r in df.iterrows()]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    mm = np.lib.format.open_memmap(NPY, mode="w+", dtype=np.float16, shape=(len(paths), 1, 224, 224))
    meta = [None] * len(paths)

    def work(i):
        arr, m = _one(paths[i])
        mm[i] = arr
        meta[i] = m

    with ThreadPoolExecutor(workers) as ex:
        for k, _ in enumerate(ex.map(work, range(len(paths)))):
            if k % 2000 == 0:
                print(f"  {k}/{len(paths)}", flush=True)
    mm.flush()
    m = pd.DataFrame(meta, columns=["age", "sex", "view"])
    m.insert(0, "image_id", df.image_id.to_numpy())
    m.to_csv(META, index=False)
    return m


if __name__ == "__main__":
    m = build_meta_only() if "--meta-only" in sys.argv else build()
    print("cached", len(m), "images;", m.sex.value_counts().to_dict(), m.view.value_counts().to_dict(), "age NaN", int(m.age.isna().sum()))
