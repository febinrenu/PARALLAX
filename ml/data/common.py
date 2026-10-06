"""Shared entry points for reading datasets and splits, identical on a laptop and on Kaggle.

    from ml.data.common import load_split, image_path
    df = load_split("ham10000")                    # one row per image, see make_splits.py for columns
    test = df[df.split == "official_test"]
    path = image_path(test.iloc[0])               # absolute path of the image on this machine

Set PARALLAX_DATA_ROOT to relocate the raw data (Kaggle notebooks point it at a folder that links the
attached inputs). Default is ml/data/raw.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = REPO_ROOT / "ml" / "data" / "splits"
CACHE_DIR = REPO_ROOT / "ml" / "data" / "cache"
REPORTS_DIR = REPO_ROOT / "reports"
SEED = 20261006


def data_root() -> Path:
    return Path(os.environ.get("PARALLAX_DATA_ROOT") or REPO_ROOT / "ml" / "data" / "raw")


def load_split(name: str, splits_dir: Path | None = None) -> pd.DataFrame:
    p = Path(splits_dir or SPLITS_DIR) / f"{name}.csv"
    if not p.is_file():
        raise FileNotFoundError(f"{p} not found. Run: python ml/data/make_splits.py {name}")
    return pd.read_csv(p, keep_default_na=False, na_values=[""], dtype={"image_id": str, "group": str})


def image_path(row, root: Path | None = None) -> Path:
    return Path(root or data_root()) / row["source_dir"] / row["relpath"]


def load_eval_index() -> dict:
    return json.loads((REPO_ROOT / "ml" / "data" / "eval_index.json").read_text(encoding="utf-8"))
