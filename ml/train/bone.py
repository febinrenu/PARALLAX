"""Bone fracture detector helpers (P2.6): YOLO dataset from our splits, prediction export, metrics with CIs.

The detector is single-class ("fracture"). Negative images are kept with empty label files so the model
learns to stay quiet. Image-level score = highest box confidence (0 when nothing is detected).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from ml.eval import bootstrap as bs
from ml.eval import detection as det
from ml.eval import metrics as mt

SPLITS = ("train", "val", "cal", "test")


def read_yolo_boxes(txt: Path) -> np.ndarray:
    """Normalised xyxy boxes from a YOLO txt file (class cx cy w h per line); (0, 4) for an empty file."""
    rows = [ln.split() for ln in Path(txt).read_text().splitlines() if ln.strip()]
    if not rows:
        return np.zeros((0, 4))
    a = np.array([[float(v) for v in r[1:5]] for r in rows])
    return np.stack([a[:, 0] - a[:, 2] / 2, a[:, 1] - a[:, 3] / 2, a[:, 0] + a[:, 2] / 2, a[:, 1] + a[:, 3] / 2], 1).clip(0, 1)


def _link(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.symlink(src, dst)
    except OSError:  # Windows without symlink rights
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def build_yolo_dataset(df: pd.DataFrame, src_root: Path, out_dir: Path) -> Path:
    """Lay out images/ and labels/ per split and write data.yaml. Rows come from ml/data/splits/fracatlas.csv (dropped rows excluded)."""
    out_dir = Path(out_dir)
    for _, r in df[df.split.isin(SPLITS)].iterrows():
        base = Path(src_root) / r["source_dir"]
        for kind in ("images", "labels"):
            (out_dir / kind / r["split"]).mkdir(parents=True, exist_ok=True)
        _link(base / r["relpath"], out_dir / "images" / r["split"] / f"{r['image_id']}.jpg")
        lbl = base / r["yolo_label"]
        dst = out_dir / "labels" / r["split"] / f"{r['image_id']}.txt"
        if lbl.is_file():
            shutil.copyfile(lbl, dst)
        else:
            dst.write_text("")
    yaml = out_dir / "data.yaml"
    yaml.write_text(f"path: {out_dir.resolve().as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: fracture\n", encoding="utf-8")
    return yaml


def read_image_bgr(path: Path) -> np.ndarray:
    """BGR array for Ultralytics. A few FracAtlas files lose their last bytes: OpenCV decodes them, PIL is the fallback."""
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is not None:
        return img
    from PIL import Image, ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    with Image.open(path) as im:
        return cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2BGR)


def predict_split(model, image_paths: list[Path], imgsz: int = 640, batch: int = 16) -> list[np.ndarray]:
    """Per image an (n, 5) array: normalised xyxy + confidence, at a very low threshold so AP and ROC see every candidate.

    Images are decoded here and passed as arrays: handing Ultralytics a list of paths makes it open each with PIL,
    which rejects the truncated files.
    """
    out = []
    for i in range(0, len(image_paths), batch):
        arrays = [read_image_bgr(p) for p in image_paths[i : i + batch]]
        for r in model.predict(arrays, imgsz=imgsz, conf=0.001, iou=0.6, max_det=100, verbose=False):
            b = r.boxes
            out.append(np.concatenate([b.xyxyn.cpu().numpy(), b.conf.cpu().numpy()[:, None]], 1) if len(b) else np.zeros((0, 5)))
    return out


def pack_dets(dets: list[np.ndarray]) -> dict[str, np.ndarray]:
    """Flat arrays for np.savez: rows + offsets (so predictions/<split>.npz needs no pickling)."""
    off = np.cumsum([0] + [len(d) for d in dets])
    return {"det_rows": np.concatenate(dets) if len(dets) else np.zeros((0, 5)), "det_offsets": off}


def unpack_dets(rows: np.ndarray, offsets: np.ndarray) -> list[np.ndarray]:
    return [rows[offsets[i] : offsets[i + 1]] for i in range(len(offsets) - 1)]


def _obj(items: list[np.ndarray]) -> np.ndarray:
    a = np.empty(len(items), dtype=object)
    for i, v in enumerate(items):
        a[i] = v
    return a


def evaluate(val: dict, test: dict, B: int = 2000) -> dict:
    """val/test: {'dets': list, 'gts': list, 'groups': array}. Threshold for sensitivity@90% specificity is fixed on val.

    Returns AP50, image-level AUROC, sensitivity/specificity at the val-fixed threshold, each with a 95% cluster-bootstrap CI.
    """
    yv = np.array([len(g) > 0 for g in val["gts"]]).astype(int)
    sv = det.image_scores(val["dets"])
    thr = mt.threshold_at_specificity(yv, sv, 0.9)
    yt = np.array([len(g) > 0 for g in test["gts"]]).astype(int)
    st = det.image_scores(test["dets"])
    kw = dict(groups=np.asarray(test["groups"]), B=B)
    return {
        "n_images": int(len(yt)), "n_fractured": int(yt.sum()), "n_boxes": int(sum(len(g) for g in test["gts"])),
        "map50": bs.ci({"d": _obj(test["dets"]), "g": _obj(test["gts"])}, lambda d, g: det.average_precision(list(d), list(g), 0.5), **kw),
        "auroc_image": bs.ci({"y": yt, "s": st}, lambda y, s: mt.auroc_binary(y, s), strata=yt, **kw),
        "sensitivity_at_val_90spec": bs.ci({"y": yt, "s": st}, lambda y, s: mt.sensitivity_at_threshold(y, s, thr), strata=yt, **kw),
        "specificity_at_val_threshold": bs.ci({"y": yt, "s": st}, lambda y, s: mt.specificity_at_threshold(y, s, thr), strata=yt, **kw),
        "val_threshold": float(thr), "val_n_images": int(len(yv)), "operating_point_note": "threshold fixed on the validation split, applied unchanged to test",
    }
