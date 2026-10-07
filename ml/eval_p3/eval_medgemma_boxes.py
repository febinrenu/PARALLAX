"""How good are MedGemma's chest boxes? (P3, the last open item of the D3 second-reader work.)

    python -m ml.eval_p3.eval_medgemma_boxes --n 100          # needs the RSNA images and a GPU with MedGemma

MedGemma is asked to box lung opacities on RSNA test images that carry ground-truth boxes (label Lung_Opacity)
and on Normal images. It answers on a 0-1000 grid as [y_min, x_min, y_max, x_max] (checked on one film before;
this measures it). For the opacity images the script reports how often it boxes anything, the share of boxes
whose centre lies inside a true box (pointing game), and the share with IoU of at least 0.1, 0.3 and 0.5. A fixed
central box is scored the same way as the baseline a model must beat, and the share of Normal images that get a
box is the false-alarm rate. Replies are cached, so a rerun costs nothing; evaluation alone needs no GPU.

Writes reports/medgemma_boxes_rsna.json and ml/artifacts/medgemma_reads/rsna_boxes.jsonl (gitignored).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))

from medproof.verify.concordance import box_iou  # noqa: E402
from ml.eval.bootstrap import ci  # noqa: E402

Box = tuple[float, float, float, float]
BASELINE_BOX: Box = (250.0, 250.0, 750.0, 750.0)  # on the 0-1000 grid: the central quarter of the image, a guess that needs no model
PROMPT = (
    "This is a chest X-ray for a research evaluation. Locate any lung opacity or consolidation. "
    'Reply with JSON only: [{"label": "opacity", "box_2d": [y_min, x_min, y_max, x_max]}] with coordinates '
    "normalized to 0-1000. If there is no opacity, reply []."
)
READS = ROOT / "ml" / "artifacts" / "medgemma_reads" / "rsna_boxes.jsonl"
LABELS_CSV = ROOT / "ml" / "data" / "raw" / "rsna_pneumonia" / "stage_2_train_labels.csv"
_BOX_RE = re.compile(r'"box_2d"\s*:\s*\[([^\]]*)\]')


def parse_boxes(reply: str) -> list[Box]:
    """Boxes from a reply as (x_min, y_min, x_max, y_max) on the 0-1000 grid; anything unusable is dropped."""
    out: list[Box] = []
    for m in _BOX_RE.finditer(reply or ""):
        try:
            vals = [float(v) for v in m.group(1).split(",")]
        except ValueError:
            continue
        if len(vals) != 4 or any(v < 0 or v > 1000 for v in vals):
            continue
        y0, x0, y1, x1 = vals
        if x1 > x0 and y1 > y0:
            out.append((x0, y0, x1, y1))
    return out


def to_pixels(box: Box, width: int, height: int) -> Box:
    x0, y0, x1, y1 = box
    return (x0 / 1000.0 * width, y0 / 1000.0 * height, x1 / 1000.0 * width, y1 / 1000.0 * height)


def pointing_hit(pred: Box, gts: list[Box]) -> bool:
    cx, cy = (pred[0] + pred[2]) / 2.0, (pred[1] + pred[3]) / 2.0
    return any(g[0] <= cx <= g[2] and g[1] <= cy <= g[3] for g in gts)


def _best_iou(preds: list[Box], gts: list[Box]) -> float:
    return max((box_iou(p, g) for p in preds for g in gts), default=0.0)


def _fmt(c: dict[str, Any]) -> dict[str, float]:
    return {k: round(float(c[k]), 4) for k in ("point", "lo", "hi")}


def _rate(values: np.ndarray, ids: np.ndarray, n_boot: int) -> dict[str, float]:
    return _fmt(ci({"v": values.astype(float)}, lambda v: float(np.mean(v)), groups=ids, B=n_boot))


def _opacity_metrics(rows: list[dict[str, Any]], n_boot: int) -> dict[str, Any]:
    ids = np.array([r["id"] for r in rows], dtype=object)
    best = np.array([_best_iou(r["pred"], r["gt"]) for r in rows])
    return {
        "any_box": _rate(np.array([bool(r["pred"]) for r in rows]), ids, n_boot),
        "pointing_hit": _rate(np.array([any(pointing_hit(p, r["gt"]) for p in r["pred"]) for r in rows]), ids, n_boot),
        "iou_ge_0.1": _rate(best >= 0.1, ids, n_boot),
        "iou_ge_0.3": _rate(best >= 0.3, ids, n_boot),
        "iou_ge_0.5": _rate(best >= 0.5, ids, n_boot),
        "mean_best_iou": round(float(best.mean()), 4),
    }


def evaluate(rows: list[dict[str, Any]], *, size: int = 1024, n_boot: int = 1000) -> dict[str, Any]:
    """Rows hold pixel boxes: {id, label, gt: [boxes], pred: [boxes]}. Opacity images are scored for localisation,
    Normal images only for false alarms."""
    opac = [r for r in rows if r["label"] == "Lung_Opacity"]
    norm = [r for r in rows if r["label"] == "Normal"]
    out: dict[str, Any] = {"opacity_images": {"n": len(opac)}, "normal_images": {"n": len(norm)}}
    if opac:
        out["opacity_images"].update(_opacity_metrics(opac, n_boot))
        base = to_pixels(BASELINE_BOX, size, size)
        out["baseline_central_box"] = _opacity_metrics([{**r, "pred": [base]} for r in opac], n_boot)
    if norm:
        ids = np.array([r["id"] for r in norm], dtype=object)
        out["normal_images"]["any_box"] = _rate(np.array([bool(r["pred"]) for r in norm]), ids, n_boot)
    return out


# ---------------------------------------------------------------- running the model (GPU)

def _gt_boxes() -> dict[str, list[Box]]:
    import csv

    gt: dict[str, list[Box]] = {}
    with LABELS_CSV.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["Target"] == "1":
                x, y, w, h = (float(row[k]) for k in ("x", "y", "width", "height"))
                gt.setdefault(row["patientId"], []).append((x, y, x + w, y + h))
    return gt


def _select(n: int) -> list[tuple[str, str, Path]]:
    from ml.data.common import image_path, load_eval_index, load_split

    spec = load_eval_index()["datasets"]["rsna"]
    df = load_split("rsna")
    test = df[df["split"].isin(spec["test_splits"]) & (df["eval_batch"].astype(str) == "True")]
    picks = []
    for label, k in (("Lung_Opacity", n), ("Normal", n // 2)):
        sub = test[test["label"] == label].sample(frac=1.0, random_state=20261007).head(k)
        picks += [(str(r["image_id"]), label, image_path(r)) for _, r in sub.iterrows()]
    return picks


def run_model(picks: list[tuple[str, str, Path]]) -> dict[str, str]:
    """Replies by image id, cached in READS so an interrupted run resumes."""
    from PIL import Image

    from medproof.intake.decode import load_image
    from services.medgemma.loader import TransformersBackend

    READS.parent.mkdir(parents=True, exist_ok=True)
    replies: dict[str, str] = {}
    if READS.exists():
        for line in READS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                replies[row["id"]] = row["reply"]
    todo = [p for p in picks if p[0] not in replies]
    if todo:
        backend = TransformersBackend.from_env()
        with READS.open("a", encoding="utf-8") as fh:
            for image_id, _, path in todo:
                img = Image.fromarray(load_image(path.read_bytes()).display).convert("RGB")
                reply = backend.generate(img, PROMPT, max_new_tokens=200)
                replies[image_id] = reply
                fh.write(json.dumps({"id": image_id, "reply": reply}) + "\n")
                fh.flush()
    return replies


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="opacity images (half as many Normal images are added)")
    ap.add_argument("--no-model", action="store_true", help="score cached replies only")
    args = ap.parse_args(argv)
    picks = _select(args.n)
    gt = _gt_boxes()
    if args.no_model:
        replies = {json.loads(line)["id"]: json.loads(line)["reply"] for line in READS.read_text(encoding="utf-8").splitlines() if line.strip()}
    else:
        replies = run_model(picks)
    rows = []
    for image_id, label, _ in picks:
        if image_id in replies:
            rows.append({"id": image_id, "label": label, "gt": gt.get(image_id, []),
                         "pred": [to_pixels(b, 1024, 1024) for b in parse_boxes(replies[image_id])]})
    out = {
        "what": "MedGemma 1.5 4B (4-bit) asked to box lung opacities on RSNA test images; ground truth from stage_2_train_labels.csv",
        "prompt": PROMPT, "image_size": 1024, "answered": len(rows), "requested": len(picks),
        **evaluate(rows, size=1024),
        "note": "the box convention is [y_min, x_min, y_max, x_max] on a 0-1000 grid; the baseline is a fixed central box covering a quarter of the image",
    }
    path = ROOT / "reports" / "medgemma_boxes_rsna.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
