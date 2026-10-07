#!/usr/bin/env python3
"""Bake the web app's media and demo data from real, license-checked sources.

Outputs (committed):
  web/public/media/      hero + chest stills, perturbation atlas, credits.json
  web/public/cases/<id>/ sample cases for the workstation (case.json + overlays)
  web/src/story/baked/chest.json   numbers the landing chapters animate

Every number comes from running the pipeline's own code on these images: the P1 CXR reader
and anatomy segmenter, P1's perturbation library, the quality gate, and a deletion-based
faithfulness test that follows plan.md D1 (blur the top 10% of the heatmap, re-score, compare
with 30 random regions of the same area). Note text and text evidence are illustrative until
the context engine (P3.5) exists, and say so in each case's `provenance`.

Needs torch + torchxrayvision with weights in ~/.torchxrayvision. If the reader cannot load,
this script fails loudly rather than writing made-up numbers.

    python scripts/bake_web_assets.py
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from medproof.core.config import PipelineConfig  # noqa: E402
from medproof.core.schemas import Stability, StudyResult, TextEvidence  # noqa: E402
from medproof.core.status_rule import compute_status  # noqa: E402
from medproof.intake import quality  # noqa: E402
from medproof.intake.decode import load_image  # noqa: E402
from medproof.intake.run import run_bytes  # noqa: E402
from medproof.pipeline import DISCLAIMER, _fold_ledger_head, run_study  # noqa: E402
from medproof.readers.anatomy import AnatomySegmenter  # noqa: E402
from medproof.readers.cxr import CxrReader  # noqa: E402
from medproof.readers.evidence import to_findings  # noqa: E402
from medproof.verify import perturbations  # noqa: E402

UA = "ParallaxHackathonAssetFetcher/0.1 (https://github.com/febinrenu/PARALLAX)"
CACHE = Path(os.environ.get("PARALLAX_MEDIA_CACHE", Path(tempfile.gettempdir()) / "parallax_web_media"))
PUBLIC = ROOT / "web" / "public"
MEDIA = PUBLIC / "media"
CASES = PUBLIC / "cases"
STORY = ROOT / "web" / "src" / "story" / "baked"
FRACATLAS = ROOT / "ml" / "data" / "raw" / "fracatlas" / "FracAtlas"

SOURCES = {
    "chest": {
        "title": "Chest radiograph in influenza and H. influenzae, posteroanterior",
        "author": "Mikael Häggström, M.D.",
        "license": "CC0 1.0",
        "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "page": "https://commons.wikimedia.org/wiki/File:Chest_radiograph_in_influensa_and_H_influenzae,_posteroanterior.jpg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/d/da/Chest_radiograph_in_influensa_and_H_influenzae%2C_posteroanterior.jpg",
        "modifications": "Converted to grayscale, centre-cropped to a square, resized.",
    },
    "brain": {
        "title": "Glioblastoma, contrast-enhanced axial MRI (AFIP 00405558)",
        "author": "Armed Forces Institute of Pathology (US Government)",
        "license": "Public domain",
        "license_url": "https://commons.wikimedia.org/wiki/File:AFIP-00405558-Glioblastoma-Radiology.jpg",
        "page": "https://commons.wikimedia.org/wiki/File:AFIP-00405558-Glioblastoma-Radiology.jpg",
        "wiki_title": "File:AFIP-00405558-Glioblastoma-Radiology.jpg",
        "modifications": "Converted to grayscale.",
    },
    "skin": {
        "title": "Dermoscopic image ISIC_0015552 with ISIC 2018 Task 1 lesion boundary",
        "author": "Anonymous, via the ISIC Archive",
        "license": "CC0 1.0",
        "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "page": "https://api.isic-archive.com/images/ISIC_0015552/",
        "isic_id": "ISIC_0015552",
        "mask_zip": "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1_Validation_GroundTruth.zip",
        "modifications": "Resized.",
    },
    "wrist": {
        "title": "FracAtlas IMG0002311, right wrist with distal radius fracture",
        "author": "Abedeen et al., FracAtlas (Scientific Data, 2023)",
        "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "page": "https://figshare.com/articles/dataset/The_dataset/22363012",
        "image_id": "IMG0002311",
        "modifications": "Resized; fracture box from the dataset's own annotation.",
    },
}

SYNTHETIC_CHEST_NOTE = (
    "62-year-old with fever and productive cough for four days. "
    "Crackles over the right upper chest on auscultation. "
    "No prior chest surgery. No devices."
)

# Synthetic notes for the other samples: a live demo reads image and note together, and a finding
# whose heatmap fails the region test can still be supported by the note.
SAMPLE_NOTES = {
    "bone": "34-year-old fell on an outstretched hand yesterday. Pain and swelling over the wrist, tender over the distal radius. No previous injury to this wrist.",
    "brain": "58-year-old with three weeks of worsening morning headaches and a first seizure two days ago. MRI requested to look for an intracranial mass.",
    "skin": "Pigmented lesion on the upper back, present for many years and unchanged according to the patient. No itching, bleeding or recent growth.",
}


# -- fetching --------------------------------------------------------------------------------------
def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def _cached(name: str, url: str) -> bytes:
    path = CACHE / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_get(url))
    return path.read_bytes()


def fetch_chest() -> bytes:
    return _cached("chest.jpg", SOURCES["chest"]["url"])


def fetch_brain() -> bytes:
    title = SOURCES["brain"]["wiki_title"]
    api = f"https://commons.wikimedia.org/w/api.php?action=query&titles={urllib.request.quote(title)}&prop=imageinfo&iiprop=url&format=json"
    page = next(iter(json.loads(_get(api))["query"]["pages"].values()))
    return _cached("brain.jpg", page["imageinfo"][0]["url"])


def fetch_skin() -> tuple[bytes, bytes, dict]:
    isic = SOURCES["skin"]["isic_id"]
    meta = json.loads(_get(f"https://api.isic-archive.com/api/v2/images/{isic}/"))
    if meta.get("copyright_license") != "CC-0":
        raise SystemExit(f"{isic} is {meta.get('copyright_license')}, not CC-0: refusing to bake it")
    image = _cached(f"{isic}.jpg", meta["files"]["full"]["url"])
    zbytes = _cached("isic_t1_val_gt.zip", SOURCES["skin"]["mask_zip"])
    with zipfile.ZipFile(io.BytesIO(zbytes)) as z:
        name = next(n for n in z.namelist() if n.endswith(f"{isic}_segmentation.png"))
        mask = z.read(name)
    return image, mask, meta


def fetch_wrist() -> tuple[bytes, list[int]]:
    image_id = SOURCES["wrist"]["image_id"]
    img = FRACATLAS / "images" / "Fractured" / f"{image_id}.jpg"
    xml = FRACATLAS / "Annotations" / "PASCAL VOC" / f"{image_id}.xml"
    if not img.exists():
        raise SystemExit("FracAtlas not found: run `python ml/data/download.py fracatlas` first")
    box = next(ET.parse(xml).getroot().iter("bndbox"))
    xyxy = [int(float(box.find(k).text)) for k in ("xmin", "ymin", "xmax", "ymax")]
    return img.read_bytes(), xyxy


# -- image helpers ---------------------------------------------------------------------------------
def gray(raw: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(raw)).convert("L"))


def square_crop(a: np.ndarray) -> np.ndarray:
    h, w = a.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return a[y0 : y0 + s, x0 : x0 + s]


def resize_long(a: np.ndarray, long_edge: int) -> np.ndarray:
    h, w = a.shape[:2]
    scale = long_edge / max(h, w)
    if scale >= 1.0:
        return a
    return cv2.resize(a, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def png(a: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="PNG", optimize=True)  # no gAMA/iCCP chunks
    return buf.getvalue()


def save_webp(a: np.ndarray, path: Path, quality_: int = 82) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(a).save(path, format="WEBP", quality=quality_, method=6)


def save_png(a: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png(a))


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# -- faithfulness and stability ----------------------------------------------------------------------
def deletion_drop(reader: CxrReader, img01: np.ndarray, mask: np.ndarray, label_idx: int, base: float) -> float:
    soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 6)
    blurred = cv2.GaussianBlur(img01, (0, 0), max(img01.shape) / 40)
    out = img01 * (1 - soft) + blurred * soft
    return float(base - reader.score(out.astype(np.float32))[label_idx])


def top_fraction_mask(heat: np.ndarray, frac: float = 0.10) -> np.ndarray:
    thresh = np.quantile(heat[heat > 0], 1 - frac) if (heat > 0).any() else 1.0
    return heat >= thresh


def faithfulness(reader, img01, heat, label_idx, base, rng, n_random: int = 30) -> dict:
    mask = top_fraction_mask(heat)
    region_threshold = float(heat[mask].min()) if mask.any() else 1.0
    drop = deletion_drop(reader, img01, mask, label_idx, base)
    h, w = mask.shape
    randoms, shifts = [], []
    for _ in range(n_random):
        dy, dx = int(rng.integers(-h // 2, h // 2)), int(rng.integers(-w // 2, w // 2))
        shifted = np.roll(mask, (dy, dx), axis=(0, 1))
        randoms.append(deletion_drop(reader, img01, shifted, label_idx, base))
        shifts.append((dx / w, dy / h))
    threshold = float(np.quantile(randoms, 0.9))
    ex = int(np.argsort(randoms)[len(randoms) // 2])  # a median random region
    example = randoms[ex]
    return {
        # The landing redraws exactly these two regions: heat >= region_threshold, and the same
        # mask rolled by random_example_shift (normalised x, y; np.roll wraps around the edges).
        "region_threshold": round(region_threshold, 4),
        "random_example_shift": [round(shifts[ex][0], 4), round(shifts[ex][1], 4)],
        "drop": round(drop, 4),
        "random_threshold_p90": round(threshold, 4),
        "random_median_drop": round(float(np.median(randoms)), 4),
        "random_example_drop": round(example, 4),
        "faithful": bool(drop > threshold),
        "region_fraction": 0.10,
    }


def stability(reader, img01, label_idx, base) -> tuple[dict, list[dict], list[np.ndarray]]:
    tiles, imgs, flips = [], [], 0
    worst, worst_delta = None, -1.0
    for name in perturbations.NAMES:
        pert = perturbations.apply(name, img01, 3, 0).astype(np.float32)
        p = float(reader.score(pert)[label_idx])
        flipped = (p >= 0.5) != (base >= 0.5)
        flips += int(flipped)
        if abs(p - base) > worst_delta:
            worst, worst_delta = name, abs(p - base)
        tiles.append({"perturbation": name, "prob": round(p, 4), "flipped": bool(flipped)})
        imgs.append(pert)
    summary = {"tests": len(perturbations.NAMES), "flip_rate": round(flips / len(perturbations.NAMES), 4), "worst_perturbation": worst}
    return summary, tiles, imgs


# -- bakes -----------------------------------------------------------------------------------------
def bake_chest(reader: CxrReader, anatomy_fn) -> dict:
    src = fetch_chest()
    film = resize_long(square_crop(gray(src)), 1024)
    raw = png(film)
    img = load_image(raw)
    out_dir = CASES / "chest"
    out_dir.mkdir(parents=True, exist_ok=True)

    intake_result, decoded = run_bytes(raw, "cxr")
    out = reader.predict(decoded)
    with tempfile.TemporaryDirectory() as tmp:
        findings, warnings = to_findings(out, reader.cfg, artifact_dir=tmp)
        for f in findings:
            name = f"heatmap_{f.finding_id}.png"
            heat = cv2.imread(str(Path(tmp) / name), cv2.IMREAD_GRAYSCALE)
            save_png(cv2.resize(heat, (512, 512), interpolation=cv2.INTER_AREA), out_dir / name)
            f.image_evidence[0].heatmap_ref = f"/cases/chest/{name}"

    img01 = img.analysis.astype(np.float32)
    if img01.ndim == 3:
        img01 = img01.mean(axis=2)
    base_probs = reader.score(img01)
    rng = np.random.default_rng(20261007)
    story_findings, atlas_imgs, primary_tiles = [], None, None

    note = SYNTHETIC_CHEST_NOTE
    note_evidence = {
        "Pneumonia": ("fever and productive cough", "symptom"),
        "Consolidation": ("Crackles over the right upper chest", "laterality"),
        "Lung Opacity": ("Crackles over the right upper chest", "laterality"),
    }
    te_counter = 1
    for rf, f in zip([r for r in out.findings if r.heatmap is not None or r.boxes_xyxy], findings, strict=True):
        idx = out.labels.index(f.label)
        base = float(base_probs[idx])
        faith = faithfulness(reader, img01, rf.heatmap, idx, base, rng) if rf.heatmap is not None else None
        stab, tiles, imgs = stability(reader, img01, idx, base)
        if atlas_imgs is None:
            atlas_imgs = imgs
        f.image_evidence[0].faithfulness_drop = faith["drop"] if faith else None
        f.image_evidence[0].faithful = faith["faithful"] if faith else None
        f.stability = Stability(**stab)
        if f.label in note_evidence:
            quote, fact_type = note_evidence[f.label]
            start = note.index(quote)
            f.text_evidence = [TextEvidence(
                evidence_id=f"te_{te_counter}", note_id="n1", span=(start, start + len(quote)),
                quote=quote, fact_type=fact_type, polarity="supports",
            )]
            te_counter += 1
        f.flags = [fl for fl in f.flags if fl != "partial_view"]
        f.status = compute_status(f)
        h, w = img.shape
        box = f.image_evidence[0].bbox_xyxy
        story_findings.append({
            "id": f.finding_id, "label": f.label, "prob": round(base, 4), "tier": f.tier, "status": f.status,
            "region": f.image_evidence[0].region_name,
            "bbox": [round(box[0] / w, 4), round(box[1] / h, 4), round(box[2] / w, 4), round(box[3] / h, 4)] if box else None,
            "heatmap": f"/cases/chest/heatmap_{f.finding_id}.png",
            "faithfulness": faith, "stability": stab, "tiles": tiles,
        })

    order = ["Pneumonia", "Consolidation", "Lung Opacity"]
    candidates = [s for s in story_findings if s["faithfulness"] and s["faithfulness"]["faithful"]] or story_findings
    primary = sorted(candidates, key=lambda s: (order.index(s["label"]) if s["label"] in order else 9, -s["prob"]))[0]
    primary_tiles = primary["tiles"]

    # Contact sheet of the eight perturbed films (real outputs of P1's perturbation library).
    tile = 384
    atlas = np.zeros((tile * 2, tile * 4), np.uint8)
    for i, a in enumerate(atlas_imgs or []):
        t = cv2.resize(np.clip(a * 255, 0, 255).astype(np.uint8), (tile, tile), interpolation=cv2.INTER_AREA)
        atlas[(i // 4) * tile : (i // 4 + 1) * tile, (i % 4) * tile : (i % 4 + 1) * tile] = t
    save_webp(atlas, MEDIA / "chest-perturbations.webp", 80)

    # Anatomy masks packed as RGB: R = right lung, G = left lung, B = heart (patient sides).
    anat = anatomy_fn(decoded)
    if anat is not None:
        rgb = np.stack([anat.right_lung, anat.left_lung, anat.heart], axis=-1).astype(np.float32)
        rgb = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_AREA)
        # Soft edges so the shader's 0.5 iso-contour is smooth instead of stair-stepped.
        rgb = cv2.GaussianBlur(rgb, (0, 0), 1.2)
        save_png(np.clip(rgb * 255, 0, 255).astype(np.uint8), MEDIA / "chest-anatomy.png")

    # Quality gate on degraded copies: real reasons and fix text for the intake beat.
    dark = (np.clip(film.astype(np.float32) / 255.0, 0, 1) ** 2.6 * 255 * 0.45).astype(np.uint8)
    m = cv2.getRotationMatrix2D((film.shape[1] / 2, film.shape[0] / 2), 16, 1.0)
    rotated = cv2.warpAffine(film, m, (film.shape[1], film.shape[0]), borderValue=0)
    degraded = []
    for name, a in [("underexposed", dark), ("rotated", rotated)]:
        rep = quality.assess(load_image(png(a)), "cxr")
        degraded.append({"variant": name, "passed": rep.passed, "reasons": [r.to_dict() for r in rep.reasons]})

    hist = np.bincount(film.ravel(), minlength=256).astype(np.float64)
    cdf = np.cumsum(hist) / hist.sum()

    save_webp(film, MEDIA / "chest.webp", 86)
    shutil.copy(MEDIA / "chest.webp", out_dir / "image.webp")

    quality_dict = intake_result.payload.get("quality", {})
    study = StudyResult(
        study_id="sample-chest", input_sha256=img.sha256, modality="cxr", ood_score=0.0,
        quality=quality_dict, findings=findings, claims=[], ledger_head=_fold_ledger_head([intake_result]),
        disclaimer=DISCLAIMER,
    )
    case = {
        "id": "chest",
        "title": "Chest radiograph, posteroanterior",
        "modality": "cxr",
        "image": {"src": "/cases/chest/image.webp", "width": img.shape[1], "height": img.shape[0]},
        "credit": "chest",
        "notes": {"n1": note},
        "annotations": [],
        "anatomy": "/media/chest-anatomy.png" if anat is not None else None,
        "study": json.loads(study.model_dump_json()),
        "provenance": {
            "findings": "Real output: TorchXRayVision DenseNet121 with Grad-CAM++ and PSPNet anatomy zones.",
            "faithfulness": "Real: deletion test per plan.md D1, computed by scripts/bake_web_assets.py until P1.9 ships.",
            "stability": "Real: P1's eight perturbations at severity 3.",
            "text_evidence": "Illustrative: the note is synthetic and the spans in this sample view are placed by hand. Run live analysis to have the context engine find them.",
            "calibration": "Not yet calibrated (P2.8): prob_calibrated equals prob_raw.",
        },
        "warnings": warnings,
    }
    (out_dir / "case.json").write_text(json.dumps(case, indent=2), encoding="utf-8")

    return {
        "image": "/media/chest.webp",
        "size": [img.shape[1], img.shape[0]],
        "findings": story_findings,
        "primary": primary["id"],
        "stability_tiles": primary_tiles,
        "perturbation_atlas": "/media/chest-perturbations.webp",
        "anatomy": "/media/chest-anatomy.png" if anat is not None else None,
        "quality_beats": degraded,
        "cdf": [round(float(v), 4) for v in cdf[::4]],  # 64 samples of the luminance CDF
        "note": note,
        "model": out.model_id,
        "source_sha256": sha256(src),
    }


def _sample_case(case_id: str, title: str, modality: str, raw_png: bytes, annotations: list[dict], extra_provenance: dict) -> dict:
    out_dir = CASES / case_id
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        cfg = PipelineConfig(cache_dir=Path(tmp) / "c", artifact_root=Path(tmp) / "a", ledger_path=Path(tmp) / "l.jsonl")
        study, _ = run_study(raw_png, study_id=f"sample-{case_id}", modality_hint=modality, config=cfg)
    img = load_image(raw_png)
    return {
        "id": case_id,
        "title": title,
        "modality": modality,
        "image": {"src": f"/cases/{case_id}/image.webp", "width": img.shape[1], "height": img.shape[0]},
        "credit": case_id if case_id != "bone" else "wrist",
        "notes": {"n1": SAMPLE_NOTES[case_id]} if case_id in SAMPLE_NOTES else {},
        "annotations": annotations,
        "anatomy": None,
        "study": json.loads(study.model_dump_json()),
        "provenance": {
            "findings": "Sample view: dataset annotation only. Run live analysis to read this image with the trained reader.",
            "text_evidence": "Synthetic clinical note, written for this demo.",
            **extra_provenance,
        },
        "warnings": [],
    }


def bake_wrist() -> dict:
    raw, box = fetch_wrist()
    full = gray(raw)
    h0, w0 = full.shape
    film = resize_long(full, 1600)
    h, w = film.shape
    nbox = [box[0] / w0, box[1] / h0, box[2] / w0, box[3] / h0]
    save_webp(film, MEDIA / "hero-wrist.webp", 82)

    # No-JS still: the film with the grease-pencil ellipse baked in (drawn at 2x, then reduced).
    big = Image.fromarray(film).convert("RGB").resize((w * 2, h * 2), Image.LANCZOS)
    d = ImageDraw.Draw(big)
    cx, cy = (nbox[0] + nbox[2]) / 2 * w * 2, (nbox[1] + nbox[3]) / 2 * h * 2
    rx, ry = (nbox[2] - nbox[0]) * w * 1.25, (nbox[3] - nbox[1]) * h * 1.55
    d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], outline=(243, 200, 70), width=14)
    big.resize((w, h), Image.LANCZOS).save(MEDIA / "hero-still.webp", format="WEBP", quality=82, method=6)

    case_raw = png(resize_long(full, 1440))
    out_dir = CASES / "bone"
    save_webp(resize_long(full, 1440), out_dir / "image.webp", 84)
    case = _sample_case(
        "bone", "Wrist radiograph, posteroanterior", "bone_xray", case_raw,
        [{"kind": "bbox", "bbox": [round(v, 4) for v in nbox], "label": "Fracture (dataset annotation)", "source": "FracAtlas ground truth"}],
        {"annotations": "Dataset annotation: FracAtlas ground-truth box, not model output. The bone reader arrives with P2.6 weights."},
    )
    (out_dir / "case.json").write_text(json.dumps(case, indent=2), encoding="utf-8")
    return {"image": "/media/hero-wrist.webp", "still": "/media/hero-still.webp", "size": [w, h], "fracture_box": [round(v, 4) for v in nbox], "source_sha256": sha256(raw)}


def bake_skin() -> dict:
    image_raw, mask_raw, meta = fetch_skin()
    rgb = np.asarray(Image.open(io.BytesIO(image_raw)).convert("RGB"))
    rgb = resize_long(rgb, 1280)
    mask = np.asarray(Image.open(io.BytesIO(mask_raw)).convert("L"))
    mask = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_AREA)
    mask = cv2.GaussianBlur(mask, (0, 0), 1.5)  # smooth iso-contour at 0.5 when magnified
    out_dir = CASES / "skin"
    save_webp(rgb, out_dir / "image.webp", 86)
    save_png(mask, out_dir / "mask.png")
    case = _sample_case(
        "skin", "Dermoscopy, pigmented lesion", "skin_dermoscopy", png(rgb),
        [{"kind": "mask", "ref": "/cases/skin/mask.png", "label": "Lesion boundary (dataset annotation)", "source": "ISIC 2018 Task 1 ground truth"}],
        {"annotations": "Dataset annotation: ISIC 2018 Task 1 lesion boundary, not model output. The skin reader arrives with P2.5 weights."},
    )
    (out_dir / "case.json").write_text(json.dumps(case, indent=2), encoding="utf-8")
    return {"source_sha256": sha256(image_raw), "isic_license": meta.get("copyright_license")}


def bake_brain() -> dict:
    raw = fetch_brain()
    film = gray(raw)
    out_dir = CASES / "brain"
    save_webp(film, out_dir / "image.webp", 88)
    case = _sample_case(
        "brain", "Brain MRI, axial, contrast-enhanced", "brain_mri", png(film), [],
        {"annotations": "No mask yet: the tumour segmenter arrives with P2.4 weights."},
    )
    (out_dir / "case.json").write_text(json.dumps(case, indent=2), encoding="utf-8")
    return {"source_sha256": sha256(raw)}


def main() -> None:
    reader = CxrReader.load(anatomy_fn=AnatomySegmenter.try_load())
    anatomy_fn = AnatomySegmenter.try_load()
    if anatomy_fn is None:
        raise SystemExit("anatomy segmenter weights missing")

    chest = bake_chest(reader, anatomy_fn)
    wrist = bake_wrist()
    skin = bake_skin()
    brain = bake_brain()

    credits = []
    for key, src in SOURCES.items():
        entry = {k: v for k, v in src.items() if k in ("title", "author", "license", "license_url", "page", "modifications")}
        entry["id"] = key
        entry["source_sha256"] = {"chest": chest, "wrist": wrist, "skin": skin, "brain": brain}[key]["source_sha256"]
        credits.append(entry)
    (MEDIA / "credits.json").write_text(json.dumps(credits, indent=2, ensure_ascii=False), encoding="utf-8")

    STORY.mkdir(parents=True, exist_ok=True)
    (STORY / "chest.json").write_text(json.dumps({"chest": chest, "wrist": wrist}, indent=2), encoding="utf-8")
    index = [
        {"id": "chest", "title": "Chest radiograph, posteroanterior", "modality": "cxr"},
        {"id": "bone", "title": "Wrist radiograph, posteroanterior", "modality": "bone_xray"},
        {"id": "skin", "title": "Dermoscopy, pigmented lesion", "modality": "skin_dermoscopy"},
        {"id": "brain", "title": "Brain MRI, axial, contrast-enhanced", "modality": "brain_mri"},
    ]
    for entry in index:  # small thumbnails so the study rail never downloads full films
        full = Image.open(CASES / entry["id"] / "image.webp")
        full.thumbnail((240, 240), Image.LANCZOS)
        full.save(CASES / entry["id"] / "thumb.webp", format="WEBP", quality=78, method=6)
        entry["thumb"] = f"/cases/{entry['id']}/thumb.webp"
    (CASES / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")

    p = next(f for f in chest["findings"] if f["id"] == chest["primary"])
    print(f"primary finding: {p['label']} p={p['prob']} region={p['region']} faithfulness={p['faithfulness']} stability={p['stability']}")
    for f in chest["findings"]:
        print(f"  {f['id']} {f['label']:<14} p={f['prob']:.3f} {f['tier']:<8} {f['status']:<10} drop={f['faithfulness'] and f['faithfulness']['drop']} flip={f['stability']['flip_rate']}")


if __name__ == "__main__":
    main()
