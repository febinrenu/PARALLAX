"""Register a trained model in ml/artifacts/registry.json (committed). Weights themselves are never committed.

    kaggle kernels output <user>/<slug> -p ml/artifacts/<model_id>      # pull the notebook output first
    python ml/artifacts/register.py ml/artifacts/<model_id> [--note "..."]

Reads model_meta.json + metrics.json written by the notebook, re-hashes the weights, and writes one registry
entry. P1's inference wrappers read `classes` (ordered), `preproc_spec`, `arch` and `weights_sha256` from here.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

REGISTRY = Path(__file__).resolve().parent / "registry.json"

LICENSES = {
    "skin_cls": ("Weights derived from timm ImageNet checkpoints (Apache-2.0) fine-tuned on HAM10000 (CC BY-NC 4.0): non-commercial use only", "ISIC 2018 Task 3 test set is held out; MedSAM and MedGemma are not involved"),
    "brain_cls": ("timm EfficientNet-B0 (Apache-2.0) fine-tuned on Brain Tumor MRI Dataset (CC0)", "Trained on the leakage-free split; BDNeuro-MRI never seen"),
    "brain_seg": ("smp U-Net resnet34 (MIT) fine-tuned on LGG MRI Segmentation (CC BY-NC-SA 4.0): non-commercial, share-alike", "Patient-level split; trained on TCGA-LGG only"),
    "bone_det": ("Ultralytics YOLO (AGPL-3.0) fine-tuned on FracAtlas (CC BY 4.0): weights inherit AGPL-3.0", "Trained and tested on FracAtlas; no external test set"),
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while c := f.read(1 << 20):
            h.update(c)
    return h.hexdigest()


def headline(metrics: dict) -> dict:
    """Pick the few numbers shown in the registry for a quick look (full tables stay in metrics.json)."""
    out = {}
    for k in ("headline", "official_test", "test", "leakage_free_test"):
        blk = metrics.get(k)
        if isinstance(blk, dict):
            for m in ("balanced_accuracy", "accuracy", "macro_f1", "mean_patient_dice", "map50", "auroc_image"):
                v = blk.get(m)
                if isinstance(v, dict) and "point" in v:
                    out[f"{k}.{m}"] = {x: round(v[x], 4) for x in ("point", "lo", "hi")}
    return out


def register(model_dir: Path, note: str = "", registry: Path = REGISTRY) -> dict:
    model_dir = Path(model_dir)
    meta = json.loads((model_dir / "model_meta.json").read_text(encoding="utf-8"))
    metrics = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
    wf = meta.get("weights_file")
    if wf:
        wsha = sha256_file(model_dir / wf)
        if meta.get("weights_sha256") and wsha != meta["weights_sha256"]:
            raise SystemExit(f"{wf}: sha256 {wsha[:12]} does not match model_meta.json {meta['weights_sha256'][:12]}; the download is corrupt")
        meta["weights_sha256"] = wsha
    key = meta["model_id"]
    lic, contamination = LICENSES.get(key.split("@")[0], ("see model card", "see model card"))
    entry = {
        "model_id": key, "task": meta["task"], "arch": meta["arch"], "classes": meta["classes"], "preproc_spec": meta["preproc_spec"],
        "weights_file": wf, "weights_sha256": meta.get("weights_sha256"), "weights_bytes": (model_dir / wf).stat().st_size if wf else None,
        "training_data": meta["training_data"], "split_name": meta["split_name"], "split_hash": meta["split_hash"], "git_commit": meta["git_commit"], "seed": meta.get("seed"),
        "temperature": None, "conformal": None, "license": lic, "contamination": contamination,
        "metrics_file": f"ml/artifacts/{model_dir.name}/metrics.json", "headline": headline(metrics), "kaggle_kernel": meta.get("kaggle_kernel"),
        "registered_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "note": note,
    }
    reg = json.loads(registry.read_text(encoding="utf-8")) if registry.is_file() else {"schema": 1, "models": {}}
    reg["models"][key] = entry
    registry.write_text(json.dumps(reg, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return entry


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--note", default="")
    a = ap.parse_args(argv)
    e = register(a.model_dir, a.note)
    print(f"registered {e['model_id']} sha256 {str(e['weights_sha256'])[:12]} headline {e['headline']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
