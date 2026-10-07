"""P2.12 corruption benchmark: 8 perturbations x 5 severities on every test set, plus the stability flip rate used by P2.10.

    python ml/eval/corruption.py run [model ...]    # run models on the perturbed images, cache outputs (GPU, minutes)
    python ml/eval/corruption.py evaluate           # recompute accuracy-vs-severity tables from the cached outputs

Perturbations come from backend/medproof/verify/perturbations.py, applied to the decoded image *before* preprocessing, as the
product's stability check does. Metric per model: skin and brain classifiers balanced accuracy; segmenter pooled Dice (photometric
perturbations only, since a rotation or crop would also move the ground-truth mask); bone detector and chest reader AUROC.
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for p in (REPO, REPO / "backend"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from medproof.intake.decode import load_image  # noqa: E402
from medproof.intake.preprocess import PreprocSpec, prepare  # noqa: E402
from medproof.verify.perturbations import NAMES, SEVERITIES, apply  # noqa: E402

from ml.data.common import data_root, image_path, load_split  # noqa: E402
from ml.eval import bootstrap as bs  # noqa: E402
from ml.eval import metrics as mt  # noqa: E402
from ml.eval import predict as pr  # noqa: E402

ART = REPO / "ml" / "artifacts"
SEED = 20261006
GEOMETRIC = ("rotate", "crop")  # change geometry, so excluded where a mask or box must stay aligned
B_CORRUPTION = 300  # resamples per cell: 40 cells per model, so the CIs are a little coarser than the headline ones
MODELS = ("skin_cls", "brain_cls", "brain_seg", "bone_det", "cxr_chex")


def _pool(fn, items, workers=6):
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(fn, items))


def _perturb_all(images: list[np.ndarray], name: str, sev: int) -> list[np.ndarray]:
    return _pool(lambda a: apply(name, a, sev, SEED), images)


def _prep_batch(images, spec) -> "torch.Tensor":  # noqa: F821
    import torch

    return torch.from_numpy(np.stack(_pool(lambda a: prepare(a, spec), images)))


# ----------------------------------------------------------------------------- per-model sets


def _skin():
    df = load_split("ham10000")
    df = df[(df.split == "official_test") & df.eval_batch].reset_index(drop=True)
    classes = pr.meta("skin_cls")["classes"]
    imgs = _pool(lambda r: load_image(image_path(r, data_root())).analysis, [r for _, r in df.iterrows()])
    return df.image_id.to_numpy().astype(str), df.label.map({c: i for i, c in enumerate(classes)}).to_numpy(), imgs, "skin_cls"


def _brain():
    df = load_split("brain_mri")
    df = df[df.split == "test"].reset_index(drop=True)
    classes = pr.meta("brain_cls")["classes"]
    imgs = _pool(lambda r: load_image(image_path(r, data_root())).analysis, [r for _, r in df.iterrows()])
    return df.image_id.to_numpy().astype(str), df.label.map({c: i for i, c in enumerate(classes)}).to_numpy(), imgs, "brain_effnet"


def _seg():
    from PIL import Image

    df = load_split("lgg_seg")
    df = df[df.split == "test"].reset_index(drop=True)
    root = data_root()
    imgs = [np.asarray(Image.open(image_path(r, root)).convert("RGB"), dtype=np.float32) / 255.0 for _, r in df.iterrows()]
    masks = []
    import cv2

    for _, r in df.iterrows():
        m = (np.asarray(Image.open(image_path({"source_dir": r.source_dir, "relpath": r.mask_relpath}, root))) > 0).astype(np.uint8)
        masks.append(cv2.resize(m, (256, 256), interpolation=cv2.INTER_NEAREST))
    return df.image_id.to_numpy().astype(str), np.stack(masks), imgs, "seg"


def _bone(n_neg: int = 100, long_side: int = 1024):
    import cv2

    df = load_split("fracatlas")
    t = df[df.split == "test"]
    sub = pd.concat([t[t.label == "fracture"], t[t.label == "no_fracture"].sample(n=min(n_neg, int((t.label == "no_fracture").sum())), random_state=SEED)]).sort_values("image_id").reset_index(drop=True)

    def load(r):
        a = load_image(image_path(r, data_root())).analysis
        s = long_side / max(a.shape[:2])
        return cv2.resize(a, (int(a.shape[1] * s), int(a.shape[0] * s)), interpolation=cv2.INTER_AREA) if s < 1 else a

    return sub.image_id.to_numpy().astype(str), (sub.label == "fracture").to_numpy().astype(int), _pool(load, [r for _, r in sub.iterrows()], 4), "yolo"


def _cxr(n: int = 500):
    df = load_split("rsna")
    t = df[(df.split == "test") & df.eval_batch].sample(n=n, random_state=SEED).sort_values("image_id").reset_index(drop=True)
    imgs = _pool(lambda r: load_image(image_path(r, data_root())).analysis, [r for _, r in t.iterrows()], 4)
    return t.image_id.to_numpy().astype(str), (t.label == "Lung_Opacity").to_numpy().astype(int), imgs, "cxr_xrv"


# ----------------------------------------------------------------------------- run


def run_model(name: str) -> Path:
    ids, y, imgs, spec = {"skin_cls": _skin, "brain_cls": _brain, "brain_seg": _seg, "bone_det": _bone, "cxr_chex": _cxr}[name]()
    import os

    cap = int(os.environ.get("CORRUPTION_N", "0"))
    if cap:  # dry run on a few images
        ids, y, imgs = ids[:cap], y[:cap], imgs[:cap]
    seg_spec = PreprocSpec("brain_seg", 256, 3, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    if name in ("skin_cls", "brain_cls"):
        model = pr.load_timm(name)
        infer = lambda arrs: pr.logits(model, _prep_batch(arrs, spec))  # noqa: E731
    elif name == "brain_seg":
        model = pr.load_unet(name)
        infer = lambda arrs: (pr.seg_probs(model, _prep_batch(arrs, seg_spec)) > 0.5)  # noqa: E731
    elif name == "bone_det":
        model = pr.load_yolo(name)
        infer = lambda arrs: pr.yolo_scores(model, arrs)  # noqa: E731
    else:
        model = pr.load_cxr("chex")
        li = list(model.pathologies).index("Lung Opacity")
        infer = lambda arrs: pr.logits(model, _prep_batch(arrs, spec))[:, li]  # noqa: E731

    clean = infer(imgs)
    cells = np.zeros((len(NAMES), len(SEVERITIES)) + clean.shape, dtype=clean.dtype if clean.dtype == bool else np.float32)
    for pi, pname in enumerate(NAMES):
        for si, sev in enumerate(SEVERITIES):
            if name == "brain_seg" and pname in GEOMETRIC:
                continue
            cells[pi, si] = infer(_perturb_all(imgs, pname, sev))
            print(f"  {name} {pname} sev{sev}", flush=True)
    out = ART / name / "predictions"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "corruption.npz", ids=ids, y=y, clean=clean, cells=cells, names=np.array(NAMES), severities=np.array(SEVERITIES), skipped=np.array(list(GEOMETRIC) if name == "brain_seg" else []))
    return out / "corruption.npz"


# ----------------------------------------------------------------------------- evaluate


def _metric_fn(name: str, classes: int):
    if name in ("skin_cls", "brain_cls"):
        return lambda y, o: mt.balanced_accuracy(y, o.argmax(1), classes), "balanced accuracy"
    if name == "brain_seg":
        return None, "pooled Dice"
    return (lambda y, o: mt.auroc_binary(y, o)), "AUROC"


def evaluate(name: str) -> dict:
    z = np.load(ART / name / "predictions" / "corruption.npz", allow_pickle=False)
    y, clean, cells, names, sevs = z["y"], z["clean"], z["cells"], list(z["names"]), list(z["severities"])
    skipped = set(str(s) for s in z["skipped"])
    n = len(z["ids"])
    out = {"model": name, "n_images": int(n), "perturbations": names, "severities": [int(s) for s in sevs], "cells": {}, "bootstrap_B": B_CORRUPTION}
    if name == "brain_seg":
        gt = y.astype(bool)

        def counts(pred):  # per-slice TP, FP, FN so each bootstrap replicate is three sums, not a pass over the masks
            return np.stack([(pred & gt).sum((-2, -1)), (pred & ~gt).sum((-2, -1)), (~pred & gt).sum((-2, -1))], -1).astype(float)

        def dice_of(c):
            return mt.dice(c[:, 0].sum(), c[:, 1].sum(), c[:, 2].sum())

        out["metric"] = "pooled Dice"
        c0 = counts(clean.astype(bool))
        ci0 = bs.ci({"c": c0}, lambda c: dice_of(c), B=B_CORRUPTION)
        out["clean"] = {"point": ci0["point"], "lo": ci0["lo"], "hi": ci0["hi"]}
        for pi, pname in enumerate(names):
            for si, sev in enumerate(sevs):
                if pname in skipped:
                    out["cells"][f"{pname}/{sev}"] = None
                    continue
                ci = bs.ci({"c": counts(cells[pi, si].astype(bool))}, lambda c: dice_of(c), B=B_CORRUPTION)
                out["cells"][f"{pname}/{sev}"] = {"point": ci["point"], "lo": ci["lo"], "hi": ci["hi"]}
        return out
    k = int(y.max() + 1) if name in ("skin_cls", "brain_cls") else 2
    fn, label = _metric_fn(name, k)
    out["metric"] = label
    arrays_clean = {"y": y, "o": clean}
    c = bs.ci(arrays_clean, fn, strata=y, B=B_CORRUPTION)
    out["clean"] = {"point": c["point"], "lo": c["lo"], "hi": c["hi"]}
    for pi, pname in enumerate(names):
        for si, sev in enumerate(sevs):
            ci = bs.ci({"y": y, "o": cells[pi, si]}, fn, strata=y, B=B_CORRUPTION)
            out["cells"][f"{pname}/{sev}"] = {"point": ci["point"], "lo": ci["lo"], "hi": ci["hi"]}
    # stability: share of images whose predicted class changes under each perturbation at severity 2 (a mild, realistic setting)
    if name in ("skin_cls", "brain_cls"):
        base = clean.argmax(1)
        flips = np.stack([cells[pi, 1].argmax(1) != base for pi in range(len(names))], 1)
        out["flip_rate_severity2"] = {"mean_per_image": float(flips.mean()), "share_images_flip_rate_above_0.25": float((flips.mean(1) > 0.25).mean())}
    return out


def main(argv=None) -> int:
    a = argv or sys.argv[1:]
    cmd = a[0] if a else "evaluate"
    names = a[1:] or list(MODELS)
    if cmd == "run":
        for m in names:
            if not (ART / m / ("model_meta.json")).is_file() and not m.startswith("cxr_"):
                print("skip", m)
                continue
            print("running", m, run_model(m), flush=True)
    elif cmd == "evaluate":
        res = {"seed": SEED, "models": {}}
        for m in names:
            if (ART / m / "predictions" / "corruption.npz").is_file():
                res["models"][m] = evaluate(m)
                r = res["models"][m]
                worst = min((v["point"], k) for k, v in r["cells"].items() if v)
                print(f"{m:10s} clean {r['clean']['point']:.3f} worst {worst[1]} {worst[0]:.3f}")
        (REPO / "reports" / "corruption.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
