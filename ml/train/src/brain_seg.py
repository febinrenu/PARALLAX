# %% [markdown]
# # P2.4 Brain tumour segmenter (U-Net, ResNet34 encoder)
# Attach the Kaggle dataset **mateuszbuda/lgg-mri-segmentation** as notebook input (110 TCGA-LGG patients, 3 channels: pre-contrast, FLAIR, post-contrast).
# Patient-level 10-fold split defined in `ml/data/splits/lgg_seg.csv`; **fold 0 = 88 / 11 / 11 patients** and is the shipped model.
# Set `FOLDS=0,1,2,...` to also run other folds for out-of-fold per-patient Dice (tighter CI).
# Out-of-domain note: the product sees single-channel T1-CE slices, TCGA has three channels; training randomly replicates one channel to all three.
# Decision support only. Not a medical device.

# %%
import json, os, subprocess, sys, time
from pathlib import Path

REPO_URL = os.environ.get("PARALLAX_REPO_URL", "https://github.com/febinrenu/PARALLAX.git")
BRANCH = os.environ.get("PARALLAX_BRANCH", "main")
SMOKE = os.environ.get("SMOKE") == "1"
SLUG = "brain_seg"
ON_KAGGLE = Path("/kaggle/working").is_dir()
if ON_KAGGLE:
    SCRATCH = Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else Path("/tmp")  # anything under /kaggle/working is saved as notebook output
    REPO = SCRATCH / "PARALLAX"
    if not REPO.exists():
        subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, str(REPO)], check=True)
    ROOT, OUT = Path(os.environ.get("PARALLAX_DATA_ROOT", SCRATCH / "raw")), Path(f"/kaggle/working/{SLUG}")
else:
    REPO = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "CLAUDE.md").exists())
    ROOT = Path(os.environ.get("PARALLAX_DATA_ROOT", REPO / "ml" / "data" / "raw"))
    OUT = Path(os.environ.get("PARALLAX_OUT", REPO / "ml" / "artifacts" / "_local" / SLUG))
os.environ["PARALLAX_DATA_ROOT"] = str(ROOT)
sys.path[:0] = [str(REPO), str(REPO / "backend")]
if os.environ.get("PARALLAX_SKIP_PIP") != "1":
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pydicom", "timm", "segmentation-models-pytorch"], check=False)
OUT.mkdir(parents=True, exist_ok=True)
from ml.train import common as tc

if ON_KAGGLE:
    tc.link_attached(ROOT, "lgg_seg", "kaggle_3m")
HEADER = tc.print_header(ROOT, ["lgg_seg"])

# %%
FOLDS = [int(f) for f in os.environ.get("FOLDS", "0").split(",")]
CFG = dict(encoder="resnet34", epochs=int(os.environ.get("EPOCHS", 2 if SMOKE else 20)), lr=3e-4, batch_size=16, chan_aug_p=0.3, threshold=0.5, bootstrap_B=100 if SMOKE else 2000)
print(json.dumps(CFG), "folds", FOLDS)

# %%
import numpy as np, pandas as pd, torch
from ml.data.common import load_split, image_path, SPLITS_DIR
from ml.eval import bootstrap as bs
from ml.train import seg

full = load_split("lgg_seg").reset_index(drop=True)
t0 = time.time()
X_all, M_all = seg.load_slices([image_path(r, ROOT) for _, r in full.iterrows()], [image_path({"source_dir": r.source_dir, "relpath": r.mask_relpath}, ROOT) for _, r in full.iterrows()])
print(f"loaded {len(full)} slices in {time.time() - t0:.0f}s", tuple(X_all.shape), "tumour slices", int(M_all.flatten(1).any(1).sum()))


def take(mask):
    idx = np.nonzero(np.asarray(mask))[0]
    return idx, X_all[torch.from_numpy(idx)], M_all[torch.from_numpy(idx)]


# %%
run_t0 = time.time()
results, oof_rows, shipped = {}, [], None
nf = int(full.fold.max()) + 1
for k in FOLDS:
    test_f, val_f = k, (k + 1) % nf
    _, Xtr, Mtr = take(~full.fold.isin([test_f, val_f]))
    _, Xva, Mva = take(full.fold == val_f)
    ti, Xte, Mte = take(full.fold == test_f)
    print(f"fold {k}: train {len(Xtr)} val {len(Xva)} test {len(Xte)} slices")
    res = seg.train_unet(Xtr, Mtr, Xva, Mva, encoder=CFG["encoder"], epochs=CFG["epochs"], lr=CFG["lr"], batch_size=CFG["batch_size"], pretrained=not SMOKE, chan_aug_p=CFG["chan_aug_p"])
    probs = seg.predict_probs(res["model"], Xte)
    pats = full.iloc[ti].group.to_numpy().astype(str)
    ev = seg.evaluate_patients(probs, Mte[:, 0].numpy(), pats, CFG["threshold"])
    dices = np.array([r["dice"] for r in ev["patients"]])
    hds = np.array([r["hd95_px"] for r in ev["patients"]])
    ci = bs.ci({"d": dices}, lambda d: float(np.mean(d)), B=CFG["bootstrap_B"], method="bca")
    ci_hd = bs.ci({"d": hds[np.isfinite(hds)]}, lambda d: float(np.mean(d)), B=CFG["bootstrap_B"], method="bca") if np.isfinite(hds).sum() > 2 else None
    print(f"fold {k}: mean per-patient Dice {ci['point']:.4f} [{ci['lo']:.4f}, {ci['hi']:.4f}] over {len(dices)} patients; slice AUROC {ev['slice_auroc']:.3f}")
    results[f"fold_{k}"] = {"n_patients": len(dices), "mean_patient_dice": ci, "mean_patient_hd95_px": ci_hd, "slice_detection_auroc": ev["slice_auroc"], "pooled_dice": ev["pooled_dice"],
                            "patients": ev["patients"], "best_epoch": res["best_epoch"], "history": res["history"]}
    oof_rows += [(r["patient"], r["dice"]) for r in ev["patients"]]
    if k == 0 or shipped is None:
        shipped = dict(k=k, res=res, probs=probs, ti=ti, Mte=Mte)

# %%
metrics = {"config": CFG, "folds_run": FOLDS, **results}
shipped_block = results[f"fold_{shipped['k']}"]
metrics["headline"] = {"mean_patient_dice": shipped_block["mean_patient_dice"], "fold": shipped["k"], "n_patients": shipped_block["n_patients"]}
metrics["test"] = metrics["headline"]
if len(FOLDS) > 1:
    d = np.array([v for _, v in oof_rows])
    metrics["out_of_fold"] = {"n_patients": int(len(d)), "mean_patient_dice": bs.ci({"d": d}, lambda d: float(np.mean(d)), B=CFG["bootstrap_B"], method="bca")}
metrics["notes"] = {
    "dice": "per-patient volume Dice (TP/FP/FN pooled over a patient's slices), threshold 0.5, no test-time augmentation",
    "ci": f"95% BCa bootstrap over patients, B={CFG['bootstrap_B']}", "hd95": "per-slice 2D HD95 in pixels (no spacing in the dataset), averaged over slices where both masks are non-empty",
    "limitation": "trained on TCGA-LGG only (low-grade glioma, FLAIR-weighted ground truth); out of domain for meningioma, pituitary and single-channel T1-CE uploads",
    "reference_baseline": "mateuszbuda/brain-segmentation-pytorch was trained on this same data and is not reported as our result",
}
ids = full.iloc[shipped["ti"]].image_id.to_numpy().astype(str)
preds = {"test": dict(ids=ids, patients=full.iloc[shipped["ti"]].group.to_numpy().astype(str), probs_u8=(shipped["probs"] * 255).astype(np.uint8),
                      masks_packed=np.packbits(shipped["Mte"][:, 0].numpy().astype(bool), axis=None), mask_shape=np.array(shipped["Mte"][:, 0].shape))}
split_hash = json.load(open(SPLITS_DIR / "split_hashes.json"))["lgg_seg"]["split_hash"]
tc.write_run(OUT, model_id=f"brain_seg@{tc.git_commit()[:8]}", task="segmentation", arch=f"smp.Unet({CFG['encoder']})", classes=["background", "tumour"], preproc_spec=seg.SEG_SPEC,
             state_dict=shipped["res"]["state_dict"], weights_name="weights.pt", split_name="lgg_seg", split_hash=split_hash, training_data=["lgg_seg (Kaggle mateuszbuda, TCGA-LGG)"], metrics=metrics,
             predictions=preds, data_info=HEADER, wall_sec=time.time() - run_t0, extra={"threshold": CFG["threshold"], "input_size": seg.SEG_SIZE, "shipped_fold": shipped["k"]})
print("wrote", OUT, sorted(p.name for p in OUT.iterdir()))
