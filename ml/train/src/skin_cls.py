# %% [markdown]
# # P2.5 Skin lesion classifier (ConvNeXt-Tiny, 7 classes)
# Trains on HAM10000 (ISIC 2018 Task 3 training images), split by `lesion_id` plus duplicate groups.
# Headline number: **balanced multiclass accuracy on the official ISIC 2018 Task 3 test set**, with a 95% cluster-bootstrap CI.
# Decision support only. Not a medical device.

# %%
import json, os, subprocess, sys, time
from pathlib import Path

REPO_URL = os.environ.get("PARALLAX_REPO_URL", "https://github.com/febinrenu/PARALLAX.git")
BRANCH = os.environ.get("PARALLAX_BRANCH", "main")
SMOKE = os.environ.get("SMOKE") == "1"
SUFFIX = os.environ.get("SLUG_SUFFIX", "")  # "_s2", "_s3": extra seeds for the ensemble
SLUG = "skin_cls" + SUFFIX
SEED = int(os.environ.get("SEED", "20261006"))
ON_KAGGLE = Path("/kaggle/working").is_dir()
if ON_KAGGLE:
    SCRATCH = Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else Path("/tmp")  # anything under /kaggle/working is saved as notebook output
    REPO = SCRATCH / "PARALLAX"
    if not REPO.exists():
        subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, str(REPO)], check=True)
    ROOT, OUT = Path(os.environ.get("PARALLAX_DATA_ROOT", SCRATCH / "raw")), Path(f"/kaggle/working/{SLUG}")
else:
    REPO = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "plan.md").exists())
    ROOT = Path(os.environ.get("PARALLAX_DATA_ROOT", REPO / "ml" / "data" / "raw"))
    OUT = Path(os.environ.get("PARALLAX_OUT", REPO / "ml" / "artifacts" / "_local" / SLUG))
os.environ["PARALLAX_DATA_ROOT"] = str(ROOT)
sys.path[:0] = [str(REPO), str(REPO / "backend")]
if os.environ.get("PARALLAX_SKIP_PIP") != "1":
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pydicom", "timm"], check=False)
if ON_KAGGLE or not (ROOT / "isic2018_t3_train" / "manifest.json").exists():
    subprocess.run([sys.executable, str(REPO / "ml/data/download.py"), "isic2018_t3_train", "isic2018_t3_test", "--root", str(ROOT)], check=True)
OUT.mkdir(parents=True, exist_ok=True)
from ml.train import common as tc

HEADER = tc.print_header(ROOT, ["isic2018_t3_train", "isic2018_t3_test"])

# %%
CFG = dict(
    arch="convnext_tiny.fb_in22k_ft_in1k", epochs=int(os.environ.get("EPOCHS", 2 if SMOKE else 15)), lr=2e-4, weight_decay=0.05, batch_size=48, label_smoothing=0.1, balance_power=0.5,
    aug=dict(rot_deg=180.0, scale=(0.85, 1.15), shift=0.08, flip_h=True, flip_v=True, brightness=0.15, contrast=0.15), bootstrap_B=100 if SMOKE else 2000, smoke_per_split=60,
)
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
SPEC = "skin_cls"  # backend/medproof/intake/preprocess.py
print(json.dumps(CFG, indent=1, default=str))

# %%
import numpy as np, pandas as pd
from ml.data.common import load_split, image_path
from ml.eval import metrics as mt

df = load_split("ham10000")
df = df[df.split != "dropped"].reset_index(drop=True)
if SMOKE:  # tiny, fast, still touches every code path
    df = pd.concat([g.head(CFG["smoke_per_split"]) for _, g in df.groupby("split")]).reset_index(drop=True)
cls_idx = {c: i for i, c in enumerate(CLASSES)}
t0 = time.time()
data = {}
for split in ["train", "val", "cal", "test", "official_test"]:
    d = df[df.split == split].reset_index(drop=True)
    X = tc.load_prepared([image_path(r, ROOT) for _, r in d.iterrows()], SPEC, workers=1 if SMOKE else 4)
    data[split] = dict(df=d, X=X, y=d.label.map(cls_idx).to_numpy())
    print(split, len(d), tuple(X.shape), "class counts", np.bincount(data[split]["y"], minlength=7).tolist())
print(f"preprocessed in {time.time() - t0:.0f}s")

# %%
run_t0 = time.time()
res = tc.train_classifier(
    data["train"]["X"], data["train"]["y"], data["val"]["X"], data["val"]["y"], arch=CFG["arch"], n_classes=7, epochs=CFG["epochs"], lr=CFG["lr"], weight_decay=CFG["weight_decay"],
    batch_size=CFG["batch_size"], label_smoothing=CFG["label_smoothing"], balance_power=CFG["balance_power"], aug=CFG["aug"], pretrained=not SMOKE, seed=SEED,
)
print("best epoch", res["best_epoch"])

# %%
preds, metrics = {}, {"config": CFG, "history": res["history"], "best_epoch": res["best_epoch"]}
for split, d in data.items():
    logits = tc.predict_logits(res["model"], d["X"])
    p = mt.softmax(logits)
    sub = d["df"]
    preds[split] = dict(ids=sub.image_id.to_numpy().astype(str), logits=logits.astype(np.float32), y=d["y"], group=sub.group.to_numpy().astype(str),
                        age=sub.age.fillna(-1).to_numpy(float), sex=sub.sex.fillna("unknown").to_numpy().astype(str), site=sub.site_general.fillna("unknown").to_numpy().astype(str))
    if split in ("official_test", "test", "val", "cal"):
        metrics[split] = tc.bootstrap_report(d["y"], p, sub.group.to_numpy().astype(str), 7, CLASSES, B=CFG["bootstrap_B"], positive="mel")
        r = metrics[split]
        print(split, "BMA", round(r["balanced_accuracy"]["point"], 4), [round(r["balanced_accuracy"][k], 4) for k in ("lo", "hi")], "mel recall", round(r["per_class_recall"].get("mel", {}).get("point", float("nan")), 3))
metrics["headline"] = metrics["official_test"]
metrics["notes"] = {
    "primary_metric": "balanced multiclass accuracy (mean per-class recall) on the official ISIC 2018 Task 3 test set",
    "ci": f"95% percentile cluster bootstrap, groups = lesion_id from the ISIC API, resampled within class, B={CFG['bootstrap_B']}",
    "contamination": "official test images never used for training, validation or calibration; train copies of test images were removed by the leakage audit",
}

# %%
from ml.data.common import SPLITS_DIR
split_hash = json.load(open(SPLITS_DIR / "split_hashes.json"))["ham10000"]["split_hash"]
model_id = f"skin_cls{SUFFIX}@{tc.git_commit()[:8]}"
tc.write_run(OUT, model_id=model_id, task="classification", arch=CFG["arch"], classes=CLASSES, preproc_spec=SPEC, state_dict=res["state_dict"], weights_name="weights.pt", split_name="ham10000",
             split_hash=split_hash, training_data=["isic2018_t3_train (HAM10000)"], metrics=metrics, predictions=preds, data_info=HEADER, wall_sec=time.time() - run_t0,
             extra={"kaggle_kernel": os.environ.get("KAGGLE_KERNEL_RUN_TYPE") and "parallax-skin-cls", "seed": SEED})
print("wrote", OUT, sorted(p.name for p in OUT.iterdir()))
