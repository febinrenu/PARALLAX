# %% [markdown]
# # P2.3 Brain MRI classifier (EfficientNet-B0, 4 classes) with a leakage audit
# Attach the Kaggle dataset **masoudnickparvar/brain-tumor-mri-dataset** as notebook input.
# Two models are trained so the benchmark inflation is *measured*:
# * **A** trains on the original Kaggle Training folder and is scored on the original Testing folder (how most people report it).
# * **B** trains on our leakage-free group split and is scored on its own held-out test split. **B ships.**
# The leakage-free number for A is also computed by removing Testing images that have a near-duplicate in Training.
# Decision support only. Not a medical device.

# %%
import json, os, subprocess, sys, time
from pathlib import Path

REPO_URL = os.environ.get("PARALLAX_REPO_URL", "https://github.com/febinrenu/PARALLAX.git")
BRANCH = os.environ.get("PARALLAX_BRANCH", "main")
SMOKE = os.environ.get("SMOKE") == "1"
SLUG = "brain_cls"
ON_KAGGLE = Path("/kaggle/working").is_dir()
if ON_KAGGLE:
    REPO = Path("/kaggle/working/PARALLAX")
    if not REPO.exists():
        subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, str(REPO)], check=True)
    SCRATCH = Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else Path("/kaggle/working")  # not part of the saved output
    ROOT, OUT = Path(os.environ.get("PARALLAX_DATA_ROOT", SCRATCH / "raw")), Path(f"/kaggle/working/{SLUG}")
else:
    REPO = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "CLAUDE.md").exists())
    ROOT = Path(os.environ.get("PARALLAX_DATA_ROOT", REPO / "ml" / "data" / "raw"))
    OUT = Path(os.environ.get("PARALLAX_OUT", REPO / "ml" / "artifacts" / "_local" / SLUG))
os.environ["PARALLAX_DATA_ROOT"] = str(ROOT)
sys.path[:0] = [str(REPO), str(REPO / "backend")]
if os.environ.get("PARALLAX_SKIP_PIP") != "1":
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pydicom", "timm"], check=False)
OUT.mkdir(parents=True, exist_ok=True)
from ml.train import common as tc

if ON_KAGGLE:
    tc.link_attached(ROOT, "brain_mri", "Training")
HEADER = tc.print_header(ROOT, ["brain_mri"])

# %%
CFG = dict(arch="efficientnet_b0.ra_in1k", epochs=int(os.environ.get("EPOCHS", 2 if SMOKE else 10)), lr=3e-4, weight_decay=0.01, batch_size=64, label_smoothing=0.1,
           aug=dict(rot_deg=10.0, scale=(0.9, 1.1), shift=0.05, flip_h=True, flip_v=False, brightness=0.1, contrast=0.1), bootstrap_B=100 if SMOKE else 2000, smoke_per_split=40)
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
SPEC = "brain_effnet"
print(json.dumps(CFG, indent=1, default=str))

# %%
import numpy as np, pandas as pd, torch
from ml.data.common import load_split, image_path
from ml.eval import bootstrap as bs, metrics as mt

full = load_split("brain_mri")  # every image incl. those the audit dropped; `split` is the leakage-free assignment
if SMOKE:
    full = pd.concat([g.head(CFG["smoke_per_split"]) for _, g in full.groupby(["orig_split", "split"])]).reset_index(drop=True)
cls_idx = {c: i for i, c in enumerate(CLASSES)}
full["y"] = full.label.map(cls_idx)

# Which original-Testing images have a near-duplicate (Hamming <= 4) or a same-scan neighbour (<= 10) in original Training?
tr_dup = set(full[full.orig_split == "training"].dup_cluster)
tr_loose = set(full[full.orig_split == "training"].loose_cluster)
full["leaked_dup"] = (full.orig_split == "testing") & full.dup_cluster.isin(tr_dup)
full["leaked_loose"] = (full.orig_split == "testing") & full.loose_cluster.isin(tr_loose)
print("original Testing images:", int((full.orig_split == "testing").sum()), "with a duplicate in Training:", int(full.leaked_dup.sum()),
      "with a same-scan neighbour in Training:", int(full.leaked_loose.sum()))

t0 = time.time()
full_X = tc.load_prepared([image_path(r, ROOT) for _, r in full.iterrows()], SPEC, workers=1 if SMOKE else 4)
print(f"preprocessed {len(full)} images in {time.time() - t0:.0f}s", tuple(full_X.shape))


def subset(mask):
    idx = np.nonzero(np.asarray(mask))[0]
    return full.iloc[idx].reset_index(drop=True), full_X[torch.from_numpy(idx)]


# %%
def fit(train_mask, val_mask, tag):
    d_tr, X_tr = subset(train_mask)
    d_va, X_va = subset(val_mask)
    print(tag, "train", len(d_tr), "val", len(d_va))
    return tc.train_classifier(X_tr, d_tr.y.to_numpy(), X_va, d_va.y.to_numpy(), arch=CFG["arch"], n_classes=4, epochs=CFG["epochs"], lr=CFG["lr"], weight_decay=CFG["weight_decay"],
                               batch_size=CFG["batch_size"], label_smoothing=CFG["label_smoothing"], aug=CFG["aug"], pretrained=not SMOKE)


def evaluate(model, mask, tag):
    d, X = subset(mask)
    p = mt.softmax(tc.predict_logits(model, X))
    rep = tc.bootstrap_report(d.y.to_numpy(), p, d.group.to_numpy().astype(str), 4, CLASSES, B=CFG["bootstrap_B"])
    print(tag, "n", len(d), "acc", round(rep["accuracy"]["point"], 4), [round(rep["accuracy"][k], 4) for k in ("lo", "hi")], "bal acc", round(rep["balanced_accuracy"]["point"], 4))
    return rep, d, p


run_t0 = time.time()
metrics = {"config": CFG}

# --- Model A: original Kaggle split. Validation = a random (image-level, as most notebooks do) 10% of Training.
rng = np.random.RandomState(tc.SEED)
is_tr = (full.orig_split == "training").to_numpy()
val_a = is_tr & (rng.rand(len(full)) < 0.10)
res_a = fit(is_tr & ~val_a, val_a, "A")
metrics["A_original_testing"], dA, pA = evaluate(res_a["model"], (full.orig_split == "testing").to_numpy(), "A on original Testing")
metrics["A_testing_without_duplicates"], _, _ = evaluate(res_a["model"], ((full.orig_split == "testing") & ~full.leaked_dup).to_numpy(), "A on Testing minus duplicates")
metrics["A_testing_without_scan_neighbours"], _, _ = evaluate(res_a["model"], ((full.orig_split == "testing") & ~full.leaked_loose).to_numpy(), "A on Testing minus same-scan neighbours")

# --- Model B: leakage-free group split. This is the shipped model.
res_b = fit((full.split == "train").to_numpy(), (full.split == "val").to_numpy(), "B")
metrics["leakage_free_test"], dB, pB = evaluate(res_b["model"], (full.split == "test").to_numpy(), "B on leakage-free test")
metrics["val"], _, _ = evaluate(res_b["model"], (full.split == "val").to_numpy(), "B val")
metrics["cal"], _, _ = evaluate(res_b["model"], (full.split == "cal").to_numpy(), "B cal")

# --- The headline: how much does the original benchmark inflate accuracy?
acc = lambda y, p: mt.accuracy(y, p.argmax(1))  # noqa: E731
gA, gB = dA.group.to_numpy().astype(str), dB.group.to_numpy().astype(str)
vA = bs.bootstrap_values({"y": dA.y.to_numpy(), "p": pA}, acc, groups=gA, strata=dA.y.to_numpy(), B=CFG["bootstrap_B"])
vB = bs.bootstrap_values({"y": dB.y.to_numpy(), "p": pB}, acc, groups=gB, strata=dB.y.to_numpy(), B=CFG["bootstrap_B"], seed=7)
metrics["inflation"] = {
    "accuracy_original_minus_leakage_free": bs.diff_ci(vA[1], vB[1], vA[0], vB[0]),
    "n_testing_with_duplicate_in_training": int(full.leaked_dup.sum()), "n_testing_with_scan_neighbour_in_training": int(full.leaked_loose.sum()),
    "note": "A and B are different models on different test sets, so the difference mixes model and test-set effects; the within-model rows above isolate the leak",
}
metrics["headline"] = metrics["leakage_free_test"]
metrics["notes"] = {
    "ci": f"95% percentile cluster bootstrap, groups = pHash similarity clusters (no patient ids exist), class-stratified, B={CFG['bootstrap_B']}",
    "limitation": "brain_mri merges Br35H, SARTAJ and Figshare; patient grouping is approximated by image similarity",
    "contamination": "BDNeuro-MRI external set never used for training",
}

# %%
preds = {}
for split in ("val", "cal", "test"):
    d, X = subset((full.split == split).to_numpy())
    lg = tc.predict_logits(res_b["model"], X)
    preds[split] = dict(ids=d.image_id.to_numpy().astype(str), logits=lg.astype(np.float32), y=d.y.to_numpy(), group=d.group.to_numpy().astype(str))
d, X = subset((full.orig_split == "testing").to_numpy())
preds["A_original_testing"] = dict(ids=d.image_id.to_numpy().astype(str), logits=tc.predict_logits(res_a["model"], X).astype(np.float32), y=d.y.to_numpy(),
                                   leaked_dup=d.leaked_dup.to_numpy(), leaked_loose=d.leaked_loose.to_numpy())
from ml.data.common import SPLITS_DIR
split_hash = json.load(open(SPLITS_DIR / "split_hashes.json"))["brain_mri"]["split_hash"]
tc.write_run(OUT, model_id=f"brain_cls@{tc.git_commit()[:8]}", task="classification", arch=CFG["arch"], classes=CLASSES, preproc_spec=SPEC, state_dict=res_b["state_dict"], weights_name="weights.pt",
             split_name="brain_mri", split_hash=split_hash, training_data=["brain_mri (Kaggle masoudnickparvar), leakage-free split"],
             metrics={**metrics, "history_A": res_a["history"], "history_B": res_b["history"]}, predictions=preds, data_info=HEADER, wall_sec=time.time() - run_t0,
             extra={"comparison_model_A_state": "not saved (diagnostic only)"})
print("wrote", OUT, sorted(p.name for p in OUT.iterdir()))
