# %% [markdown]
# # P2.6 Bone fracture detector (Ultralytics YOLO nano, FracAtlas)
# Single class ("fracture"), 640 px, negatives kept with empty labels. Group-aware, body-part-stratified split from `ml/data/splits/fracatlas.csv`.
# Metrics (all from saved detections, with 95% cluster-bootstrap CIs): AP at IoU 0.5, image-level AUROC (score = max box confidence),
# sensitivity at 90% specificity with the threshold **fixed on the validation split**.
# Licence note: Ultralytics is AGPL-3.0, so the resulting weights are AGPL-3.0 too. FracAtlas is CC BY 4.0.
# Decision support only. Not a medical device.

# %%
import json, os, shutil, subprocess, sys, time
from pathlib import Path

REPO_URL = os.environ.get("PARALLAX_REPO_URL", "https://github.com/febinrenu/PARALLAX.git")
BRANCH = os.environ.get("PARALLAX_BRANCH", "main")
SMOKE = os.environ.get("SMOKE") == "1"
SLUG = "bone_det"
ON_KAGGLE = Path("/kaggle/working").is_dir()
if ON_KAGGLE:
    SCRATCH = Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else Path("/tmp")  # anything under /kaggle/working is saved as notebook output
    REPO = SCRATCH / "PARALLAX"
    if not REPO.exists():
        subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, str(REPO)], check=True)
    ROOT, OUT, WORK = Path(os.environ.get("PARALLAX_DATA_ROOT", SCRATCH / "raw")), Path(f"/kaggle/working/{SLUG}"), SCRATCH / "yolo_work"
else:
    REPO = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "plan.md").exists())
    ROOT = Path(os.environ.get("PARALLAX_DATA_ROOT", REPO / "ml" / "data" / "raw"))
    OUT = Path(os.environ.get("PARALLAX_OUT", REPO / "ml" / "artifacts" / "_local" / SLUG))
    WORK = Path(os.environ.get("PARALLAX_WORK", OUT.parent / f"{SLUG}_work"))
os.environ["PARALLAX_DATA_ROOT"] = str(ROOT)
sys.path[:0] = [str(REPO), str(REPO / "backend")]
if os.environ.get("PARALLAX_SKIP_PIP") != "1":
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "ultralytics", "pydicom"], check=False)
if ON_KAGGLE or not (ROOT / "fracatlas" / "manifest.json").exists():
    subprocess.run([sys.executable, str(REPO / "ml/data/download.py"), "fracatlas", "--root", str(ROOT)], check=True)
OUT.mkdir(parents=True, exist_ok=True)
WORK.mkdir(parents=True, exist_ok=True)
from ml.train import common as tc

HEADER = tc.print_header(ROOT, ["fracatlas"])

# %%
CFG = dict(weights="yolo26n.pt", fallback_weights="yolo11n.pt", imgsz=320 if SMOKE else 640, epochs=int(os.environ.get("EPOCHS", 1 if SMOKE else 80)), patience=20, batch=16, bootstrap_B=100 if SMOKE else 2000, smoke_per_split=30,
           train_args=dict(mosaic=0.0, close_mosaic=0, fliplr=0.5, flipud=0.0, degrees=5.0, translate=0.1, scale=0.3, hsv_h=0.0, hsv_s=0.0, hsv_v=0.3, workers=2, amp=True, plots=False, cache=False))
print(json.dumps(CFG, indent=1))

# %%
import numpy as np, pandas as pd
from ml.data.common import load_split, image_path, SPLITS_DIR
from ml.train import bone

df = load_split("fracatlas")
df = df[df.split != "dropped"].reset_index(drop=True)
if SMOKE:
    df = pd.concat([g.head(CFG["smoke_per_split"]) for _, g in df.groupby("split")]).reset_index(drop=True)
print(df.groupby("split").label.value_counts().unstack(fill_value=0))
yaml = bone.build_yolo_dataset(df, ROOT, WORK / "dataset")

# %%
from ultralytics import YOLO

run_t0 = time.time()
try:
    model, used = YOLO(CFG["weights"]), CFG["weights"]
except Exception as e:  # older ultralytics without YOLO26
    print("could not load", CFG["weights"], "->", e, "; falling back to", CFG["fallback_weights"])
    model, used = YOLO(CFG["fallback_weights"]), CFG["fallback_weights"]
model.train(data=str(yaml), epochs=CFG["epochs"], patience=CFG["patience"], batch=CFG["batch"], imgsz=CFG["imgsz"], seed=tc.SEED, deterministic=True,
            project=str(WORK / "runs"), name="fracatlas", exist_ok=True, **CFG["train_args"])
best = WORK / "runs" / "fracatlas" / "weights" / "best.pt"
print("best weights", best, best.exists())

# %%
best_model = YOLO(str(best))
sets = {}
for split in ("val", "cal", "test"):
    d = df[df.split == split].reset_index(drop=True)
    paths = [WORK / "dataset" / "images" / split / f"{i}.jpg" for i in d.image_id]
    dets = bone.predict_split(best_model, paths, imgsz=CFG["imgsz"], batch=CFG["batch"])
    gts = [bone.read_yolo_boxes(WORK / "dataset" / "labels" / split / f"{i}.txt") for i in d.image_id]
    sets[split] = dict(df=d, dets=dets, gts=gts, groups=d.group.to_numpy().astype(str))
    print(split, len(d), "images,", sum(len(g) for g in gts), "GT boxes,", sum(len(x) for x in dets), "detections")

metrics = {"config": {k: v for k, v in CFG.items()}, "weights_start": used}
metrics["test"] = bone.evaluate(sets["val"], sets["test"], B=CFG["bootstrap_B"])
metrics["headline"] = metrics["test"]
for k in ("map50", "auroc_image", "sensitivity_at_val_90spec", "specificity_at_val_threshold"):
    c = metrics["test"][k]
    print(k, round(c["point"], 4), [round(c["lo"], 4), round(c["hi"], 4)])
try:  # cross-check against Ultralytics' own AP50 (different matching details, expect a close value, not an identical one)
    v = best_model.val(data=str(yaml), split="test", conf=0.001, iou=0.6, imgsz=CFG["imgsz"], plots=False, verbose=False)
    metrics["ultralytics_map50_crosscheck"] = float(v.box.map50)
    print("ultralytics map50", metrics["ultralytics_map50_crosscheck"])
except Exception as e:
    metrics["ultralytics_map50_crosscheck"] = f"unavailable: {e}"
metrics["notes"] = {
    "ci": f"95% percentile cluster bootstrap, groups = duplicate clusters (no patient ids exist), B={CFG['bootstrap_B']}; AP50 resamples whole images with their detections",
    "operating_point": "threshold for sensitivity at 90% specificity is fixed on the validation split and applied unchanged to test",
    "limitation": "no patient identifiers: different views of one patient may sit in different splits; 59 negative images were truncated by a few bytes at the source and are flagged `truncated_file`",
    "contamination": "trained and tested on FracAtlas only; no external test set",
    "license": "AGPL-3.0 (Ultralytics)",
}

# %%
preds = {}
for split, s in sets.items():
    gt_off = np.cumsum([0] + [len(g) for g in s["gts"]])
    preds[split] = dict(ids=s["df"].image_id.to_numpy().astype(str), groups=s["groups"], body_part=s["df"].body_part.to_numpy().astype(str), **bone.pack_dets(s["dets"]),
                        gt_rows=np.concatenate(s["gts"]) if len(s["gts"]) else np.zeros((0, 4)), gt_offsets=gt_off)
shutil.copyfile(best, OUT / "weights.pt")
split_hash = json.load(open(SPLITS_DIR / "split_hashes.json"))["fracatlas"]["split_hash"]
tc.write_run(OUT, model_id=f"bone_det@{tc.git_commit()[:8]}", task="detection", arch=used.replace(".pt", ""), classes=["fracture"], preproc_spec="ultralytics_letterbox_640", state_dict=None, weights_name="weights.pt",
             split_name="fracatlas", split_hash=split_hash, training_data=["fracatlas (figshare, CC BY 4.0)"], metrics=metrics, predictions=preds, data_info=HEADER, wall_sec=time.time() - run_t0,
             extra={"imgsz": CFG["imgsz"], "inference": "load weights.pt with ultralytics.YOLO and call predict(imgsz=640, conf=0.001); the library applies its own letterbox"})
meta = json.load(open(OUT / "model_meta.json"))
meta["weights_file"], meta["weights_sha256"] = "weights.pt", tc.sha256_file(OUT / "weights.pt")
json.dump(meta, open(OUT / "model_meta.json", "w"), indent=1)
print("wrote", OUT, sorted(p.name for p in OUT.iterdir()))
