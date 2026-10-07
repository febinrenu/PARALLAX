"""P2.14 `make eval`: recompute every number from cached predictions and write reports/metrics.json.

    python ml/eval/run_all.py            # about a minute or two on a laptop CPU, no GPU, no raw datasets, no network
    python ml/eval/run_all.py --check    # fail if the result differs from the committed reports/metrics.json

Inputs are only committed files: ml/artifacts/<model>/predictions/*.npz, calibration and quality files, ml/data/splits/*.csv and
reports/{leakage,ood,concordance}.json. Everything is seeded (ml/eval/bootstrap.py SEED), so two runs give identical files.
reports/metrics.json is the single file the web app reads: headline rows for the validation page plus every section in full.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for p in (REPO, REPO / "backend"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

ART = REPO / "ml" / "artifacts"
REPORTS = REPO / "reports"
B = 1000  # resamples for the headline CIs here (the training notebooks used 2000; intervals agree to the third decimal)
B_LIGHT = 300  # validation and calibration splits, which only feed calibration, not the headline table


def _j(name: str):
    p = REPORTS / name
    return json.loads(p.read_text()) if p.is_file() else None


# ----------------------------------------------------------------------------- tasks (top-level so a process pool can run them)


def task_split(name: str, split: str, b: int) -> tuple:
    from ml.eval import metrics as mt
    from ml.train import common as tc

    meta = json.loads((ART / name / "model_meta.json").read_text())
    classes = meta["classes"]
    z = np.load(ART / name / "predictions" / f"{split}.npz", allow_pickle=False)
    rep = tc.bootstrap_report(z["y"], mt.softmax(z["logits"]), z["group"], len(classes), classes, B=b, positive="mel" if name == "skin_cls" else None)
    return ("split", name, split, meta["model_id"], classes, rep)


def task_rest() -> tuple:
    from ml.eval import bootstrap as bs
    from ml.eval import external
    from ml.train import bone as bn
    from ml.train import seg

    out: dict = {}
    nb = json.loads((ART / "brain_cls" / "metrics.json").read_text())
    infl = {k: nb[k] for k in ("A_original_testing", "A_testing_without_duplicates", "A_testing_without_scan_neighbours", "inflation") if k in nb}
    infl["note"] = "computed in the training notebook (bootstrap B=2000); model A is the diagnostic model trained on the original Kaggle split"
    out["brain_inflation"] = infl
    if (ART / "skin_cls" / "predictions" / "milk10k.npz").is_file():
        out["skin_external"] = external.evaluate_skin(B)
    if (ART / "brain_cls" / "predictions" / "bdneuro.npz").is_file():
        out["brain_external"] = external.evaluate(B)
    meta = json.loads((ART / "bone_det" / "model_meta.json").read_text())

    def split(s):
        z = np.load(ART / "bone_det" / "predictions" / f"{s}.npz", allow_pickle=False)
        dets = bn.unpack_dets(z["det_rows"], z["det_offsets"])
        gts = [z["gt_rows"][z["gt_offsets"][i] : z["gt_offsets"][i + 1]] for i in range(len(z["gt_offsets"]) - 1)]
        return {"dets": dets, "gts": gts, "groups": z["groups"]}

    out["bone_det"] = {"model_id": meta["model_id"], "test": bn.evaluate(split("val"), split("test"), B=B)}
    meta = json.loads((ART / "brain_seg" / "model_meta.json").read_text())
    z = np.load(ART / "brain_seg" / "predictions" / "test.npz", allow_pickle=False)
    shape = tuple(z["mask_shape"])
    masks = np.unpackbits(z["masks_packed"])[: int(np.prod(shape))].reshape(shape).astype(bool)
    ev = seg.evaluate_patients(z["probs_u8"].astype(np.float32) / 255.0, masks, z["patients"])
    d = np.array([r["dice"] for r in ev["patients"]])
    h = np.array([r["hd95_px"] for r in ev["patients"]])
    out["brain_seg"] = {"model_id": meta["model_id"], "n_test_groups": int(len(d)), "mean_patient_dice": bs.ci({"d": d}, lambda d: float(np.mean(d)), B=B, method="bca"),
                        "mean_patient_hd95_px": bs.ci({"d": h[np.isfinite(h)]}, lambda d: float(np.mean(d)), B=B, method="bca") if np.isfinite(h).sum() > 2 else None,
                        "slice_detection_auroc": ev["slice_auroc"], "pooled_dice": ev["pooled_dice"], "patients": ev["patients"],
                        "note": "one fold (fold 0) of a 10-fold patient-level split; groups are TCGA patients, some merged when tumour slices were near-duplicates"}
    return ("rest", out)


def task_calibration() -> tuple:
    from ml.eval import calibration as cal

    out = {"seed": 20261006, "models": {}}
    for name, splits in cal.MODELS.items():
        out["models"][name] = cal.calibrate_classifier(name, splits, B=B)
    out["models"]["bone_det"] = cal.calibrate_bone(B=B)
    return ("calibration", out)


def task_cxr(tag: str) -> tuple:
    from ml.eval import cxr

    r = cxr.evaluate(tag)
    if (ART / f"cxr_{tag}" / "predictions" / "localization.npz").is_file():
        r["localization"] = cxr.evaluate_localization(tag)
    return ("cxr", tag, r)


def task_subgroup(name: str) -> tuple:
    from ml.eval import subgroups as sg

    sg.B = 500  # per-subgroup resamples: many subgroups x metrics, so a lighter bootstrap than the headline numbers
    return ("subgroup", name, {"skin_cls": sg.skin, "skin_cls_milk10k": sg.skin_milk, "brain_cls": sg.brain, "bone_det": sg.bone, "cxr_chex": sg.cxr}[name]())


def task_corruption(name: str) -> tuple:
    from ml.eval import corruption as co

    return ("corruption", name, co.evaluate(name))


def task_signals() -> tuple:
    from ml.eval import signals as si

    return ("signals", si.evaluate())


# ----------------------------------------------------------------------------- assembly


def _ci(c: dict) -> list:
    return [round(c["lo"], 4), round(c["hi"], 4)]


def headline(s: dict) -> list[dict]:
    rows = []
    m = s["models"]
    t = m["skin_cls"]["splits"]["official_test"]
    rows.append({"model": "skin_cls", "task": "skin lesion classification (7 classes)", "eval_set": "ISIC 2018 Task 3 official test", "n": t["n"], "metric": "balanced multiclass accuracy", "value": round(t["balanced_accuracy"]["point"], 4), "ci95": _ci(t["balanced_accuracy"]), "external": False, "contaminated": False,
                 "also": {"melanoma sensitivity": round(t["per_class_recall"]["mel"]["point"], 4), "melanoma AUROC": round(t["auroc_mel_vs_rest"]["point"], 4)}})
    if "external_milk10k" in m["skin_cls"]:
        e = m["skin_cls"]["external_milk10k"]
        rows.append({"model": "skin_cls", "task": "skin lesion classification (7 classes)", "eval_set": "MILK10k dermoscopic images (external)", "n": e["n"], "metric": "balanced multiclass accuracy", "value": round(e["balanced_accuracy"]["point"], 4), "ci95": _ci(e["balanced_accuracy"]),
                     "external": True, "contaminated": False, "also": {"accuracy": round(e["accuracy"]["point"], 4), "melanoma sensitivity": round(e["per_class_recall"]["mel"]["point"], 4)}})
    t = m["brain_cls"]["splits"]["test"]
    rows.append({"model": "brain_cls", "task": "brain MRI classification (4 classes)", "eval_set": "leakage-free test split", "n": t["n"], "metric": "accuracy", "value": round(t["accuracy"]["point"], 4), "ci95": _ci(t["accuracy"]), "external": False, "contaminated": False,
                 "also": {"balanced accuracy": round(t["balanced_accuracy"]["point"], 4), "macro F1": round(t["macro_f1"]["point"], 4)}})
    if "external_bdneuro" in m["brain_cls"]:
        e = m["brain_cls"]["external_bdneuro"]
        rows.append({"model": "brain_cls", "task": "brain MRI classification (4 classes)", "eval_set": "BDNeuro-MRI, non-duplicate images (external)", "n": e["n"], "metric": "accuracy", "value": round(e["accuracy"]["point"], 4), "ci95": _ci(e["accuracy"]), "external": True, "contaminated": False,
                     "also": {"balanced accuracy": round(e["balanced_accuracy"]["point"], 4), "pituitary recall": round(e["per_class_recall"]["pituitary"]["point"], 4)}})
    b = m["brain_cls"].get("benchmark_inflation")
    if b:
        a = b["A_original_testing"]["accuracy"]
        rows.append({"model": "brain_cls (diagnostic model A)", "task": "benchmark inflation", "eval_set": "original Kaggle Testing folder", "n": b["A_original_testing"]["n"], "metric": "accuracy", "value": round(a["point"], 4), "ci95": _ci(a), "external": False, "contaminated": True,
                     "also": {"after removing images with a duplicate in Training": round(b["A_testing_without_duplicates"]["accuracy"]["point"], 4), "after also removing same-scan neighbours": round(b["A_testing_without_scan_neighbours"]["accuracy"]["point"], 4)}})
    g = m["brain_seg"]["mean_patient_dice"]
    rows.append({"model": "brain_seg", "task": "brain tumour segmentation", "eval_set": "LGG held-out patient groups", "n": m["brain_seg"]["n_test_groups"], "metric": "mean per-patient Dice", "value": round(g["point"], 4), "ci95": _ci(g), "external": False, "contaminated": False,
                 "also": {"slice detection AUROC": round(m["brain_seg"]["slice_detection_auroc"], 4)}})
    t = m["bone_det"]["test"]
    rows.append({"model": "bone_det", "task": "bone fracture detection", "eval_set": "FracAtlas test split", "n": t["n_images"], "metric": "image-level AUROC", "value": round(t["auroc_image"]["point"], 4), "ci95": _ci(t["auroc_image"]), "external": False, "contaminated": False,
                 "also": {"mAP50": round(t["map50"]["point"], 4), "sensitivity at validation-fixed 90% specificity": round(t["sensitivity_at_val_90spec"]["point"], 4)}})
    for tag, cm in s["cxr"]["models"].items():
        c = cm["labels"]["Lung Opacity"]["auroc"]
        rows.append({"model": f"chest reader, TorchXRayVision {tag}", "task": "chest X-ray, Lung Opacity vs rest", "eval_set": "RSNA test (patient-level)", "n": cm["n_test"], "metric": "AUROC", "value": round(c["point"], 4), "ci95": _ci(c), "external": not cm["trained_on_rsna"],
                     "contaminated": cm["trained_on_rsna"], "also": {"FNR at 10% target": round(cm["labels"]["Lung Opacity"]["fnr_control"]["fnr_test"], 4)}})
    return rows


def contamination_ledger() -> list[dict]:
    L = [
        ("skin_cls", "ISIC 2018 Task 3 official test", "clean", "never used for training, validation or calibration; 115 training copies of test images were removed by the audit"),
        ("skin_cls", "HAM10000 internal test split", "clean", "lesion-grouped split"),
        ("skin_cls", "MILK10k dermoscopic images", "clean (external)", "94 images that duplicate HAM10000 images and 576 images of classes the model does not cover are excluded"),
        ("brain_cls", "leakage-free test split", "clean", "similarity-grouped split; no duplicate crosses splits"),
        ("brain_cls", "original Kaggle Testing folder", "contaminated", "727 of 1,600 images have a near-duplicate in the Kaggle Training folder; shown only to measure inflation"),
        ("brain_cls", "BDNeuro-MRI non-duplicate images", "clean (external)", "71% of BDNeuro duplicates the Kaggle training source and is excluded; the remaining 1,644 images are independent"),
        ("brain_seg", "LGG held-out patient groups", "clean", "patient-level split; trained on TCGA-LGG only"),
        ("bone_det", "FracAtlas test split", "clean, patient grouping unavailable", "no patient identifiers exist, so another view of a test patient may sit in training"),
        ("xrv densenet121-chex", "RSNA test", "clean (external)", "trained on CheXpert only"),
        ("xrv densenet121-mimic_ch", "RSNA test", "clean (external)", "trained on MIMIC-CXR with CheXpert labels only"),
        ("xrv densenet121-all", "RSNA test", "contaminated", "trained on RSNA among six other datasets; reported for reference and labelled in the UI"),
        ("MedGemma 1.5 4B", "FracAtlas, HAM10000, RSNA", "unverified", "training data overlap with our test sets has not been checked (P3 to confirm from the model card)"),
        ("MedSAM", "ISIC 2018 Task 1", "unverified", "its training corpus included public skin-lesion data; exact sources not yet checked"),
    ]
    return [{"model": a, "eval_set": b, "status": c, "note": d} for a, b, c, d in L]


def run() -> dict:
    from ml.eval import corruption as co
    from ml.eval import cxr as cxr_mod
    from ml.eval import subgroups as sg

    workers = max(2, min(8, (os.cpu_count() or 4) - 2))
    got: dict = {"models": {}, "cxr": {}, "subgroup": {}, "corruption": {}}
    rest: dict = {}
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        # The calibration and chest tasks write the fitted calibration files that the signal and chest-subgroup tasks read, so those
        # two start only after the writers finish; everything else is independent and starts immediately.
        writers = [ex.submit(task_calibration)] + [ex.submit(task_cxr, t) for t in cxr_mod.WEIGHTS if (ART / f"cxr_{t}" / "predictions" / "test.npz").is_file()]
        p2 = [ex.submit(task_split, "skin_cls", s_, B if s_ in ("official_test", "test") else B_LIGHT) for s_ in ("official_test", "test", "val", "cal")]
        p2 += [ex.submit(task_split, "brain_cls", s_, B if s_ == "test" else B_LIGHT) for s_ in ("test", "val", "cal")]
        p2 += [ex.submit(task_rest)]
        p2 += [ex.submit(task_subgroup, n) for n in ("skin_cls", "skin_cls_milk10k", "brain_cls", "bone_det")]
        p2 += [ex.submit(task_corruption, m) for m in co.MODELS if (ART / m / "predictions" / "corruption.npz").is_file()]
        for f in writers:
            r = f.result()
            if r[0] == "calibration":
                got["calibration"] = r[1]
            else:
                got["cxr"][r[1]] = r[2]
        p2 += [ex.submit(task_signals), ex.submit(task_subgroup, "cxr_chex")]
        for f in p2:
            r = f.result()
            if r[0] == "split":
                _, name, split, mid, classes, rep = r
                blk = got["models"].setdefault(name, {"model_id": mid, "classes": classes, "splits": {}})
                blk["splits"][split] = rep
            elif r[0] == "rest":
                rest = r[1]
            elif r[0] == "signals":
                got["signals"] = r[1]
            else:
                got[r[0]][r[1]] = r[2]
    order = ["official_test", "test", "val", "cal"]
    for blk in got["models"].values():
        blk["splits"] = {k: blk["splits"][k] for k in order if k in blk["splits"]}
    got["models"]["brain_cls"]["benchmark_inflation"] = rest["brain_inflation"]
    if "brain_external" in rest:
        got["models"]["brain_cls"]["external_bdneuro"] = rest["brain_external"]
    if "skin_external" in rest:
        got["models"]["skin_cls"]["external_milk10k"] = rest["skin_external"]
    got["models"]["bone_det"] = rest["bone_det"]
    got["models"]["brain_seg"] = rest["brain_seg"]
    s = {
        "models": {k: got["models"][k] for k in ("skin_cls", "brain_cls", "brain_seg", "bone_det")}, "calibration": got["calibration"],
        "cxr": {"dataset": "RSNA Pneumonia Detection Challenge (stage 2 train images), patient-level split", "target": "RSNA class Lung_Opacity vs the other two classes", "models": {t: got["cxr"][t] for t in cxr_mod.WEIGHTS if t in got["cxr"]}},
        "subgroups": {"min_n": sg.MIN_N, "min_positives": sg.MIN_POS, "bootstrap_B": 500, "datasets": {n: got["subgroup"][n] for n in ("skin_cls", "skin_cls_milk10k", "brain_cls", "bone_det", "cxr_chex")}},
        "corruption": {"seed": co.SEED, "models": {m: got["corruption"][m] for m in co.MODELS if m in got["corruption"]}}, "signals": got["signals"],
    }
    reg = json.loads((ART / "registry.json").read_text())
    return {
        "generated_by": "ml/eval/run_all.py", "seed": 20261006, "bootstrap_resamples": B,
        "ci_method": "95% percentile cluster bootstrap (BCa for per-patient Dice); resampling unit = lesion / patient / similarity cluster, stratified by class where a metric needs every class present",
        "disclaimer": "Decision support only. Not a medical device. All numbers are on held-out data and carry confidence intervals.",
        "headline": headline(s), "contamination_ledger": contamination_ledger(),
        "models": s["models"], "calibration": s["calibration"], "chest_reader": s["cxr"], "subgroups": s["subgroups"], "corruption": s["corruption"], "trust_signals": s["signals"],
        "leakage": _j("leakage.json"), "ood": _j("ood.json"), "second_reader": _j("concordance.json"),
        "registry": {k: {x: v[x] for x in ("task", "arch", "classes", "weights_sha256", "split_hash", "git_commit", "license", "contamination", "headline")} for k, v in reg["models"].items()},
    }


def main(argv=None) -> int:
    a = argv or sys.argv[1:]
    t0 = time.time()
    res = run()
    text = json.dumps(res, indent=1) + "\n"
    target = REPORTS / "metrics.json"
    if "--check" in a:
        old = target.read_text(encoding="utf-8") if target.is_file() else ""
        if old != text:
            print("reports/metrics.json differs from a fresh recomputation", file=sys.stderr)
            return 1
        print(f"reports/metrics.json reproduced exactly in {time.time() - t0:.0f}s")
        return 0
    target.write_text(text, encoding="utf-8")
    for k, fname in (("calibration", "calibration"), ("chest_reader", "cxr"), ("subgroups", "subgroups"), ("corruption", "corruption"), ("trust_signals", "signals")):
        (REPORTS / f"{fname}.json").write_text(json.dumps(res[k], indent=1) + "\n", encoding="utf-8")
    print(f"wrote {target} ({len(text) / 1e6:.2f} MB) in {time.time() - t0:.0f}s")
    for r in res["headline"]:
        print(f"  {r['model']:34s} {r['metric']:30s} {r['value']:.3f} {r['ci95']}  {'(contaminated)' if r['contaminated'] else '(external)' if r['external'] else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
