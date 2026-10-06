"""P2.7: chest X-ray reader validation on RSNA with a contamination-honest table.

    python ml/eval/cxr.py score            # run the three TorchXRayVision weight sets once, cache scores
    python ml/eval/cxr.py evaluate         # recompute every metric from the cached scores (what `make eval` calls)
    python ml/eval/cxr.py localize         # Grad-CAM pointing game and box IoU vs RSNA boxes (needs the weights and a GPU)

No model is trained here. Weight sets: `chex` and `mimic_ch` never saw RSNA (external validation); `all` was trained on
RSNA (and NIH, from which RSNA images derive), so its numbers are labelled contaminated and shown for reference only.
Target = RSNA "Lung Opacity" class (the pneumonia-challenge positive). The readers' outputs are operating-point-scaled
probabilities, not logits, so calibration is Platt scaling on logit(p) fitted on half of the RSNA calibration split and the
FNR threshold on the other half; nothing is fitted on the test split.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for p in (REPO, REPO / "backend"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from medproof.calibrate import abstain, conformal, selective, temperature  # noqa: E402
from medproof.calibrate.binary import BinaryCalibrator  # noqa: E402

from ml.data.common import CACHE_DIR, data_root  # noqa: E402
from ml.eval import bootstrap as bs  # noqa: E402
from ml.eval import metrics as mt  # noqa: E402
from ml.eval.cxr_cache import NPY, kept_rows  # noqa: E402

ART = REPO / "ml" / "artifacts"
WEIGHTS = {  # tag -> (torchxrayvision weights, trained on RSNA?, note)
    "chex": ("densenet121-res224-chex", False, "CheXpert; never saw RSNA"),
    "mimic_ch": ("densenet121-res224-mimic_ch", False, "MIMIC-CXR with CheXpert labels; never saw RSNA"),
    "all": ("densenet121-res224-all", True, "NIH, PadChest, CheXpert, MIMIC-CXR, Google, OpenI and RSNA: CONTAMINATED on RSNA, reference only"),
}
LABELS = ["Lung Opacity", "Pneumonia"]
B = 1000


def _split_arrays(df: pd.DataFrame, split: str) -> np.ndarray:
    return np.nonzero((df.split == split).to_numpy())[0]


# ----------------------------------------------------------------------------- scoring


def score(tag: str, batch: int = 128) -> Path:
    import torch
    import torchxrayvision as xrv

    weights, contaminated, note = WEIGHTS[tag]
    df = kept_rows()
    X = np.load(NPY, mmap_mode="r")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = xrv.models.DenseNet(weights=weights).to(dev).eval()
    out = np.zeros((len(df), len(model.pathologies)), np.float32)
    with torch.no_grad():
        for i in range(0, len(df), batch):
            out[i : i + batch] = model(torch.from_numpy(np.asarray(X[i : i + batch], dtype=np.float32)).to(dev)).cpu().numpy()
    mdir = ART / f"cxr_{tag}"
    (mdir / "predictions").mkdir(parents=True, exist_ok=True)
    for split in ("cal", "test"):
        idx = _split_arrays(df, split)
        np.savez_compressed(mdir / "predictions" / f"{split}.npz", ids=df.image_id.to_numpy()[idx].astype(str), probs=out[idx], labels=np.array(model.pathologies), cls=df.label.to_numpy()[idx].astype(str),
                            y=(df.label.to_numpy()[idx] == "Lung_Opacity").astype(int), group=df.group.to_numpy()[idx].astype(str))
    (mdir / "model_meta.json").write_text(json.dumps({"model_id": f"xrv-{weights.replace('-res224', '')}", "task": "classification (multi-label, pretrained)", "weights": weights, "trained_on_rsna": contaminated, "note": note,
                                                      "labels": list(model.pathologies), "output_kind": "probability (operating-point scaled), not logits", "preproc_spec": "cxr_xrv", "torchxrayvision": xrv.__version__,
                                                      "training": "none in this project (pretrained, Apache-2.0)"}, indent=1), encoding="utf-8")
    return mdir


# ----------------------------------------------------------------------------- metrics


def _load(tag: str, split: str) -> dict:
    z = np.load(ART / f"cxr_{tag}" / "predictions" / f"{split}.npz", allow_pickle=False)
    return {k: z[k] for k in z.files}


def _col(d: dict, label: str) -> np.ndarray:
    return d["probs"][:, list(d["labels"]).index(label)]


def _ece_binary(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    return mt.ece(y, np.stack([1 - p, p], 1), bins)


def _reliability(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict]:
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    return [{"lo": float(edges[b]), "hi": float(edges[b + 1]), "n": int((which == b).sum()), "mean_predicted": float(p[which == b].mean()) if (which == b).any() else None,
             "observed": float(y[which == b].mean()) if (which == b).any() else None} for b in range(bins)]


def evaluate(tag: str) -> dict:
    meta = json.loads((ART / f"cxr_{tag}" / "model_meta.json").read_text())
    cal, test = _load(tag, "cal"), _load(tag, "test")
    y_c, y_t, g_t = cal["y"], test["y"], test["group"]
    half = np.arange(len(y_c)) % 2 == 0  # alternate rows: a deterministic split of the calibration set into A (Platt) and B (FNR threshold)
    out: dict = {"model_id": meta["model_id"], "trained_on_rsna": meta["trained_on_rsna"], "contamination": meta["note"], "n_cal": int(len(y_c)), "n_test": int(len(y_t)), "prevalence_test": float(y_t.mean()), "labels": {}}
    kw = dict(groups=g_t, strata=y_t, B=B)
    for label in LABELS + ["max(Lung Opacity, Pneumonia)"]:
        if label.startswith("max"):
            s_c = np.maximum(_col(cal, LABELS[0]), _col(cal, LABELS[1]))
            s_t = np.maximum(_col(test, LABELS[0]), _col(test, LABELS[1]))
        else:
            s_c, s_t = _col(cal, label), _col(test, label)
        blk: dict = {}
        blk["auroc"] = bs.ci({"y": y_t, "s": s_t}, lambda y, s: mt.auroc_binary(y, s), **kw)
        from sklearn.metrics import average_precision_score

        blk["auprc"] = bs.ci({"y": y_t, "s": s_t}, lambda y, s: float(average_precision_score(y, s)), **kw)
        bc = BinaryCalibrator.fit(label, s_c[half], y_c[half], s_c[~half], y_c[~half], alpha=0.1)
        p_t = bc.prob(s_t)
        z_c = temperature.logit(s_c[half])
        T = temperature.fit_temperature_binary(z_c, y_c[half])
        p_temp = temperature.sigmoid(temperature.logit(s_t) / T)
        blk["calibration"] = {"method": "Platt on logit(score)", "a": bc.a, "b": bc.b, "temperature_only_T": T,
                              "ece_raw": bs.ci({"y": y_t, "p": s_t}, lambda y, p: _ece_binary(y, p), **kw), "ece_temperature_only": bs.ci({"y": y_t, "p": p_temp}, lambda y, p: _ece_binary(y, p), **kw),
                              "ece_platt": bs.ci({"y": y_t, "p": p_t}, lambda y, p: _ece_binary(y, p), **kw), "brier_raw": float(np.mean((s_t - y_t) ** 2)), "brier_platt": float(np.mean((p_t - y_t) ** 2)),
                              "reliability_raw": _reliability(y_t, s_t), "reliability_platt": _reliability(y_t, p_t)}
        pos = y_t == 1
        miss = ~(p_t >= bc.thr_fnr)
        k, n = int(miss[pos].sum()), int(pos.sum())
        blk["fnr_control"] = {"target_fnr": 0.1, "threshold_calibrated_prob": bc.thr_fnr, "fnr_test": float(k / n), "fnr_ci95_clopper_pearson": list(conformal.clopper_pearson(k, n)),
                              "within_3_points_of_target": bool(abs(k / n - 0.1) <= 0.03), "fpr_test": float(np.mean(~miss[~pos])), "ppv_test": float(np.mean(pos[~miss])) if (~miss).any() else None,
                              "fraction_reported": float(np.mean(~miss)), "n_positives_in_calibration_B": int(bc.meta["positives_b"])}
        conf = np.maximum(p_t, 1 - p_t)
        correct = ((p_t >= 0.5) == pos).astype(float)
        inband = (p_t >= bc.abstain_low) & (p_t <= bc.abstain_high)
        blk["abstention"] = {"band": [bc.abstain_low, bc.abstain_high], "fraction_abstained": float(inband.mean()), "accuracy_outside_band": float(correct[~inband].mean()), "accuracy_inside_band": float(correct[inband].mean()) if inband.any() else None,
                             "accuracy_overall": float(correct.mean())}
        blk["selective"] = {"aurc": bs.ci({"c": conf, "k": correct}, lambda c, k: selective.aurc(c, k), groups=g_t, B=B), "oracle_aurc": selective.oracle_aurc(correct),
                            "coverage_at_5pct_error": selective.coverage_at_risk(conf, correct, 0.05), "coverage_at_10pct_error": selective.coverage_at_risk(conf, correct, 0.10)}
        out["labels"][label] = blk
        bc.save(ART / f"cxr_{tag}" / f"calibration_{label.split(' ')[0].lower().replace('(', '').replace(',', '')}.json")
    # three-class view: which RSNA classes does the Lung Opacity score separate from the rest?
    cls_t = test["cls"]
    s = _col(test, "Lung Opacity")
    out["auroc_lung_opacity_vs_each_class"] = {c: float(mt.auroc_binary((cls_t == "Lung_Opacity")[(cls_t == "Lung_Opacity") | (cls_t == c)], s[(cls_t == "Lung_Opacity") | (cls_t == c)])) for c in sorted(set(cls_t)) if c != "Lung_Opacity"}
    return out


# ----------------------------------------------------------------------------- localisation


def _gt_boxes() -> dict[str, np.ndarray]:
    lab = next(data_root().joinpath("rsna_pneumonia").rglob("stage_2_train_labels.csv"))
    d = pd.read_csv(lab)
    d = d[d.Target == 1]
    return {pid: g[["x", "y", "width", "height"]].to_numpy(float) for pid, g in d.groupby("patientId")}


def _iou_xywh(box_xyxy, gts_xywh) -> float:
    x1, y1, x2, y2 = box_xyxy
    best = 0.0
    for gx, gy, gw, gh in gts_xywh:
        ix = max(0.0, min(x2, gx + gw) - max(x1, gx))
        iy = max(0.0, min(y2, gy + gh) - max(y1, gy))
        inter = ix * iy
        union = (x2 - x1) * (y2 - y1) + gw * gh - inter
        best = max(best, inter / union if union > 0 else 0.0)
    return best


def localize(tag: str, labels=("Lung Opacity", "Pneumonia"), cap: int = 1000, seed: int = 20261006) -> Path:
    """Grad-CAM++ peak and region vs RSNA ground-truth boxes on a seeded sample of test positives."""
    import torch
    import torchxrayvision as xrv

    from medproof.readers.cam import CamExtractor, resize_cam
    from medproof.readers.cxr import largest_region

    weights, _, _ = WEIGHTS[tag]
    df = kept_rows()
    X = np.load(NPY, mmap_mode="r")
    gts = _gt_boxes()
    idx = [i for i in _split_arrays(df, "test") if df.image_id.iloc[i] in gts]
    rng = np.random.RandomState(seed)
    idx = sorted(rng.choice(idx, size=min(cap, len(idx)), replace=False).tolist())
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = xrv.models.DenseNet(weights=weights).to(dev).eval()
    names = list(model.pathologies)
    res = {lab: {"ids": [], "peak_in_box": [], "best_iou": [], "box_area_frac": [], "center_in_box": []} for lab in labels}
    with CamExtractor(model, model.features) as cx:
        for n, i in enumerate(idx):
            pid = df.image_id.iloc[i]
            x = torch.from_numpy(np.asarray(X[i : i + 1], dtype=np.float32)).to(dev)
            cams = cx.maps(x, [names.index(lab) for lab in labels], "gradcam++")
            g = gts[pid]
            union_frac = min(1.0, float(sum(b[2] * b[3] for b in g)) / (1024.0 * 1024.0))
            for lab, cam_small in zip(labels, cams):
                cam = resize_cam(cam_small, (1024, 1024))
                py, px = np.unravel_index(int(cam.argmax()), cam.shape)
                r = res[lab]
                r["ids"].append(pid)
                r["peak_in_box"].append(bool(any(b[0] <= px < b[0] + b[2] and b[1] <= py < b[1] + b[3] for b in g)) if cam.max() > 0 else False)
                reg = largest_region(cam, 0.5)
                r["best_iou"].append(_iou_xywh(reg[1], g) if reg else 0.0)
                r["box_area_frac"].append(union_frac)
                r["center_in_box"].append(bool(any(b[0] <= 512 < b[0] + b[2] and b[1] <= 512 < b[1] + b[3] for b in g)))
            if n % 200 == 0:
                print(f"  {tag}: {n}/{len(idx)}", flush=True)
    mdir = ART / f"cxr_{tag}"
    np.savez_compressed(mdir / "predictions" / "localization.npz", **{f"{lab.replace(' ', '_')}__{k}": np.array(v) for lab, r in res.items() for k, v in r.items()})
    return mdir


def evaluate_localization(tag: str) -> dict:
    z = np.load(ART / f"cxr_{tag}" / "predictions" / "localization.npz", allow_pickle=False)
    out = {}
    for lab in LABELS:
        key = lab.replace(" ", "_")
        if f"{key}__ids" not in z.files:
            continue
        ids = z[f"{key}__ids"]
        pk, iou, frac, ctr = (z[f"{key}__{k}"] for k in ("peak_in_box", "best_iou", "box_area_frac", "center_in_box"))
        arrs = {"pk": pk.astype(float), "iou": iou, "frac": frac, "ctr": ctr.astype(float)}
        out[lab] = {
            "n_positives_with_boxes": int(len(ids)),
            "pointing_game": bs.ci({"pk": arrs["pk"]}, lambda pk: float(np.mean(pk)), groups=ids, B=B),
            "pointing_game_chance_if_random_point": float(frac.mean()), "pointing_game_image_centre_baseline": float(ctr.mean()),
            "mean_best_iou": bs.ci({"i": iou}, lambda i: float(np.mean(i)), groups=ids, B=B),
            "iou_at_least_0.1": float(np.mean(iou >= 0.1)), "iou_at_least_0.3": float(np.mean(iou >= 0.3)), "iou_at_least_0.5": float(np.mean(iou >= 0.5)),
            "method": "Grad-CAM++ on DenseNet features, region = connected area >= 50% of the peak, original 1024 px grid",
        }
    return out


def main(argv=None) -> int:
    cmd = (argv or sys.argv[1:] or ["evaluate"])[0]
    tags = list(WEIGHTS)
    if cmd == "score":
        for t in tags:
            print("scoring", t, score(t))
    elif cmd == "localize":
        for t in tags:
            print("localizing", t, localize(t))
    elif cmd == "evaluate":
        res = {"dataset": "RSNA Pneumonia Detection Challenge (stage 2 train images), patient-level split", "target": "RSNA class Lung_Opacity vs the other two classes", "models": {}}
        for t in tags:
            if not (ART / f"cxr_{t}" / "predictions" / "test.npz").is_file():
                continue
            res["models"][t] = evaluate(t)
            if (ART / f"cxr_{t}" / "predictions" / "localization.npz").is_file():
                res["models"][t]["localization"] = evaluate_localization(t)
            b = res["models"][t]["labels"]["Lung Opacity"]
            print(f"{t:9s} contaminated={res['models'][t]['trained_on_rsna']!s:5s} AUROC(Lung Opacity) {b['auroc']['point']:.3f} [{b['auroc']['lo']:.3f}, {b['auroc']['hi']:.3f}]  FNR@ctrl {b['fnr_control']['fnr_test']:.3f}  ECE {b['calibration']['ece_raw']['point']:.3f}->{b['calibration']['ece_platt']['point']:.3f}")
        (REPO / "reports" / "cxr.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
