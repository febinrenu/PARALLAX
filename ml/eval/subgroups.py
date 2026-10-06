"""P2.11 subgroup audit (D16): metrics by age band, sex, body site, body part and view, each with a 95% CI; the worst subgroup is named.

    python ml/eval/subgroups.py

Only held-out test predictions are used. A subgroup is reported when it has at least MIN_N images (and, for sensitivity-type
metrics, at least MIN_POS positives); smaller ones appear as "too few" rather than as a noisy number. The worst subgroup per
metric is the lowest point estimate among the reported ones; `ci_entirely_worse_than_overall` says whether the evidence is firm. Rows with missing metadata ("unknown") are reported but never named as the worst subgroup.
Datasets without demographic metadata (brain MRI Kaggle set, LGG) have none to audit and say so.
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

from medproof.calibrate.binary import BinaryCalibrator  # noqa: E402

from ml.eval import bootstrap as bs  # noqa: E402
from ml.eval import detection as det  # noqa: E402
from ml.eval import metrics as mt  # noqa: E402

ART = REPO / "ml" / "artifacts"
MIN_N, MIN_POS, B = 30, 15, 1000
HIGHER_IS_WORSE = {"false_negative_rate_at_control_threshold"}  # every other metric is "higher is better"
MISSING = {"unknown"}  # missing metadata is reported but not named as a demographic subgroup


def age_band(a: float) -> str:
    if a is None or not np.isfinite(a) or a < 0:
        return "unknown"
    return "<30" if a < 30 else "30-49" if a < 50 else "50-69" if a < 70 else "70+"


def _block(df: pd.DataFrame, col: str, metrics: dict, overall: dict) -> dict:
    """metrics: name -> (fn(sub_df) -> float, min_positives_required or 0, positive_col or None)."""
    rows = {}
    for val, sub in df.groupby(col, dropna=False):
        val = str(val)
        r: dict = {"n": int(len(sub))}
        if len(sub) < MIN_N:
            r["status"] = f"too few images (n < {MIN_N})"
            rows[val] = r
            continue
        for name, (fn, need_pos, pos_col) in metrics.items():
            if need_pos and int(sub[pos_col].sum()) < need_pos:
                r[name] = {"status": f"too few positives (< {need_pos})", "n_positives": int(sub[pos_col].sum())}
                continue
            try:
                c = bs.ci({"i": np.arange(len(sub))}, lambda i, sub=sub, fn=fn: fn(sub.iloc[i]), groups=sub["group"].to_numpy(), B=B)
            except ValueError:
                r[name] = {"status": "fewer than 2 groups"}
                continue
            c["ci_entirely_worse_than_overall"] = bool(c["lo"] > overall[name]) if name in HIGHER_IS_WORSE else bool(c["hi"] < overall[name])
            r[name] = c
        rows[val] = r
    worst = {}
    for name in metrics:
        sign = -1.0 if name in HIGHER_IS_WORSE else 1.0
        elig = [(sign * v["point"], k) for k, r in rows.items() if k not in MISSING and isinstance((v := r.get(name)), dict) and "point" in v and np.isfinite(v["point"])]
        if elig:
            _, k = min(elig)
            worst[name] = {"subgroup": k, "value": rows[k][name]["point"], "overall": overall[name], "ci_entirely_worse_than_overall": rows[k][name]["ci_entirely_worse_than_overall"], "direction": "higher is worse" if name in HIGHER_IS_WORSE else "lower is worse"}
    return {"subgroups": rows, "worst": worst}


def _audit(df: pd.DataFrame, cols: list[str], metrics: dict) -> dict:
    overall = {}
    for name, (fn, _, _) in metrics.items():
        try:
            overall[name] = float(fn(df))
        except Exception:
            overall[name] = float("nan")
    out = {"n": int(len(df)), "overall": overall, "by": {}}
    for c in cols:
        out["by"][c] = _block(df, c, metrics, overall)
    return out


# ----------------------------------------------------------------------------- skin


def skin() -> dict:
    meta = json.loads((ART / "skin_cls" / "model_meta.json").read_text())
    classes = meta["classes"]
    z = np.load(ART / "skin_cls" / "predictions" / "official_test.npz", allow_pickle=False)
    logits, y = z["logits"], z["y"]
    mel = classes.index("mel")
    df = pd.DataFrame({"y": y, "pred": logits.argmax(1), "group": z["group"].astype(str), "age_band": [age_band(a) for a in z["age"]], "sex": z["sex"], "site": z["site"]})
    df["correct"] = (df.y == df.pred).astype(int)
    df["is_mel"] = (df.y == mel).astype(int)
    metrics = {
        "accuracy": (lambda s: float(s.correct.mean()), 0, None),
        "balanced_accuracy": (lambda s: mt.balanced_accuracy(s.y.to_numpy(), s.pred.to_numpy(), len(classes)), 0, None),
        "melanoma_sensitivity": (lambda s: float((s.pred[s.is_mel == 1] == mel).mean()), MIN_POS, "is_mel"),
    }
    out = _audit(df, ["age_band", "sex", "site"], metrics)
    out.update({"dataset": "ISIC 2018 Task 3 official test", "metadata_source": "ISIC API (age is rounded to 5 years; ~19% of age, sex and site values are missing and shown as 'unknown')", "model": meta["model_id"]})
    return out


# ----------------------------------------------------------------------------- brain


def brain() -> dict:
    meta = json.loads((ART / "brain_cls" / "model_meta.json").read_text())
    z = np.load(ART / "brain_cls" / "predictions" / "test.npz", allow_pickle=False)
    y, pred = z["y"], z["logits"].argmax(1)
    classes = meta["classes"]
    df = pd.DataFrame({"y": y, "pred": pred, "group": z["group"].astype(str), "class": [classes[i] for i in y]})
    df["correct"] = (df.y == df.pred).astype(int)
    metrics = {"recall": (lambda s: float(s.correct.mean()), 0, None)}
    out = _audit(df, ["class"], metrics)
    out.update({"dataset": "Brain Tumor MRI Dataset, leakage-free test split", "model": meta["model_id"], "note": "the dataset carries no age, sex or scanner metadata, so only the per-class audit is possible; recall per class is the subgroup metric"})
    return out


# ----------------------------------------------------------------------------- bone


def bone() -> dict:
    from ml.train import bone as bn

    meta = json.loads((ART / "bone_det" / "model_meta.json").read_text())
    zt, zv = (np.load(ART / "bone_det" / "predictions" / f"{s}.npz", allow_pickle=False) for s in ("test", "val"))
    def split_arrays(z):
        dets = bn.unpack_dets(z["det_rows"], z["det_offsets"])
        gts = [z["gt_rows"][z["gt_offsets"][i] : z["gt_offsets"][i + 1]] for i in range(len(z["gt_offsets"]) - 1)]
        return det.image_scores(dets), np.array([len(g) > 0 for g in gts]).astype(int)

    sv, yv = split_arrays(zv)
    st, yt = split_arrays(zt)
    thr = mt.threshold_at_specificity(yv, sv, 0.9)
    from ml.data.common import load_split

    sp = load_split("fracatlas").set_index("image_id")
    ids = zt["ids"].astype(str)
    df = pd.DataFrame({"y": yt, "score": st, "group": zt["groups"].astype(str), "body_part": sp.loc[ids, "body_part"].to_numpy(), "view": sp.loc[ids, "view"].to_numpy(), "hardware": np.where(sp.loc[ids, "hardware"].to_numpy() == 1, "implant present", "no implant")})
    df["pos"] = df.y
    metrics = {
        "auroc": (lambda s: mt.auroc_binary(s.y.to_numpy(), s.score.to_numpy()), MIN_POS, "pos"),
        "sensitivity_at_val_threshold": (lambda s: mt.sensitivity_at_threshold(s.y.to_numpy(), s.score.to_numpy(), thr), MIN_POS, "pos"),
        "specificity_at_val_threshold": (lambda s: mt.specificity_at_threshold(s.y.to_numpy(), s.score.to_numpy(), thr), 0, None),
    }
    out = _audit(df, ["body_part", "view", "hardware"], metrics)
    out.update({"dataset": "FracAtlas test split", "model": meta["model_id"], "operating_point": "threshold fixed on the validation split (90% specificity there)", "threshold": float(thr)})
    return out


# ----------------------------------------------------------------------------- chest


def cxr(tag: str = "chex") -> dict:
    name = f"cxr_{tag}"
    z = np.load(ART / name / "predictions" / "test.npz", allow_pickle=False)
    labels = list(z["labels"])
    s = z["probs"][:, labels.index("Lung Opacity")]
    y = z["y"]
    meta = pd.read_csv(REPO / "ml" / "data" / "cache" / "rsna_meta.csv", dtype={"image_id": str}).set_index("image_id")
    ids = z["ids"].astype(str)
    bc = BinaryCalibrator.load(ART / name / "calibration_lung.json")
    miss = ~(bc.prob(s) >= bc.thr_fnr)
    df = pd.DataFrame({"y": y, "score": s, "miss": miss.astype(int), "group": z["group"].astype(str), "age_band": [age_band(a) for a in meta.loc[ids, "age"]], "sex": meta.loc[ids, "sex"].to_numpy(), "view": meta.loc[ids, "view"].to_numpy()})
    df["pos"] = df.y
    metrics = {
        "auroc": (lambda s_: mt.auroc_binary(s_.y.to_numpy(), s_.score.to_numpy()), MIN_POS, "pos"),
        "false_negative_rate_at_control_threshold": (lambda s_: float(s_.miss[s_.y == 1].mean()), MIN_POS, "pos"),
    }
    out = _audit(df, ["age_band", "sex", "view"], metrics)
    out.update({"dataset": "RSNA test split (patient-level)", "model": f"xrv-densenet121-{tag}", "label": "Lung Opacity", "note": "ages above 90 are kept as recorded; the FNR column uses the threshold that targets a 10% miss rate overall"})
    return out


def main() -> int:
    res = {"min_n": MIN_N, "min_positives": MIN_POS, "bootstrap_B": B, "datasets": {}}
    for name, fn in (("skin_cls", skin), ("brain_cls", brain), ("bone_det", bone), ("cxr_chex", cxr)):
        try:
            res["datasets"][name] = fn()
        except FileNotFoundError as e:
            res["datasets"][name] = {"status": f"pending: {e.filename}"}
            continue
        for col, blk in res["datasets"][name]["by"].items():
            for metric, w in blk["worst"].items():
                print(f"{name:9s} {col:10s} {metric:36s} worst {w['subgroup']:>16s} {w['value']:.3f} (overall {w['overall']:.3f}){'  CI entirely worse than overall' if w['ci_entirely_worse_than_overall'] else ''}")
    (REPO / "reports" / "subgroups.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
