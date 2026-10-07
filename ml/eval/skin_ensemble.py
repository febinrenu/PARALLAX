"""Optional extra: does a 3-seed ConvNeXt-Tiny ensemble with flip test-time augmentation beat the single skin model?

    python ml/eval/skin_ensemble.py score      # GPU: MILK10k logits per member and flip-TTA logits on the official test and MILK10k
    python ml/eval/skin_ensemble.py evaluate   # CPU: compare single, TTA, ensemble and ensemble+TTA with bootstrap CIs, fit the ensemble calibrator

Members are ml/artifacts/skin_cls, skin_cls_s2 and skin_cls_s3 (same architecture and data, different seeds). Averaging logits keeps
the temperature and conformal machinery unchanged. The ensemble is only worth shipping if it improves on both the official test set and
the external MILK10k set; the comparison below decides, and the result is written to reports/skin_ensemble.json either way.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for p in (REPO, REPO / "backend"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

ART = REPO / "ml" / "artifacts"
MEMBERS = ["skin_cls", "skin_cls_s2", "skin_cls_s3"]
FLIPS = [(), (3,), (2,), (2, 3)]  # identity, horizontal, vertical, both (tensor dims of NCHW)


def _images(split_name: str, split: str):
    from medproof.intake.decode import load_image
    from medproof.intake.preprocess import prepare
    from ml.data.common import data_root, image_path, load_split
    import torch

    df = load_split(split_name)
    df = df[df.split == split].reset_index(drop=True)
    return df, torch.from_numpy(np.stack([prepare(load_image(image_path(r, data_root())), "skin_cls") for _, r in df.iterrows()]))


def score() -> None:
    import torch

    from ml.eval import predict as pr

    sets = {"official_test": ("ham10000", "official_test"), "milk10k": ("milk10k", "external_test")}
    cache = {k: _images(*v) for k, v in sets.items()}
    for m in MEMBERS:
        if not (ART / m / "weights.pt").is_file():
            print("skip", m, "(no weights)")
            continue
        model = pr.load_timm(m)
        for key, (df, X) in cache.items():
            logits = np.stack([pr.logits(model, torch.flip(X, f) if f else X) for f in FLIPS])  # (4, N, 7)
            np.savez_compressed(ART / m / "predictions" / f"tta_{key}.npz", ids=df.image_id.to_numpy().astype(str), logits=logits.astype(np.float32))
        print("scored", m, flush=True)


def _load(m: str, key: str):
    z = np.load(ART / m / "predictions" / f"tta_{key}.npz", allow_pickle=False)
    return z["ids"], z["logits"]


def evaluate(B: int = 1000) -> dict:
    from medproof.calibrate import conformal, temperature
    from medproof.calibrate.calibrator import Calibrator

    from ml.eval import bootstrap as bs
    from ml.eval import metrics as mt

    meta = json.loads((ART / "skin_cls" / "model_meta.json").read_text())
    classes = meta["classes"]
    have = [m for m in MEMBERS if (ART / m / "predictions" / "tta_official_test.npz").is_file()]
    truth = {k: np.load(ART / "skin_cls" / "predictions" / f"{k}.npz", allow_pickle=False) for k in ("official_test", "milk10k")}
    out: dict = {"members": have, "variants": {}}

    def stat(y):
        return lambda y, p: mt.balanced_accuracy(y, p.argmax(1), 7)

    for key in ("official_test", "milk10k"):
        z = truth[key]
        y, groups = z["y"], z["group"]
        for m in have:
            ids, _ = _load(m, key)
            assert (ids == z["ids"]).all(), "member predictions must be in the same image order"
        L = {m: _load(m, key)[1] for m in have}  # (4, N, 7) each
        variants = {
            **{f"single, {m.replace('skin_cls_s', 'seed ').replace('skin_cls', 'seed 1')}": L[m][0] for m in have},
            "single (seed 1) + flip TTA": L["skin_cls"].mean(0),
            f"ensemble of {len(have)}": np.mean([L[m][0] for m in have], 0),
            f"ensemble of {len(have)} + flip TTA": np.mean([L[m].mean(0) for m in have], 0),
        }
        for name, lg in variants.items():
            p = mt.softmax(lg)
            out["variants"].setdefault(name, {})[key] = {
                "balanced_accuracy": bs.ci({"y": y, "p": p}, lambda y, p: mt.balanced_accuracy(y, p.argmax(1), 7), groups=groups, strata=y, B=B),
                "accuracy": bs.ci({"y": y, "p": p}, lambda y, p: mt.accuracy(y, p.argmax(1)), groups=groups, strata=y, B=B),
                "melanoma_recall": bs.ci({"y": y, "p": p}, lambda y, p: float(np.mean(p[y == 4].argmax(1) == 4)), groups=groups, strata=y, B=B),
                "ece_15": bs.ci({"y": y, "p": p}, lambda y, p: mt.ece(y, p, 15), groups=groups, strata=y, B=B),
            }
    best = f"ensemble of {len(have)} + flip TTA"
    singles = [f"single, {m.replace('skin_cls_s', 'seed ').replace('skin_cls', 'seed 1')}" for m in have]
    mean_single = {k: float(np.mean([out["variants"][n][k]["balanced_accuracy"]["point"] for n in singles])) for k in ("official_test", "milk10k")}
    out["mean_of_single_seeds"] = mean_single
    out["gain_over_mean_single_seed"] = {k: out["variants"][best][k]["balanced_accuracy"]["point"] - mean_single[k] for k in mean_single}
    out["gain_over_seed_1"] = {k: out["variants"][best][k]["balanced_accuracy"]["point"] - out["variants"][singles[0]][k]["balanced_accuracy"]["point"] for k in mean_single}
    g = out["gain_over_mean_single_seed"]
    out["recommendation"] = ("ship the ensemble" if g["official_test"] >= 0.015 and g["milk10k"] >= 0.01 else "keep a single model: the ensemble does not beat the average single seed by a margin worth three times the weights and inference time")
    out["note"] = "seeds differ by about 0.04 balanced accuracy on the official test (seed 1 was the weakest), so part of the gain over seed 1 is seed luck; the fair baseline is the mean of the single seeds. Swapping the shipped weights for the best single seed costs nothing but is selection on the test set, so it is not recommended as evidence of better generalisation"
    # calibrate the ensemble on the validation and calibration splits (members' own cached predictions, plain average, no TTA to match serving cost)
    vals = {s: [np.load(ART / m / "predictions" / f"{s}.npz", allow_pickle=False) for m in have] for s in ("val", "cal")}
    if all(len(v) == len(have) for v in vals.values()) and len(have) > 1:
        lv, lc = (np.mean([z["logits"] for z in vals[s]], 0) for s in ("val", "cal"))
        cal = Calibrator.fit(classes, lv, vals["val"][0]["y"], lc, vals["cal"][0]["y"], alpha=0.1, mondrian=True)
        cal.meta.update({"members": have, "note": "temperature on val, conformal on cal, logits averaged over members"})
        (ART / "skin_ens").mkdir(exist_ok=True)
        cal.save(ART / "skin_ens" / "calibration.json")
        z = truth["official_test"]
        L = np.mean([_load(m, "official_test")[1][0] for m in have], 0)
        res = cal.calibrate(L)
        sets = [[classes.index(c) for c in s] for s in res.conformal_sets]
        out["ensemble_calibration"] = {"temperature": cal.temperature, "official_test_coverage_at_90": conformal.coverage(sets, z["y"]), "mean_set_size": conformal.mean_set_size(sets)}
    return out


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "evaluate"
    if cmd == "score":
        score()
        return 0
    res = evaluate()
    (REPO / "reports" / "skin_ensemble.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    for name, v in res["variants"].items():
        print(f"{name:34s} official {v['official_test']['balanced_accuracy']['point']:.3f}  milk10k {v['milk10k']['balanced_accuracy']['point']:.3f}")
    print("gain over mean single seed", res["gain_over_mean_single_seed"], "->", res["recommendation"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
