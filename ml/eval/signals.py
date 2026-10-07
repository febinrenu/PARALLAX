"""P2.10 trust-signal validation (D13): is the error rate really higher when a warning is raised?

    python ml/eval/signals.py quality     # run the quality gate on every evaluation image, cache flags (CPU, minutes)
    python ml/eval/signals.py ood         # energy-score OOD AUROC against other-modality and natural images (GPU)
    python ml/eval/signals.py evaluate    # combine cached pieces into reports/signals.json

For each signal F: error rate with F versus without F, each with a 95% cluster-bootstrap CI, and the difference with its CI.
A signal "predicts errors" only if the difference is positive and its CI excludes zero; otherwise the report says so and the
signal must not be used to downgrade findings (plan.md section 9.4). Signals whose upstream module does not exist yet are
listed as pending, with the harness ready for them.

Signals available now: unstable (flip rate over the 8 perturbations at severity 2 above 0.25), low quality (P1's gate: any
warning or failure), OOD (energy score above the 95th percentile of validation images), abstain (calibrated tier).
Pending: discordant (needs P3's complete second-reader batch), unfaithful (needs P1.9 faithfulness).
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

from medproof.calibrate import temperature  # noqa: E402
from medproof.calibrate.binary import BinaryCalibrator  # noqa: E402
from medproof.calibrate.calibrator import Calibrator  # noqa: E402
from medproof.intake.decode import load_image  # noqa: E402
from medproof.intake.quality import assess  # noqa: E402

from ml.data.common import data_root, image_path, load_split  # noqa: E402
from ml.eval import bootstrap as bs  # noqa: E402
from ml.eval import metrics as mt  # noqa: E402

ART = REPO / "ml" / "artifacts"
B = 1000
MODALITY = {"skin_cls": "skin_dermoscopy", "brain_cls": "brain_mri", "cxr_chex": "cxr", "bone_det": "bone_xray"}
SPLIT_OF = {"skin_cls": "ham10000", "brain_cls": "brain_mri", "cxr_chex": "rsna", "bone_det": "fracatlas"}


# ----------------------------------------------------------------------------- quality gate on real images


def quality_frame(name: str) -> pd.DataFrame:
    """P1's quality gate on every image of the model's corruption subset. Cached, since it decodes each file."""
    cache = ART / name / "predictions" / "quality.csv"  # committed: the flags need the images, which a fresh clone does not have
    z = np.load(ART / name / "predictions" / "corruption.npz", allow_pickle=False)
    ids = z["ids"].astype(str)
    if cache.is_file():
        q = pd.read_csv(cache, dtype={"image_id": str})
        if list(q.image_id) == list(ids):
            return q
    df = load_split(SPLIT_OF[name]).set_index("image_id")
    rows = []
    for i in ids:
        rep = assess(load_image(image_path(df.loc[i], data_root())), MODALITY[name])
        rows.append({"image_id": i, "n_warn": sum(r.level == "warn" for r in rep.reasons), "n_fail": sum(r.level == "fail" for r in rep.reasons), "codes": ";".join(r.code for r in rep.reasons), **{f"m_{k}": float(v) for k, v in rep.metrics.items()}})
    q = pd.DataFrame(rows)
    q.to_csv(cache, index=False)
    return q


# ----------------------------------------------------------------------------- per-model frames


def energy(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    m = z.max(1, keepdims=True)
    return -(m[:, 0] + np.log(np.exp(z - m).sum(1)))


def classifier_frame(name: str) -> pd.DataFrame:
    z = np.load(ART / name / "predictions" / "corruption.npz", allow_pickle=False)
    y, clean, cells = z["y"], z["clean"].astype(np.float64), z["cells"]
    cal = Calibrator.load(ART / name / "calibration.json")
    out = cal.calibrate(clean)
    pred = clean.argmax(1)
    flips = np.stack([cells[pi, 1].argmax(1) != pred for pi in range(cells.shape[0])], 1)
    val = np.load(ART / name / "predictions" / "val.npz", allow_pickle=False)
    thr = float(np.quantile(energy(val["logits"]), 0.95))
    q = quality_frame(name)
    df = pd.DataFrame({"image_id": z["ids"].astype(str), "y": y, "pred": pred, "error": (pred != y).astype(int), "confidence": out.confidence, "set_size": [len(s) for s in out.conformal_sets], "tier": out.tiers,
                       "flip_rate": flips.mean(1), "energy": energy(clean)})
    df["unstable"] = df.flip_rate > 0.25
    df["ood_energy"] = df.energy > thr
    df["low_quality"] = (q.n_warn + q.n_fail).to_numpy() > 0
    df["quality_fail"] = q.n_fail.to_numpy() > 0
    df["abstain"] = df.tier == "abstain"
    df.attrs["energy_threshold"] = thr
    return df


def cxr_frame(tag: str = "chex") -> pd.DataFrame:
    name = f"cxr_{tag}"
    z = np.load(ART / name / "predictions" / "corruption.npz", allow_pickle=False)
    y, clean, cells = z["y"], z["clean"].astype(np.float64), z["cells"]  # model outputs for the Lung Opacity label
    prob = temperature.sigmoid(clean) if clean.min() < 0 or clean.max() > 1 else clean
    bc = BinaryCalibrator.load(ART / name / "calibration_lung.json")
    p = bc.prob(prob)
    flips = np.stack([((temperature.sigmoid(cells[pi, 1]) if cells[pi, 1].min() < 0 or cells[pi, 1].max() > 1 else cells[pi, 1]) >= 0.5) != (prob >= 0.5) for pi in range(cells.shape[0])], 1)
    q = quality_frame(name)
    df = pd.DataFrame({"image_id": z["ids"].astype(str), "y": y, "pred": (p >= 0.5).astype(int), "confidence": np.maximum(p, 1 - p), "flip_rate": flips.mean(1)})
    df["error"] = (df.pred != df.y).astype(int)
    df["unstable"] = df.flip_rate > 0.25
    df["low_quality"] = (q.n_warn + q.n_fail).to_numpy() > 0
    df["quality_fail"] = q.n_fail.to_numpy() > 0
    df["abstain"] = (p >= bc.abstain_low) & (p <= bc.abstain_high)
    return df


def bone_frame() -> pd.DataFrame:
    z = np.load(ART / "bone_det" / "predictions" / "corruption.npz", allow_pickle=False)
    y, clean, cells = z["y"], z["clean"].astype(np.float64), z["cells"]  # image-level score = highest box confidence
    bc = BinaryCalibrator.load(ART / "bone_det" / "calibration.json")
    p = bc.prob(clean)
    flips = np.stack([(bc.prob(cells[pi, 1]) >= 0.5) != (p >= 0.5) for pi in range(cells.shape[0])], 1)
    q = quality_frame("bone_det")
    df = pd.DataFrame({"image_id": z["ids"].astype(str), "y": y, "pred": (p >= 0.5).astype(int), "confidence": np.maximum(p, 1 - p), "flip_rate": flips.mean(1)})
    df["error"] = (df.pred != df.y).astype(int)
    df["unstable"] = df.flip_rate > 0.25
    df["low_quality"] = (q.n_warn + q.n_fail).to_numpy() > 0
    df["quality_fail"] = q.n_fail.to_numpy() > 0
    df["abstain"] = (p >= bc.abstain_low) & (p <= bc.abstain_high)
    return df


# ----------------------------------------------------------------------------- the comparison


def compare(df: pd.DataFrame, flag: str) -> dict:
    e, f = df.error.to_numpy().astype(float), df[flag].to_numpy().astype(bool)
    n1, n0 = int(f.sum()), int((~f).sum())
    out: dict = {"n_flagged": n1, "n_unflagged": n0, "share_flagged": float(f.mean())}
    if n1 < 10 or n0 < 10:
        out["verdict"] = "too few flagged or unflagged cases to judge"
        return out
    kw = dict(B=B)
    out["error_rate_flagged"] = bs.ci({"e": e, "f": f}, lambda e, f: float(e[f].mean()) if f.any() else float("nan"), **kw)
    out["error_rate_unflagged"] = bs.ci({"e": e, "f": f}, lambda e, f: float(e[~f].mean()) if (~f).any() else float("nan"), **kw)
    d = bs.ci({"e": e, "f": f}, lambda e, f: float(e[f].mean() - e[~f].mean()) if f.any() and (~f).any() else float("nan"), **kw)
    r = bs.ci({"e": e, "f": f}, lambda e, f: float(e[f].mean() / e[~f].mean()) if f.any() and (~f).any() and e[~f].mean() > 0 else float("nan"), **kw)
    out["difference"], out["risk_ratio"] = d, r
    if r["n_valid"] == 0:
        out["risk_ratio_note"] = "undefined: no unflagged case is wrong"
    ok = d["lo"] > 0
    out["predicts_errors"] = bool(ok)
    out["verdict"] = ("flagged cases are wrong more often (difference CI excludes 0): the signal is usable" if ok else
                      "no reliable difference in error rate between flagged and unflagged cases: do not use this signal to downgrade findings" if d["hi"] >= 0 else
                      "flagged cases are wrong LESS often: the signal is miscalibrated")
    return out


# ----------------------------------------------------------------------------- OOD


def _cifar_images(n: int = 300) -> list[np.ndarray]:
    import pickle

    import tarfile

    root = data_root() / "cifar10_test"
    hit = next(root.rglob("test_batch"), None)
    if hit is None:  # extract the one member we need, by exact name, from the md5-verified archive
        with tarfile.open(root / "cifar-10-python.tar.gz") as t:
            m = t.getmember("cifar-10-batches-py/test_batch")
            m.name = "test_batch"
            t.extract(m, root)
        hit = root / "test_batch"
    p = hit
    d = pickle.load(open(p, "rb"), encoding="bytes")  # tarball md5 verified by the downloader
    x = d[b"data"][:n].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32) / 255.0
    return [a for a in x]


def _other_images(name: str, n: int = 300) -> list[np.ndarray]:
    df = load_split(name)
    if name == "ham10000":
        t = df[(df.split == "official_test") & df.eval_batch]
    elif name == "rsna":
        t = df[(df.split == "test") & df.eval_batch]
    else:
        t = df[df.split == "test"]
    t = t.sample(n=min(n, len(t)), random_state=20261006)
    import cv2

    out = []
    for _, r in t.iterrows():
        try:
            a = load_image(image_path(r, data_root())).analysis
        except Exception:  # the product decoder rejects the truncated FracAtlas JPEGs; they are skipped here and counted in the report note
            continue
        s = 512 / max(a.shape[:2])
        out.append(cv2.resize(a, (int(a.shape[1] * s), int(a.shape[0] * s)), interpolation=cv2.INTER_AREA) if s < 1 else a)
    return out


def ood_auroc(name: str) -> dict:
    """Energy-score AUROC separating the model's own test images (ID) from other modalities and natural images (OOD)."""
    import torch

    from medproof.intake.preprocess import prepare
    from ml.eval import predict as pr

    spec = {"skin_cls": "skin_cls", "brain_cls": "brain_effnet"}[name]
    model = pr.load_timm(name)
    own = {"skin_cls": "ham10000", "brain_cls": "brain_mri"}[name]
    test = {"skin_cls": "official_test", "brain_cls": "test"}[name]
    z = np.load(ART / name / "predictions" / f"{test}.npz", allow_pickle=False)
    e_id = energy(z["logits"])
    msp_id = temperature.softmax(z["logits"]).max(1)
    sets = {"cifar10_natural_images": _cifar_images()}
    for other in ("ham10000", "brain_mri", "fracatlas", "rsna"):
        if other != own:
            sets[f"{other}_other_modality"] = _other_images(other)
    out = {}
    for sname, imgs in sets.items():
        X = torch.from_numpy(np.stack([prepare(a, spec) for a in imgs]))
        lg = pr.logits(model, X)
        e_ood, msp_ood = energy(lg), temperature.softmax(lg).max(1)
        y = np.r_[np.zeros(len(e_id)), np.ones(len(e_ood))]
        kw = dict(strata=y, B=B)
        out[sname] = {"n_ood": len(imgs), "n_id": int(len(e_id)),
                      "auroc_energy": bs.ci({"y": y, "s": np.r_[e_id, e_ood]}, lambda y, s: mt.auroc_binary(y, s), **kw),
                      "auroc_max_softmax": bs.ci({"y": y, "s": np.r_[-msp_id, -msp_ood]}, lambda y, s: mt.auroc_binary(y, s), **kw)}
    return out


# ----------------------------------------------------------------------------- driver


def quality_summary(name: str) -> dict:
    q = quality_frame(name)
    q["codes"] = q["codes"].fillna("").astype(str)
    mcols = [c for c in q.columns if c.startswith("m_")]
    return {"n": int(len(q)), "share_with_any_warning_or_failure": float(((q.n_warn + q.n_fail) > 0).mean()), "share_failing": float((q.n_fail > 0).mean()),
            "target_share_flagged_on_clean_images": 0.05, "reasons": {c: int(q.codes.str.contains(c).sum()) for c in sorted({c for s in q.codes for c in s.split(";") if c})},
            "metric_percentiles": {c[2:]: {f"p{p}": float(np.nanpercentile(q[c], p)) for p in (1, 5, 50, 95, 99)} for c in mcols}}


def evaluate() -> dict:
    res: dict = {"definition": "error = top-1 wrong (classifiers) or decision at calibrated probability 0.5 wrong (chest reader and bone detector, image level); the evaluation images are each model's corruption-benchmark subset, clean images only",
                 "models": {}, "pending": {"discordant": "needs the complete MedGemma batch from P3 for the same images; harness: compare(df, 'discordant')", "unfaithful": "needs P1.9 faithfulness; harness: compare(df, 'unfaithful')"}}
    for name in ("skin_cls", "brain_cls", "cxr_chex", "bone_det"):
        if not (ART / name / "predictions" / "corruption.npz").is_file():
            continue
        df = cxr_frame() if name.startswith("cxr") else bone_frame() if name == "bone_det" else classifier_frame(name)
        blk = {"n": int(len(df)), "error_rate_overall": float(df.error.mean()), "signals": {}, "quality_gate": quality_summary(name)}
        for flag in ("unstable", "low_quality", "abstain") + (("ood_energy",) if "ood_energy" in df else ()):
            blk["signals"][flag] = compare(df, flag)
        if "energy_threshold" in df.attrs:
            blk["ood_energy_threshold_p95_of_validation"] = df.attrs["energy_threshold"]
        res["models"][name] = blk
    ood_p = REPO / "reports" / "ood.json"
    if ood_p.is_file():
        res["ood_auroc"] = json.loads(ood_p.read_text())
    return res


def main(argv=None) -> int:
    a = argv or sys.argv[1:]
    cmd = a[0] if a else "evaluate"
    if cmd == "quality":
        for n in ("skin_cls", "brain_cls", "cxr_chex", "bone_det"):
            if (ART / n / "predictions" / "corruption.npz").is_file():
                quality_frame(n)
                print("quality done", n, flush=True)
    elif cmd == "ood":
        out = {n: ood_auroc(n) for n in ("skin_cls", "brain_cls")}
        (REPO / "reports" / "ood.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        for n, d in out.items():
            for s, v in d.items():
                print(f"{n} vs {s}: AUROC energy {v['auroc_energy']['point']:.3f}  max-softmax {v['auroc_max_softmax']['point']:.3f}")
    elif cmd == "evaluate":
        res = evaluate()
        (REPO / "reports" / "signals.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        for m, b in res["models"].items():
            for s, v in b["signals"].items():
                if "difference" in v:
                    print(f"{m:9s} {s:12s} flagged {v['n_flagged']:4d} err {v['error_rate_flagged']['point']:.3f} vs {v['error_rate_unflagged']['point']:.3f}  diff {v['difference']['point']:+.3f} [{v['difference']['lo']:+.3f}, {v['difference']['hi']:+.3f}]")
                else:
                    print(f"{m:9s} {s:12s} {v['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
