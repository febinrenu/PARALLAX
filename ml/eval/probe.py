"""Optional extra: MedSigLIP linear probes as an additional skin and brain reader, alone and fused with the CNNs.

    python ml/eval/probe.py embed       # GPU, needs HF_TOKEN with access to google/medsiglip-448; caches embeddings in ml/data/cache (not committed)
    python ml/eval/probe.py fit         # CPU: logistic-regression probes, writes ml/artifacts/{skin,brain}_probe/{probe.npz,predictions/*.npz}
    python ml/eval/probe.py evaluate    # CPU, from committed predictions: probe alone vs CNN alone vs fusion, with bootstrap CIs (reports/probe.json)

The probe is a multinomial logistic regression on L2-normalised MedSigLIP image embeddings (frozen encoder, 448 px). Its regularisation is chosen
on the validation split; fusion weights are chosen on the validation split too, never on a test set. Embeddings are large and stay out of git;
the probe weights (a few KB) and its per-split logits are committed so `make eval` reproduces everything.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for p in (REPO, REPO / "backend"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

ART = REPO / "ml" / "artifacts"
CACHE = Path(os.environ.get("PARALLAX_EMB_OUT") or REPO / "ml" / "data" / "cache")
MODEL = "google/medsiglip-448"
# probe name -> (CNN artifact, class list source, [(split csv, split value, prediction file name)], train split csv)
SETS = {
    "skin_probe": {"cnn": "skin_cls", "csv": "ham10000", "splits": {"train": "train", "val": "val", "cal": "cal", "test": "test", "official_test": "official_test"}, "external": ("milk10k", "external_test", "milk10k")},
    "brain_probe": {"cnn": "brain_cls", "csv": "brain_mri", "splits": {"train": "train", "val": "val", "cal": "cal", "test": "test"}, "external": ("bdneuro", "external_test", "bdneuro")},
}


def _token() -> str:
    """HF token from the environment, the local .env, or (on Kaggle) the notebook's Secrets. Never written into any file."""
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    env = REPO / ".env"
    if env.is_file():
        for line in env.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip()
    try:
        from kaggle_secrets import UserSecretsClient

        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        raise RuntimeError("HF_TOKEN is not set (needs access to google/medsiglip-448): export it, put it in .env, or add a Kaggle notebook secret named HF_TOKEN") from None


def embed() -> None:
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    from medproof.intake.decode import load_image
    from ml.data.common import data_root, image_path, load_split

    src = os.environ.get("PARALLAX_MEDSIGLIP_DIR")  # a local copy of the model folder avoids the 3.5 GB hub download
    tok = None if src else _token()
    model = AutoModel.from_pretrained(src or MODEL, token=tok, torch_dtype=torch.float16).cuda().eval()
    proc = AutoProcessor.from_pretrained(src or MODEL, token=tok)
    CACHE.mkdir(parents=True, exist_ok=True)
    jobs = [("ham10000", None), ("milk10k", "external_test"), ("brain_mri", None), ("bdneuro", "external_test")]
    for name, only in jobs:
        out = CACHE / f"medsiglip_{name}.npz"
        if out.is_file() or not (data_root() / name).exists():
            continue
        df = load_split(name)
        df = df[df.split != "dropped"].reset_index(drop=True) if only is None else df[df.split == only].reset_index(drop=True)
        embs = []
        for i in range(0, len(df), 32):
            ims = []
            for _, r in df.iloc[i : i + 32].iterrows():
                d = load_image(image_path(r, data_root())).display
                ims.append(Image.fromarray(d).convert("RGB"))
            x = proc(images=ims, return_tensors="pt")["pixel_values"].cuda().half()
            with torch.no_grad():
                embs.append(model.get_image_features(pixel_values=x).float().cpu().numpy())
            if (i // 32) % 20 == 0:
                print(f"  {name}: {i}/{len(df)}", flush=True)
        np.savez_compressed(out, ids=df.image_id.to_numpy().astype(str), split=df.split.to_numpy().astype(str), emb=np.concatenate(embs).astype(np.float16))
        print("embedded", name, len(df), flush=True)


def _norm(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def _load_emb(name: str):
    z = np.load(CACHE / f"medsiglip_{name}.npz", allow_pickle=False)
    return z["ids"], z["split"], _norm(z["emb"])


def fit() -> None:
    from sklearn.linear_model import LogisticRegression

    from ml.eval import metrics as mt

    for probe, cfg in SETS.items():
        meta = json.loads((ART / cfg["cnn"] / "model_meta.json").read_text())
        classes = meta["classes"]
        cidx = {c: i for i, c in enumerate(classes)}
        from ml.data.common import load_split

        df = load_split(cfg["csv"]).set_index("image_id")
        ids, spl, E = _load_emb(cfg["csv"])
        y_all = df.loc[ids, "label"].map(cidx).to_numpy()
        tr, va = spl == "train", spl == "val"
        best = None
        for C in (0.3, 1, 3, 10, 30, 100):
            clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(E[tr], y_all[tr])
            bma = mt.balanced_accuracy(y_all[va], clf.predict(E[va]), len(classes))
            print(f"  {probe} C={C}: val balanced accuracy {bma:.3f}")
            if best is None or bma > best[0]:
                best = (bma, C, clf)
        _, C, clf = best
        out = ART / probe
        (out / "predictions").mkdir(parents=True, exist_ok=True)
        np.savez(out / "probe.npz", coef=clf.coef_.astype(np.float32), intercept=clf.intercept_.astype(np.float32), classes=np.array(classes), C=C, encoder=MODEL, normalise="l2")
        for key, sp in cfg["splits"].items():
            m = spl == sp
            _save_pred(out, key, ids[m], clf, E[m], y_all[m], df.loc[ids[m], "group"].to_numpy().astype(str))
        name, sp, key = cfg["external"]
        if (CACHE / f"medsiglip_{name}.npz").is_file():
            ids2, spl2, E2 = _load_emb(name)
            d2 = load_split(name).set_index("image_id")
            _save_pred(out, key, ids2, clf, E2, d2.loc[ids2, "label"].map(cidx).to_numpy(), d2.loc[ids2, "group"].to_numpy().astype(str))
        (out / "model_meta.json").write_text(json.dumps({"model_id": probe, "task": "classification (linear probe)", "arch": f"logistic regression on {MODEL} image embeddings", "classes": classes, "preproc_spec": "medsiglip-448 processor", "C": C,
                                                         "val_balanced_accuracy": best[0], "license": "probe weights: derived from HAI-DEF terms model embeddings and non-commercial data; see model card", "contamination": "MedSigLIP training data overlap with our test sets is unverified"}, indent=1))
        print("fitted", probe, "C", C)


def _save_pred(out: Path, key: str, ids, clf, E, y, groups) -> None:
    np.savez_compressed(out / "predictions" / f"{key}.npz", ids=ids, logits=clf.predict_log_proba(E).astype(np.float32), y=y, group=groups)


# ----------------------------------------------------------------------------- evaluation from committed predictions


def evaluate(B: int = 500) -> dict:
    from ml.eval import bootstrap as bs
    from ml.eval import metrics as mt

    res: dict = {"note": "the probe's regularisation and the fusion weight are chosen on the validation split only", "models": {}}
    for probe, cfg in SETS.items():
        cnn = cfg["cnn"]
        K = len(json.loads((ART / cnn / "model_meta.json").read_text())["classes"])
        loadp = lambda m, s: np.load(ART / m / "predictions" / f"{s}.npz", allow_pickle=False)  # noqa: E731
        # fusion weight on validation: p = w * softmax(cnn) + (1 - w) * softmax(probe)
        v_c, v_p = loadp(cnn, "val"), loadp(probe, "val")
        assert (v_c["ids"] == v_p["ids"]).all()
        best = max(((mt.balanced_accuracy(v_c["y"], (w * mt.softmax(v_c["logits"]) + (1 - w) * mt.softmax(v_p["logits"])).argmax(1), K), w) for w in np.linspace(0, 1, 11)))
        w = best[1]
        blk = {"fusion_weight_on_cnn": float(w), "val_balanced_accuracy_fused": float(best[0]), "splits": {}}
        splits = [s for s in (list(cfg["splits"].values())[3:] + [cfg["external"][2]]) if (ART / probe / "predictions" / f"{s}.npz").is_file()]
        for s in splits:
            c, p = loadp(cnn, s), loadp(probe, s)
            assert (c["ids"] == p["ids"]).all(), f"{probe}/{s}: id order differs"
            y, g = c["y"], c["group"]
            pc, pp = mt.softmax(c["logits"]), mt.softmax(p["logits"])
            fused = w * pc + (1 - w) * pp
            rows = {}
            for name, prob in (("CNN", pc), ("MedSigLIP probe", pp), ("fusion", fused)):
                rows[name] = {"balanced_accuracy": bs.ci({"y": y, "p": prob}, lambda y, p: mt.balanced_accuracy(y, p.argmax(1), K), groups=g, strata=y, B=B),
                              "accuracy": bs.ci({"y": y, "p": prob}, lambda y, p: mt.accuracy(y, p.argmax(1)), groups=g, strata=y, B=B)}
            blk["splits"][s] = rows
        res["models"][probe] = blk
    return res


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "evaluate"
    if cmd == "embed":
        embed()
    elif cmd == "fit":
        fit()
    else:
        r = evaluate()
        (REPO / "reports" / "probe.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
        for m, b in r["models"].items():
            print(m, "fusion weight on CNN", b["fusion_weight_on_cnn"])
            for s, rows in b["splits"].items():
                print(f"  {s:14s} " + "  ".join(f"{k} {v['balanced_accuracy']['point']:.3f}" for k, v in rows.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
