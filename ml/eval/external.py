"""D15 external validation of the brain classifier on BDNeuro-MRI (the images that do not duplicate the Kaggle training source).

    python ml/eval/external.py score      # run brain_cls on ml/data/splits/bdneuro.csv (GPU), cache logits
    python ml/eval/external.py evaluate   # metrics with CIs from the cached logits

71% of BDNeuro-MRI duplicates images in the Kaggle brain dataset (see reports/leakage.json), so only the remaining images are
independent. Labels follow the folder names; "no_tumor" maps to "notumor". The set is glioma-heavy, so read per-class recall.
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


def score() -> Path:
    import torch

    from medproof.intake.preprocess import prepare
    from medproof.intake.decode import load_image
    from ml.data.common import data_root, image_path, load_split
    from ml.eval import predict as pr

    df = load_split("bdneuro")
    df = df[df.split == "external_test"].reset_index(drop=True)
    classes = pr.meta("brain_cls")["classes"]
    model = pr.load_timm("brain_cls")
    X = torch.from_numpy(np.stack([prepare(load_image(image_path(r, data_root())), "brain_effnet") for _, r in df.iterrows()]))
    lg = pr.logits(model, X)
    out = ART / "brain_cls" / "predictions" / "bdneuro.npz"
    np.savez_compressed(out, ids=df.image_id.to_numpy().astype(str), logits=lg.astype(np.float32), y=df.label.map({c: i for i, c in enumerate(classes)}).to_numpy(), group=df.group.to_numpy().astype(str))
    return out


def evaluate(B: int = 1000) -> dict:
    from ml.eval import metrics as mt
    from ml.train import common as tc

    meta = json.loads((ART / "brain_cls" / "model_meta.json").read_text())
    z = np.load(ART / "brain_cls" / "predictions" / "bdneuro.npz", allow_pickle=False)
    p = mt.softmax(z["logits"])
    rep = tc.bootstrap_report(z["y"], p, z["group"], 4, meta["classes"], B=B)
    rep.update({"dataset": "BDNeuro-MRI, images with no duplicate in the Kaggle brain dataset", "model": meta["model_id"], "class_counts": {c: int((z["y"] == i).sum()) for i, c in enumerate(meta["classes"])}})
    return rep


def score_skin() -> Path:
    """skin_cls on the MILK10k dermoscopic images that are not in HAM10000 (external test, with skin-tone grades)."""
    import torch

    from medproof.intake.decode import load_image
    from medproof.intake.preprocess import prepare
    from ml.data.common import data_root, image_path, load_split
    from ml.eval import predict as pr

    df = load_split("milk10k")
    df = df[df.split == "external_test"].reset_index(drop=True)
    classes = pr.meta("skin_cls")["classes"]
    model = pr.load_timm("skin_cls")
    X = torch.from_numpy(np.stack([prepare(load_image(image_path(r, data_root())), "skin_cls") for _, r in df.iterrows()]))
    lg = pr.logits(model, X)
    out = ART / "skin_cls" / "predictions" / "milk10k.npz"
    np.savez_compressed(out, ids=df.image_id.to_numpy().astype(str), logits=lg.astype(np.float32), y=df.label.map({c: i for i, c in enumerate(classes)}).to_numpy(), group=df.group.to_numpy().astype(str),
                        age=df.age.fillna(-1).to_numpy(float), sex=df.sex.fillna("unknown").to_numpy().astype(str), site=df.site_general.fillna("unknown").to_numpy().astype(str), skin_tone=df.skin_tone.astype(str).to_numpy().astype(str))
    return out


def evaluate_skin(B: int = 1000) -> dict:
    from ml.eval import metrics as mt
    from ml.train import common as tc

    meta = json.loads((ART / "skin_cls" / "model_meta.json").read_text())
    z = np.load(ART / "skin_cls" / "predictions" / "milk10k.npz", allow_pickle=False)
    rep = tc.bootstrap_report(z["y"], mt.softmax(z["logits"]), z["group"], 7, meta["classes"], B=B, positive="mel")
    rep.update({"dataset": "MILK10k dermoscopic images (external; classes the model does not cover and duplicates of HAM10000 excluded)", "model": meta["model_id"], "class_counts": {c: int((z["y"] == i).sum()) for i, c in enumerate(meta["classes"])}})
    return rep


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "evaluate"
    if cmd == "score":
        print("wrote", score())
    elif cmd == "score_skin":
        print("wrote", score_skin())
    elif cmd == "evaluate_skin":
        r = evaluate_skin()
        print(f"MILK10k external skin: n={r['n']} BMA {r['balanced_accuracy']['point']:.3f} [{r['balanced_accuracy']['lo']:.3f}, {r['balanced_accuracy']['hi']:.3f}] accuracy {r['accuracy']['point']:.3f} mel sens {r['per_class_recall']['mel']['point']:.3f}", {c: round(v['point'], 2) for c, v in r['per_class_recall'].items()})
    else:
        r = evaluate()
        a, b = r["accuracy"], r["balanced_accuracy"]
        print(f"BDNeuro external: n={r['n']} accuracy {a['point']:.3f} [{a['lo']:.3f}, {a['hi']:.3f}]  balanced accuracy {b['point']:.3f} [{b['lo']:.3f}, {b['hi']:.3f}]", {c: round(v['point'], 3) for c, v in r['per_class_recall'].items()})
