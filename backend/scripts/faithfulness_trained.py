"""Faithfulness and stability of the trained brain and skin classifiers on staged images.

    python scripts/faithfulness_trained.py [--images DIR] [-n 12] [--out FILE]

Writes the share of positive findings that pass the deletion permutation test and the share that flip under
perturbation, per model. Staged images are public datasets; nothing here is committed except the summary JSON.
"""

from __future__ import annotations

import argparse
import glob
import json
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=str(ROOT / "backend" / "data" / "router"))
    ap.add_argument("--artifacts", default=str(ROOT / "ml" / "artifacts"))
    ap.add_argument("-n", type=int, default=12)
    ap.add_argument("--only", choices=("brain_cls", "skin_cls", "bone_det"), default=None, help="assess a single model")
    ap.add_argument("--min-prob", type=float, default=0.3, help="only findings at least this probable are assessed")
    ap.add_argument("--out", default=str(ROOT / "backend" / "medproof" / "verify" / "results" / "faithfulness_trained.json"))
    a = ap.parse_args(argv)
    from medproof.intake.decode import load_image
    from medproof.readers import bone, classifier
    from medproof.verify import faithfulness as FA
    from medproof.verify import stability as ST

    report = {}
    for name, modality, folder in (("brain_cls", "brain_mri", "brain_mri"), ("skin_cls", "skin_dermoscopy", "skin_dermoscopy"), ("bone_det", "bone_xray", "bone_xray")):
        if a.only and a.only != name:
            continue
        if name == "bone_det":  # box-only findings: the deletion test runs on the box region
            reader = bone.BoneReader.from_dir(Path(a.artifacts) / name)
        else:
            reader = classifier.ImageClassifierReader.from_dir(Path(a.artifacts) / name, modality=modality, positive_threshold=0.0)
        files = sorted(glob.glob(str(Path(a.images) / folder / "**" / "*.*"), recursive=True))
        files = files[:: max(1, len(files) // a.n)][: a.n]
        rows = []
        for f in files:
            img = load_image(f)
            out = reader.predict(img)
            out.findings = [x for x in out.findings if x.prob_raw >= a.min_prob]
            if not out.findings:
                continue
            fa = FA.assess_output(reader, img, out, FA.FaithfulnessConfig(n_random=9))
            st = ST.assess_output(reader, img, out)
            for k in fa:
                rows.append({"image": Path(f).name, "label": k, "prob": round(float(next(x.prob_raw for x in out.findings if x.label == k)), 3),
                             "faithful": bool(fa[k].faithful), "drop": round(float(fa[k].drop), 3), "flip_rate": float(st[k].flip_rate)})
            print(name, Path(f).name, [(r["label"], r["faithful"], r["flip_rate"]) for r in rows if r["image"] == Path(f).name], flush=True)
        n = len(rows)
        report[name] = {"model_id": reader.model_id, "images": len({r["image"] for r in rows}), "findings": n,
                        "faithful_rate": round(sum(r["faithful"] for r in rows) / n, 3) if n else None,
                        "unstable_rate": round(sum(r["flip_rate"] > 0.25 for r in rows) / n, 3) if n else None,
                        "mean_drop": round(float(np.mean([r["drop"] for r in rows])), 3) if n else None, "rows": rows}
    Path(a.out).write_text(json.dumps(report, indent=1))
    print(json.dumps({k: {x: v[x] for x in v if x != "rows"} for k, v in report.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
