"""Smoke-check the trained brain, skin and bone models once their weight files are in place.

    python scripts/check_trained_models.py [--images DIR] [--sam PATH]

Looks for ml/artifacts/<name>/weights.pt (the registry lists the expected hashes; weights are never committed).
For each model that is present it loads the reader through the same code the pipeline uses, runs a few images
from DIR/<modality>/ (default data/router, as staged by stage_router_data.py), and prints probabilities, findings,
faithfulness and stability. Missing weights are reported, not treated as failures.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "ml" / "artifacts"
SAMPLES = {"brain_cls": "brain_mri", "brain_seg": "brain_mri", "skin_cls": "skin_dermoscopy", "bone_det": "bone_xray"}


def _images(base: Path, modality: str, n: int):
    from medproof.intake.decode import load_image

    files = sorted(glob.glob(str(base / modality / "**" / "*.*"), recursive=True))[:n]
    return [(Path(f).name, load_image(f)) for f in files]


def _report(name: str, reader, imgs, kind: str):
    from medproof.verify import faithfulness as FA
    from medproof.verify import stability as ST

    for fname, img in imgs:
        out = reader.predict(img)
        top = [(f.label, round(f.prob_raw, 3)) for f in out.findings]
        print(f"  {fname}: probs={np.round(out.probs, 3).tolist()} findings={top}")
        if out.findings and kind == "classifier":
            fa = FA.assess_output(reader, img, out, FA.FaithfulnessConfig(n_random=9))
            st = ST.assess_output(reader, img, out)
            for k in fa:
                print(f"    {k}: faithful={fa[k].faithful} drop10={fa[k].drop} flip_rate={st[k].flip_rate} worst={st[k].worst_perturbation}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=str(ROOT / "backend" / "data" / "router"))
    ap.add_argument("--sam", default=None, help="local MedSAM folder (default: download from the hub)")
    ap.add_argument("-n", type=int, default=3)
    ap.add_argument("--artifacts", default=str(ARTIFACTS), help="folder with <model>/model_meta.json and weights.pt")
    a = ap.parse_args(argv)
    base, artifacts = Path(a.images), Path(a.artifacts)
    status = {}
    from medproof.readers import bone, classifier, segmenter

    for name in ("brain_cls", "skin_cls", "brain_seg", "bone_det"):
        d = artifacts / name
        meta = d / "model_meta.json"
        if not meta.is_file():
            status[name] = "no model_meta.json"
            continue
        if not (d / "weights.pt").is_file():
            status[name] = "weights.pt missing: place the training output in ml/artifacts/%s/" % name
            continue
        imgs = _images(base, SAMPLES[name], a.n)
        if not imgs:
            status[name] = f"no sample images under {base / SAMPLES[name]}"
            continue
        print(f"== {name}")
        try:
            if name in ("brain_cls", "skin_cls"):
                modality = SAMPLES[name]
                reader = classifier.ImageClassifierReader.from_dir(d, modality=modality, positive_threshold=0.0)
                _report(name, reader, imgs, "classifier")
                if name == "skin_cls":
                    from medproof.segment.medsam import MedSAM

                    sam = MedSAM.load(model_id=a.sam) if a.sam else MedSAM.load()
                    for fname, img in imgs[:1]:
                        out = reader.predict(img)
                        segmenter.attach_medsam_masks(out, img, sam)
                        print(f"  MedSAM masks on {fname}: {[(f.label, int(f.mask.sum()) if f.mask is not None else 0) for f in out.findings]}")
            elif name == "brain_seg":
                seg = segmenter.UNetSegmenter.from_dir(d)
                for fname, img in imgs:
                    m = seg.segment(img)
                    print(f"  {fname}: tumour mask covers {m.mean():.3%} of the slice")
            else:
                reader = bone.BoneReader.from_dir(d)
                _report(name, reader, imgs, "detector")
            status[name] = "ok"
        except Exception as exc:  # report and keep checking the others
            status[name] = f"FAILED: {type(exc).__name__}: {str(exc)[:160]}"
    print(json.dumps(status, indent=1))
    return 0 if all(v == "ok" or "missing" in v or v.startswith("no ") for v in status.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
