"""Builds the precedent index for a dataset from its TRAIN split and scores it on the test batch (P3.8).

    python -m ml.eval_p3.build_retrieval --dataset fracatlas
    python -m ml.eval_p3.build_retrieval --dataset ham10000 --embedder biomedclip

The index holds training images only, so a precedent can never be a test image. Each query leaves out
its own id and its own lesion/patient group. Metrics: precision@5 (share of the 5 precedents whose
true label equals the query's), how that compares with picking 5 images by chance, and the accuracy
of a majority vote over the 5 (a kNN classifier built from the same index), with cluster-bootstrap
intervals. Writes reports/retrieval_<dataset>_<embedder>.json and a few demo precedents with thumbnails.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))

from medproof.intake.decode import load_image  # noqa: E402
from medproof.retrieval.embedder import embed_images  # noqa: E402
from medproof.retrieval.index import PrecedentIndex, to_precedents  # noqa: E402

from ml.data.common import image_path, load_eval_index, load_split  # noqa: E402
from ml.eval.bootstrap import ci  # noqa: E402

OUT = ROOT / "ml" / "artifacts" / "retrieval"
# RSNA has no train split (the chest model is pretrained), so its reference set is the calibration split; queries come
# from the test split, and the split is by image with each image its own patient, after near-duplicates were removed.
INDEX_SPLIT = {"rsna": "cal"}
K = 5


def make_embedder(name: str):  # noqa: ANN201
    if name == "biomedclip":
        from medproof.retrieval.embedder import OpenClipEmbedder

        return OpenClipEmbedder()
    from medproof.retrieval.embedder import MedSigLipEmbedder

    return MedSigLipEmbedder()


def open_image(path: Path) -> Image.Image:
    """PIL for ordinary images; the product's own decoder for DICOM (rescale, windowing, MONOCHROME1)."""
    if Path(path).suffix.lower() == ".dcm":
        return Image.fromarray(load_image(Path(path).read_bytes()).display).convert("RGB")
    return Image.open(path)


def items_for(df) -> list:  # noqa: ANN001
    return [(str(r["image_id"]), (lambda p=image_path(r): open_image(p))) for _, r in df.iterrows()]


def thumbnail(src: Path, dst: Path, size: int = 160) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    im = open_image(src).convert("RGB")
    im.thumbnail((size, size))
    im.save(dst, "JPEG", quality=80)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["fracatlas", "ham10000", "brain_mri", "rsna"], required=True)
    ap.add_argument("--embedder", choices=["biomedclip", "medsiglip"], default="biomedclip")
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--limit-test", type=int, default=None)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args(argv)

    df = load_split(args.dataset)
    spec = load_eval_index()["datasets"][args.dataset]
    train = df[(df["split"] == INDEX_SPLIT.get(args.dataset, "train")) & (df["dropped_reason"].isna())]
    test = df[df["split"].isin(spec["test_splits"]) & (df["eval_batch"].astype(str) == "True")]
    if args.limit_train:
        train = train.sample(n=min(args.limit_train, len(train)), random_state=1)
    if args.limit_test:
        test = test.sample(n=min(args.limit_test, len(test)), random_state=1)
    # keep only images that exist on this machine (the HAM10000 training zip may still be downloading)
    train = train[[image_path(r).is_file() for _, r in train.iterrows()]]
    emb = make_embedder(args.embedder)
    print(f"embedding {len(train)} train and {len(test)} test images with {emb.name}", flush=True)

    ids, vecs, skipped = embed_images(emb, items_for(train), args.batch)
    labels = dict(zip(train["image_id"].astype(str), train["label"], strict=True))
    groups = dict(zip(train["image_id"].astype(str), train["group"].astype(str), strict=True))
    index = PrecedentIndex.build(args.dataset, emb.name, ids, [labels[i] for i in ids], vecs, [groups[i] for i in ids])
    index.save(OUT / f"{args.dataset}_{emb.name}")

    qids, qvecs, qskipped = embed_images(emb, items_for(test), args.batch)
    qlabel = dict(zip(test["image_id"].astype(str), test["label"], strict=True))
    qgroup = dict(zip(test["image_id"].astype(str), test["group"].astype(str), strict=True))
    neighbours = [index.search(v, K, exclude_ids=[i], exclude_groups=[qgroup[i]]) for i, v in zip(qids, qvecs, strict=True)]
    y = np.array([qlabel[i] for i in qids], dtype=object)
    per_query = np.array([np.mean([h.label == yl for h in hits]) if hits else 0.0 for hits, yl in zip(neighbours, y, strict=True)])
    vote = np.array([Counter(h.label for h in hits).most_common(1)[0][0] if hits else "none" for hits in neighbours], dtype=object)
    classes = sorted(set(y))
    prior = Counter(index.labels)
    n_idx = sum(prior.values())
    chance = float(sum((c / n_idx) ** 2 for c in prior.values())) if n_idx else float("nan")
    g = np.array([qgroup[i] for i in qids], dtype=object)

    def fmt(c: dict) -> dict:
        return {k: round(c[k], 4) for k in ("point", "lo", "hi")}

    def bal_acc(t, p):  # noqa: ANN001
        return float(np.mean([(p[t == c] == c).mean() for c in classes if (t == c).any()]))

    report = {
        "dataset": args.dataset, "embedder": emb.name, "k": K,
        "index_size": len(index), "queries": len(qids), "skipped_unreadable": len(skipped) + len(qskipped),
        "train_only": True, "excludes_query_id_and_group": True,
        "precision_at_5": fmt(ci({"p": per_query}, lambda p: float(p.mean()), groups=g, B=1000)),
        "precision_at_5_if_chosen_by_chance": round(chance, 4),
        "majority_vote_accuracy": fmt(ci({"t": y, "p": vote}, lambda t, p: float((t == p).mean()), groups=g, B=1000)),
        "majority_vote_balanced_accuracy": fmt(ci({"t": y, "p": vote}, bal_acc, groups=g, strata=y, B=1000)),
        "precision_at_5_by_class": {c: round(float(per_query[y == c].mean()), 4) for c in classes},
        "train_label_counts": dict(prior),
        "note": ("BiomedCLIP, the ungated fallback embedder" if emb.name == "biomedclip" else "MedSigLIP, the plan's preferred embedder"),
    }
    demo = []
    for i, hits in list(zip(qids, neighbours, strict=True))[:5]:
        precedents = to_precedents(hits, args.dataset, lambda h: f"ml/artifacts/retrieval/thumbs/{args.dataset}/{h.id}.jpg")
        for p, h in zip(precedents, hits, strict=True):
            src = image_path(train[train["image_id"].astype(str) == h.id].iloc[0])
            try:
                thumbnail(src, ROOT / p.thumb_ref)
            except OSError:
                pass  # an unreadable image gets no thumbnail; the precedent is still listed
        demo.append({"query": i, "query_label": qlabel[i], "precedents": [p.model_dump() for p in precedents]})
    report["demo"] = demo
    (ROOT / "reports").mkdir(exist_ok=True)
    path = ROOT / "reports" / f"retrieval_{args.dataset}_{emb.name}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "demo"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
