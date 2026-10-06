"""Batch reads to JSONL, resumable, for the second-reader evaluation and the offline demo.

    python -m services.medgemma.batch --dataset ham10000          # P2's eval batch, official test split
    python -m services.medgemma.batch --dataset fracatlas
    python -m services.medgemma.batch --images DIR --modality cxr --out reads.jsonl   # any folder

Rows are keyed by image id (dataset mode) or relative path (folder mode); re-running skips what is
already in the file, so a dropped notebook or a closed laptop loses nothing.
"""

import argparse
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

from PIL import Image

from services.medgemma.reader import Backend, read_image

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
DATASET_MODALITY = {"ham10000": "skin_dermoscopy", "fracatlas": "bone_xray"}
REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "ml" / "artifacts" / "medgemma_reads"
SHUFFLE_SEED = 20261006  # P2's seed; a fixed shuffle makes any prefix of the batch a random sample

Item = tuple[str, Path, dict]  # (id, image path, extra columns such as the true label)


def folder_items(images: Path) -> list[Item]:
    paths = [p for p in sorted(images.rglob("*")) if p.suffix.lower() in EXTS]
    return [(p.relative_to(images).as_posix(), p, {}) for p in paths]


def eval_items(dataset: str, splits_dir: Path | None = None, data_root: Path | None = None) -> list[Item]:
    """The images P2 fixed for the second-reader jobs (ml/data/eval_index.json), in a fixed random order."""
    from ml.data.common import image_path, load_eval_index, load_split

    spec = load_eval_index()["datasets"][dataset]
    df = load_split(dataset, splits_dir)
    sel = df[df["split"].isin(spec["test_splits"]) & (df["eval_batch"].astype(str) == "True")]
    return [
        (str(r["image_id"]), image_path(r, data_root), {"label": r["label"], "group": r.get("group")})
        for _, r in sel.sample(frac=1.0, random_state=SHUFFLE_SEED).iterrows()
    ]


def run_items(
    backend: Backend, items: Iterable[Item], out: Path, *, modality: str, limit: int | None = None
) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if out.exists():
        done = {json.loads(line)["image"] for line in out.read_text(encoding="utf-8").splitlines() if line.strip()}
    todo = [it for it in items if it[0] not in done]
    if limit is not None:
        todo = todo[:limit]
    with out.open("a", encoding="utf-8") as fh:
        for image_id, path, extra in todo:
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            res = read_image(backend, Image.open(path).convert("RGB"), modality)
            row = {"image": image_id, "sha256": sha, "modality": modality, **extra, **res.model_dump()}
            fh.write(json.dumps(row) + "\n")
            fh.flush()
    return len(todo)


def run_batch(
    backend: Backend, images: Path, out: Path, *, modality: str, limit: int | None = None
) -> int:
    return run_items(backend, folder_items(images), out, modality=modality, limit=limit)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=sorted(DATASET_MODALITY))
    ap.add_argument("--images", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--modality", default="cxr")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    if bool(args.dataset) == bool(args.images):
        ap.error("give exactly one of --dataset or --images")
    if args.dataset:
        items, modality = eval_items(args.dataset), DATASET_MODALITY[args.dataset]
        out = args.out or OUT_DIR / f"{args.dataset}.jsonl"
    else:
        if not args.out:
            ap.error("--out is required with --images")
        items, modality, out = folder_items(args.images), args.modality, args.out
    from services.medgemma.loader import TransformersBackend

    n = run_items(TransformersBackend.from_env(), items, out, modality=modality, limit=args.limit)
    print(f"read {n} images -> {out}")


if __name__ == "__main__":
    main()
