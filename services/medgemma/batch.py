"""Batch reads to JSONL, resumable by image path.

    python -m services.medgemma.batch --images DIR --modality skin_dermoscopy \
        --out ml/artifacts/medgemma_reads/ham.jsonl
"""

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image

from services.medgemma.reader import Backend, read_image

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def run_batch(
    backend: Backend, images: Path, out: Path, *, modality: str, limit: int | None = None
) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if out.exists():
        done = {json.loads(line)["image"] for line in out.read_text().splitlines() if line.strip()}
    todo = [p for p in sorted(images.rglob("*")) if p.suffix.lower() in EXTS]
    todo = [p for p in todo if p.relative_to(images).as_posix() not in done]
    if limit is not None:
        todo = todo[:limit]
    with out.open("a", encoding="utf-8") as fh:
        for path in todo:
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            res = read_image(backend, Image.open(path).convert("RGB"), modality)
            row = {
                "image": path.relative_to(images).as_posix(),
                "sha256": sha,
                "modality": modality,
                **res.model_dump(),
            }
            fh.write(json.dumps(row) + "\n")
            fh.flush()
    return len(todo)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--modality", default="cxr")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    from services.medgemma.loader import TransformersBackend

    n = run_batch(
        TransformersBackend.from_env(), args.images, args.out, modality=args.modality, limit=args.limit
    )
    print(f"read {n} images -> {args.out}")


if __name__ == "__main__":
    main()
