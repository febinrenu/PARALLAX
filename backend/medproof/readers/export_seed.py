"""Write MedGemma reads for demo images as small files that can be committed.

    MEDGEMMA_URL=http://localhost:8001 python -m medproof.readers.export_seed \
        --images demo/cases/cxr --modality cxr --out demo/medgemma_reads

Needs the MedGemma service once (a GPU, Kaggle or Colab). Afterwards set MEDGEMMA_SEED_DIR to the output
folder and the reader answers those images with no GPU, no tunnel and no cache, which is what the offline
demo needs. Images are matched by their SHA-256, so the files only apply to the exact same image.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from medproof.intake.decode import DecodeError, load_image
from medproof.readers.generalist import GeneralistReader

EXTS = {".png", ".jpg", ".jpeg", ".dcm", ".bmp", ".webp"}


def export_folder(reader: GeneralistReader, folder: Path, modality: str, out: Path) -> tuple[int, list[str]]:
    """Export a read for every image in `folder`. Returns (files written, problems)."""
    written, problems = 0, []
    for path in sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in EXTS):
        try:
            image = load_image(path.read_bytes())
            reader.export_seed(image, modality, out)
            written += 1
        except (DecodeError, ValueError, OSError) as exc:
            problems.append(f"{path.name}: {exc}")
    return written, problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--modality", required=True, choices=["cxr", "brain_mri", "skin_dermoscopy", "bone_xray", "other"])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    reader = GeneralistReader()
    written, problems = export_folder(reader, args.images, args.modality, args.out)
    print(f"wrote {written} read(s) to {args.out}")
    for p in problems:
        print("skipped:", p)
    return 0 if written and not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
