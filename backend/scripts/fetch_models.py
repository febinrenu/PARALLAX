"""Put MedSigLIP and MedSAM in local folders and print the environment variables that point the app at them.

    python scripts/fetch_models.py [--dest DIR] [--only medsiglip|medsam] [--from-cache]

Downloads from the Hugging Face hub with HF_TOKEN (from the environment or the repo's git-ignored .env; never printed).
MedSigLIP is gated: the account behind the token must have accepted its terms. With --from-cache nothing is
downloaded: the folders are copied out of the local hub cache, which is how a machine that already holds the models
can hand them to another one (copy the dest folder and the two .npz files from backend/artifacts/).

Nothing here is committed: weights stay out of git.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

MODELS = {
    "medsiglip": ("google/medsiglip-448", "MEDPROOF_ROUTER_MODEL"),
    "medsam": ("flaviagiammarino/medsam-vit-base", "MEDPROOF_MEDSAM_MODEL"),
}
KEEP = ("*.json", "*.safetensors", "*.bin", "*.txt", "*.model", "*.pt")


def _token() -> str | None:
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    try:
        from medproof.intake.embedder import _hf_token

        return _hf_token()
    except Exception:
        return None


def fetch(name: str, dest: Path, from_cache: bool) -> Path:
    from huggingface_hub import snapshot_download

    repo, _ = MODELS[name]
    target = dest / name
    target.mkdir(parents=True, exist_ok=True)
    if from_cache:
        src = Path(snapshot_download(repo, local_files_only=True))
        for f in src.iterdir():
            shutil.copy2(f.resolve(), target / f.name)  # resolve: cache entries are links to blobs
    else:
        snapshot_download(repo, local_dir=str(target), token=_token(), allow_patterns=list(KEEP))
    return target


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=str(Path(__file__).resolve().parents[1] / "models"), help="parent folder (default backend/models, git-ignored)")
    ap.add_argument("--only", choices=sorted(MODELS), default=None)
    ap.add_argument("--from-cache", action="store_true")
    a = ap.parse_args(argv)
    dest = Path(a.dest)
    lines = []
    for name in ([a.only] if a.only else sorted(MODELS)):
        try:
            path = fetch(name, dest, a.from_cache)
        except Exception as exc:
            print(f"{name}: failed ({type(exc).__name__}). Gated model? Accept its terms on the hub with the token's account.", file=sys.stderr)
            return 1
        lines.append(f"{MODELS[name][1]}={path.resolve()}")
        print(f"{name}: {path}")
    print("\nAdd to the root .env (or the shell):")
    print("\n".join(lines))
    print("and keep router_probe.npz and ood.npz under backend/artifacts/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
