"""Real-model smoke test: one CXR through the loaded model. Needs HF_TOKEN and a GPU.

    python -m services.medgemma.smoke --image services/medgemma/fixtures/smoke_cxr.png
"""

import argparse
import sys
import time

from pathlib import Path

from PIL import Image


def load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal .env reader so the smoke test works without extra packages."""
    import os

    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.partition("=")
            if sep and value.strip() and not key.startswith("#"):
                os.environ.setdefault(key.strip(), value.strip())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    args = ap.parse_args()
    load_dotenv()

    import torch

    from services.medgemma.loader import TransformersBackend
    from services.medgemma.reader import read_image

    t0 = time.perf_counter()
    backend = TransformersBackend.from_env()
    load_s = time.perf_counter() - t0
    res = read_image(backend, Image.open(args.image).convert("RGB"), "cxr")
    vram = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    print(
        f"model={backend.name} quant={backend.quant} load={load_s:.1f}s "
        f"read={res.ms}ms vram={vram:.2f}GiB"
    )
    print(f"ok={res.ok} labels={[x.name for x in res.labels]} boxes={len(res.boxes)}")
    print(f"impression={res.impression!r}\nwarnings={res.warnings}\nraw={res.raw[:600]!r}")
    if not (res.ok and res.impression):
        print("SMOKE FAILED", file=sys.stderr)
        return 1
    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
