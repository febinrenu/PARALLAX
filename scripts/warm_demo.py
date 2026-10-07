"""Run the four sample cases through a running API once, so a live demo is fast.

The first upload of an image pays for model loading and the faithfulness test (about 50 s for a
chest film on CPU). Faithfulness and stability are cached by image and modality, and the readers
stay loaded in the server process, so uploading the same files during the demo takes about 20 s.

    python scripts/warm_demo.py            # API on http://127.0.0.1:8000
    python scripts/warm_demo.py --api URL

Uses only the standard library. Images and notes come from web/public/cases/*/ (CC0 / CC BY /
public domain, see web/public/media/credits.json).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "web" / "public" / "cases"
ORDER = ["chest", "skin", "brain", "bone"]


def _multipart(fields: dict[str, str], file_field: str, path: Path) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; filename="{path.name}"\r\n'
        f"Content-Type: image/webp\r\n\r\n".encode()
    )
    parts.append(path.read_bytes())
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read())


def warm(api: str, case: str, timeout_s: float) -> str:
    meta = json.loads((CASES / case / "case.json").read_text(encoding="utf-8"))
    fields = {"modality_hint": meta["study"]["modality"]}
    note = (meta.get("notes") or {}).get("n1")
    if note:
        fields["notes"] = note
    body, ctype = _multipart(fields, "image", CASES / case / "image.webp")
    req = urllib.request.Request(f"{api}/studies", data=body, headers={"Content-Type": ctype}, method="POST")
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=30) as resp:
        study_id = json.loads(resp.read())["study_id"]
    while time.monotonic() - t0 < timeout_s:
        study = _get(f"{api}/studies/{study_id}")
        if "findings" in study:
            shown = [f["label"] for f in study["findings"] if f["status"] != "rejected"]
            return f"{time.monotonic() - t0:5.1f}s  {len(study['findings'])} findings, {len(shown)} shown, {len(study['claims'])} claims"
        if study.get("status") == "failed":
            return f"failed: {study.get('error')}"
        time.sleep(1)
    return f"still running after {timeout_s:.0f}s"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--timeout", type=float, default=300.0, help="seconds to wait per case")
    args = ap.parse_args()
    for _ in range(120):
        try:
            _get(f"{args.api}/ledger/verify")
            break
        except (urllib.error.URLError, OSError):
            time.sleep(0.5)
    else:
        print(f"API not reachable at {args.api}", file=sys.stderr)
        return 1
    print(f"Warming the demo cases against {args.api} ...")
    for case in ORDER:
        print(f"  {case:6s} {warm(args.api, case, args.timeout)}", flush=True)
    print("Done. Uploading these same files in the demo now uses the cached checks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
