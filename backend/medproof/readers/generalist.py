"""Client for the MedGemma read service: the independent second reader.

Always returns a GeneralistRead and never raises. When the service is down, slow or refuses, the
result is marked unavailable and the study carries on without a second read (plan.md 7.4, P4.3).
Replies are cached on disk by (image sha256, modality, prompt version), so evaluation and the
offline demo never depend on a live tunnel.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from diskcache import Cache  # type: ignore[import-untyped]
from PIL import Image

from medproof.intake.decode import DecodedImage
from medproof.verify.report_labels import ReportLabel, labels_from_report

PROMPT_VERSION = "v4"  # keep in step with services/medgemma/prompts/*.md


@dataclass
class GeneralistRead:
    ok: bool
    modality: str
    source: str  # "live" | "cache" | "seed" | "unavailable"
    model: str = ""
    impression: str = ""
    findings_text: str = ""
    labels: list[ReportLabel] = field(default_factory=list)
    says_normal: bool = False
    boxes: list[tuple[str, tuple[float, float, float, float]]] = field(default_factory=list)  # label, xyxy pixels
    raw_ref: str = ""
    warnings: list[str] = field(default_factory=list)

    def labels_state(self, label: str) -> str | None:
        """'present', 'hedged', 'absent' or None when the report never mentions the label."""
        for lab in self.labels:
            if lab.label == label:
                return "absent" if not lab.present else ("hedged" if lab.hedged else "present")
        return None


def norm_box_to_pixels(
    box: tuple[float, float, float, float], *, height: int, width: int
) -> tuple[float, float, float, float]:
    """Service boxes are x_min, y_min, x_max, y_max on a 0..1000 grid; return pixels on the display image."""
    x0, y0, x1, y1 = box
    return (x0 / 1000.0 * width, y0 / 1000.0 * height, x1 / 1000.0 * width, y1 / 1000.0 * height)


def _png(display: np.ndarray) -> bytes:
    arr = display
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return buf.getvalue()


class GeneralistReader:
    def __init__(
        self,
        url: str | None = None,
        *,
        cache_dir: Path | str = Path(".cache/medgemma"),
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
        prompt_version: str = PROMPT_VERSION,
        env: Mapping[str, str] | None = None,
        seed_dir: Path | str | None = None,
    ) -> None:
        env = os.environ if env is None else env
        seed = seed_dir or env.get("MEDGEMMA_SEED_DIR")
        self.seed_dir = Path(seed) if seed else None
        self.url = (url or env.get("MEDGEMMA_URL") or "").rstrip("/")
        self.prompt_version = prompt_version
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        self._cache = Cache(str(cache_dir))
        self._client = httpx.Client(timeout=timeout, transport=transport)

    def close(self) -> None:
        self._client.close()
        self._cache.close()

    def read(self, image: DecodedImage, modality: str) -> GeneralistRead:
        key = hashlib.sha256(f"{image.sha256}|{modality}|{self.prompt_version}".encode()).hexdigest()
        raw_ref = f"medgemma:{key[:16]}"
        cached = self._cache.get(key)
        if cached is not None:
            return self._build(cached, modality, image.shape, "cache", raw_ref)
        shipped = self._load_seed(image.sha256, modality)
        if shipped is not None:
            return self._build(shipped, modality, image.shape, "seed", raw_ref)
        if not self.url:
            return self._unavailable(modality, raw_ref, "MEDGEMMA_URL is not set")
        try:
            resp = self._client.post(
                f"{self.url}/read",
                files={"image": ("image.png", _png(image.display), "image/png")},
                data={"modality": modality},
            )
            resp.raise_for_status()
            body: dict[str, Any] = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            return self._unavailable(modality, raw_ref, type(exc).__name__)
        if not body.get("ok"):
            return self._unavailable(modality, raw_ref, str(body.get("error") or "the service returned no read"))
        self._cache.set(key, body)
        return self._build(body, modality, image.shape, "live", raw_ref)

    def _seed_path(self, directory: Path, sha256: str, modality: str) -> Path:
        return directory / f"{sha256}.{modality}.{self.prompt_version}.json"

    def _load_seed(self, sha256: str, modality: str) -> dict[str, Any] | None:
        """A read shipped with the repo for a demo image; ignored when missing, corrupt or for another prompt."""
        if self.seed_dir is None:
            return None
        path = self._seed_path(self.seed_dir, sha256, modality)
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return body if isinstance(body, dict) and body.get("ok") else None

    def export_seed(self, image: DecodedImage, modality: str, directory: Path | str) -> Path:
        """Write this image's read (from the cache or the live service) as a small committable file."""
        read = self.read(image, modality)
        if not read.ok:
            raise ValueError(f"no usable read to export: {read.warnings}")
        key = hashlib.sha256(f"{image.sha256}|{modality}|{self.prompt_version}".encode()).hexdigest()
        body = self._cache.get(key) or self._load_seed(image.sha256, modality)
        if body is None:
            raise ValueError("the read is not in the cache")
        out = self._seed_path(Path(directory), image.sha256, modality)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(body, indent=1), encoding="utf-8")
        return out

    @staticmethod
    def _unavailable(modality: str, raw_ref: str, why: str) -> GeneralistRead:
        return GeneralistRead(
            ok=False, modality=modality, source="unavailable", raw_ref=raw_ref,
            warnings=[f"second read unavailable: {why}"],
        )

    @staticmethod
    def _build(
        body: dict[str, Any], modality: str, shape: tuple[int, int], source: str, raw_ref: str
    ) -> GeneralistRead:
        h, w = shape
        names = [str(x.get("name", "")) for x in body.get("labels", []) if x.get("present", True)]
        text = " ".join(
            part
            for part in (body.get("findings_text", ""), ("Findings: " + "; ".join(names) + ".") if names else "", body.get("impression", ""))
            if part
        )
        parsed = labels_from_report(text, modality)
        boxes = [
            (str(b.get("label", "")), norm_box_to_pixels(tuple(b["xyxy_norm"]), height=h, width=w))
            for b in body.get("boxes", [])
        ]
        return GeneralistRead(
            ok=True, modality=modality, source=source, model=str(body.get("model", "")),
            impression=str(body.get("impression", "")), findings_text=str(body.get("findings_text", "")),
            labels=parsed.labels, says_normal=parsed.says_normal, boxes=boxes, raw_ref=raw_ref,
            warnings=list(body.get("warnings", [])),
        )
