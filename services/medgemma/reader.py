"""One image in, one validated ReadResult out. Never raises."""

import time
from pathlib import Path
from typing import Protocol

from PIL import Image

from services.medgemma.schema import Box, Label, ReadResult, extract_json

PROMPTS = Path(__file__).resolve().parent / "prompts"


class Backend(Protocol):
    name: str
    quant: str

    def generate(self, image: Image.Image, prompt: str, max_new_tokens: int = 700) -> str: ...


def load_prompt(name: str) -> str:
    raw = (PROMPTS / f"{name}.md").read_text(encoding="utf-8")
    if raw.startswith("---\n"):
        raw = raw.split("\n---\n", 1)[1]
    return raw.strip()


def build_prompt(modality: str, extra: str | None) -> str:
    if modality == "cxr":
        prompt = load_prompt("read_cxr")
    else:
        prompt = load_prompt("read_generic").replace("{modality}", modality.replace("_", " "))
    return f"{prompt}\n\nAdditional instruction from the system: {extra}" if extra else prompt


def _valid_box(box: list[int]) -> bool:
    if len(box) != 4 or not all(isinstance(v, int) and 0 <= v <= 1000 for v in box):
        return False
    return box[2] > box[0] and box[3] > box[1]


def _parse(text: str, modality: str) -> tuple[list[Label], list[Box], str, list[str]]:
    obj = extract_json(text)
    labels = [Label.model_validate(x) for x in obj.get("labels", [])]
    warnings: list[str] = []
    boxes: list[Box] = []
    for item in obj.get("boxes", []) if modality == "cxr" else []:
        raw_box = [round(v) if isinstance(v, float) else v for v in item.get("box", [])]
        if _valid_box(raw_box):
            a, b, c, d = raw_box
            boxes.append(Box(label=str(item.get("label", "")), xyxy_norm=(a, b, c, d)))
        else:
            warnings.append(f"dropped invalid box for '{item.get('label', '?')}'")
    return labels, boxes, str(obj.get("impression", "")).strip(), warnings


def read_image(
    backend: Backend, image: Image.Image, modality: str, extra: str | None = None
) -> ReadResult:
    t0 = time.perf_counter()
    base = ReadResult(ok=False, model=backend.name, quant=backend.quant)
    prompt = build_prompt(modality, extra)
    text = ""
    try:
        for attempt in range(2):
            text = backend.generate(
                image, prompt if attempt == 0 else f"{prompt}\n\n{load_prompt('repair')}"
            )
            try:
                labels, boxes, impression, warnings = _parse(text, modality)
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                base.error = f"unparseable output: {exc}"
                continue
            return base.model_copy(
                update={
                    "ok": True,
                    "labels": labels,
                    "boxes": boxes,
                    "impression": impression,
                    "raw": text,
                    "error": None,
                    "warnings": warnings,
                    "ms": int((time.perf_counter() - t0) * 1000),
                }
            )
    except Exception as exc:  # noqa: BLE001 - backend failures (OOM, driver) must not take the service down
        base.error = f"backend error: {exc}"
    base.raw = text
    base.ms = int((time.perf_counter() - t0) * 1000)
    return base
