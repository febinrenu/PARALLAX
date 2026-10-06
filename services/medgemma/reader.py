"""One image in, one validated ReadResult out. Never raises."""

import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from PIL import Image

from services.medgemma.schema import Box, Label, ReadResult, extract_json

PROMPTS = Path(__file__).resolve().parent / "prompts"


class Backend(Protocol):
    name: str
    quant: str

    def generate(self, image: Image.Image, prompt: str, max_new_tokens: int = 300) -> str: ...


@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    # Read once: a long batch must not depend on the working tree staying unchanged (a git rebase broke one).
    raw = (PROMPTS / f"{name}.md").read_text(encoding="utf-8")
    if raw.startswith("---\n"):
        raw = raw.split("\n---\n", 1)[1]
    return raw.strip()


def preload_prompts() -> None:
    """Read every prompt now, so later reads never touch the disk."""
    for path in PROMPTS.glob("*.md"):
        load_prompt(path.stem)


def build_prompt(modality: str, extra: str | None) -> str:
    if modality == "cxr":
        prompt = load_prompt("read_cxr")
    else:
        prompt = load_prompt("read_generic").replace("{modality}", modality.replace("_", " "))
    return f"{prompt}\n\nAdditional instruction from the system: {extra}" if extra else prompt


_HEDGE_PREFIX = re.compile(r"^(possible|probable|likely|suspected|questionable)\s+", re.IGNORECASE)


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union


def _clean_name(name: str) -> tuple[str, str | None]:
    """Move a leading hedge word out of a label name: 'possible nodule' -> ('nodule', 'possible')."""
    m = _HEDGE_PREFIX.match(name.strip())
    return (name.strip()[m.end():], m.group(1).lower()) if m else (name.strip(), None)


def _valid_box(box: list[int]) -> bool:
    if len(box) != 4 or not all(isinstance(v, int) and 0 <= v <= 1000 for v in box):
        return False
    return box[2] > box[0] and box[3] > box[1]


_REFUSAL = re.compile(
    r"unable to provide|cannot provide|can't provide|not able to provide|"
    r"i am an ai|as an ai|cannot give medical advice|not a substitute for",
    re.IGNORECASE,
)
_SECTIONS = re.compile(r"findings\s*:?\s*(?P<f>.*?)\s*impression\s*:?\s*(?P<i>.*)\Z", re.IGNORECASE | re.DOTALL)
_ANSWER_START = "<unused95>"  # Gemma thinking mode closes its reasoning with this token


class Refusal(ValueError):
    """The model declined to read the image."""


def _strip_thinking(text: str) -> str:
    """Drop a leading reasoning block; an unfinished one means there is no answer yet."""
    if "<unused94>" in text:
        if _ANSWER_START not in text:
            raise ValueError("model was still reasoning when output ended")
        text = text.split(_ANSWER_START, 1)[1]
    return text.strip()


def _parse_report(text: str) -> tuple[str, str]:
    """(findings_text, impression) from a plain FINDINGS / IMPRESSION report."""
    m = _SECTIONS.search(text)
    if m:
        return m.group("f").strip(), m.group("i").strip()
    if re.search(r"findings|impression", text, re.IGNORECASE) is None:
        raise ValueError("reply is neither JSON nor a FINDINGS/IMPRESSION report")
    tail = re.split(r"impression\s*:?", text, flags=re.IGNORECASE)
    return (text.strip(), tail[-1].strip()) if len(tail) > 1 else (text.strip(), "")


def _parse(text: str, modality: str) -> tuple[list[Label], list[Box], str, str, list[str]]:
    text = _strip_thinking(text)
    if _REFUSAL.search(text):
        raise Refusal("model refused to read the image")
    if "{" not in text:
        findings, impression = _parse_report(text)
        return [], [], impression, findings, ["labels not provided; derive them from the report text"]
    obj = extract_json(text)
    if not any(k in obj for k in ("labels", "impression", "boxes")):
        raise ValueError("reply has none of labels, boxes or impression")
    warnings: list[str] = []
    labels: list[Label] = []
    seen: set[str] = set()
    for raw_label in obj.get("labels", []):
        lab = Label.model_validate(raw_label)
        name, hedge = _clean_name(lab.name)
        if name.lower() in seen:
            continue
        seen.add(name.lower())
        labels.append(lab.model_copy(update={"name": name, "confidence_text": lab.confidence_text or hedge}))
    if len(labels) < len(obj.get("labels", [])):
        warnings.append("collapsed repeated labels")
    boxes: list[Box] = []
    for item in obj.get("boxes", []) if modality == "cxr" else []:
        raw_box = [round(v) if isinstance(v, float) else v for v in item.get("box", [])]
        if _valid_box(raw_box):
            a, b, c, d = raw_box
            name, _ = _clean_name(str(item.get("label", "")))
            if any(x.label == name and _iou(x.xyxy_norm, (a, b, c, d)) > 0.5 for x in boxes):
                continue
            boxes.append(Box(label=name, xyxy_norm=(a, b, c, d)))
        else:
            warnings.append(f"dropped invalid box for '{item.get('label', '?')}'")
    return labels, boxes, str(obj.get("impression", "")).strip(), "", warnings


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
                labels, boxes, impression, findings_text, warnings = _parse(text, modality)
            except Refusal as exc:
                base.error = f"refusal: {exc}"
                continue
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                base.error = f"unparseable output: {exc}"
                continue
            return base.model_copy(
                update={
                    "ok": True,
                    "labels": labels,
                    "boxes": boxes,
                    "impression": impression,
                    "findings_text": findings_text,
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
