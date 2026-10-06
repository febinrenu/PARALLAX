"""Strict output types for the MedGemma service and a tolerant JSON extractor."""

import json
from typing import Literal

from pydantic import BaseModel, Field

Modality = Literal["cxr", "brain_mri", "skin_dermoscopy", "bone_xray", "other"]


class Label(BaseModel):
    name: str
    present: bool = True
    confidence_text: str | None = None


class Box(BaseModel):
    label: str
    # x_min, y_min, x_max, y_max on a 0..1000 grid; pixel conversion is the reader's job.
    xyxy_norm: tuple[int, int, int, int]


class ReadResult(BaseModel):
    ok: bool
    model: str = ""
    quant: str = ""
    labels: list[Label] = Field(default_factory=list)
    boxes: list[Box] = Field(default_factory=list)
    impression: str = ""
    findings_text: str = ""  # report-style reply; labels are derived from it downstream
    raw: str = ""
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)
    ms: int = 0


def extract_json(text: str) -> dict:
    """Return the first complete JSON object in `text`, ignoring braces inside strings.

    A reply that is cut off before its outer object closes raises instead of returning an
    inner object, which would look like a valid but empty result.
    """
    start = text.find("{")
    while start != -1:
        depth, in_str, esc, end = 0, False, False, -1
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end == -1:
            raise ValueError("model output ends before its JSON object closes (truncated)")
        try:
            obj = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict):
            return obj
        start = text.find("{", end + 1)
    raise ValueError("no JSON object found in model output")
