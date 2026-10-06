"""Two-reader concordance (D3): compare a specialist finding with the MedGemma second read.

This stage only records the second reader's opinion on each finding (`second_read`) and adds
audit flags. Turning a disagreement into the `discordant` status is the status rule's job (P4).
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Hashable, Sequence

from medproof.core.schemas import Finding, SecondRead, StageResult
from medproof.readers.generalist import GeneralistRead
from medproof.verify.report_labels import ReportLabel

BoxXYXY = tuple[float, float, float, float]
MIN_IOU = 0.1  # plan.md status rule: below this the two readers localise different things


def box_iou(a: BoxXYXY, b: BoxXYXY) -> float:
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _positives(read: GeneralistRead) -> list[ReportLabel]:
    return [lab for lab in read.labels if lab.present]


def _agrees(read: GeneralistRead, label: str) -> bool | None:
    state = read.labels_state(label)
    if state in ("present", "hedged"):
        return True
    if state == "absent":
        return False
    if read.says_normal or _positives(read):
        return False  # the second reader looked and reported something else, or nothing abnormal
    return None  # silent: neither support nor contradiction


def _matching_label(read: GeneralistRead, label: str) -> str | None:
    if read.labels_state(label) in ("present", "hedged"):
        return label
    pos = _positives(read)
    return pos[0].label if pos else None


def _best_iou(finding: Finding, read: GeneralistRead) -> float | None:
    mine = [e.bbox_xyxy for e in finding.image_evidence if e.bbox_xyxy is not None]
    theirs = [b for name, b in read.boxes if name == finding.label]
    if not mine or not theirs:
        return None
    return max(box_iou(m, t) for m in mine for t in theirs)


def apply(findings: list[Finding], read: GeneralistRead) -> tuple[list[Finding], StageResult]:
    t0 = time.perf_counter()
    out: list[Finding] = []
    counts: Counter[str] = Counter()
    if not read.ok:
        for f in findings:
            out.append(f.model_copy(update={"flags": list(dict.fromkeys([*f.flags, "second_read_unavailable"]))}))
        why = read.warnings or ["second read unavailable"]
        stage = StageResult(
            stage="concordance", ok=False, ms=int((time.perf_counter() - t0) * 1000),
            payload={"source": read.source, "findings": len(findings)}, warnings=list(why),
        )
        return out, stage
    for f in findings:
        agrees = _agrees(read, f.label)
        iou = _best_iou(f, read)
        flags = list(f.flags)
        if agrees is False:
            flags.append("second_reader_disagrees")
        if iou is not None and iou < MIN_IOU:
            flags.append("second_box_mismatch")
        counts["agree" if agrees else "disagree" if agrees is False else "inconclusive"] += 1
        sr = SecondRead(model=read.model, label=_matching_label(read, f.label), agrees=agrees, box_iou=iou, raw_ref=read.raw_ref)
        out.append(f.model_copy(update={"second_read": sr, "flags": list(dict.fromkeys(flags))}))
    stage = StageResult(
        stage="concordance", ok=True, ms=int((time.perf_counter() - t0) * 1000),
        payload={"source": read.source, "findings": len(findings), "agree": counts["agree"],
                 "disagree": counts["disagree"], "inconclusive": counts["inconclusive"]},
        warnings=list(read.warnings),
    )
    return out, stage


def cohen_kappa(a: Sequence[Hashable], b: Sequence[Hashable]) -> float:
    """Cohen's kappa for two raters over any categories."""
    if len(a) != len(b):
        raise ValueError("raters must label the same items")
    n = len(a)
    if n == 0:
        raise ValueError("no items")
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb.get(k, 0) for k in ca) / (n * n)
    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1.0 - pe)
