"""Two-reader concordance (D3): compare a specialist finding with the MedGemma second read.

This stage records the second reader's opinion on each finding (`second_read`) and adds audit
flags. P4's status rule turns `second_read.agrees is False` (or a box IoU below 0.1) into the
`discordant` status, so this stage decides whether a disagreement may reach those fields at all:
plan 9.4 says a flag must be shown to predict errors before it may downgrade anything.

Measured on the cached batches (reports/concordance.json): on bone the flag predicts detector
errors but 83% of the detector's correct fracture reports are flagged too (the second reader names
a fracture on about 4% of images), and on skin the difference in error rate is not significant. No
modality is validated, so a disagreement is recorded as an audit flag and `agrees` stays None.
Add a modality to DOWNGRADE_VALIDATED only with a report showing the flag is usable.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Hashable, Sequence

from medproof.core.schemas import Finding, SecondRead, StageResult
from medproof.readers.generalist import GeneralistRead
from medproof.verify.report_labels import ReportLabel, canonical_label

BoxXYXY = tuple[float, float, float, float]
MIN_IOU = 0.1  # plan.md status rule: below this the two readers localise different things
DOWNGRADE_VALIDATED: frozenset[str] = frozenset()  # modalities where second-reader disagreement may demote a finding


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


def _best_iou(finding: Finding, read: GeneralistRead, name: str) -> float | None:
    mine = [e.bbox_xyxy for e in finding.image_evidence if e.bbox_xyxy is not None]
    theirs = [b for box_name, b in read.boxes if box_name == name]
    if not mine or not theirs:
        return None
    return max(box_iou(m, t) for m in mine for t in theirs)


def apply(
    findings: list[Finding], read: GeneralistRead, *, downgrade_modalities: frozenset[str] | None = None
) -> tuple[list[Finding], StageResult]:
    """Attach the second read. `downgrade_modalities` defaults to DOWNGRADE_VALIDATED."""
    validated = DOWNGRADE_VALIDATED if downgrade_modalities is None else downgrade_modalities
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
        name = canonical_label(f.modality, f.label)
        agrees = _agrees(read, name)
        iou = _best_iou(f, read, name)
        flags = list(f.flags)
        if agrees is False:
            flags.append("second_reader_disagrees")
        if iou is not None and iou < MIN_IOU:
            flags.append("second_box_mismatch")
        counts["agree" if agrees else "disagree" if agrees is False else "inconclusive"] += 1
        if f.modality not in validated:
            # audit only: the status rule reads these two fields, so they must not carry the disagreement
            agrees = True if agrees else None
            iou = None
        sr = SecondRead(model=read.model, label=_matching_label(read, name), agrees=agrees, box_iou=iou, raw_ref=read.raw_ref)
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
