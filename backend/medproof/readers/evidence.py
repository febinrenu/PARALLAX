"""Turn a ReaderOutput into contract Finding / ImageEvidence objects.

Reader-only status: a reader cannot verify its own output, so every finding leaves here as
"uncertain" with `faithful=None`. The faithfulness test, stability score, second reader and
the status rule decide whether it is upgraded. Calibration fields are placeholders until the
calibration stage replaces them.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from medproof.core.schemas import Finding, ImageEvidence
from medproof.readers.base import ReaderFinding, ReaderOutput
from medproof.readers.cxr_config import CxrConfig


def _write_heatmap(heat: np.ndarray, path: Path) -> None:
    u8 = np.rint(np.clip(heat, 0.0, 1.0) * 255.0).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), u8):
        raise OSError(f"could not write heatmap to {path}")


def to_findings(
    out: ReaderOutput,
    cfg: CxrConfig,
    *,
    finding_start: int = 1,
    evidence_start: int = 1,
    artifact_dir: str | Path | None = None,
) -> tuple[list[Finding], list[str]]:
    """Return (findings, warnings). Ids are f{n} and ie_{n}, counted from the given starts."""
    warnings = list(out.warnings)
    findings: list[Finding] = []
    fn, en = finding_start, evidence_start
    for rf in out.findings:
        box = rf.boxes_xyxy[0] if rf.boxes_xyxy else None
        if box is None and rf.heatmap is None:
            warnings.append(f"{rf.label}: no image location available, finding withheld")
            continue
        fid, eid = f"f{fn}", f"ie_{en}"
        heat_ref = None
        if artifact_dir is not None and rf.heatmap is not None:
            name = f"heatmap_{fid}.png"
            _write_heatmap(rf.heatmap, Path(artifact_dir) / name)
            heat_ref = f"{Path(artifact_dir).as_posix()}/{name}"
        evidence = ImageEvidence(
            evidence_id=eid,
            kind="heatmap" if rf.heatmap is not None else "bbox",
            bbox_xyxy=tuple(float(v) for v in box) if box is not None else None,
            heatmap_ref=heat_ref,
            source_model=out.model_id,
            method=rf.method,
            region_name=rf.region_name,
        )
        evidences = [evidence]
        if rf.mask_source and rf.mask is not None and artifact_dir is not None and rf.mask.any():
            mname = f"mask_{fid}.png"
            _write_heatmap(rf.mask.astype(np.float32), Path(artifact_dir) / mname)
            en += 1
            evidences.append(ImageEvidence(
                evidence_id=f"ie_{en}", kind="mask", bbox_xyxy=evidence.bbox_xyxy,
                mask_ref=f"{Path(artifact_dir).as_posix()}/{mname}", source_model=out.model_id, method=rf.mask_source,
                region_name=rf.region_name,
            ))
        flags = ["uncalibrated", *out.flags, *rf.flags]
        findings.append(
            Finding(
                finding_id=fid,
                modality=out.modality,
                label=rf.label,
                prob_raw=float(rf.prob_raw),
                prob_calibrated=float(rf.prob_raw),
                conformal_set=[],
                tier=cfg.tier(float(rf.prob_raw)),
                status="uncertain",
                image_evidence=evidences,
                flags=list(dict.fromkeys(flags)),
            )
        )
        fn += 1
        en += 1
    return findings, warnings
