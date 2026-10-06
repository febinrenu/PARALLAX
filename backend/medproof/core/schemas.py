# Contract types (plan.md section 4). Copied verbatim from the frozen sketch so intake
# can build against it; the platform owner finalizes this file and may replace it.
from typing import Literal

from pydantic import BaseModel, Field

Modality = Literal["cxr", "brain_mri", "skin_dermoscopy", "bone_xray", "other"]
Status = Literal["verified", "uncertain", "discordant", "rejected"]


class ImageEvidence(BaseModel):
    evidence_id: str
    kind: Literal["heatmap", "mask", "bbox"]
    bbox_xyxy: tuple[float, float, float, float] | None = None
    mask_ref: str | None = None
    heatmap_ref: str | None = None
    source_model: str
    method: str
    region_name: str | None = None
    faithfulness_drop: float | None = None
    faithful: bool | None = None


class TextEvidence(BaseModel):
    evidence_id: str
    note_id: str
    span: tuple[int, int]
    quote: str
    fact_type: str
    polarity: Literal["supports", "contradicts", "neutral"]


class SecondRead(BaseModel):
    model: str
    label: str | None
    agrees: bool | None
    box_iou: float | None
    raw_ref: str


class Stability(BaseModel):
    tests: int
    flip_rate: float
    worst_perturbation: str | None


class Precedent(BaseModel):
    case_id: str
    dataset: str
    label: str
    similarity: float
    thumb_ref: str


class Finding(BaseModel):
    finding_id: str
    modality: Modality
    label: str
    prob_raw: float
    prob_calibrated: float
    conformal_set: list[str]
    coverage_target: float = 0.90
    tier: Literal["high", "moderate", "low", "abstain"]
    status: Status
    image_evidence: list[ImageEvidence] = []
    text_evidence: list[TextEvidence] = []
    second_read: SecondRead | None = None
    stability: Stability | None = None
    precedents: list[Precedent] = []
    flags: list[str] = []


class Claim(BaseModel):
    claim_id: str
    template: str
    evidence_ids: list[str] = Field(min_length=1)
    rendered: str | None = None
    entailed: bool | None = None
    blocked_reason: str | None = None


class StageResult(BaseModel):
    stage: str
    ok: bool
    ms: int
    payload: dict
    warnings: list[str] = []


class StudyResult(BaseModel):
    study_id: str
    input_sha256: str
    modality: Modality
    ood_score: float
    quality: dict
    findings: list[Finding]
    claims: list[Claim]
    ledger_head: str
    disclaimer: str
