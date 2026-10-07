/* AUTO-GENERATED from contracts/*.schema.json by scripts/gen_contracts_ts.mjs.
 * Do not edit by hand — run `make contracts` after changing backend/medproof/core/schemas.py. */

export interface StudyResult {
  claims: Claim[];
  disclaimer: string;
  findings: Finding[];
  input_sha256: string;
  ledger_head: string;
  modality: "cxr" | "brain_mri" | "skin_dermoscopy" | "bone_xray" | "other";
  ood_score: number;
  quality: {
    [k: string]: unknown;
  };
  study_id: string;
}
export interface Claim {
  blocked_reason?: string | null;
  claim_id: string;
  entailed?: boolean | null;
  /**
   * @minItems 1
   */
  evidence_ids: [string, ...string[]];
  rendered?: string | null;
  template: string;
}
export interface Finding {
  conformal_set: string[];
  coverage_target?: number;
  finding_id: string;
  flags?: string[];
  image_evidence?: ImageEvidence[];
  label: string;
  modality: "cxr" | "brain_mri" | "skin_dermoscopy" | "bone_xray" | "other";
  precedents?: Precedent[];
  prob_calibrated: number;
  prob_raw: number;
  second_read?: SecondRead | null;
  stability?: Stability | null;
  status: "verified" | "uncertain" | "discordant" | "rejected";
  text_evidence?: TextEvidence[];
  tier: "high" | "moderate" | "low" | "abstain";
}
export interface ImageEvidence {
  bbox_xyxy?: [number, number, number, number] | null;
  evidence_id: string;
  faithful?: boolean | null;
  faithfulness_drop?: number | null;
  heatmap_ref?: string | null;
  kind: "heatmap" | "mask" | "bbox";
  mask_ref?: string | null;
  method: string;
  region_name?: string | null;
  source_model: string;
}
export interface Precedent {
  case_id: string;
  dataset: string;
  label: string;
  similarity: number;
  thumb_ref: string;
}
export interface SecondRead {
  agrees: boolean | null;
  box_iou: number | null;
  label: string | null;
  model: string;
  raw_ref: string;
}
export interface Stability {
  flip_rate: number;
  tests: number;
  worst_perturbation: string | null;
}
export interface TextEvidence {
  evidence_id: string;
  fact_type: string;
  note_id: string;
  polarity: "supports" | "contradicts" | "neutral";
  quote: string;
  /**
   * @minItems 2
   * @maxItems 2
   */
  span: [number, number];
}

export interface StageResult {
  ms: number;
  ok: boolean;
  payload: {
    [k: string]: unknown;
  };
  stage: string;
  warnings?: string[];
}
