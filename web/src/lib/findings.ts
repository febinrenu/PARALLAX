import type { Finding, StageResult, Status, StudyResult } from "../api/types";
import { gl as tokens, type Rgb } from "../design/tokens";

/** Status is drawn as stroke style, never as a new colour (plan.md section 10.1). */
export interface Stroke {
  color: Rgb;
  cssColor: string;
  dashed: boolean;
  label: string;
}

export function strokeFor(status: Status): Stroke {
  switch (status) {
    case "verified":
      return { color: tokens.pencilYellow, cssColor: "var(--color-pencil-yellow)", dashed: false, label: "Verified" };
    case "uncertain":
      return { color: tokens.pencilYellow, cssColor: "var(--color-pencil-yellow)", dashed: true, label: "Uncertain" };
    case "discordant":
      return { color: tokens.pencilRed, cssColor: "var(--color-pencil-red)", dashed: false, label: "Discordant" };
    case "rejected":
      return { color: tokens.pencilRed, cssColor: "var(--color-pencil-red)", dashed: true, label: "Rejected" };
  }
}

/** Lead-marker letter for a patient-side region name ("right upper zone" gives "R"). */
export function leadMarker(region: string | null | undefined): "R" | "L" | null {
  if (!region) return null;
  if (/^right\b/i.test(region)) return "R";
  if (/^left\b/i.test(region)) return "L";
  return null;
}

export function regionOf(f: Finding): string | null {
  return f.image_evidence?.find((e) => e.region_name)?.region_name ?? null;
}

export function boxOf(f: Finding, imageSize: [number, number]): [number, number, number, number] | null {
  const b = f.image_evidence?.find((e) => e.bbox_xyxy)?.bbox_xyxy;
  if (!b) return null;
  const [w, h] = imageSize;
  return [b[0] / w, b[1] / h, b[2] / w, b[3] / h];
}

export function heatmapOf(f: Finding): string | null {
  return f.image_evidence?.find((e) => e.heatmap_ref)?.heatmap_ref ?? null;
}

export function maskOf(f: Finding): string | null {
  return f.image_evidence?.find((e) => e.mask_ref)?.mask_ref ?? null;
}

export const TIER_WORD: Record<Finding["tier"], string> = {
  high: "high confidence",
  moderate: "moderate confidence",
  low: "low confidence",
  abstain: "abstained",
};

/** Sentence-case display label: "Lung Opacity" stays, "Pleural_Thickening" becomes "Pleural thickening". */
export function displayLabel(label: string): string {
  const s = label.replace(/_/g, " ").toLowerCase();
  return s.charAt(0).toUpperCase() + s.slice(1);
}

// -- verification chain ------------------------------------------------------------------------
export const CHAIN = ["Intake", "Reader", "Faithfulness", "Stability", "Second read", "Context", "Calibration", "Firewall"] as const;
export type ChainStage = (typeof CHAIN)[number];
export type ChainState = "pass" | "fail" | "pending" | "unavailable";

export interface ChainStep {
  stage: ChainStage;
  state: ChainState;
  detail: string;
}

/**
 * Where each finding stands on the verification chain. `live` is true while the SSE stream is
 * still running: stages that have not reported yet are "pending" rather than "unavailable".
 */
export function chainFor(f: Finding, study: StudyResult | null, stages: StageResult[], live: boolean): ChainStep[] {
  const seen = new Set(stages.map((s) => s.stage));
  const quality = (study?.quality ?? {}) as { passed?: boolean };
  const missing = (detail: string): Pick<ChainStep, "state" | "detail"> =>
    live ? { state: "pending", detail: "Waiting for this stage" } : { state: "unavailable", detail };

  const steps: Record<ChainStage, Pick<ChainStep, "state" | "detail">> = {
    Intake:
      quality.passed === false
        ? { state: "fail", detail: "Image quality gate raised a failure" }
        : { state: "pass", detail: "Decoded and passed the quality gate" },
    Reader: f.image_evidence?.length
      ? { state: "pass", detail: `Located: ${regionOf(f) ?? "region on the image"}` }
      : { state: "fail", detail: "No image location" },
    Faithfulness: (() => {
      const ev = f.image_evidence?.find((e) => e.faithful !== null && e.faithful !== undefined);
      if (!ev) return missing("Faithfulness test not run for this finding");
      const drop = ev.faithfulness_drop != null ? ` (confidence drop ${ev.faithfulness_drop.toFixed(2)})` : "";
      return ev.faithful ? { state: "pass" as const, detail: `Deleting the region lowers confidence${drop}` } : { state: "fail" as const, detail: `Deleting the region barely changes confidence${drop}` };
    })(),
    Stability: f.stability
      ? f.stability.flip_rate <= 0.25
        ? { state: "pass", detail: `Flipped in ${Math.round(f.stability.flip_rate * f.stability.tests)} of ${f.stability.tests} perturbations` }
        : { state: "fail", detail: `Flipped in ${Math.round(f.stability.flip_rate * f.stability.tests)} of ${f.stability.tests} perturbations` }
      : missing("Stability not tested"),
    "Second read": f.second_read
      ? f.second_read.agrees === false || (f.second_read.box_iou ?? 1) < 0.1
        ? { state: "fail", detail: `Second reader disagrees${f.second_read.label ? `: ${f.second_read.label}` : ""}` }
        : { state: "pass", detail: "Second reader agrees" }
      : missing("Second reader not available yet"),
    Context: f.text_evidence?.length
      ? f.text_evidence.some((t) => t.polarity === "contradicts")
        ? { state: "fail", detail: "Notes contradict this finding" }
        : f.text_evidence.some((t) => t.polarity === "supports")
          ? { state: "pass", detail: "Notes support this finding" }
          : { state: "unavailable", detail: "Notes are neutral" }
      : missing("No supporting note text"),
    Calibration: f.flags?.includes("uncalibrated")
      ? { state: "unavailable", detail: "Not calibrated yet" }
      : f.conformal_set.length > 2
        ? { state: "fail", detail: `Conformal set has ${f.conformal_set.length} labels` }
        : { state: "pass", detail: "Calibrated, conformal set is small" },
    Firewall: study?.claims?.length ? { state: "pass", detail: "Report claims passed the firewall" } : missing("Report not generated yet"),
  };
  void seen; // stage names map onto chain steps once more stages exist; today only intake and reader report
  return CHAIN.map((stage) => ({ stage, ...steps[stage] }));
}
