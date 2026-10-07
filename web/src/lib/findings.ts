import type { Finding, StageResult, Status, StudyResult } from "../api/types";
import { gl as tokens, type Rgb } from "../design/tokens";

/** Status is drawn as stroke style, never as a new colour (plan.md section 10.1). */
export interface Stroke {
  color: Rgb;
  cssColor: string;
  dashed: boolean;
  label: string;
}

/** The finding's own image region failed the deletion test: where the model looked is not why it decided. */
export function regionFailed(f: Finding): boolean {
  return f.image_evidence?.[0]?.faithful === false;
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
export function chainFor(f: Finding, study: StudyResult | null, _stages: StageResult[], live: boolean): ChainStep[] {
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
      if (!ev)
        return f.image_evidence?.[0]?.kind === "bbox"
          ? missing("Not assessable: a box has no heatmap to delete and re-score")
          : missing("Not tested: only the most probable findings get the region test");
      const drop = ev.faithfulness_drop != null ? ` (confidence drop ${ev.faithfulness_drop.toFixed(2)})` : "";
      if (ev.faithful) return { state: "pass" as const, detail: `Deleting the region lowers confidence${drop}` };
      // P1's test fails a region whose drop is under 0.05, or no larger than deleting random regions.
      const bigDrop = (ev.faithfulness_drop ?? 0) >= 0.05;
      return { state: "fail" as const, detail: bigDrop ? `Deleting random regions lowers confidence just as much${drop}` : `Deleting the region barely changes confidence${drop}` };
    })(),
    Stability: f.stability
      ? f.stability.flip_rate <= 0.25
        ? { state: "pass", detail: `Flipped in ${Math.round(f.stability.flip_rate * f.stability.tests)} of ${f.stability.tests} perturbations` }
        : { state: "fail", detail: `Flipped in ${Math.round(f.stability.flip_rate * f.stability.tests)} of ${f.stability.tests} perturbations` }
      : missing("Stability not tested"),
    "Second read": (() => {
      const sr = f.second_read;
      if (!sr) return missing("Second reader not available yet");
      if (sr.agrees === false || (sr.box_iou != null && sr.box_iou < 0.1))
        return { state: "fail" as const, detail: `Second reader disagrees${sr.label ? `: ${sr.label}` : ""}` };
      if (sr.agrees === true) return { state: "pass" as const, detail: `Second reader agrees${sr.box_iou != null ? ` (box overlap ${sr.box_iou.toFixed(2)})` : ""}` };
      // Agreement is left empty where the second reader is not validated for this modality: its
      // read is kept as an audit flag and never changes the status.
      const noted = f.flags?.includes("second_reader_disagrees") ? "; its disagreement is logged as an audit flag" : "";
      return { state: "unavailable" as const, detail: `Inconclusive: the second reader is not validated for this modality${noted}` };
    })(),
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
    Firewall: (() => {
      const claims = study?.claims ?? [];
      if (!claims.length) return missing("Report not generated yet");
      const passed = claims.filter((c) => c.blocked_reason == null).length;
      const blocked = claims.length - passed;
      if (!passed) return { state: "fail" as const, detail: `All ${claims.length} report claims were blocked` };
      return { state: "pass" as const, detail: `${passed} report ${passed === 1 ? "claim" : "claims"} passed${blocked ? `, ${blocked} blocked` : ""}` };
    })(),
  };
  return CHAIN.map((stage) => ({ stage, ...steps[stage] }));
}
