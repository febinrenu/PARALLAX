import { describe, expect, it } from "vitest";
import chest from "../../public/cases/chest/case.json";
import type { CaseFile } from "../api/types";
import type { Claim } from "../contracts";
import { chainFor, displayLabel, leadMarker, regionFailed, strokeFor } from "./findings";

const study = (chest as unknown as CaseFile).study;

describe("status is drawn as stroke style, not new colours", () => {
  it("uses yellow for verified and uncertain, red for discordant and rejected", () => {
    expect(strokeFor("verified").cssColor).toBe(strokeFor("uncertain").cssColor);
    expect(strokeFor("discordant").cssColor).toBe(strokeFor("rejected").cssColor);
    expect(strokeFor("verified").cssColor).not.toBe(strokeFor("discordant").cssColor);
  });

  it("dashes uncertainty", () => {
    expect(strokeFor("verified").dashed).toBe(false);
    expect(strokeFor("uncertain").dashed).toBe(true);
  });
});

describe("labels", () => {
  it("derives the lead marker from the patient-side region", () => {
    expect(leadMarker("right upper zone")).toBe("R");
    expect(leadMarker("left lower zone")).toBe("L");
    expect(leadMarker("cardiac region")).toBeNull();
  });

  it("puts labels in sentence case", () => {
    expect(displayLabel("Pleural_Thickening")).toBe("Pleural thickening");
    expect(displayLabel("Lung Opacity")).toBe("Lung opacity");
  });
});

describe("verification chain on the real baked chest case", () => {
  const pneumonia = study.findings.find((f) => f.label === "Pneumonia")!;

  it("passes reader, faithfulness and stability with real numbers", () => {
    const chain = chainFor(pneumonia, study, [], false);
    const by = Object.fromEntries(chain.map((s) => [s.stage, s.state]));
    expect(by.Reader).toBe("pass");
    expect(by.Faithfulness).toBe("pass");
    expect(by.Stability).toBe("pass");
  });

  it("marks stages that do not exist yet as unavailable after the run, pending during it", () => {
    expect(chainFor(pneumonia, study, [], false).find((s) => s.stage === "Second read")!.state).toBe("unavailable");
    expect(chainFor(pneumonia, study, [], true).find((s) => s.stage === "Second read")!.state).toBe("pending");
  });
});

describe("second read and firewall steps from P3's stages", () => {
  const base = study.findings.find((f) => f.label === "Pneumonia")!;
  const step = (f: typeof base, s: typeof study, name: string) => chainFor(f, s, [], false).find((x) => x.stage === name)!;
  const read = (agrees: boolean | null, box_iou: number | null) => ({ agrees, box_iou, label: "pneumonia", model: "medgemma", raw_ref: "r" });

  it("shows an unvalidated second read as inconclusive, never as agreement", () => {
    const f = { ...base, second_read: read(null, null), flags: [...(base.flags ?? []), "second_reader_disagrees"] };
    const s = step(f, study, "Second read");
    expect(s.state).toBe("unavailable");
    expect(s.detail).toMatch(/^Inconclusive/);
    expect(s.detail).toMatch(/audit flag/);
  });

  it("passes on agreement and fails on disagreement or a box overlap under 0.1", () => {
    expect(step({ ...base, second_read: read(true, 0.6) }, study, "Second read").state).toBe("pass");
    expect(step({ ...base, second_read: read(false, null) }, study, "Second read").state).toBe("fail");
    expect(step({ ...base, second_read: read(true, 0.05) }, study, "Second read").state).toBe("fail");
  });

  it("counts passed and blocked claims on the firewall step", () => {
    const claim = (id: string, blocked: string | null): Claim => ({ claim_id: id, template: "t", evidence_ids: ["e1"], rendered: "r", entailed: null, blocked_reason: blocked });
    expect(step(base, { ...study, claims: [claim("c1", null), claim("c2", "no evidence id")] }, "Firewall")).toMatchObject({ state: "pass", detail: "1 report claim passed, 1 blocked" });
    expect(step(base, { ...study, claims: [claim("c1", "diagnostic phrasing")] }, "Firewall").state).toBe("fail");
  });
});

describe("failed and untestable regions", () => {
  const base = study.findings.find((f) => f.label === "Pneumonia")!;
  const withEvidence = (patch: Partial<NonNullable<typeof base.image_evidence>[number]>) => ({
    ...base,
    image_evidence: [{ ...base.image_evidence![0], ...patch }, ...base.image_evidence!.slice(1)],
  });

  it("flags a region that failed the deletion test, even on a finding the notes support", () => {
    expect(regionFailed(withEvidence({ faithful: false }))).toBe(true);
    expect(regionFailed(withEvidence({ faithful: true }))).toBe(false);
    expect(regionFailed(withEvidence({ faithful: null }))).toBe(false);
  });

  it("explains that a box-only finding cannot take the region test", () => {
    const f = withEvidence({ faithful: null, faithfulness_drop: null, kind: "bbox" });
    const step = chainFor(f, study, [], false).find((s) => s.stage === "Faithfulness")!;
    expect(step.detail).toMatch(/^Not assessable/);
  });
});

describe("calibration step", () => {
  const base = study.findings.find((f) => f.label === "Pneumonia")!;
  const cal = (patch: Partial<typeof base>) => chainFor({ ...base, flags: [], ...patch }, study, [], false).find((s) => s.stage === "Calibration")!;

  it("fails when the calibrated model abstains", () => {
    expect(cal({ tier: "abstain", conformal_set: ["Pneumonia"] }).state).toBe("fail");
  });

  it("names the plausible set for multi-class models and says Platt for binary ones", () => {
    expect(cal({ conformal_set: ["glioma"], prob_calibrated: 0.95 }).detail).toBe("Plausible set: Glioma (calibrated probability 0.95)");
    expect(cal({ conformal_set: [], prob_calibrated: 0.89 }).detail).toBe("Platt-scaled, calibrated probability 0.89");
  });

  it("says so when the model is not calibrated", () => {
    expect(cal({ flags: ["uncalibrated"] }).state).toBe("unavailable");
  });
});
