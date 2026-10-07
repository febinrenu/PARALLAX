import { primary } from "./data";

/** Demo ledger for the landing. Payloads hold only strings, integers and booleans, so the
 * browser hashes them exactly as core/ledger.py would (see lib/pyjson.ts). */
export const LEDGER_FRAMES = [
  { event: "stage_completed", actor: "system", title: "Intake", body: "Decoded, quality gate passed", payload: { stage: "intake", ok: true, ms: 177 } },
  { event: "stage_completed", actor: "system", title: "Reader", body: `${primary.label} located, ${primary.region}`, payload: { stage: "reader", label: primary.label, region: primary.region ?? "" } },
  { event: "stage_completed", actor: "system", title: "Faithfulness", body: "Deleting the region drops confidence", payload: { stage: "faithfulness", verdict: "faithful" } },
  { event: "stage_completed", actor: "system", title: "Stability", body: "Held in 7 of 8 perturbations", payload: { stage: "stability", held: 7, tests: 8 } },
  { event: "finding_feedback", actor: "dr_ines", title: "Doctor", body: "Finding accepted", payload: { finding_id: primary.id, decision: "accept" } },
];

export const LEDGER_TS0 = "2026-10-07T09:14:02+00:00";
export const tsFor = (i: number) => LEDGER_TS0.replace(":02+", `:${String(2 + i * 3).padStart(2, "0")}+`);
