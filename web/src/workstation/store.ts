import { useSyncExternalStore } from "react";
import { createStudy, getStudy, loadCase, openStudyEvents, postFeedback, studyImageUrl } from "../api/client";
import type { Annotation, CaseFile, Finding, Modality, StageResult, StudyResult } from "../api/types";

export interface Session {
  mode: "empty" | "sample" | "live";
  title: string;
  modality: Modality | null;
  caseId: string | null; // sample case id
  studyId: string | null; // live study id
  study: StudyResult | null;
  /** Findings to show: the final study's, or the reader's while the stream is still running. */
  findings: Finding[];
  stages: StageResult[];
  streaming: boolean;
  error: string | null;
  notes: Record<string, string>;
  image: { url: string; width: number; height: number } | null;
  annotations: Annotation[];
  anatomyUrl: string | null;
  provenance: Record<string, string>;
  selected: string | null;
  hoverFinding: string | null;
  decisions: Record<string, "accept" | "reject">;
  firstStageMs: number | null;
}

const EMPTY: Session = {
  mode: "empty",
  title: "",
  modality: null,
  caseId: null,
  studyId: null,
  study: null,
  findings: [],
  stages: [],
  streaming: false,
  error: null,
  notes: {},
  image: null,
  annotations: [],
  anatomyUrl: null,
  provenance: {},
  selected: null,
  hoverFinding: null,
  decisions: {},
  firstStageMs: null,
};

let state: Session = EMPTY;
const subs = new Set<() => void>();
let closeStream: (() => void) | null = null;
let loadToken = 0;

function set(patch: Partial<Session>) {
  state = { ...state, ...patch };
  subs.forEach((fn) => fn());
}

export function useSession<T>(select: (s: Session) => T): T {
  return useSyncExternalStore(
    (fn) => {
      subs.add(fn);
      return () => subs.delete(fn);
    },
    () => select(state),
    () => select(EMPTY),
  );
}

export const getSession = () => state;

// -- actions -------------------------------------------------------------------------------------
export async function openSample(id: string) {
  closeStream?.();
  const token = ++loadToken;
  try {
    const c: CaseFile = await loadCase(id);
    if (token !== loadToken) return;
    set({
      ...EMPTY,
      mode: "sample",
      title: c.title,
      modality: c.modality,
      caseId: c.id,
      study: c.study,
      findings: c.study.findings,
      notes: c.notes,
      image: { url: c.image.src, width: c.image.width, height: c.image.height },
      annotations: c.annotations,
      anatomyUrl: c.anatomy,
      provenance: c.provenance,
      selected: pickInitial(c.study.findings),
    });
  } catch (e) {
    if (token === loadToken) set({ ...EMPTY, error: `Could not open the sample case: ${(e as Error).message}` });
  }
}

/**
 * Sends the open sample case (its exact image bytes, modality and note) through the live pipeline.
 * Same bytes means the demo second read applies and the cached faithfulness and stability checks
 * are reused, so a rehearsed demo runs in seconds.
 */
export async function runSampleLive(): Promise<string | null> {
  const { mode, caseId, image, modality, notes, title } = state;
  if (mode !== "sample" || !caseId || !image) return null;
  try {
    const res = await fetch(image.url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const file = new File([await res.blob()], `${caseId}.webp`, { type: "image/webp" });
    await uploadStudy(file, modality ?? "", notes.n1 ?? "");
    if (state.mode === "live") set({ title: `${title}, live` });
    return null;
  } catch (e) {
    return `Could not start the live analysis: ${(e as Error).message}`;
  }
}

export async function uploadStudy(file: File, modalityHint: Modality | "", notes: string) {
  closeStream?.();
  const token = ++loadToken;
  const t0 = performance.now();
  set({
    ...EMPTY,
    mode: "live",
    title: file.name,
    modality: modalityHint || null,
    streaming: true,
    notes: notes ? { n1: notes } : {},
  });
  try {
    const res = await createStudy(file, { modalityHint: modalityHint || undefined, notes: notes || undefined });
    if (token !== loadToken) return;
    set({ studyId: res.study_id, image: { url: studyImageUrl(res.study_id), width: 0, height: 0 } });
    closeStream = openStudyEvents(res.study_id, {
      onStage: (s) => {
        if (token !== loadToken) return;
        const stages = [...state.stages, s];
        const readerFindings = (s.payload?.findings as Finding[] | undefined) ?? null;
        set({
          stages,
          firstStageMs: state.firstStageMs ?? Math.round(performance.now() - t0),
          findings: readerFindings ?? state.findings,
          selected: state.selected ?? (readerFindings ? pickInitial(readerFindings) : null),
        });
      },
      onDone: (study) => {
        if (token !== loadToken) return;
        const stillReportable = study.findings.some((f) => f.finding_id === state.selected && f.status !== "rejected");
        set({ study, findings: study.findings, streaming: false, selected: stillReportable ? state.selected : pickInitial(study.findings) });
      },
      onError: (message) => {
        if (token !== loadToken) return;
        set({ streaming: false, error: message });
        void getStudy(res.study_id).catch(() => undefined);
      },
    });
  } catch (e) {
    if (token === loadToken) set({ streaming: false, error: `Upload failed: ${(e as Error).message}. Check that the analysis server is running.` });
  }
}

export function selectFinding(id: string | null) {
  if (id !== state.selected) set({ selected: id });
}

export function stepFinding(delta: 1 | -1) {
  const reportable = state.findings.filter((f) => f.status !== "rejected");
  const list = reportable.length ? reportable : state.findings;
  if (!list.length) return;
  const i = list.findIndex((f) => f.finding_id === state.selected);
  const next = list[(i + delta + list.length) % list.length];
  set({ selected: next.finding_id });
}

export function hoverFinding(id: string | null) {
  if (id !== state.hoverFinding) set({ hoverFinding: id });
}

/** Returns the toast text, worded the same way as the audit log entry. */
export async function decide(decision: "accept" | "reject"): Promise<string | null> {
  const id = state.selected;
  if (!id) return null;
  const verb = decision === "accept" ? "accepted" : "rejected";
  set({ decisions: { ...state.decisions, [id]: decision } });
  if (state.mode === "live" && state.studyId) {
    try {
      const r = await postFeedback(state.studyId, { finding_id: id, decision });
      return `Finding ${verb}. Logged to ledger ${r.ledger_hash.slice(0, 8)}.`;
    } catch (e) {
      return `Finding ${verb} here, but logging failed: ${(e as Error).message}.`;
    }
  }
  return `Finding ${verb}. Sample cases are not logged; upload a study to record decisions in the ledger.`;
}

function pickInitial(findings: Finding[]): string | null {
  const order = { verified: 0, discordant: 1, uncertain: 2, rejected: 3 } as const;
  // Never open on a withheld finding: if nothing is reportable, show the plain image.
  const best = [...findings].filter((f) => f.status !== "rejected").sort((a, b) => order[a.status] - order[b.status] || b.prob_calibrated - a.prob_calibrated)[0];
  return best?.finding_id ?? null;
}
