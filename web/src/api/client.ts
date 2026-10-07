import type { CaseFile, CaseIndexEntry, CreateStudyResponse, Credit, LedgerEntry, StageResult, StudyResult, StudyStatus } from "./types";

/** The FastAPI server, behind Vite's dev/preview proxy (and the production reverse proxy). */
export const API = "/api";

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = String(body.detail);
    } catch {
      /* body was not JSON */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

/** Overlay refs from the API are `/studies/...`; refs from sample cases are static `/cases/...`. */
export function assetUrl(ref: string): string {
  return ref.startsWith("/studies/") ? `${API}${ref}` : ref;
}

export async function createStudy(file: File, opts: { modalityHint?: string; notes?: string } = {}): Promise<CreateStudyResponse> {
  const body = new FormData();
  body.append("image", file);
  if (opts.modalityHint) body.append("modality_hint", opts.modalityHint);
  if (opts.notes) body.append("notes", opts.notes);
  return json(await fetch(`${API}/studies`, { method: "POST", body }));
}

export interface TranscribeResult {
  ok: boolean;
  note: string;
  transcript: string;
  language: string;
  duration_s: number | null;
  warnings: string[];
}

/** Dictation in, the typed note with the transcript appended out. The audio is not stored. */
export async function transcribe(audio: Blob, typed: string): Promise<TranscribeResult> {
  const body = new FormData();
  body.append("audio", audio, audio.type.includes("ogg") ? "dictation.ogg" : audio.type.includes("mp4") ? "dictation.m4a" : "dictation.webm");
  if (typed) body.append("typed", typed);
  return json(await fetch(`${API}/transcribe`, { method: "POST", body }));
}

export async function getStudy(id: string): Promise<StudyResult | StudyStatus> {
  return json(await fetch(`${API}/studies/${encodeURIComponent(id)}`));
}

export const studyImageUrl = (id: string) => `${API}/studies/${encodeURIComponent(id)}/image`;

/** Full-depth luminance: float16 little-endian, plus the overlay grid's original size. */
export async function getPixels(id: string, signal?: AbortSignal) {
  const res = await fetch(`${API}/studies/${encodeURIComponent(id)}/pixels`, { signal });
  if (!res.ok) throw new Error(`pixels: HTTP ${res.status}`);
  const width = Number(res.headers.get("x-width"));
  const height = Number(res.headers.get("x-height"));
  const original: [number, number] = [Number(res.headers.get("x-original-width")), Number(res.headers.get("x-original-height"))];
  const data = new Uint16Array(await res.arrayBuffer());
  if (data.length !== width * height) throw new Error("pixels: size mismatch");
  return { data, width, height, original };
}

export async function postFeedback(id: string, body: { finding_id: string; decision: "accept" | "reject"; actor?: string; note?: string }) {
  return json<{ ledger_hash: string; ledger_head: string }>(
    await fetch(`${API}/studies/${encodeURIComponent(id)}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  );
}

export async function getStudyLedger(id: string) {
  return json<{ entries: LedgerEntry[]; head: string }>(await fetch(`${API}/studies/${encodeURIComponent(id)}/ledger`));
}

export async function verifyLedger() {
  return json<{ ok: boolean; entries: number; head?: string; broken_at?: number; reason?: string }>(await fetch(`${API}/ledger/verify`));
}

export async function getMetrics() {
  return json<Record<string, unknown>>(await fetch(`${API}/metrics`));
}

export async function getModelCards() {
  return json<{ cards: Record<string, unknown>[]; note?: string }>(await fetch(`${API}/model-cards`));
}

export interface StudyEventHandlers {
  onStage: (s: StageResult) => void;
  onDone: (s: StudyResult) => void;
  onError: (message: string) => void;
}

/** Subscribe to a study's SSE stream. Returns a function that closes it. */
export function openStudyEvents(id: string, h: StudyEventHandlers): () => void {
  const es = new EventSource(`${API}/studies/${encodeURIComponent(id)}/events`);
  let finished = false;
  es.addEventListener("stage", (e) => h.onStage(JSON.parse((e as MessageEvent).data)));
  es.addEventListener("done", (e) => {
    finished = true;
    h.onDone(JSON.parse((e as MessageEvent).data));
    es.close();
  });
  es.addEventListener("error", (e) => {
    const data = (e as MessageEvent).data;
    if (typeof data === "string" && data) {
      finished = true;
      h.onError(data); // the pipeline itself reported a failure
      es.close();
    } else if (!finished) {
      h.onError("Lost the connection to the analysis server. Check that it is running, then try again.");
      es.close();
    }
  });
  return () => {
    finished = true;
    es.close();
  };
}

export async function loadCaseIndex(): Promise<CaseIndexEntry[]> {
  return json(await fetch("/cases/index.json"));
}

export async function loadCase(id: string): Promise<CaseFile> {
  return json(await fetch(`/cases/${encodeURIComponent(id)}/case.json`));
}

export async function loadCredits(): Promise<Credit[]> {
  return json(await fetch("/media/credits.json"));
}
