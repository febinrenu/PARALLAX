import type { Finding, StageResult, StudyResult, TextEvidence } from "../contracts";

export type { Finding, StageResult, StudyResult, TextEvidence };
export type Modality = StudyResult["modality"];
export type Status = Finding["status"];

/** A reference annotation shipped with a sample case (dataset ground truth, never model output). */
export interface Annotation {
  kind: "bbox" | "mask";
  bbox?: [number, number, number, number]; // normalised xyxy
  ref?: string;
  label: string;
  source: string;
}

export interface CaseIndexEntry {
  id: string;
  title: string;
  modality: Modality;
  thumb?: string;
}

export interface CaseFile extends CaseIndexEntry {
  image: { src: string; width: number; height: number };
  credit: string;
  notes: Record<string, string>;
  annotations: Annotation[];
  anatomy: string | null;
  study: StudyResult;
  provenance: Record<string, string>;
  warnings: string[];
}

export interface Credit {
  id: string;
  title: string;
  author: string;
  license: string;
  license_url: string;
  page: string;
  modifications: string;
}

export interface LedgerEntry {
  index: number;
  prev_hash: string;
  hash: string;
  ts: string;
  actor: string;
  event: string;
  study_id: string | null;
  payload_hash: string;
  payload: Record<string, unknown>;
}

export interface CreateStudyResponse {
  study_id: string;
  events_url: string;
  status_url: string;
}

export type StudyStatus =
  | { status: "running"; study_id: string; stages_completed: string[] }
  | { status: "failed"; study_id: string; error: string };
