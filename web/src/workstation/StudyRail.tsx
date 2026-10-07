import { useEffect, useId, useRef, useState } from "react";
import { Microphone, Stop, UploadSimple } from "@phosphor-icons/react";
import { loadCaseIndex, transcribe } from "../api/client";
import type { CaseIndexEntry, Modality } from "../api/types";
import { openSample, uploadStudy, useSession } from "./store";
import { NO_API_NOTE, useApiAvailable } from "./useApi";

const MODALITIES: { value: Modality | ""; label: string }[] = [
  { value: "cxr", label: "Chest X-ray" },
  { value: "bone_xray", label: "Bone X-ray" },
  { value: "brain_mri", label: "Brain MRI" },
  { value: "skin_dermoscopy", label: "Skin dermoscopy" },
  { value: "", label: "Not sure" },
];

type DictationState = { phase: "idle" | "recording" | "transcribing"; message: string | null };

/**
 * Record a dictation and append its transcript to the notes box (P3's voice path). The doctor
 * reviews the merged text before analysing; the audio itself never becomes part of the study.
 */
function useDictation(notes: string, setNotes: (s: string) => void) {
  const [state, setState] = useState<DictationState>({ phase: "idle", message: null });
  const recorder = useRef<MediaRecorder | null>(null);
  const supported = typeof window !== "undefined" && typeof window.MediaRecorder !== "undefined" && Boolean(navigator.mediaDevices?.getUserMedia);

  async function start() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const rec = new MediaRecorder(stream);
      const chunks: Blob[] = [];
      rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
      rec.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        setState({ phase: "transcribing", message: null });
        try {
          const out = await transcribe(new Blob(chunks, { type: rec.mimeType || "audio/webm" }), notes.trim());
          setNotes(out.note);
          setState({ phase: "idle", message: out.ok ? "Transcript added below your notes. Review it before analysing." : `Not transcribed: ${out.warnings[0] ?? "the audio service is unavailable"}.` });
        } catch (e) {
          setState({ phase: "idle", message: `Not transcribed: ${(e as Error).message}.` });
        }
      };
      recorder.current = rec;
      rec.start();
      setState({ phase: "recording", message: null });
    } catch {
      setState({ phase: "idle", message: "Microphone not available. Allow microphone access, or type the notes." });
    }
  }

  function stop() {
    recorder.current?.stop();
    recorder.current = null;
  }

  return { state, supported, start, stop };
}

export function StudyRail({ pendingFile, setPendingFile }: { pendingFile: File | null; setPendingFile: (f: File | null) => void }) {
  const [cases, setCases] = useState<CaseIndexEntry[]>([]);
  const [modality, setModality] = useState<Modality | "">("cxr");
  const [notes, setNotes] = useState("");
  const caseId = useSession((s) => s.caseId);
  const streaming = useSession((s) => s.streaming);
  const fileInput = useRef<HTMLInputElement>(null);
  const dictation = useDictation(notes, setNotes);
  const api = useApiAvailable();
  const ids = { file: useId(), modality: useId(), notes: useId(), help: useId() };

  useEffect(() => {
    loadCaseIndex().then(setCases).catch(() => setCases([]));
  }, []);

  const submit = () => {
    if (!pendingFile) return fileInput.current?.click();
    void uploadStudy(pendingFile, modality, notes.trim());
    setPendingFile(null);
  };

  return (
    <aside aria-label="Studies" className="flex h-full min-h-0 flex-col gap-5 overflow-y-auto border-r border-film-line/60 bg-film-panel px-3 py-3" data-lenis-prevent>
      <section>
        <h2 className="px-1 text-[13px] font-medium text-ink">Sample cases</h2>
        <ul className="mt-2 space-y-1">
          {cases.map((c) => (
            <li key={c.id}>
              <button
                type="button"
                onClick={() => void openSample(c.id)}
                aria-current={caseId === c.id ? "true" : undefined}
                className={`flex w-full items-center gap-2.5 rounded-[var(--radius-panel)] p-1.5 text-left transition-colors duration-150 ${caseId === c.id ? "bg-film-raised" : "hover:bg-film-raised/50"}`}
              >
                <img src={c.thumb} alt="" width={44} height={44} loading="lazy" decoding="async" className="size-11 shrink-0 rounded-[3px] bg-film-base object-cover" />
                <span className="text-[13px] leading-snug text-ink">{c.title}</span>
              </button>
            </li>
          ))}
        </ul>
      </section>

      <section aria-labelledby={ids.file}>
        <h2 id={ids.file} className="px-1 text-[13px] font-medium text-ink">Upload a study</h2>
        <div className="mt-2 space-y-3 px-1">
          <input
            ref={fileInput}
            type="file"
            aria-label="Choose a study file"
            tabIndex={-1}
            accept=".png,.jpg,.jpeg,.dcm,image/png,image/jpeg,application/dicom"
            className="sr-only"
            onChange={(e) => setPendingFile(e.target.files?.[0] ?? null)}
          />
          <button
            type="button"
            onClick={() => fileInput.current?.click()}
            className="flex w-full items-center gap-2 rounded-[var(--radius-control)] border border-dashed border-film-line px-2.5 py-2 text-left text-[13px] text-ink-dim hover:border-ink-dim hover:text-ink"
          >
            <UploadSimple size={16} />
            <span className="truncate">{pendingFile ? pendingFile.name : "Choose PNG, JPEG or DICOM"}</span>
          </button>

          <div className="space-y-1.5">
            <label htmlFor={ids.modality} className="block text-[12.5px] text-ink">Modality</label>
            <select
              id={ids.modality}
              value={modality}
              aria-describedby={ids.help}
              onChange={(e) => setModality(e.target.value as Modality | "")}
              className="w-full rounded-[var(--radius-control)] border border-film-line bg-film-base px-2 py-1.5 text-[13px] text-ink"
            >
              {MODALITIES.map((m) => (
                <option key={m.label} value={m.value}>{m.label}</option>
              ))}
            </select>
            <p id={ids.help} className="text-[12px] leading-snug text-ink-dim">Pick the modality, or Not sure to let the router decide (it needs the router model on the server).</p>
          </div>

          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <label htmlFor={ids.notes} className="block text-[12.5px] text-ink">Clinical notes (optional)</label>
              {dictation.supported && api !== false && (
                <button
                  type="button"
                  onClick={() => (dictation.state.phase === "recording" ? dictation.stop() : void dictation.start())}
                  disabled={dictation.state.phase === "transcribing" || streaming}
                  aria-pressed={dictation.state.phase === "recording"}
                  className={`flex items-center gap-1 rounded-[var(--radius-control)] px-1.5 py-0.5 text-[12px] disabled:opacity-50 ${dictation.state.phase === "recording" ? "text-pencil-red-ink" : "text-ink-dim hover:text-ink"}`}
                >
                  {dictation.state.phase === "recording" ? <Stop size={13} weight="fill" /> : <Microphone size={13} />}
                  {dictation.state.phase === "recording" ? "Stop" : dictation.state.phase === "transcribing" ? "Transcribing…" : "Dictate"}
                </button>
              )}
            </div>
            <textarea
              id={ids.notes}
              rows={3}
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              className="w-full resize-y rounded-[var(--radius-control)] border border-film-line bg-film-base px-2 py-1.5 text-[13px] leading-snug text-ink"
            />
            {dictation.state.message && (
              <p role="status" className="text-[12px] leading-snug text-ink-dim">
                {dictation.state.message}
              </p>
            )}
          </div>

          {api === false && <p className="text-[12px] leading-snug text-ink-dim">{NO_API_NOTE}</p>}
          <button
            type="button"
            onClick={submit}
            disabled={streaming || api === false}
            className="w-full rounded-[var(--radius-control)] bg-pencil-yellow px-3 py-2 text-[13px] font-medium text-lightbox-ink transition-transform duration-100 active:scale-[0.98] disabled:opacity-60"
          >
            {streaming ? "Analysing…" : pendingFile ? "Analyse study" : "Choose a file"}
          </button>
        </div>
      </section>
    </aside>
  );
}
