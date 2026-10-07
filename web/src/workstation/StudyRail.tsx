import { useEffect, useId, useRef, useState } from "react";
import { UploadSimple } from "@phosphor-icons/react";
import { loadCaseIndex } from "../api/client";
import type { CaseIndexEntry, Modality } from "../api/types";
import { openSample, uploadStudy, useSession } from "./store";

const MODALITIES: { value: Modality | ""; label: string }[] = [
  { value: "cxr", label: "Chest X-ray" },
  { value: "bone_xray", label: "Bone X-ray" },
  { value: "brain_mri", label: "Brain MRI" },
  { value: "skin_dermoscopy", label: "Skin dermoscopy" },
  { value: "", label: "Not sure" },
];

export function StudyRail({ pendingFile, setPendingFile }: { pendingFile: File | null; setPendingFile: (f: File | null) => void }) {
  const [cases, setCases] = useState<CaseIndexEntry[]>([]);
  const [modality, setModality] = useState<Modality | "">("cxr");
  const [notes, setNotes] = useState("");
  const caseId = useSession((s) => s.caseId);
  const streaming = useSession((s) => s.streaming);
  const fileInput = useRef<HTMLInputElement>(null);
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
            <p id={ids.help} className="text-[12px] leading-snug text-ink-dim">Pick the modality. Automatic routing needs the router model, which is not installed on this machine.</p>
          </div>

          <div className="space-y-1.5">
            <label htmlFor={ids.notes} className="block text-[12.5px] text-ink">Clinical notes (optional)</label>
            <textarea
              id={ids.notes}
              rows={3}
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              className="w-full resize-y rounded-[var(--radius-control)] border border-film-line bg-film-base px-2 py-1.5 text-[13px] leading-snug text-ink"
            />
          </div>

          <button
            type="button"
            onClick={submit}
            disabled={streaming}
            className="w-full rounded-[var(--radius-control)] bg-pencil-yellow px-3 py-2 text-[13px] font-medium text-lightbox-ink transition-transform duration-100 active:scale-[0.98] disabled:opacity-60"
          >
            {streaming ? "Analysing…" : pendingFile ? "Analyse study" : "Choose a file"}
          </button>
        </div>
      </section>
    </aside>
  );
}
