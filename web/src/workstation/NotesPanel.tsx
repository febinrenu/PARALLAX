import { Fragment, useMemo } from "react";
import type { StageResult } from "../api/types";
import { displayLabel } from "../lib/findings";
import { segments, type Span } from "../lib/notes";
import { hoverFinding, selectFinding, useSession } from "./store";

/** What P3's context stage reported for this study: quarantined spans per note and history prompts. */
function contextOutput(stages: StageResult[]) {
  const payload = (stages.find((s) => s.stage === "context")?.payload ?? {}) as {
    flagged_spans?: Record<string, [number, number][]>;
    missing_context?: string[];
  };
  return { flagged: payload.flagged_spans ?? {}, missing: payload.missing_context ?? [] };
}

export function NotesPanel() {
  const notes = useSession((s) => s.notes);
  const findings = useSession((s) => s.findings);
  const selected = useSession((s) => s.selected);
  const provenance = useSession((s) => s.provenance);
  const mode = useSession((s) => s.mode);
  const stages = useSession((s) => s.stages);

  const byNote = useMemo(() => {
    const m = new Map<string, Span[]>();
    const ordered = [...findings].sort((a, b) => Number(b.finding_id === selected) - Number(a.finding_id === selected));
    for (const f of ordered) for (const te of f.text_evidence ?? []) m.set(te.note_id, [...(m.get(te.note_id) ?? []), { te, finding: f }]);
    return m;
  }, [findings, selected]);
  const { flagged, missing } = useMemo(() => contextOutput(stages), [stages]);

  const ids = Object.keys(notes);
  const quarantinedCount = Object.values(flagged).reduce((n, spans) => n + spans.length, 0);
  return (
    <section aria-label="Clinical notes" className="min-h-0 overflow-y-auto border-t border-film-line/60 bg-film-panel px-4 py-3" data-lenis-prevent>
      <div className="flex items-baseline gap-3">
        <h2 className="text-[13px] font-medium text-ink">Notes</h2>
        {provenance.text_evidence && mode === "sample" && <p className="truncate text-[12px] text-ink-dim">{provenance.text_evidence}</p>}
      </div>
      {ids.length === 0 && (
        <p className="mt-2 text-[13px] text-ink-dim">{mode === "empty" ? "Notes you add with an upload appear here, with the facts that support each finding underlined." : "No notes were provided for this study."}</p>
      )}
      {ids.map((id) => (
        <p key={id} className="mt-2 max-w-[75ch] text-[15px] leading-relaxed text-ink">
          {segments(notes[id], byNote.get(id) ?? [], flagged[id] ?? []).map((seg, i) =>
            seg.kind === "text" ? (
              <Fragment key={i}>{seg.text}</Fragment>
            ) : seg.kind === "quarantine" ? (
              <mark key={i} title="Quarantined: read as data, never followed" className="rounded-[2px] bg-transparent text-ink-dim line-through decoration-pencil-red decoration-2 outline outline-1 outline-pencil-red/70">
                {seg.text}
              </mark>
            ) : (
              <button
                key={i}
                type="button"
                onMouseEnter={() => hoverFinding(seg.span.finding.finding_id)}
                onMouseLeave={() => hoverFinding(null)}
                onFocus={() => hoverFinding(seg.span.finding.finding_id)}
                onBlur={() => hoverFinding(null)}
                onClick={() => selectFinding(seg.span.finding.finding_id)}
                title={`${seg.span.te.polarity === "supports" ? "Supports" : seg.span.te.polarity === "contradicts" ? "Contradicts" : "Neutral for"} ${displayLabel(seg.span.finding.label)}`}
                className={`inline rounded-[2px] text-left underline decoration-2 underline-offset-4 transition-colors duration-100 ${
                  seg.span.te.polarity === "contradicts" ? "decoration-pencil-red" : seg.span.te.polarity === "supports" ? "decoration-pencil-yellow" : "decoration-ink-dim decoration-dotted"
                } ${seg.span.finding.finding_id === selected ? "bg-pencil-yellow/15" : "hover:bg-pencil-yellow/10"}`}
              >
                {seg.text}
              </button>
            ),
          )}
        </p>
      ))}
      {quarantinedCount > 0 && (
        <p className="mt-2 text-[12.5px] text-pencil-red-ink">
          Quarantined: {quarantinedCount === 1 ? "an instruction was" : `${quarantinedCount} instructions were`} found in the notes. It is read as data and never followed.
        </p>
      )}
      {missing.length > 0 && (
        <ul className="mt-3 space-y-1 border-t border-film-line/50 pt-2 text-[13px] text-ink-dim">
          {missing.map((m) => (
            <li key={m}>{m}</li>
          ))}
        </ul>
      )}
    </section>
  );
}
