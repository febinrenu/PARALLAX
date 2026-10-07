import { Fragment, useMemo } from "react";
import type { Finding, TextEvidence } from "../api/types";
import { displayLabel } from "../lib/findings";
import { hoverFinding, selectFinding, useSession } from "./store";

interface Span {
  te: TextEvidence;
  finding: Finding;
}

/** Split note text into plain runs and evidence spans. Notes render as text, never as HTML. */
function segments(text: string, spans: Span[]) {
  const sorted = [...spans].sort((a, b) => a.te.span[0] - b.te.span[0]);
  const out: ({ kind: "text"; text: string } | { kind: "span"; text: string; span: Span })[] = [];
  let at = 0;
  for (const s of sorted) {
    const [a, b] = s.te.span;
    if (a < at || b > text.length) continue; // overlapping or out-of-range spans are skipped, not mangled
    if (a > at) out.push({ kind: "text", text: text.slice(at, a) });
    out.push({ kind: "span", text: text.slice(a, b), span: s });
    at = b;
  }
  if (at < text.length) out.push({ kind: "text", text: text.slice(at) });
  return out;
}

export function NotesPanel() {
  const notes = useSession((s) => s.notes);
  const findings = useSession((s) => s.findings);
  const selected = useSession((s) => s.selected);
  const provenance = useSession((s) => s.provenance);
  const mode = useSession((s) => s.mode);

  const byNote = useMemo(() => {
    const m = new Map<string, Span[]>();
    for (const f of findings) for (const te of f.text_evidence ?? []) m.set(te.note_id, [...(m.get(te.note_id) ?? []), { te, finding: f }]);
    return m;
  }, [findings]);

  const ids = Object.keys(notes);
  return (
    <section aria-label="Clinical notes" className="min-h-0 overflow-y-auto border-t border-film-line/60 bg-film-panel px-4 py-3" data-lenis-prevent>
      <div className="flex items-baseline gap-3">
        <h2 className="text-[13px] font-medium text-ink">Notes</h2>
        {provenance.text_evidence && <p className="truncate text-[12px] text-ink-dim">{provenance.text_evidence}</p>}
      </div>
      {ids.length === 0 && (
        <p className="mt-2 text-[13px] text-ink-dim">{mode === "empty" ? "Notes you add with an upload appear here, with the facts that support each finding underlined." : "No notes were provided for this study."}</p>
      )}
      {ids.map((id) => (
        <p key={id} className="mt-2 max-w-[75ch] text-[15px] leading-relaxed text-ink">
          {segments(notes[id], byNote.get(id) ?? []).map((seg, i) =>
            seg.kind === "text" ? (
              <Fragment key={i}>{seg.text}</Fragment>
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
    </section>
  );
}
