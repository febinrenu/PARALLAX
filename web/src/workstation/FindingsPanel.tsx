import { useEffect, useRef, useState } from "react";
import type { Finding } from "../api/types";
import { chainFor, displayLabel, leadMarker, regionOf, strokeFor, TIER_WORD } from "../lib/findings";
import { ChainBar, ChainDetail } from "./VerificationChain";
import { selectFinding, useSession } from "./store";

function StrokeSwatch({ status }: { status: Finding["status"] }) {
  const s = strokeFor(status);
  return (
    <svg width="22" height="6" aria-hidden="true" className="shrink-0">
      <line x1="1" y1="3" x2="21" y2="3" stroke={s.cssColor} strokeWidth="2.5" strokeLinecap="round" strokeDasharray={s.dashed ? "4 4" : undefined} />
    </svg>
  );
}

export function FindingsPanel() {
  const findings = useSession((s) => s.findings);
  const selected = useSession((s) => s.selected);
  const study = useSession((s) => s.study);
  const stages = useSession((s) => s.stages);
  const streaming = useSession((s) => s.streaming);
  const mode = useSession((s) => s.mode);
  const decisions = useSession((s) => s.decisions);
  const annotations = useSession((s) => s.annotations);
  const error = useSession((s) => s.error);
  const firstStageMs = useSession((s) => s.firstStageMs);
  const listRef = useRef<HTMLOListElement>(null);
  const [showWithheld, setShowWithheld] = useState(false);
  // Rejected findings never reach the report (plan.md section 4); they stay inspectable here,
  // collapsed and labelled, so the doctor can see what was withheld and why.
  const reportable = findings.filter((f) => f.status !== "rejected");
  const withheld = findings.filter((f) => f.status === "rejected");
  useEffect(() => {
    if (selected && withheld.some((f) => f.finding_id === selected)) setShowWithheld(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected]);

  useEffect(() => {
    listRef.current?.querySelector<HTMLElement>(`[data-finding="${selected}"]`)?.scrollIntoView({ block: "nearest" });
  }, [selected]);

  const warnings = stages.flatMap((s) => s.warnings ?? []);

  return (
    <aside aria-label="Findings" className="flex min-h-0 flex-col border-l border-film-line/60 bg-film-panel">
      <div className="flex items-baseline justify-between px-4 pb-2 pt-3">
        <h2 className="text-[13px] font-medium text-ink">Findings</h2>
        {mode === "live" && (
          <span className="font-mono text-[11px] tabular text-ink-dim">
            {streaming ? `${stages.length} stages in` : firstStageMs != null ? `first stage ${firstStageMs} ms` : ""}
          </span>
        )}
      </div>

      {error && <p className="mx-4 mb-3 rounded-[var(--radius-control)] bg-pencil-red/10 px-3 py-2 text-[13px] text-ink">{error}</p>}

      <ol ref={listRef} className="min-h-0 flex-1 overflow-y-auto px-2 pb-3" data-lenis-prevent>
        {[...reportable, ...(showWithheld ? withheld : [])].map((f, i) => {
          const isSel = f.finding_id === selected;
          const region = regionOf(f);
          const marker = leadMarker(region);
          const steps = chainFor(f, study, stages, streaming);
          const decision = decisions[f.finding_id];
          const firstWithheld = f.status === "rejected" && i === reportable.length;
          return (
            <li key={f.finding_id} data-finding={f.finding_id}>
              {firstWithheld && <p className="px-2.5 pb-1 pt-3 text-[12px] text-ink-dim">Withheld from the report</p>}
              <button
                type="button"
                onClick={() => selectFinding(f.finding_id)}
                aria-current={isSel ? "true" : undefined}
                className={`w-full rounded-[var(--radius-panel)] px-2.5 py-2.5 text-left transition-colors duration-150 ${isSel ? "bg-film-raised" : "hover:bg-film-raised/50"}`}
              >
                <div className="flex items-center gap-2">
                  <StrokeSwatch status={f.status} />
                  <span className="text-[14px] font-medium text-ink">{displayLabel(f.label)}</span>
                  <span className="ml-auto font-mono text-[13px] tabular text-ink">{f.prob_calibrated.toFixed(2)}</span>
                </div>
                <div className="mt-1 flex items-center gap-2 pl-[30px] text-[12.5px] text-ink-dim">
                  {marker && <span className="grid size-4 place-items-center rounded-[3px] border border-ink-dim/70 font-mono text-[10px] text-ink">{marker}</span>}
                  <span className="truncate">
                    {region ?? "no named region"}, {TIER_WORD[f.tier]}
                  </span>
                </div>
                <div className="mt-2 flex items-center gap-3 pl-[30px]">
                  <ChainBar steps={steps} />
                  <span className={`text-[12px] ${f.status === "verified" ? "text-ink" : f.status === "uncertain" ? "text-ink-dim" : "text-pencil-red"}`}>{strokeFor(f.status).label}</span>
                  {decision && <span className="ml-auto text-[12px] text-ink-dim">{decision === "accept" ? "Accepted" : "Rejected"}</span>}
                </div>
                {f.conformal_set.length > 0 && (
                  <div className="mt-2 flex flex-wrap gap-1 pl-[30px]">
                    {f.conformal_set.map((l) => (
                      <span key={l} className="rounded-full border border-film-line px-2 py-0.5 text-[11.5px] text-ink-dim">{displayLabel(l)}</span>
                    ))}
                  </div>
                )}
                {isSel && <div className="pl-[30px]"><ChainDetail steps={steps} /></div>}
              </button>
            </li>
          );
        })}

        {withheld.length > 0 && (
          <li className="px-2.5 pt-3">
            <button type="button" onClick={() => setShowWithheld((v) => !v)} aria-expanded={showWithheld} className="text-left text-[12.5px] leading-snug text-ink-dim hover:text-ink">
              {showWithheld ? "Hide" : "Show"} {withheld.length} withheld {withheld.length === 1 ? "finding" : "findings"}
              <span className="mt-0.5 block text-ink-dim/80">
                {reportable.length === 0 && !streaming
                  ? "None of them has verified evidence: the region test failed or was not run (only the most probable findings are tested), and no note text supports them."
                  : "No verified image or note evidence."}
              </span>
            </button>
          </li>
        )}
        {!findings.length && mode !== "empty" && !streaming && (
          <li className="px-3 py-4 text-[13px] leading-relaxed text-ink-dim">
            {annotations.length ? "No model findings for this modality yet. The outline on the image is a dataset annotation, shown for reference." : "No findings were reported for this study."}
            {warnings.length > 0 && <span className="mt-2 block text-ink-dim/90">{warnings[warnings.length - 1]}</span>}
          </li>
        )}
        {!findings.length && streaming && <li className="px-3 py-4 text-[13px] text-ink-dim">Reading the image…</li>}
        {mode === "empty" && <li className="px-3 py-4 text-[13px] text-ink-dim">Findings appear here with the evidence behind each one.</li>}
      </ol>
    </aside>
  );
}
