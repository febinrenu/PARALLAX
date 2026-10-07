import { Check, FileText, Scroll, X } from "@phosphor-icons/react";
import { Link } from "react-router";
import { decide, useSession } from "./store";
import { toast } from "./Toasts";

export async function runDecision(decision: "accept" | "reject") {
  const message = await decide(decision);
  if (message) toast(message);
}

export function ActionBar() {
  const selected = useSession((s) => s.selected);
  const mode = useSession((s) => s.mode);
  const studyId = useSession((s) => s.studyId);
  const caseId = useSession((s) => s.caseId);
  const reportId = mode === "live" ? studyId : caseId ? `sample-${caseId}` : null;

  return (
    <div className="flex flex-wrap items-center gap-2 border-t border-film-line/60 bg-film-panel px-4 py-3">
      <button
        type="button"
        disabled={!selected}
        onClick={() => void runDecision("accept")}
        aria-keyshortcuts="A"
        className="flex items-center gap-1.5 rounded-[var(--radius-control)] bg-ink px-3 py-1.5 text-[13px] font-medium text-film-base transition-transform duration-100 active:scale-[0.98] disabled:opacity-40"
      >
        <Check size={14} weight="bold" /> Accept finding
      </button>
      <button
        type="button"
        disabled={!selected}
        onClick={() => void runDecision("reject")}
        aria-keyshortcuts="R"
        className="flex items-center gap-1.5 rounded-[var(--radius-control)] border border-film-line px-3 py-1.5 text-[13px] text-ink transition-transform duration-100 active:scale-[0.98] disabled:opacity-40"
      >
        <X size={14} weight="bold" /> Reject finding
      </button>
      <span className="ml-auto flex items-center gap-3 text-[13px]">
        {reportId && (
          <Link to={`/report/${reportId}`} className="flex items-center gap-1 text-ink-dim hover:text-ink">
            <FileText size={14} /> Report
          </Link>
        )}
        {mode === "live" && studyId && (
          <Link to={`/audit/${studyId}`} className="flex items-center gap-1 text-ink-dim hover:text-ink">
            <Scroll size={14} /> Audit trail
          </Link>
        )}
      </span>
    </div>
  );
}
