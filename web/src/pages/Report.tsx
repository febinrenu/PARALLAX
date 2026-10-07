import { useEffect, useState } from "react";
import { Link, useParams } from "react-router";
import { API, getStudy, loadCase } from "../api/client";
import type { Finding, StudyResult } from "../api/types";
import { displayLabel, regionOf, strokeFor, TIER_WORD } from "../lib/findings";
import { Mark } from "../design/Mark";

/**
 * Fallback sentence for one finding when the study has no report-stage claims: a fixed template
 * whose slots are filled from the evidence.
 */
function sentence(f: Finding): string {
  const region = regionOf(f);
  const set = f.conformal_set.length > 1 ? `; the plausible set is ${f.conformal_set.map(displayLabel).join(", ")}` : "";
  return `Doctor, consider ${displayLabel(f.label).toLowerCase()}${region ? ` in the ${region}` : ""} (${TIER_WORD[f.tier]}${set}).`;
}

export function Report() {
  const { id = "" } = useParams();
  const [study, setStudy] = useState<StudyResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    document.title = "Report, Parallax";
    const load = id.startsWith("sample-") ? loadCase(id.slice(7)).then((c) => c.study) : getStudy(id).then((s) => ("findings" in s ? s : Promise.reject(new Error("The analysis is still running."))));
    load.then(setStudy).catch((e) => setError((e as Error).message));
  }, [id]);

  if (error) return <p className="mx-auto max-w-xl px-6 py-24 text-ink-dim">Could not load this report: {error}</p>;
  if (!study) return <p className="mx-auto max-w-xl px-6 py-24 text-ink-dim">Loading the report…</p>;

  const reportable = study.findings.filter((f) => f.status !== "rejected");
  // Only claims the firewall and entailment judge let through; blocked ones live in the audit view.
  const claims = (study.claims ?? []).filter((c) => c.blocked_reason == null && c.rendered);
  const blocked = (study.claims ?? []).length - claims.length;
  const live = !id.startsWith("sample-");
  const generated = new Date().toLocaleString();

  return (
    <div className="report-page mx-auto w-full max-w-[210mm] bg-lightbox px-[16mm] py-[14mm] text-lightbox-ink print:max-w-none print:p-0">
      <style>{`@page { size: A4; margin: 16mm; } @media print { html, body { background: #fff !important; } }`}</style>
      <header className="flex items-start justify-between border-b border-lightbox-ink/20 pb-4">
        <div className="flex items-center gap-2 text-lightbox-ink">
          <Mark size={22} />
          <span className="text-[16px] font-medium">Parallax</span>
        </div>
        <div className="text-right text-[12px] leading-snug text-lightbox-ink/70">
          <p>Study {study.study_id}</p>
          <p>Generated {generated}</p>
        </div>
      </header>

      <h1 className="mt-8 text-[28px] font-light leading-tight">Second-reader report</h1>
      <p className="mt-2 text-[13px] text-lightbox-ink/70">
        Modality {study.modality}. Input SHA-256 <span className="break-all font-mono text-[11.5px]">{study.input_sha256}</span>
      </p>

      {claims.length > 0 && (
        <section className="mt-8">
          <h2 className="text-[14px] font-medium">Report</h2>
          <ol className="mt-3 space-y-3">
            {claims.map((c) => (
              <li key={c.claim_id} className="break-inside-avoid text-[15px] leading-relaxed">
                {c.rendered}
                <span className="ml-2 font-mono text-[11.5px] text-lightbox-ink/60">{c.evidence_ids.join(", ")}</span>
              </li>
            ))}
          </ol>
          <p className="mt-4 text-[12px] leading-snug text-lightbox-ink/70">
            The language model wrote templates only; code filled every value from the evidence ids shown.
            {blocked > 0 ? ` ${blocked} unsupported ${blocked === 1 ? "claim was" : "claims were"} blocked and appear struck through in the audit view.` : ""}
          </p>
        </section>
      )}

      <section className="mt-8">
        <h2 className="text-[14px] font-medium">Findings for review</h2>
        {reportable.length === 0 ? (
          <p className="mt-3 text-[14px] text-lightbox-ink/80">No findings were reported.</p>
        ) : (
          <ol className="mt-3 space-y-4">
            {reportable.map((f) => (
              <li key={f.finding_id} className="break-inside-avoid">
                <p className="text-[15px] leading-relaxed">{sentence(f)}</p>
                <p className="mt-1 text-[12px] text-lightbox-ink/70">
                  {strokeFor(f.status).label}. Calibrated probability {f.prob_calibrated.toFixed(2)}. Evidence{" "}
                  {[...(f.image_evidence ?? []).map((e) => e.evidence_id), ...(f.text_evidence ?? []).map((t) => t.evidence_id)].join(", ") || "none"}.
                </p>
              </li>
            ))}
          </ol>
        )}
        {claims.length === 0 && (
          <p className="mt-6 text-[12px] leading-snug text-lightbox-ink/70">
            This study has no report-stage output, so each sentence comes from a fixed template filled from the evidence.
          </p>
        )}
      </section>

      <footer className="mt-12 border-t border-lightbox-ink/20 pt-4 text-[11.5px] leading-snug text-lightbox-ink/70">
        <p>{study.disclaimer}</p>
        <p className="mt-2">
          Ledger head <span className="break-all font-mono">{study.ledger_head}</span>
        </p>
      </footer>

      <div className="mt-8 flex flex-wrap gap-3 print:hidden">
        <button type="button" onClick={() => window.print()} className="rounded-[var(--radius-control)] bg-lightbox-ink px-3 py-1.5 text-[13px] text-lightbox">
          Print report
        </button>
        {live ? (
          <a href={`${API}/studies/${id}/fhir`} download={`parallax-${id}.fhir.json`} className="rounded-[var(--radius-control)] border border-lightbox-ink/40 px-3 py-1.5 text-[13px] text-lightbox-ink">
            Download FHIR bundle
          </a>
        ) : (
          <span className="rounded-[var(--radius-control)] border border-lightbox-ink/20 px-3 py-1.5 text-[13px] text-lightbox-ink/60">FHIR export is available for uploaded studies</span>
        )}
        <Link to="/read" className="px-1 py-1.5 text-[13px] text-lightbox-ink underline underline-offset-4">
          Back to the workstation
        </Link>
      </div>
    </div>
  );
}
