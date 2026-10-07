import { useEffect, useState } from "react";
import { useParams } from "react-router";
import { getStudy, getStudyLedger, verifyLedger } from "../api/client";
import type { LedgerEntry, StudyResult } from "../api/types";
import { verifyChain, type VerifyResult } from "../lib/ledger";
import { Page, Pending, Section } from "./Page";

type ServerVerify = Awaited<ReturnType<typeof verifyLedger>>;

function summary(e: LedgerEntry): string {
  const p = e.payload as Record<string, unknown>;
  if (e.event === "stage_completed") return `Stage ${String(p.stage)} ${p.ok ? "completed" : "failed"} in ${String(p.ms)} ms`;
  if (e.event === "finding_feedback") return `Finding ${String(p.finding_id)} ${p.decision === "accept" ? "accepted" : "rejected"}`;
  return e.event;
}

export function Audit() {
  const { id = "" } = useParams();
  const isSample = id.startsWith("sample-");
  const [entries, setEntries] = useState<LedgerEntry[] | null>(null);
  const [study, setStudy] = useState<StudyResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [server, setServer] = useState<ServerVerify | null>(null);
  const [local, setLocal] = useState<VerifyResult | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    document.title = "Audit trail, Parallax";
    if (isSample) return;
    getStudyLedger(id).then((r) => setEntries(r.entries)).catch((e) => setError((e as Error).message));
    getStudy(id).then((s) => ("findings" in s ? setStudy(s) : undefined)).catch(() => undefined);
  }, [id, isSample]);

  const verify = async () => {
    if (!entries) return;
    setBusy(true);
    try {
      const [s, l] = await Promise.all([verifyLedger(), verifyChain(entries, { checkLinks: false })]);
      setServer(s);
      setLocal(l);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const blocked = study?.claims.filter((c) => c.blocked_reason) ?? [];

  return (
    <Page title="Audit trail" lead={<>Every stage and every decision for study <span className="font-mono text-[15px] text-ink">{id}</span>, hash-chained so that changing any entry breaks verification.</>}>
      {isSample && <Pending>Sample cases are static files and have no ledger entries. Upload a study in the workstation to create a real audit trail.</Pending>}
      {error && <Pending>Could not load the audit trail: {error}</Pending>}

      {entries && (
        <Section
          title={`${entries.length} entries`}
          aside={
            <button type="button" onClick={() => void verify()} disabled={busy} className="rounded-[var(--radius-control)] bg-ink px-3 py-1.5 text-[13px] font-medium text-film-base disabled:opacity-50">
              {busy ? "Verifying…" : "Verify chain"}
            </button>
          }
        >
          <div aria-live="polite" className="mb-6 space-y-1 text-[14px]">
            {server && (
              <p className={server.ok ? "text-ink" : "text-pencil-red"}>
                {server.ok ? `Server recomputed the whole ledger: ${server.entries} entries, chain intact.` : `Server verification failed at entry ${server.broken_at}: ${server.reason}.`}
              </p>
            )}
            {local && (
              <p className={local.ok ? "text-ink" : "text-pencil-red"}>
                {local.ok
                  ? `This browser recomputed this study's ${entries.length} entry hashes with WebCrypto: all match.${local.payloadSkipped.length ? ` ${local.payloadSkipped.length} payloads contain decimals and were checked by the server only.` : ""}`
                  : `This browser found a mismatch at entry ${local.brokenAt}: ${local.reason}.`}
              </p>
            )}
          </div>

          <ol className="space-y-0">
            {entries.map((e) => (
              <li key={e.hash} className="grid grid-cols-[48px_1fr] gap-4 border-t border-film-line/50 py-3 md:grid-cols-[48px_1fr_220px]">
                <span className="font-mono text-[12px] tabular text-ink-dim">#{e.index}</span>
                <div>
                  <p className="text-[14px] text-ink">{summary(e)}</p>
                  <p className="mt-0.5 text-[12.5px] text-ink-dim">
                    {e.actor}, {new Date(e.ts).toLocaleString()}
                  </p>
                </div>
                <p className="col-start-2 break-all font-mono text-[11.5px] leading-snug text-ink-dim md:col-start-3">
                  {e.hash.slice(0, 16)}…<br />
                  prev {e.prev_hash.slice(0, 12)}…
                </p>
              </li>
            ))}
          </ol>
        </Section>
      )}

      {blocked.length > 0 && (
        <Section title="Blocked claims">
          <ul className="space-y-3">
            {blocked.map((c) => (
              <li key={c.claim_id}>
                <p className="text-[15px] text-ink-dim line-through decoration-pencil-red decoration-2">{c.rendered ?? c.template}</p>
                <p className="mt-1 text-[13px] text-pencil-red">Blocked: {c.blocked_reason}</p>
              </li>
            ))}
          </ul>
        </Section>
      )}
    </Page>
  );
}
