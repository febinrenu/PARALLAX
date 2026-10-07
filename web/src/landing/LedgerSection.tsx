import { useEffect, useId, useMemo, useState, type CSSProperties } from "react";
import type { LedgerEntry } from "../api/types";
import { buildChain, verifyChain, type VerifyResult } from "../lib/ledger";
import { LEDGER_FRAMES, tsFor } from "../story/ledgerFrames";

/**
 * The ledger chapter: a sticky film strip of five real hash-chained entries, then an unpinned
 * "try it" panel. Editing a value re-hashes in the browser with WebCrypto and verification
 * fails at that entry; everything after it turns red because it can no longer be trusted.
 */
export function LedgerSection() {
  const [chain, setChain] = useState<LedgerEntry[] | null>(null);
  const [label, setLabel] = useState(String(LEDGER_FRAMES[1].payload.label));
  const [decision, setDecision] = useState(String(LEDGER_FRAMES[4].payload.decision));
  const [result, setResult] = useState<VerifyResult | null>(null);
  const ids = { label: useId(), decision: useId(), status: useId() };

  useEffect(() => {
    if (!globalThis.crypto?.subtle) return; // insecure context: the strip keeps "hash pending"
    void buildChain(LEDGER_FRAMES.map((f, i) => ({ event: f.event, actor: f.actor, ts: tsFor(i), study_id: "demo", payload: f.payload }))).then(setChain);
  }, []);

  const tampered = useMemo(() => {
    if (!chain) return null;
    return chain.map((e, i) => {
      if (i === 1) return { ...e, payload: { ...e.payload, label } };
      if (i === 4) return { ...e, payload: { ...e.payload, decision } };
      return e;
    });
  }, [chain, label, decision]);

  useEffect(() => {
    if (!tampered) return;
    let live = true;
    void verifyChain(tampered).then((r) => live && setResult(r));
    return () => {
      live = false;
    };
  }, [tampered]);

  const brokenAt = result && !result.ok ? result.brokenAt : null;
  const frameState = (i: number) => (brokenAt === null ? "ok" : i >= brokenAt ? "broken" : "ok");
  const pristine = label === LEDGER_FRAMES[1].payload.label && decision === LEDGER_FRAMES[4].payload.decision;

  return (
    <>
      <section id="ledger" aria-label="Evidence ledger" className="chapter" data-chapter="ledger" data-stage={8} style={{ "--len": "240svh" } as CSSProperties}>
        <div className="stage ledger-stage">
          <div className="mx-auto w-full max-w-[1400px] px-[var(--gutter)] pt-[14svh]">
            <h2 className="display-md">Every step leaves a&nbsp;receipt.</h2>
            <p className="lede mt-5">Stages and doctor decisions join a hash chain. Change one character and verification fails. Try it below.</p>
          </div>
          <div className="film-strip" data-strip>
            <ol className="film-strip-track" data-strip-track>
              {LEDGER_FRAMES.map((f, i) => (
                <li key={f.title} className="strip-frame" data-state={frameState(i)}>
                  <span className="frame-index font-mono">#{i}</span>
                  <p className="frame-title">{f.title}</p>
                  <p className="frame-body">{f.body}</p>
                  <p className="frame-hash font-mono">{chain ? `${chain[i].hash.slice(0, 20)}…` : "hash pending"}</p>
                  <p className="frame-prev font-mono">{chain ? `prev ${chain[i].prev_hash.slice(0, 12)}…` : ""}</p>
                </li>
              ))}
            </ol>
          </div>
        </div>
      </section>

      <section aria-labelledby={ids.status} className="ledger-try">
        <div className="ledger-try-inner">
          <h3 className="text-[21px] font-light text-ink">Try to rewrite history.</h3>
          <p className="mt-2 max-w-[56ch] text-[15px] text-ink-dim">These two values sit inside entries #1 and #4. Change either one and the browser recomputes every SHA-256 the way the server does.</p>
          <div className="mt-6 grid gap-5 sm:grid-cols-2">
            <div className="space-y-1.5">
              <label htmlFor={ids.label} className="block text-[13px] text-ink">Reader label, entry #1</label>
              <input id={ids.label} value={label} onChange={(e) => setLabel(e.target.value)} spellCheck={false} className="ledger-input" />
            </div>
            <div className="space-y-1.5">
              <label htmlFor={ids.decision} className="block text-[13px] text-ink">Doctor decision, entry #4</label>
              <input id={ids.decision} value={decision} onChange={(e) => setDecision(e.target.value)} spellCheck={false} className="ledger-input" />
            </div>
          </div>
          <p id={ids.status} role="status" aria-live="polite" className={`mt-6 text-[16px] ${brokenAt === null ? "text-ink" : "text-pencil-red"}`}>
            {!chain
              ? "Computing hashes…"
              : brokenAt === null
                ? "Chain verified: all five entries match their hashes."
                : `Verification failed at entry ${brokenAt}: ${result?.reason === "payload_hash mismatch" ? "its contents no longer match its hash" : result?.reason}. Every entry after it is untrusted.`}
          </p>
          {!pristine && (
            <button
              type="button"
              onClick={() => {
                setLabel(String(LEDGER_FRAMES[1].payload.label));
                setDecision(String(LEDGER_FRAMES[4].payload.decision));
              }}
              className="mt-4 rounded-[var(--radius-control)] border border-film-line px-3 py-1.5 text-[13px] text-ink hover:border-ink-dim"
            >
              Restore the original values
            </button>
          )}
        </div>
      </section>
    </>
  );
}
