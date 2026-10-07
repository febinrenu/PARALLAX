import type { ChainStep } from "../lib/findings";

const SEG: Record<ChainStep["state"], string> = {
  pass: "bg-ink",
  fail: "bg-pencil-red",
  pending: "bg-ink-dim/40 motion-safe:animate-pulse",
  unavailable: "border border-film-line bg-transparent",
};

const WORD: Record<ChainStep["state"], string> = {
  pass: "passed",
  fail: "failed",
  pending: "waiting",
  unavailable: "not run",
};

/** Eight segments, one per stage, in pipeline order. The order is a real sequence. */
export function ChainBar({ steps }: { steps: ChainStep[] }) {
  return (
    <div className="flex gap-[3px]" role="img" aria-label={steps.map((s) => `${s.stage} ${WORD[s.state]}`).join(", ")}>
      {steps.map((s) => (
        <span key={s.stage} title={`${s.stage}: ${s.detail}`} className={`h-[3px] w-[14px] rounded-full ${SEG[s.state]}`} />
      ))}
    </div>
  );
}

export function ChainDetail({ steps }: { steps: ChainStep[] }) {
  return (
    <ol className="mt-3 space-y-1.5" aria-label="Verification chain">
      {steps.map((s, i) => (
        <li key={s.stage} className="grid grid-cols-[18px_96px_1fr] items-baseline gap-2 text-[12.5px] leading-snug">
          <span className="font-mono text-[11px] tabular text-ink-dim">{i + 1}</span>
          <span className={s.state === "fail" ? "text-pencil-red-ink" : s.state === "pass" ? "text-ink" : "text-ink-dim"}>{s.stage}</span>
          <span className="text-ink-dim">{s.detail}</span>
        </li>
      ))}
    </ol>
  );
}
