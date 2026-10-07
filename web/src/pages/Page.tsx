import type { ReactNode } from "react";

/** Quiet document layout for the secondary pages: one column, readable measure. */
export function Page({ title, lead, children }: { title: string; lead?: ReactNode; children: ReactNode }) {
  return (
    <div className="mx-auto w-full max-w-[1080px] px-4 py-10 md:px-8 md:py-14">
      <h1 className="text-[36px] font-light leading-[1.1] tracking-[-0.02em] text-ink md:text-[48px]">{title}</h1>
      {lead && <p className="mt-4 max-w-[65ch] text-[17px] leading-relaxed text-ink-dim">{lead}</p>}
      <div className="mt-10 space-y-14">{children}</div>
    </div>
  );
}

export function Section({ title, children, aside }: { title: string; children: ReactNode; aside?: ReactNode }) {
  return (
    <section>
      <div className="flex flex-wrap items-baseline justify-between gap-2 border-b border-film-line/70 pb-2">
        <h2 className="text-[18px] font-medium text-ink">{title}</h2>
        {aside && <div className="text-[13px] text-ink-dim">{aside}</div>}
      </div>
      <div className="mt-5">{children}</div>
    </section>
  );
}

export function Pending({ children }: { children: ReactNode }) {
  return <p className="max-w-[65ch] rounded-[var(--radius-panel)] bg-film-panel px-4 py-3 text-[14px] leading-relaxed text-ink-dim">{children}</p>;
}
