import { useEffect, useState } from "react";
import { getModelCards, loadCredits } from "../api/client";
import type { Credit } from "../api/types";
import { RESOURCES } from "../lib/resources";
import { Page, Pending, Section } from "./Page";

export function Models() {
  const [cards, setCards] = useState<Record<string, unknown>[] | null>(null);
  const [credits, setCredits] = useState<Credit[]>([]);
  useEffect(() => {
    document.title = "Models and data, Parallax";
    getModelCards().then((r) => setCards(r.cards)).catch(() => setCards([]));
    loadCredits().then(setCredits).catch(() => setCredits([]));
  }, []);

  return (
    <Page title="Models and data" lead="What each model was trained on, what it is used for, and under which licence. Contaminated evaluations are labelled where they occur.">
      <Section title="Model cards">
        {cards && cards.length > 0 ? (
          <div className="space-y-4">
            {cards.map((c, i) => (
              <pre key={i} className="overflow-x-auto rounded-[var(--radius-panel)] bg-film-panel p-4 font-mono text-[12px] text-ink">{JSON.stringify(c, null, 2)}</pre>
            ))}
          </div>
        ) : (
          <Pending>Model cards and datasheets publish with the trained weights (plan P2.13). The declared resources below are complete today.</Pending>
        )}
      </Section>

      <Section title="Declared resources">
        <div className="grid gap-x-10 gap-y-6 md:grid-cols-2">
          {(["Model", "Dataset", "Service"] as const).map((kind) => (
            <div key={kind} className={kind === "Dataset" ? "md:row-span-2" : ""}>
              <h3 className="text-[13px] text-ink-dim">{kind === "Model" ? "Models" : kind === "Dataset" ? "Datasets" : "Services"}</h3>
              <dl className="mt-3 space-y-4">
                {RESOURCES.filter((r) => r.kind === kind).map((r) => (
                  <div key={r.name}>
                    <dt className="text-[14.5px] text-ink">{r.name}</dt>
                    <dd className="mt-0.5 text-[13px] leading-snug text-ink-dim">
                      {r.use}. <span className="text-ink">{r.license}</span>.{r.note ? ` ${r.note}` : ""}
                    </dd>
                  </div>
                ))}
              </dl>
            </div>
          ))}
        </div>
      </Section>

      <Section title="Images on this site">
        <ul className="space-y-3 text-[14px]">
          {credits.map((c) => (
            <li key={c.id} className="leading-snug">
              <a href={c.page} className="text-ink underline decoration-film-line underline-offset-4 hover:decoration-ink-dim">{c.title}</a>
              <span className="text-ink-dim">, {c.author}. {c.license}. {c.modifications}</span>
            </li>
          ))}
        </ul>
      </Section>
    </Page>
  );
}
