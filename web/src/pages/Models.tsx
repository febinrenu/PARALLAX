import { useEffect, useState } from "react";
import { getModelCards, loadCredits } from "../api/client";
import type { Credit } from "../api/types";
import datasheets from "../../../docs/datasheets_index.json";
import { RESOURCES } from "../lib/resources";
import { Page, Pending, Section } from "./Page";

/** P2.13 card as served by GET /model-cards (docs/model_cards/*.json). */
interface ModelCard {
  id?: string;
  title?: string;
  registry_id?: string;
  license?: string;
  contamination?: string;
  headline?: Record<string, { point: number; lo?: number; hi?: number }>;
  markdown?: string;
}

const METRIC_NAMES: Record<string, string> = { auroc_image: "AUROC (image level)", map50: "mAP@50", macro_f1: "Macro F1", mean_patient_dice: "Mean patient Dice" };
const METRIC = (key: string) => {
  const k = key.replace(/^headline\./, "");
  const plain = k.replace(/_/g, " ");
  return METRIC_NAMES[k] ?? plain.charAt(0).toUpperCase() + plain.slice(1);
};

function CardSummary({ card }: { card: ModelCard }) {
  // Only the "headline.*" rows: the per-split duplicates are in the full card.
  const metrics = Object.entries(card.headline ?? {}).filter(([k]) => k.startsWith("headline."));
  return (
    <article className="rounded-[var(--radius-panel)] bg-film-panel p-4">
      <h3 className="text-[15px] text-ink">{card.title ?? card.id}</h3>
      {card.registry_id && <p className="mt-0.5 font-mono text-[12px] text-ink-dim">{card.registry_id}</p>}
      {metrics.length > 0 && (
        <dl className="mt-3 grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-[13px]">
          {metrics.map(([k, v]) => (
            <div key={k} className="contents">
              <dt className="text-ink-dim">{METRIC(k)}</dt>
              <dd className="font-mono text-ink">
                {v.point.toFixed(3)}
                {v.lo != null && v.hi != null ? <span className="text-ink-dim"> [{v.lo.toFixed(3)}, {v.hi.toFixed(3)}]</span> : null}
              </dd>
            </div>
          ))}
        </dl>
      )}
      {card.contamination && <p className="mt-3 text-[13px] leading-snug text-ink-dim">Evaluation: {card.contamination}.</p>}
      {card.license && <p className="mt-1 text-[13px] leading-snug text-ink-dim">Licence: {card.license}.</p>}
      {card.markdown && (
        <details className="mt-3 text-[13px]">
          <summary className="cursor-pointer text-ink hover:underline">Full model card</summary>
          <div className="mt-2 whitespace-pre-wrap break-words font-mono text-[12px] leading-relaxed text-ink-dim">{card.markdown}</div>
        </details>
      )}
    </article>
  );
}

export function Models() {
  const [cards, setCards] = useState<ModelCard[] | null>(null);
  const [credits, setCredits] = useState<Credit[]>([]);
  useEffect(() => {
    document.title = "Models and data, Parallax";
    getModelCards().then((r) => setCards(r.cards as ModelCard[])).catch(() => setCards([]));
    loadCredits().then(setCredits).catch(() => setCredits([]));
  }, []);

  return (
    <Page title="Models and data" lead="What each model was trained on, what it is used for, and under which licence. Contaminated evaluations are labelled where they occur.">
      <Section title="Model cards">
        {cards && cards.length > 0 ? (
          <div className="grid gap-4 lg:grid-cols-2">
            {cards.map((c) => (
              <CardSummary key={c.id ?? c.title} card={c} />
            ))}
          </div>
        ) : (
          <Pending>Model cards and datasheets publish with the trained weights (plan P2.13). The declared resources below are complete today.</Pending>
        )}
      </Section>

      <Section title="Datasheets" aside="docs/datasheets">
        <p className="mb-4 max-w-[65ch] text-[15px] leading-relaxed text-ink-dim">One datasheet per dataset used for training or evaluation: where it comes from, how it was split, and the licence that limits what the weights may be used for.</p>
        <dl className="grid gap-x-10 gap-y-3 md:grid-cols-2">
          {datasheets.map((d) => (
            <div key={d.id}>
              <dt className="text-[14.5px] text-ink">{d.title}</dt>
              <dd className="mt-0.5 text-[13px] text-ink-dim">
                Licence: <span className={d.license === "unconfirmed" ? "text-pencil-red-ink" : "text-ink"}>{d.license}</span>
              </dd>
            </div>
          ))}
        </dl>
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
