import { useEffect, useState } from "react";
import { getMetrics } from "../api/client";
import { leakageMethod, leakageRows } from "../lib/leakage";
import { Page, Pending, Section } from "./Page";

const fmt = new Intl.NumberFormat("en-US");

export function Validation() {
  const [metrics, setMetrics] = useState<Record<string, unknown> | null | "offline">(null);
  useEffect(() => {
    document.title = "Validation, Parallax";
    getMetrics().then(setMetrics).catch(() => setMetrics("offline"));
  }, []);
  const published = metrics && metrics !== "offline" && metrics.available !== false;

  return (
    <Page
      title="Validation"
      lead="Every number here is reproducible from committed code and cached predictions. Where a result is not measured yet, the page says so instead of showing a placeholder."
    >
      <Section title="Leakage audit" aside={`${leakageMethod.hash}, duplicates at Hamming distance up to ${leakageMethod.duplicate_max_hamming}`}>
        <p className="mb-5 max-w-[65ch] text-[15px] leading-relaxed text-ink-dim">
          Near-duplicate images that land in both training and test sets inflate accuracy. We hash every image, group duplicates, and keep each group on one side of the split.
        </p>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[640px] text-left text-[14px]">
            <thead className="text-[12.5px] text-ink-dim">
              <tr>
                <th scope="col" className="py-2 pr-4 font-normal">Dataset</th>
                <th scope="col" className="py-2 pr-4 text-right font-normal">Images</th>
                <th scope="col" className="py-2 pr-4 text-right font-normal">Duplicate pairs</th>
                <th scope="col" className="py-2 pr-4 text-right font-normal">Pairs across original splits</th>
                <th scope="col" className="py-2 text-right font-normal">Images removed</th>
              </tr>
            </thead>
            <tbody>
              {leakageRows.map((r) => (
                <tr key={r.id} className="border-t border-film-line/50 align-top">
                  <th scope="row" className="py-3 pr-4 font-normal text-ink">
                    {r.name}
                    <span className="mt-1 block max-w-[46ch] text-[12.5px] leading-snug text-ink-dim">{r.note}</span>
                  </th>
                  <td className="py-3 pr-4 text-right font-mono tabular text-ink">{fmt.format(r.images)}</td>
                  <td className="py-3 pr-4 text-right font-mono tabular text-ink">{fmt.format(r.duplicatePairs)}</td>
                  <td className={`py-3 pr-4 text-right font-mono tabular ${r.pairsAcrossSplits > 0 ? "text-pencil-yellow" : "text-ink"}`}>{fmt.format(r.pairsAcrossSplits)}</td>
                  <td className="py-3 text-right font-mono tabular text-ink">{fmt.format(r.removed)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      <Section title="Model performance and calibration">
        {published ? (
          <pre className="overflow-x-auto rounded-[var(--radius-panel)] bg-film-panel p-4 font-mono text-[12px] text-ink">{JSON.stringify(metrics, null, 2)}</pre>
        ) : (
          <Pending>
            Metrics with bootstrap confidence intervals publish at gate G3, when <span className="font-mono text-ink">make eval</span> writes <span className="font-mono text-ink">reports/metrics.json</span>.
            {metrics === "offline" && " The analysis server is not reachable from this page right now."}
          </Pending>
        )}
      </Section>

      <Section title="Do the warnings mean anything?">
        <p className="mb-4 max-w-[65ch] text-[15px] leading-relaxed text-ink-dim">
          For each flag (discordant, unstable, unfaithful, out of distribution, low quality) we compare error rates on findings with and without it. A flag that does not predict errors is dropped from the status rule and reported here as such.
        </p>
        <Pending>The trust-signal comparison publishes with the evaluation run at gate G3.</Pending>
      </Section>
    </Page>
  );
}
