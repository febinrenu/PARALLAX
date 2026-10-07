import type { Finding, TextEvidence } from "../api/types";

export interface Span {
  te: TextEvidence;
  finding: Finding;
}

export type Segment =
  | { kind: "text"; text: string }
  | { kind: "span"; text: string; span: Span }
  | { kind: "quarantine"; text: string };

/**
 * Split note text into plain runs, quarantined runs and evidence spans. Notes render as text, never
 * as HTML. Quarantined text (an instruction the injection guard found) wins over evidence; among
 * evidence spans, the earlier one in `spans` wins, so callers put the selected finding's first.
 */
export function segments(text: string, spans: Span[], quarantined: [number, number][] = []): Segment[] {
  const items = [
    ...quarantined.map(([a, b]) => ({ a, b, q: true as const })),
    ...spans.map((s) => ({ a: s.te.span[0], b: s.te.span[1], q: false as const, s })),
  ];
  // Stable sort: at the same start, quarantine first, then the caller's order.
  const sorted = items.map((it, i) => ({ it, i })).sort((x, y) => x.it.a - y.it.a || Number(y.it.q) - Number(x.it.q) || x.i - y.i);
  const out: Segment[] = [];
  let at = 0;
  for (const { it } of sorted) {
    if (it.a < at || it.b > text.length || it.b <= it.a) continue; // overlapping or out-of-range runs are skipped, not mangled
    if (it.a > at) out.push({ kind: "text", text: text.slice(at, it.a) });
    out.push(it.q ? { kind: "quarantine", text: text.slice(it.a, it.b) } : { kind: "span", text: text.slice(it.a, it.b), span: it.s });
    at = it.b;
  }
  if (at < text.length) out.push({ kind: "text", text: text.slice(at) });
  return out;
}
