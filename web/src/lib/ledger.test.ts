import { describe, expect, it } from "vitest";
import type { LedgerEntry } from "../api/types";
import fixture from "./__fixtures__/ledger_parity.json";
import { buildChain, entryHash, GENESIS_HASH, payloadHash, verifyChain } from "./ledger";
import { FloatNotSupported, pyJsonDumps } from "./pyjson";

const entries = fixture as LedgerEntry[];

describe("parity with backend/medproof/core/ledger.py", () => {
  it("recomputes every payload hash Python wrote", async () => {
    for (const e of entries) expect(await payloadHash(e.payload)).toBe(e.payload_hash);
  });

  it("recomputes every entry hash, including a null study_id", async () => {
    for (const e of entries) expect(await entryHash(e)).toBe(e.hash);
    expect(entries.some((e) => e.study_id === null)).toBe(true);
  });

  it("verifies the whole chain from genesis", async () => {
    const r = await verifyChain(entries);
    expect(r).toEqual({ ok: true, brokenAt: null, reason: null, payloadSkipped: [] });
  });
});

describe("tamper detection", () => {
  it("flags a changed payload at the right index", async () => {
    const t = structuredClone(entries);
    (t[1].payload as Record<string, unknown>).note = "edited";
    const r = await verifyChain(t);
    expect(r.ok).toBe(false);
    expect(r.brokenAt).toBe(1);
    expect(r.reason).toBe("payload_hash mismatch");
  });

  it("flags a changed actor even when the payload is untouched", async () => {
    const t = structuredClone(entries);
    t[2].actor = "someone else";
    const r = await verifyChain(t);
    expect(r).toMatchObject({ ok: false, brokenAt: 2, reason: "hash mismatch" });
  });

  it("flags a removed entry as a broken link", async () => {
    const t = structuredClone(entries);
    t.splice(1, 1);
    const r = await verifyChain(t);
    expect(r).toMatchObject({ ok: false, brokenAt: 1, reason: "prev_hash mismatch" });
  });
});

describe("pyJsonDumps", () => {
  it("matches Python's default separators and key order", () => {
    expect(pyJsonDumps({ b: 1, a: [true, null, "x"] })).toBe('{"a": [true, null, "x"], "b": 1}');
  });

  it("escapes non-ASCII like ensure_ascii=True", () => {
    expect(pyJsonDumps("café 😀")).toBe('"caf\\u00e9 \\ud83d\\ude00"');
  });

  it("refuses floats instead of guessing Python's repr", () => {
    expect(() => pyJsonDumps({ p: 0.5 })).toThrow(FloatNotSupported);
  });
});

describe("buildChain", () => {
  it("produces a chain that verifies and starts at genesis", async () => {
    const chain = await buildChain([
      { event: "stage_completed", actor: "system", ts: "2026-10-07T00:00:00+00:00", study_id: "demo", payload: { stage: "intake" } },
      { event: "stage_completed", actor: "system", ts: "2026-10-07T00:00:01+00:00", study_id: "demo", payload: { stage: "reader" } },
    ]);
    expect(chain[0].prev_hash).toBe(GENESIS_HASH);
    expect((await verifyChain(chain)).ok).toBe(true);
  });
});
