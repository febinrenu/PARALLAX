// Client-side verification of the evidence ledger, matching backend/medproof/core/ledger.py:
//   payload_hash = sha256(json.dumps(payload, sort_keys=True, default=str))
//   hash         = sha256(f"{prev_hash}:{event}:{payload_hash}:{ts}:{actor}:{study_id}")
// The entry-hash chain uses only strings, so it verifies exactly for any real entry. Payload
// re-hashing needs Python-identical JSON, which pyJsonDumps gives for payloads without floats.
import type { LedgerEntry } from "../api/types";
import { FloatNotSupported, pyJsonDumps } from "./pyjson";

export const GENESIS_HASH = "0".repeat(64);

export async function sha256Hex(text: string): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join("");
}

export async function payloadHash(payload: unknown): Promise<string> {
  return sha256Hex(pyJsonDumps(payload));
}

export async function entryHash(e: Pick<LedgerEntry, "prev_hash" | "event" | "payload_hash" | "ts" | "actor" | "study_id">): Promise<string> {
  // Python formats None as "None" inside the f-string.
  const sid = e.study_id === null ? "None" : e.study_id;
  return sha256Hex(`${e.prev_hash}:${e.event}:${e.payload_hash}:${e.ts}:${e.actor}:${sid}`);
}

export interface VerifyResult {
  ok: boolean;
  brokenAt: number | null;
  reason: string | null;
  /** Indexes whose payload could not be re-hashed in the browser (contains floats). */
  payloadSkipped: number[];
}

/**
 * Verify a contiguous chain. `startPrev` is the prev_hash the first entry must point at
 * (GENESIS for a full ledger, or the known predecessor for a slice such as one study's entries).
 */
export async function verifyChain(
  entries: LedgerEntry[],
  opts: { startPrev?: string | null; checkPayload?: boolean; checkLinks?: boolean } = {},
): Promise<VerifyResult> {
  let prev = opts.startPrev === undefined ? GENESIS_HASH : opts.startPrev;
  const payloadSkipped: number[] = [];
  for (let i = 0; i < entries.length; i++) {
    const e = entries[i];
    // One study's entries are a non-contiguous slice of the global chain: skip link checks there
    // and let the server's /ledger/verify vouch for the links.
    if (opts.checkLinks !== false && prev !== null && e.prev_hash !== prev) return { ok: false, brokenAt: i, reason: "prev_hash mismatch", payloadSkipped };
    if (opts.checkPayload !== false) {
      try {
        if ((await payloadHash(e.payload)) !== e.payload_hash) return { ok: false, brokenAt: i, reason: "payload_hash mismatch", payloadSkipped };
      } catch (err) {
        if (err instanceof FloatNotSupported) payloadSkipped.push(i);
        else throw err;
      }
    }
    if ((await entryHash(e)) !== e.hash) return { ok: false, brokenAt: i, reason: "hash mismatch", payloadSkipped };
    prev = e.hash;
  }
  return { ok: true, brokenAt: null, reason: null, payloadSkipped };
}

/** Build a chain from scratch (used by the landing's "try it" ledger, with float-free payloads). */
export async function buildChain(items: { event: string; actor: string; ts: string; study_id: string | null; payload: Record<string, unknown> }[]): Promise<LedgerEntry[]> {
  const out: LedgerEntry[] = [];
  let prev = GENESIS_HASH;
  for (let i = 0; i < items.length; i++) {
    const it = items[i];
    const ph = await payloadHash(it.payload);
    const h = await entryHash({ prev_hash: prev, event: it.event, payload_hash: ph, ts: it.ts, actor: it.actor, study_id: it.study_id });
    out.push({ index: i, prev_hash: prev, hash: h, ts: it.ts, actor: it.actor, event: it.event, study_id: it.study_id, payload_hash: ph, payload: it.payload });
    prev = h;
  }
  return out;
}
