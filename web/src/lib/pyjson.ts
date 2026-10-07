// Byte-for-byte equivalent of Python's `json.dumps(obj, sort_keys=True, default=str)` for
// strings, integers, booleans, null, arrays and objects. Python's defaults: ", " and ": "
// separators, ensure_ascii=True (non-ASCII escaped as \uXXXX, lowercase hex, surrogate pairs).
// Floats are deliberately refused: Python prints 1.0 where JS prints 1, so any payload with a
// float must be verified server-side (core/ledger.py stays authoritative).

export class FloatNotSupported extends Error {}

function str(s: string): string {
  let out = '"';
  for (let i = 0; i < s.length; i++) {
    const c = s.charCodeAt(i);
    const ch = s[i];
    if (ch === '"') out += '\\"';
    else if (ch === "\\") out += "\\\\";
    else if (ch === "\n") out += "\\n";
    else if (ch === "\r") out += "\\r";
    else if (ch === "\t") out += "\\t";
    else if (ch === "\b") out += "\\b";
    else if (ch === "\f") out += "\\f";
    else if (c < 0x20 || c > 0x7e) out += "\\u" + c.toString(16).padStart(4, "0");
    else out += ch;
  }
  return out + '"';
}

export function pyJsonDumps(value: unknown): string {
  if (value === null || value === undefined) return "null";
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "number") {
    if (!Number.isInteger(value)) throw new FloatNotSupported(`float ${value} has no portable Python repr`);
    return String(value);
  }
  if (typeof value === "string") return str(value);
  if (Array.isArray(value)) return "[" + value.map(pyJsonDumps).join(", ") + "]";
  if (typeof value === "object") {
    const keys = Object.keys(value as object).sort();
    return "{" + keys.map((k) => `${str(k)}: ${pyJsonDumps((value as Record<string, unknown>)[k])}`).join(", ") + "}";
  }
  throw new TypeError(`cannot serialise ${typeof value}`);
}
