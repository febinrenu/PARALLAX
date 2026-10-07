#!/usr/bin/env node
// Regenerate web/src/contracts.ts from contracts/study_result.schema.json (which embeds every
// nested contract type under $defs). Run via `make contracts` or `pnpm run generate:contracts`;
// run `python scripts/export_contract.py` first if schemas.py changed.

import { compile } from "json-schema-to-typescript";
import { readFile, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = path.resolve(fileURLToPath(import.meta.url), "..", "..");
const schemaPath = path.join(root, "contracts", "study_result.schema.json");
const outPath = path.join(root, "web", "src", "contracts.ts");

// json-schema-to-typescript doesn't resolve 2020-12 "prefixItems" tuple schemas (it emits
// `unknown` per slot); it does understand the older draft-07 `items: [...]` tuple form, which
// is otherwise equivalent here. Rewrite tuples before compiling. Only affects this generation
// step — the committed JSON Schema files keep the standard `prefixItems` form.
function prefixItemsToTuple(node) {
  if (Array.isArray(node)) return node.map(prefixItemsToTuple);
  if (node && typeof node === "object") {
    const out = {};
    for (const [k, v] of Object.entries(node)) {
      if (k === "prefixItems") continue;
      out[k] = prefixItemsToTuple(v);
    }
    if (Array.isArray(node.prefixItems)) out.items = node.prefixItems.map(prefixItemsToTuple);
    return out;
  }
  return node;
}

const load = async (name) =>
  prefixItemsToTuple(JSON.parse(await readFile(path.join(root, "contracts", name), "utf-8")));

const study = await compile(await load("study_result.schema.json"), "StudyResult", {
  additionalProperties: false,
  bannerComment:
    "/* AUTO-GENERATED from contracts/*.schema.json by scripts/gen_contracts_ts.mjs.\n" +
    " * Do not edit by hand — run `make contracts` after changing backend/medproof/core/schemas.py. */",
});
const stage = await compile(await load("stage_result.schema.json"), "StageResult", {
  additionalProperties: false,
  bannerComment: "",
});

await writeFile(outPath, `${study}\n${stage}`, "utf-8");
console.log(`wrote ${outPath}`);
