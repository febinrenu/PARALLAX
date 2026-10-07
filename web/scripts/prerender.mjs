#!/usr/bin/env node
// Inject the server-rendered landing into dist/index.html, so the page reads without
// JavaScript and its text paints before any script runs. Runs after both Vite builds.
import { readFile, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const ssrEntry = path.join(root, "dist-ssr", "entry-server.js");
const indexPath = path.join(root, "dist", "index.html");

const { render } = await import(pathToFileURL(ssrEntry).href);
const html = await readFile(indexPath, "utf-8");
if (!html.includes("<!--ssr-outlet-->")) throw new Error("dist/index.html has no <!--ssr-outlet--> marker");
const markup = render();
await writeFile(indexPath, html.replace("<!--ssr-outlet-->", markup), "utf-8");
await rm(path.join(root, "dist-ssr"), { recursive: true, force: true });
console.log(`prerendered landing: ${(markup.length / 1024).toFixed(1)} KB of HTML into dist/index.html`);
