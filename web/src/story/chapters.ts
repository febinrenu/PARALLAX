// The DOM side of each chapter, written as pure functions of chapter progress. Writes go
// through a cache so an unchanged value never touches the DOM (no style recalculation).
import { faith, primary, probAfterDeletion, probAfterRandom } from "./data";
import { contain, ease, iou, secondReaderBox, seg, type Frame, type Layout, type Rect } from "./scene";

const last = new WeakMap<Element, Map<string, string>>();

function put(el: Element | null | undefined, prop: string, value: string) {
  if (!el) return;
  let m = last.get(el);
  if (!m) last.set(el, (m = new Map()));
  if (m.get(prop) === value) return;
  m.set(prop, value);
  if (prop === "text") el.textContent = value;
  else if (prop.startsWith("--")) (el as HTMLElement).style.setProperty(prop, value);
  else if (prop === "class+") el.classList.add(value);
  else if (prop === "class-") el.classList.remove(value);
  else ((el as HTMLElement).style as unknown as Record<string, string>)[prop] = value;
}

function reveal(el: Element | null | undefined, k: number, dy = 10) {
  put(el, "opacity", k.toFixed(3));
  put(el, "transform", k >= 1 ? "none" : `translate3d(0, ${((1 - k) * dy).toFixed(2)}px, 0)`);
}

const PRIMARY_BOX = (primary.bbox ?? [0.3, 0.2, 0.5, 0.45]) as [number, number, number, number];

export interface Dom {
  rail: HTMLElement | null;
  railSteps: HTMLElement[];
  heroAside: HTMLElement[];
  quality: Record<string, HTMLElement | null>;
  pin: HTMLElement | null;
  faithNumber: HTMLElement | null;
  faithRandom: HTMLElement | null;
  faithVerdict: HTMLElement | null;
  flipped: HTMLElement | null;
  tileLabels: HTMLElement[];
  stabilityResult: HTMLElement | null;
  iou: HTMLElement | null;
  iouVerdict: HTMLElement | null;
  iouText: HTMLElement | null;
  noteSupport: HTMLElement | null;
  noteSide: HTMLElement | null;
  noteInject: HTMLElement | null;
  checks: Record<string, HTMLElement | null>;
  connectors: Record<string, SVGPathElement | null>;
  chips: HTMLElement[];
  blocked: HTMLElement | null;
  stripTrack: HTMLElement | null;
  stripDistance: number;
}

export function collectDom(): Dom {
  const q = <T extends Element = HTMLElement>(s: string) => document.querySelector<T>(s);
  const qa = <T extends Element = HTMLElement>(s: string) => Array.from(document.querySelectorAll<T>(s));
  return {
    rail: q("[data-rail]"),
    railSteps: qa("[data-rail-step]"),
    heroAside: qa(".hero-report, .hero-caption, .pencil"),
    quality: { underexposed: q('[data-quality="underexposed"]'), rotated: q('[data-quality="rotated"]'), pass: q('[data-quality="pass"]') },
    pin: q('[data-pin="region"]'),
    faithNumber: q("[data-faith-number]"),
    faithRandom: q("[data-faith-random]"),
    faithVerdict: q("[data-faith-verdict]"),
    flipped: q(".contact-sheet .flipped"),
    tileLabels: qa(".contact-sheet .tile-label"),
    stabilityResult: q("[data-stability-result]"),
    iou: q("[data-iou]"),
    iouVerdict: q("[data-iou-verdict]"),
    iouText: q("[data-iou-text]"),
    noteSupport: q('[data-note="support"]'),
    noteSide: q('[data-note="side"]'),
    noteInject: q('[data-note="inject"]'),
    checks: { support: q('[data-check="support"]'), side: q('[data-check="side"]'), inject: q('[data-check="inject"]') },
    connectors: { support: q<SVGPathElement>('[data-connector="support"]'), side: q<SVGPathElement>('[data-connector="side"]') },
    chips: qa("[data-slot-chip]"),
    blocked: q("[data-blocked]"),
    stripTrack: q("[data-strip-track]"),
    stripDistance: 0,
  };
}

/** Geometry that only changes on resize: connector paths and the film strip's travel. */
export function measureDom(dom: Dom, layout: Layout) {
  const ctx = layout.parts.find((p) => p.id === "context");
  const stage = document.querySelector<HTMLElement>("#context .stage");
  if (ctx && stage) {
    const s = stage.getBoundingClientRect();
    const film = contain(ctx.slots.context, 1);
    const tx = film[0] + ((PRIMARY_BOX[0] + PRIMARY_BOX[2]) / 2) * film[2];
    const ty = film[1] + ((PRIMARY_BOX[1] + PRIMARY_BOX[3]) / 2) * film[3];
    for (const key of ["support", "side"] as const) {
      const span = key === "support" ? dom.noteSupport : dom.noteSide;
      const path = dom.connectors[key];
      if (!span || !path) continue;
      // Leave from the card's edge at the span's line, so the curve never crosses note text.
      const rects = span.getClientRects();
      const r = rects[rects.length - 1] ?? span.getBoundingClientRect();
      const card = span.closest(".note-card")?.getBoundingClientRect() ?? r;
      const x1 = card.right - s.left + 10;
      const y1 = r.top - s.top + r.height * 0.8;
      const ty2 = ty + (key === "support" ? -10 : 10);
      path.setAttribute("d", `M ${x1.toFixed(1)} ${y1.toFixed(1)} C ${(x1 + 90).toFixed(1)} ${y1.toFixed(1)}, ${(tx - 140).toFixed(1)} ${ty2.toFixed(1)}, ${tx.toFixed(1)} ${ty2.toFixed(1)}`);
    }
  }
  if (dom.stripTrack) {
    const track = dom.stripTrack.scrollWidth;
    dom.stripDistance = Math.max(0, track - layout.vw + layout.vw * 0.08);
  }
}

export function initialDom(dom: Dom) {
  // Below-the-fold start states, applied once the engine runs (never in the no-JS article).
  const zero: Frame["progress"] = { reader: 0, faithfulness: 0, stability: 0, second: 0, context: 0, firewall: 0, ledger: 0 };
  updateDom(dom, { progress: zero, slotsOnScreen: {}, railStage: -1, pastHero: 0, heroLift: 0 } as unknown as Frame);
}

export function updateDom(dom: Dom, f: Frame) {
  const p = f.progress;

  // Hero: the report and caption leave as the film lifts off the lightbox.
  const lift = 1 - ease(seg(f.heroLift ?? 0, 0.04, 0.35));
  dom.heroAside.forEach((el) => put(el, "opacity", lift.toFixed(3)));

  // Rail.
  put(dom.rail, f.pastHero > 0.5 ? "class+" : "class-", "is-visible");
  dom.railSteps.forEach((li, i) => {
    put(li, i === f.railStage ? "class+" : "class-", "is-active");
    put(li, i < f.railStage ? "class+" : "class-", "is-done");
  });

  // Reader: the quality gate's real reasons, then the pinned region label.
  const r = p.reader ?? 0;
  reveal(dom.quality.underexposed, ease(seg(r, 0.04, 0.09)));
  reveal(dom.quality.rotated, ease(seg(r, 0.17, 0.22)));
  reveal(dom.quality.pass, ease(seg(r, 0.31, 0.36)));
  const slot = f.slotsOnScreen["reader:reader"];
  if (dom.pin && slot) {
    // The pin's container is the slot itself, so its offset comes from the slot rect alone:
    // no layout read during scroll.
    const film = contain(slot, 1);
    const x = film[0] - slot[0] + PRIMARY_BOX[0] * film[2] + (film[1] + PRIMARY_BOX[1] * film[3] - 30 < 76 ? 8 : 0);
    // Above the box like a lead marker, unless the box touches the film's top edge (then inside).
    const above = film[1] + PRIMARY_BOX[1] * film[3] - 30;
    const y = (above < 76 ? above + 38 : above) - slot[1];
    put(dom.pin, "transform", `translate3d(${Math.round(x)}px, ${Math.round(y)}px, 0)`);
    put(dom.pin, "opacity", ease(seg(r, 0.66, 0.74)).toFixed(3));
  }

  // Faithfulness: the real readout counts down as the region is deleted.
  const fa = p.faithfulness ?? 0;
  const value = primary.prob - faith.drop * ease(seg(fa, 0.12, 0.42));
  put(dom.faithNumber, "text", value.toFixed(2));
  reveal(dom.faithRandom, ease(seg(fa, 0.55, 0.62)));
  const randSpan = dom.faithRandom?.querySelectorAll(".font-mono")[1];
  if (randSpan) put(randSpan, "text", (primary.prob - (primary.prob - probAfterRandom) * ease(seg(fa, 0.56, 0.78))).toFixed(2));
  reveal(dom.faithVerdict, ease(seg(fa, 0.8, 0.86)));
  void probAfterDeletion;

  // Stability.
  const st = p.stability ?? 0;
  dom.tileLabels.forEach((el, i) => reveal(el, ease(seg(st, 0.3 + i * 0.02, 0.38 + i * 0.02)), 6));
  put(dom.flipped, "--flip", ease(seg(st, 0.55, 0.64)).toFixed(3));
  reveal(dom.stabilityResult, ease(seg(st, 0.62, 0.7)));

  // Second read: the overlap is computed from the two boxes being drawn.
  const s2 = p.second ?? 0;
  const overlap = iou(PRIMARY_BOX, secondReaderBox(s2));
  const discordant = overlap < 0.1;
  put(dom.iou, "text", overlap.toFixed(2));
  put(dom.iouVerdict, discordant ? "class+" : "class-", "is-red");
  put(dom.iouText, "text", discordant ? "Discordant. A human reviews this first." : "Readers agree.");
  reveal(dom.iouVerdict, ease(seg(s2, 0.5, 0.56)));
  reveal(dom.iou, ease(seg(s2, 0.48, 0.54)));

  // Context: spans underline, connectors draw to the region, the injection is quarantined.
  const c = p.context ?? 0;
  put(dom.noteSupport, "--u", ease(seg(c, 0.12, 0.22)).toFixed(3));
  put(dom.noteSide, "--u", ease(seg(c, 0.28, 0.38)).toFixed(3));
  put(dom.connectors.support, "strokeDashoffset", (1 - ease(seg(c, 0.18, 0.32))).toFixed(3));
  put(dom.connectors.side, "strokeDashoffset", (1 - ease(seg(c, 0.34, 0.48))).toFixed(3));
  put(dom.noteInject, "--q", ease(seg(c, 0.56, 0.66)).toFixed(3));
  reveal(dom.checks.support, ease(seg(c, 0.2, 0.26)), 6);
  reveal(dom.checks.side, ease(seg(c, 0.36, 0.42)), 6);
  reveal(dom.checks.inject, ease(seg(c, 0.62, 0.68)), 6);

  // Firewall: slots fill from evidence, then the unsupported sentence is struck.
  const fw = p.firewall ?? 0;
  dom.chips.forEach((chip, i) => put(chip, "--fill", ease(seg(fw, 0.14 + i * 0.09, 0.24 + i * 0.09)).toFixed(3)));
  reveal(dom.blocked, ease(seg(fw, 0.5, 0.56)));
  put(dom.blocked, "--strike", ease(seg(fw, 0.62, 0.76)).toFixed(3));

  // Ledger: the film strip pans across the five entries.
  const lg = p.ledger ?? 0;
  put(dom.stripTrack, "--strip-x", `${(-dom.stripDistance * ease(seg(lg, 0.1, 0.92))).toFixed(1)}px`);
}

export type { Rect };
