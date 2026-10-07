// The choreography as a pure function: (scroll, layout, pointer, intro time) -> what to draw.
// Nothing here keeps state between frames, so a fast scroll, an anchor jump or the End key
// always produces the correct picture for where the page is.
import { gl as tokens, type Rgb } from "../design/tokens";
import { chest, faith, primary, wrist } from "./data";

export type Rect = [number, number, number, number];
type Vec3 = [number, number, number];

export interface Box {
  uv: [number, number, number, number];
  color: Rgb;
  dashed: boolean;
  alpha: number;
}

export interface FilmDraw {
  tex: "wrist" | "chest" | "atlas";
  rect: Rect;
  rot: Vec3;
  z: number;
  alpha: number;
  bright: number;
  backlit: number;
  swap: number;
  sub: [number, number, number, number];
  heat: number;
  anat: number;
  del: number;
  delThresh: number;
  delShift: [number, number];
  delRandom: number;
  gamma: number;
  layerSep: number;
  boxes: Box[];
  loupe: Vec3;
  grain: number;
}

export interface Scene {
  panel: { rect: Rect; on: number };
  films: FilmDraw[];
  time: number;
}

export interface PartLayout {
  id: string;
  kind: "sticky" | "flow";
  top: number; // document px
  len: number; // px
  /** Slots relative to the sticky stage (sticky parts) or the document (flow parts). */
  slots: Record<string, Rect>;
  stage: number; // rail index, -1 for none
}

export interface Layout {
  vw: number;
  vh: number;
  parts: PartLayout[];
}

export interface Inputs {
  scrollY: number;
  layout: Layout;
  pointer: { x: number; y: number; inside: boolean }; // client px
  parallax: { x: number; y: number }; // smoothed, -1..1
  intro: { panel: number }; // hero lightbox flicker, 0..1
  time: number;
}

export interface Frame {
  scene: Scene;
  /** How far the film has lifted off the hero lightbox (0..1). */
  heroLift: number;
  /** Local progress of every part (0..1), for the DOM side of each chapter. */
  progress: Record<string, number>;
  /** Screen rect of named slots this frame, for DOM labels pinned to the film. */
  slotsOnScreen: Record<string, Rect>;
  railStage: number;
  pastHero: number;
}

// -- helpers -------------------------------------------------------------------------------------
const clamp01 = (v: number) => Math.min(1, Math.max(0, v));
export const seg = (t: number, a: number, b: number) => clamp01((t - a) / (b - a));
export const ease = (t: number) => t * t * (3 - 2 * t);
const easeOut = (t: number) => 1 - (1 - t) ** 3;
const mix = (a: number, b: number, t: number) => a + (b - a) * t;
const mixRect = (a: Rect, b: Rect, t: number): Rect => [mix(a[0], b[0], t), mix(a[1], b[1], t), mix(a[2], b[2], t), mix(a[3], b[3], t)];
/** Rises over [a,b], holds, falls over [c,d]. */
const bump = (t: number, a: number, b: number, c: number, d: number) => ease(seg(t, a, b)) * (1 - ease(seg(t, c, d)));

/** Largest rect of the given aspect (w/h) centred in r. */
export function contain(r: Rect, aspect: number): Rect {
  const [x, y, w, h] = r;
  if (w / h > aspect) {
    const nw = h * aspect;
    return [x + (w - nw) / 2, y, nw, h];
  }
  const nh = w / aspect;
  return [x, y + (h - nh) / 2, w, nh];
}

const inset = (r: Rect, fx: number, fy: number): Rect => [r[0] + r[2] * fx, r[1] + r[3] * fy, r[2] * (1 - 2 * fx), r[3] * (1 - 2 * fy)];
const union = (a: Rect, b: Rect): Rect => {
  const x0 = Math.min(a[0], b[0]), y0 = Math.min(a[1], b[1]);
  const x1 = Math.max(a[0] + a[2], b[0] + b[2]), y1 = Math.max(a[1] + a[3], b[1] + b[3]);
  return [x0, y0, x1 - x0, y1 - y0];
};

export function iou(a: [number, number, number, number], b: [number, number, number, number]): number {
  const ix = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const iy = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const inter = ix * iy;
  const area = (r: typeof a) => (r[2] - r[0]) * (r[3] - r[1]);
  return inter / (area(a) + area(b) - inter || 1);
}

const CHEST_ASPECT = chest.size[0] / chest.size[1];
const WRIST_ASPECT = wrist.size[0] / wrist.size[1];
const PRIMARY_BOX = (primary.bbox ?? [0.3, 0.2, 0.5, 0.45]) as [number, number, number, number];
const YELLOW = tokens.pencilYellow;
const RED = tokens.pencilRed;
const INK = tokens.ink;

/** The illustrative second reader's box: starts near the specialist's, then drifts away. */
export function secondReaderBox(t: number): [number, number, number, number] {
  const w = PRIMARY_BOX[2] - PRIMARY_BOX[0];
  const h = PRIMARY_BOX[3] - PRIMARY_BOX[1];
  const start: [number, number, number, number] = [PRIMARY_BOX[0] + w * 0.22, PRIMARY_BOX[1] + h * 0.1, PRIMARY_BOX[2] + w * 0.22, PRIMARY_BOX[3] + h * 0.1];
  // Ends overlapping a little (IoU about 0.07): just under the 0.1 discordance rule, as real
  // disagreements usually look, rather than an implausibly clean 0.
  const end: [number, number, number, number] = [PRIMARY_BOX[0] + w * 0.58, PRIMARY_BOX[1] + h * 0.71, PRIMARY_BOX[0] + w * 1.5, PRIMARY_BOX[1] + h * 1.61];
  const k = ease(seg(t, 0.6, 0.9));
  return [mix(start[0], end[0], k), mix(start[1], end[1], k), mix(start[2], end[2], k), mix(start[3], end[3], k)];
}

function base(rect: Rect, tex: FilmDraw["tex"] = "chest"): FilmDraw {
  return {
    tex, rect, rot: [0, 0, 0], z: 0, alpha: 1, bright: 1, backlit: 0, swap: 0, sub: [0, 0, 1, 1], heat: 0, anat: 0, del: 0,
    delThresh: faith.region_threshold, delShift: faith.random_example_shift, delRandom: 0, gamma: 1, layerSep: 0, boxes: [], loupe: [0, 0, 0], grain: 0.5,
  };
}

const hidden = (rect: Rect): FilmDraw => ({ ...base(rect), alpha: 0 });
const FILMS = 10; // 0 main, 1..8 contact-sheet tiles, 9 second eye

/** Every part fills the same ten entries so neighbouring parts can be blended entry by entry. */
function filmSet(main: FilmDraw, extra: Partial<Record<number, FilmDraw>> = {}): FilmDraw[] {
  return Array.from({ length: FILMS }, (_, i) => (i === 0 ? main : extra[i] ?? hidden(main.rect)));
}

function blendFilm(a: FilmDraw, b: FilmDraw, t: number): FilmDraw {
  if (t <= 0) return a;
  if (t >= 1) return b;
  // An entry that is invisible on one side keeps the other side's geometry, so it fades in place.
  const ra = a.alpha < 0.002 ? b.rect : a.rect;
  const rb = b.alpha < 0.002 ? a.rect : b.rect;
  return {
    tex: t < 0.5 ? a.tex : b.tex,
    rect: mixRect(ra, rb, t),
    rot: [mix(a.rot[0], b.rot[0], t), mix(a.rot[1], b.rot[1], t), mix(a.rot[2], b.rot[2], t)],
    z: mix(a.z, b.z, t),
    alpha: mix(a.alpha, b.alpha, t),
    bright: mix(a.bright, b.bright, t),
    backlit: mix(a.backlit, b.backlit, t),
    swap: mix(a.swap, b.swap, t),
    sub: t < 0.5 ? a.sub : b.sub,
    heat: mix(a.heat, b.heat, t),
    anat: mix(a.anat, b.anat, t),
    del: mix(a.del, b.del, t),
    delThresh: a.delThresh,
    delShift: a.delShift,
    delRandom: t < 0.5 ? a.delRandom : b.delRandom,
    gamma: mix(a.gamma, b.gamma, t),
    layerSep: mix(a.layerSep, b.layerSep, t),
    boxes: [...a.boxes.map((x) => ({ ...x, alpha: x.alpha * (1 - t) })), ...b.boxes.map((x) => ({ ...x, alpha: x.alpha * t }))].slice(0, 2),
    loupe: t < 0.5 ? a.loupe : b.loupe,
    grain: mix(a.grain, b.grain, t),
  };
}

// -- chapter states ------------------------------------------------------------------------------
type Slots = (name: string) => Rect;

interface StateCtx {
  slot: Slots;
  inp: Inputs;
}

const STATES: Record<string, (t: number, c: StateCtx) => FilmDraw[]> = {
  hero(_t, { slot, inp }) {
    const rect = contain(slot("hero"), WRIST_ASPECT);
    const f = { ...base(rect, "wrist"), backlit: inp.intro.panel, grain: 0.35 };
    const p = inp.pointer;
    if (p.inside && p.x >= rect[0] && p.x <= rect[0] + rect[2] && p.y >= rect[1] && p.y <= rect[1] + rect[3]) {
      f.loupe = [(p.x - rect[0]) / rect[2], (p.y - rect[1]) / rect[3], Math.min(84, rect[2] * 0.18)];
    }
    return filmSet(f);
  },

  rule(t, { slot }) {
    const rect = contain(slot("rule"), CHEST_ASPECT);
    return filmSet({
      ...base(rect, "wrist"),
      swap: ease(seg(t, 0.1, 0.8)),
      bright: 0.4,
      rot: [0.1, -0.14 + 0.2 * t, 0],
      z: -80,
      grain: 0.6,
    });
  },

  reader(t, { slot }) {
    const f = base(contain(slot("reader"), CHEST_ASPECT));
    f.gamma = 1 + 1.9 * bump(t, 0.02, 0.08, 0.13, 0.18);
    f.rot = [0, 0, 0.279 * bump(t, 0.15, 0.21, 0.26, 0.32)];
    f.heat = ease(seg(t, 0.36, 0.56));
    f.anat = ease(seg(t, 0.48, 0.7)) * 0.9;
    f.boxes = [{ uv: PRIMARY_BOX, color: YELLOW, dashed: false, alpha: ease(seg(t, 0.6, 0.7)) }];
    return filmSet(f);
  },

  faithfulness(t, { slot }) {
    const f = base(contain(slot("faithfulness"), CHEST_ASPECT));
    f.heat = 1 - 0.6 * ease(seg(t, 0.1, 0.22));
    const real = ease(seg(t, 0.12, 0.42)) * (1 - ease(seg(t, 0.46, 0.52)));
    const rand = ease(seg(t, 0.56, 0.78));
    f.del = t < 0.5 ? real : rand;
    f.delRandom = t < 0.5 ? 0 : 1;
    f.boxes = [{ uv: PRIMARY_BOX, color: YELLOW, dashed: false, alpha: 0.85 }];
    return filmSet(f);
  },

  stability(t, { slot }) {
    const cells = Array.from({ length: 8 }, (_, i) => contain(slot(`tile-${i}`), 1));
    const sheet = cells.reduce((u, r) => union(u, r));
    const size = Math.min(sheet[3], sheet[2] * 0.5);
    const center: Rect = [sheet[0] + (sheet[2] - size) / 2, sheet[1] + (sheet[3] - size) / 2, size, size];
    const main = { ...base(center), alpha: 1 - ease(seg(t, 0.06, 0.3)) };
    const extra: Record<number, FilmDraw> = {};
    cells.forEach((cell, i) => {
      const k = easeOut(seg(t, 0.04 + i * 0.025, 0.3 + i * 0.025));
      extra[i + 1] = {
        ...base(mixRect(center, cell, k), "atlas"),
        sub: [(i % 4) / 4, Math.floor(i / 4) / 2, 0.25, 0.5],
        alpha: ease(seg(t, 0.03 + i * 0.025, 0.12 + i * 0.025)),
        grain: 0.3,
      };
    });
    return filmSet(main, extra);
  },

  second(t, { slot, inp }) {
    const l = contain(slot("eye-l"), CHEST_ASPECT);
    const r = contain(slot("eye-r"), CHEST_ASPECT);
    const u = union(l, r);
    const size = Math.min(u[3], u[2] * 0.6);
    const c: Rect = [u[0] + (u[2] - size) / 2, u[1] + (u[3] - size) / 2, size, size];
    const sep = ease(seg(t, 0.05, 0.28)) * (1 - ease(seg(t, 0.38, 0.48)));
    const split = ease(seg(t, 0.44, 0.6));
    const px = inp.parallax.x, py = inp.parallax.y;
    const second = secondReaderBox(t);
    const discordant = iou(PRIMARY_BOX, second) < 0.1;

    const main = base(mixRect(c, l, split));
    main.layerSep = sep;
    main.rot = [sep * (0.32 + py * 0.1), sep * (-0.62 + px * 0.16), sep * 0.04];
    main.z = sep * -120;
    main.heat = sep > 0.01 ? 0 : 0.35;
    main.boxes = [{ uv: PRIMARY_BOX, color: YELLOW, dashed: false, alpha: 1 }];

    const eye = base(mixRect(c, r, split));
    eye.alpha = ease(seg(t, 0.44, 0.5));
    eye.heat = 0.25;
    eye.boxes = [
      { uv: PRIMARY_BOX, color: YELLOW, dashed: false, alpha: 0.3 },
      { uv: second, color: discordant ? RED : INK, dashed: discordant, alpha: 1 },
    ];
    return filmSet(main, { 9: eye });
  },

  context(_t, { slot }) {
    const f = base(contain(slot("context"), CHEST_ASPECT));
    f.heat = 0.5;
    f.boxes = [{ uv: PRIMARY_BOX, color: YELLOW, dashed: false, alpha: 0.9 }];
    return filmSet(f);
  },

  firewall(_t, { slot }) {
    const f = base(contain(slot("firewall"), CHEST_ASPECT));
    f.bright = 0.6;
    f.heat = 0.35;
    f.rot = [0.06, -0.18, 0];
    return filmSet(f);
  },

  ledger(_t, { slot }) {
    return filmSet(hidden(contain(slot("firewall-ghost"), CHEST_ASPECT)));
  },

  closing(_t, { slot }) {
    const f = { ...base(contain(inset(slot("closing"), 0.09, 0.07), CHEST_ASPECT)), backlit: 1, grain: 0.3 };
    return filmSet(f);
  },
};

// -- composition ---------------------------------------------------------------------------------
function stageTop(p: PartLayout, scrollY: number, vh: number): number {
  if (p.kind === "flow") return p.top - scrollY;
  if (scrollY < p.top) return p.top - scrollY;
  return Math.min(0, p.top + p.len - vh - scrollY);
}

function localProgress(p: PartLayout, scrollY: number, vh: number): number {
  if (p.kind === "flow") return clamp01((scrollY - (p.top - vh)) / (p.len + vh));
  return clamp01((scrollY - p.top) / Math.max(1, p.len - vh));
}

export function compose(inp: Inputs): Frame {
  const { layout, scrollY } = inp;
  const { vh, parts } = layout;
  const progress: Record<string, number> = {};
  const slotsOnScreen: Record<string, Rect> = {};

  const slotFn = (p: PartLayout): Slots => {
    const top = stageTop(p, scrollY, vh);
    return (name) => {
      const r = p.slots[name] ?? p.slots[Object.keys(p.slots)[0]] ?? [0, 0, 1, 1];
      const screen: Rect = p.kind === "flow" ? [r[0], r[1] - scrollY, r[2], r[3]] : [r[0], r[1] + top, r[2], r[3]];
      slotsOnScreen[`${p.id}:${name}`] = screen;
      return screen;
    };
  };

  for (const p of parts) progress[p.id] = localProgress(p, scrollY, vh);

  // The part that owns the film is the last one whose top has reached the viewport top.
  let cur = 0;
  for (let i = 0; i < parts.length; i++) if (parts[i].top <= scrollY + 1) cur = i;
  const next = Math.min(cur + 1, parts.length - 1);
  const h = next === cur ? 0 : ease(clamp01((scrollY - (parts[next].top - vh)) / vh));

  const pa = parts[cur];
  const pb = parts[next];
  const stA = STATES[pa.id] ?? STATES.ledger;
  const stB = STATES[pb.id] ?? STATES.ledger;
  const a = stA(progress[pa.id], { slot: slotFn(pa), inp });
  const b = h > 0 ? stB(0, { slot: slotFn(pb), inp }) : a;
  const dip = Math.sin(Math.PI * h); // 0 at both ends of a handoff, 1 halfway
  const films = a.map((f, i) => {
    const m = blendFilm(f, b[i], h);
    if (dip > 0.001) {
      m.bright *= 1 - 0.5 * dip;
      m.z -= 240 * dip;
    }
    return m;
  });

  // The lightbox panel: the hero's, switching off as the film lifts; the closing one, switching on.
  const hero = parts.find((p) => p.id === "hero");
  const closing = parts.find((p) => p.id === "closing");
  let panel: Scene["panel"] = { rect: [0, 0, 0, 0], on: 0 };
  if (hero && scrollY < hero.top + hero.len) {
    const r = hero.slots.panel;
    // The tubes die quickly once the film lifts, so no half-lit grey panel lingers.
    const on = pa.id === "hero" ? inp.intro.panel * (1 - ease(seg(h, 0, 0.32))) : 0;
    panel = { rect: [r[0], r[1] - scrollY, r[2], r[3]], on };
  } else if (closing) {
    const r = closing.slots.closing;
    const on = ease(seg(progress.closing, 0.18, 0.4));
    panel = { rect: [r[0], r[1] - scrollY, r[2], r[3]], on };
  }

  // DOM labels need the reader slot's screen position even when another part owns the film.
  for (const p of parts) if (p.kind === "sticky") for (const name of Object.keys(p.slots)) slotFn(p)(name);

  const railStage = h > 0.5 ? pb.stage : pa.stage;
  const pastHero = hero ? clamp01((scrollY - hero.len * 0.55) / (hero.len * 0.25)) : 1;
  const heroLift = pa.id === "hero" ? h : 1;
  return { scene: { panel, films, time: inp.time }, progress, slotsOnScreen, railStage, pastHero, heroLift };
}
