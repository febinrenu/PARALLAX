// The story engine: one gsap.ticker loop drives Lenis, composes the frame (a pure function of
// scroll), draws WebGL only when something changed, and updates the DOM side of each chapter.
// A module-level singleton, so React StrictMode's double effects can never start it twice.
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { SplitText } from "gsap/SplitText";
import Lenis from "lenis";
import { collectDom, initialDom, measureDom, updateDom } from "./chapters";
import { chest, primary, wrist } from "./data";
import { StoryRenderer } from "./renderer";
import { compose, type Layout, type PartLayout, type Rect } from "./scene";

let booting: Promise<void> | null = null;

export function startStory(): Promise<void> {
  return (booting ??= boot());
}

function docRect(el: Element | null, sy: number): Rect {
  if (!el) return [0, 0, 0, 0];
  const r = el.getBoundingClientRect();
  return [r.left, r.top + sy, r.width, r.height];
}

function measure(): Layout {
  const sy = window.scrollY;
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  const parts: PartLayout[] = [];

  const hero = document.getElementById("top");
  if (hero) {
    const r = docRect(hero, sy);
    parts.push({ id: "hero", kind: "flow", top: r[1], len: r[3], stage: -1, slots: { hero: docRect(hero.querySelector('[data-slot="hero"]'), sy), panel: docRect(hero.querySelector('[data-slot="panel"]'), sy) } });
  }
  document.querySelectorAll<HTMLElement>("section[data-chapter]").forEach((sec) => {
    const stage = sec.querySelector<HTMLElement>(".stage");
    if (!stage) return;
    const r = docRect(sec, sy);
    const s = stage.getBoundingClientRect();
    const slots: Record<string, Rect> = {};
    stage.querySelectorAll<HTMLElement>("[data-slot]").forEach((el) => {
      const e = el.getBoundingClientRect();
      slots[el.dataset.slot!] = [e.left, e.top - s.top, e.width, e.height];
    });
    parts.push({ id: sec.dataset.chapter!, kind: "sticky", top: r[1], len: r[3], stage: Number(sec.dataset.stage ?? -1), slots });
  });
  const limits = document.getElementById("limits");
  if (limits) {
    const r = docRect(limits, sy);
    parts.push({ id: "closing", kind: "flow", top: r[1], len: r[3], stage: -1, slots: { closing: docRect(limits.querySelector('[data-slot="closing"]'), sy) } });
  }
  parts.sort((a, b) => a.top - b.top);
  return { vw, vh, parts };
}

async function boot(): Promise<void> {
  const html = document.documentElement;
  const canvas = document.querySelector<HTMLCanvasElement>("[data-story-canvas]");
  if (!canvas) throw new Error("no story canvas");

  gsap.registerPlugin(ScrollTrigger, SplitText);
  ScrollTrigger.config({ ignoreMobileResize: true });

  const renderer = StoryRenderer.create(canvas);
  if (!renderer) throw new Error("WebGL2 unavailable");
  await renderer.load({ wrist: wrist.image, chest: chest.image, atlas: chest.perturbation_atlas, heat: primary.heatmap, anatomy: chest.anatomy, cdf: chest.cdf });

  const dom = collectDom();
  initialDom(dom);

  let layout = measure();
  measureDom(dom, layout);

  // Lenis smooths once; every ScrollTrigger scrubs with `true` so nothing smooths twice.
  const lenis = new Lenis({ autoRaf: false, lerp: 0.1, anchors: true, syncTouch: false });
  lenis.on("scroll", ScrollTrigger.update);
  if (import.meta.env.DEV) (window as unknown as { __lenis: Lenis }).__lenis = lenis; // screenshot tests jump with it

  let dirty = true;
  let lastScroll = -1;
  let dpr = Math.min(window.devicePixelRatio || 1, 1.75);
  let lastWidth = window.innerWidth;
  renderer.resize(dpr);

  ScrollTrigger.addEventListener("refresh", () => {
    layout = measure();
    measureDom(dom, layout);
    if (window.innerWidth !== lastWidth) lastWidth = window.innerWidth;
    renderer.resize(dpr);
    dirty = true;
  });

  // Pointer: the hero loupe and the second-read parallax.
  const pointer = { x: -1, y: -1, inside: false };
  const parallax = { x: 0, y: 0 };
  const aim = { x: 0, y: 0 };
  window.addEventListener(
    "pointermove",
    (e) => {
      pointer.x = e.clientX;
      pointer.y = e.clientY;
      pointer.inside = true;
      aim.x = (e.clientX / window.innerWidth) * 2 - 1;
      aim.y = (e.clientY / window.innerHeight) * 2 - 1;
      dirty = true;
    },
    { passive: true },
  );
  document.documentElement.addEventListener("pointerleave", () => {
    pointer.inside = false;
    dirty = true;
  });

  // Hero intro: the tubes stutter on (two dips, under the three-flashes-per-second limit),
  // the grease pencil circles the fracture, a sentence assembles, the next one is struck.
  const intro = { panel: 1 };
  let introPlaying = false;
  const pencil = document.querySelector<SVGEllipseElement>(".pencil-stroke");
  const sentences = Array.from(document.querySelectorAll<HTMLElement>(".hero-sentence"));
  const strike = document.querySelector<HTMLElement>('[data-sentence="blocked"] s');
  if (strike) strike.style.setProperty("--strike", "0");
  const heroInView = window.scrollY < window.innerHeight * 0.5;
  if (heroInView) {
    introPlaying = true;
    gsap
      .timeline({ delay: 0.15, onUpdate: () => (dirty = true), onComplete: () => (introPlaying = false) })
      .to(intro, { panel: 0.06, duration: 0.05, ease: "none" })
      .to(intro, { panel: 0.06, duration: 0.22 })
      .to(intro, { panel: 0.7, duration: 0.04, ease: "none" })
      .to(intro, { panel: 0.2, duration: 0.07, ease: "none" })
      .to(intro, { panel: 1, duration: 0.32, ease: "power2.out" })
      .to(pencil, { "--draw": 0, duration: 0.9, ease: "power2.inOut" }, "+=0.15")
      .add(() => sentences[0]?.classList.add("is-in"), "+=0.25")
      .add(() => sentences[1]?.classList.add("is-in"), "+=0.65")
      .to(strike, { "--strike": 1, duration: 0.5, ease: "power2.inOut" }, "+=0.5");
  } else {
    pencil?.style.setProperty("--draw", "0");
    strike?.style.setProperty("--strike", "1");
    sentences.forEach((s) => s.classList.add("is-in"));
  }

  // Chapter headings rise line by line as each chapter slides in (they own only their lines).
  document.querySelectorAll<HTMLElement>("[data-split]").forEach((el) => {
    const chapter = el.closest("section");
    SplitText.create(el, {
      type: "lines",
      mask: "lines",
      aria: "auto",
      autoSplit: true,
      onSplit: (self) =>
        gsap.from(self.lines, {
          yPercent: 110,
          stagger: 0.12,
          ease: "none",
          scrollTrigger: { trigger: chapter, start: "top 85%", end: "top 15%", scrub: true },
        }),
    });
  });

  // Adaptive resolution: step the DPR down when frames run long, back up with headroom.
  const frameTimes: number[] = [];
  let lastDraw = 0;
  const adapt = (now: number) => {
    const dt = now - lastDraw;
    lastDraw = now;
    if (dt > 50) return; // an idle gap, not a slow frame
    frameTimes.push(dt);
    if (frameTimes.length < 30) return;
    const sorted = [...frameTimes].sort((a, b) => a - b);
    const p90 = sorted[Math.floor(sorted.length * 0.9)];
    frameTimes.length = 0;
    const next = p90 > 20 ? Math.max(1, dpr - 0.375) : p90 < 13 ? Math.min(Math.min(window.devicePixelRatio || 1, 1.75), dpr + 0.25) : dpr;
    if (next !== dpr) {
      dpr = next;
      renderer.resize(dpr);
      dirty = true;
    }
  };

  const draw = (now: number) => {
    const y = lenis.scroll;
    parallax.x += (aim.x - parallax.x) * 0.08;
    parallax.y += (aim.y - parallax.y) * 0.08;
    const settling = Math.abs(aim.x - parallax.x) + Math.abs(aim.y - parallax.y) > 0.002;
    if (!dirty && y === lastScroll && !introPlaying && !lenis.isScrolling && !settling) return;
    dirty = false;
    lastScroll = y;
    const frame = compose({ scrollY: y, layout, pointer, parallax, intro, time: now / 1000 });
    renderer.draw(frame.scene);
    updateDom(dom, frame);
    adapt(now);
  };

  // First frame is drawn before the swap, so the WebGL film replaces the <img> seamlessly.
  draw(performance.now());
  canvas.style.transition = "none";
  html.classList.add("story-gl");
  requestAnimationFrame(() => (canvas.style.transition = ""));

  gsap.ticker.add((time) => {
    lenis.raf(time * 1000);
    draw(performance.now());
  });
  gsap.ticker.lagSmoothing(0);

  document.fonts?.ready.then(() => ScrollTrigger.refresh());
  window.addEventListener("load", () => ScrollTrigger.refresh(), { once: true });
}
