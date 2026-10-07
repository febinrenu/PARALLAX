import type { Viewer } from "./viewer";
import { dragWindow, type Windowing } from "./windowing";

export interface ViewerInputOptions {
  /** A click (not a drag) at a normalised image position, e.g. to select a finding's box. */
  onPick?: (u: number, v: number) => void;
}

/**
 * Pointer map for the viewport: left-drag or space+drag pans, right-drag sets window/level,
 * wheel zooms about the cursor, two-finger pinch zooms on touch, double-click fits.
 * The canvas rect is read once per gesture, never during a move.
 */
export function attachViewerInput(canvas: HTMLCanvasElement, viewer: Viewer, opts: ViewerInputOptions = {}): () => void {
  let rect = canvas.getBoundingClientRect();
  let mode: "pan" | "window" | null = null;
  let last = { x: 0, y: 0 };
  let downAt = { x: 0, y: 0, t: 0 };
  let startWindow: Windowing = viewer.getState().windowing;
  const touches = new Map<number, { x: number; y: number }>();
  let pinchDist = 0;

  const local = (e: PointerEvent | WheelEvent) => ({ x: e.clientX - rect.left, y: e.clientY - rect.top });

  const onDown = (e: PointerEvent) => {
    rect = canvas.getBoundingClientRect();
    canvas.setPointerCapture(e.pointerId);
    const p = local(e);
    if (e.pointerType === "touch") {
      touches.set(e.pointerId, p);
      if (touches.size === 2) {
        const [a, b] = [...touches.values()];
        pinchDist = Math.hypot(a.x - b.x, a.y - b.y);
      }
    }
    mode = e.button === 2 ? "window" : "pan";
    startWindow = viewer.getState().windowing;
    last = p;
    downAt = { ...p, t: performance.now() };
  };

  const onMove = (e: PointerEvent) => {
    if (!mode) return;
    const p = local(e);
    if (e.pointerType === "touch" && touches.has(e.pointerId)) {
      touches.set(e.pointerId, p);
      if (touches.size === 2) {
        const [a, b] = [...touches.values()];
        const d = Math.hypot(a.x - b.x, a.y - b.y);
        if (pinchDist > 0) viewer.zoomAt((a.x + b.x) / 2, (a.y + b.y) / 2, d / pinchDist);
        pinchDist = d;
        last = p;
        return;
      }
    }
    if (mode === "pan") viewer.panBy(p.x - last.x, p.y - last.y);
    else viewer.setWindow(dragWindow(startWindow, p.x - downAt.x, p.y - downAt.y, rect.width, rect.height));
    last = p;
  };

  const onUp = (e: PointerEvent) => {
    if (canvas.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId);
    touches.delete(e.pointerId);
    if (touches.size < 2) pinchDist = 0;
    const p = local(e);
    const moved = Math.hypot(p.x - downAt.x, p.y - downAt.y);
    if (mode === "pan" && moved < 4 && performance.now() - downAt.t < 400 && opts.onPick) {
      const [u, v] = viewer.screenToUv(p.x, p.y);
      opts.onPick(u, v);
    }
    if (touches.size === 0) mode = null;
  };

  const onWheel = (e: WheelEvent) => {
    e.preventDefault();
    rect = canvas.getBoundingClientRect();
    const p = local(e);
    const unit = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? rect.height : 1;
    viewer.zoomAt(p.x, p.y, Math.exp(-e.deltaY * unit * 0.0015));
  };

  const onContext = (e: Event) => e.preventDefault();
  const onDbl = () => viewer.fit();

  canvas.addEventListener("pointerdown", onDown);
  canvas.addEventListener("pointermove", onMove);
  canvas.addEventListener("pointerup", onUp);
  canvas.addEventListener("pointercancel", onUp);
  canvas.addEventListener("wheel", onWheel, { passive: false });
  canvas.addEventListener("contextmenu", onContext);
  canvas.addEventListener("dblclick", onDbl);
  return () => {
    canvas.removeEventListener("pointerdown", onDown);
    canvas.removeEventListener("pointermove", onMove);
    canvas.removeEventListener("pointerup", onUp);
    canvas.removeEventListener("pointercancel", onUp);
    canvas.removeEventListener("wheel", onWheel);
    canvas.removeEventListener("contextmenu", onContext);
    canvas.removeEventListener("dblclick", onDbl);
  };
}
