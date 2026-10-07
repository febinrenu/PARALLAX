// Window/level in normalised intensity units (0..1 of the image's stored range).
export interface Windowing {
  level: number;
  width: number;
}

export const DEFAULT_WINDOW: Windowing = { level: 0.5, width: 1 };

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

/**
 * Right-drag convention used by radiology viewers: horizontal drag widens or narrows the
 * window (contrast), vertical drag moves the level (brightness). A full viewport drag spans
 * the whole range.
 */
export function dragWindow(start: Windowing, dx: number, dy: number, viewW: number, viewH: number): Windowing {
  return {
    width: clamp(start.width + (dx / Math.max(viewW, 1)) * 2, 0.01, 2),
    level: clamp(start.level - (dy / Math.max(viewH, 1)) * 1.2, -0.5, 1.5),
  };
}

/** Display mapping, identical to the shader: returns the grey value for a stored intensity. */
export function applyWindow(v: number, w: Windowing): number {
  return clamp((v - (w.level - w.width / 2)) / Math.max(w.width, 1e-4), 0, 1);
}

/** Readout in the same units a radiologist reads them: percent of range, rounded. */
export function formatWindow(w: Windowing): { W: string; L: string } {
  return { W: (w.width * 100).toFixed(0), L: (w.level * 100).toFixed(0) };
}
