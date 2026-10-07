// 2D view transform for the viewer. All values are CSS pixels; `scale` is CSS px per image px.
export interface View {
  offsetX: number;
  offsetY: number;
  scale: number;
}

export interface ZoomLimits {
  min: number;
  max: number;
}

export interface Padding {
  top: number;
  right: number;
  bottom: number;
  left: number;
}

export function fitView(viewW: number, viewH: number, imgW: number, imgH: number, pad: number | Padding = 16): View {
  const p = typeof pad === "number" ? { top: pad, right: pad, bottom: pad, left: pad } : pad;
  const availW = viewW - p.left - p.right;
  const availH = viewH - p.top - p.bottom;
  const scale = Math.max(1e-3, Math.min(availW / imgW, availH / imgH));
  return {
    scale,
    offsetX: p.left + (availW - imgW * scale) / 2,
    offsetY: p.top + (availH - imgH * scale) / 2,
  };
}

export function imageToScreen(v: View, ix: number, iy: number): [number, number] {
  return [v.offsetX + ix * v.scale, v.offsetY + iy * v.scale];
}

export function screenToImage(v: View, x: number, y: number): [number, number] {
  return [(x - v.offsetX) / v.scale, (y - v.offsetY) / v.scale];
}

/** Zoom by `factor` keeping the image point under (x, y) fixed on screen. */
export function zoomAt(v: View, x: number, y: number, factor: number, limits: ZoomLimits): View {
  const scale = Math.min(limits.max, Math.max(limits.min, v.scale * factor));
  const k = scale / v.scale;
  return {
    scale,
    offsetX: x - (x - v.offsetX) * k,
    offsetY: y - (y - v.offsetY) * k,
  };
}

export function panBy(v: View, dx: number, dy: number): View {
  return { ...v, offsetX: v.offsetX + dx, offsetY: v.offsetY + dy };
}

/** Zoom limits relative to the fit scale: from half the fit to 32 screen px per image px. */
export function limitsFor(fit: View): ZoomLimits {
  return { min: fit.scale * 0.5, max: Math.max(fit.scale * 4, 32) };
}
