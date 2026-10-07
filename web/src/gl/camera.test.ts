import { describe, expect, it } from "vitest";
import { fitView, limitsFor, panBy, screenToImage, zoomAt } from "./camera";
import { applyWindow, dragWindow } from "./windowing";

describe("camera", () => {
  it("fits and centres the image inside the padded viewport", () => {
    const v = fitView(1000, 500, 2000, 2000, 10);
    expect(v.scale).toBeCloseTo(0.24);
    expect(v.offsetX).toBeCloseTo((1000 - 2000 * 0.24) / 2);
    expect(v.offsetY).toBeCloseTo(10);
  });

  it("keeps the image point under the cursor fixed while zooming", () => {
    const v = fitView(800, 600, 1024, 1024);
    for (const [x, y, f] of [[100, 120, 2], [700, 50, 0.6], [400, 300, 3.7]] as const) {
      const before = screenToImage(v, x, y);
      const after = screenToImage(zoomAt(v, x, y, f, { min: 0.01, max: 100 }), x, y);
      expect(after[0]).toBeCloseTo(before[0], 6);
      expect(after[1]).toBeCloseTo(before[1], 6);
    }
  });

  it("clamps zoom to the limits", () => {
    const fit = fitView(800, 600, 1024, 1024);
    const lim = limitsFor(fit);
    expect(zoomAt(fit, 0, 0, 1e6, lim).scale).toBe(lim.max);
    expect(zoomAt(fit, 0, 0, 1e-6, lim).scale).toBe(lim.min);
  });

  it("pans by screen pixels", () => {
    const v = panBy({ offsetX: 1, offsetY: 2, scale: 3 }, 10, -5);
    expect(v).toEqual({ offsetX: 11, offsetY: -3, scale: 3 });
  });
});

describe("window/level", () => {
  it("maps the window edges to black and white", () => {
    const w = { level: 0.4, width: 0.2 };
    expect(applyWindow(0.3, w)).toBeCloseTo(0, 9);
    expect(applyWindow(0.5, w)).toBeCloseTo(1, 9);
    expect(applyWindow(0.4, w)).toBeCloseTo(0.5);
  });

  it("widens with a rightward drag and brightens with an upward drag", () => {
    const start = { level: 0.5, width: 0.5 };
    const w = dragWindow(start, 100, -100, 1000, 1000);
    expect(w.width).toBeGreaterThan(start.width);
    expect(w.level).toBeGreaterThan(start.level);
  });
});
