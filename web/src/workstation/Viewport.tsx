import { useEffect, useRef, useState, type MutableRefObject } from "react";
import { ArrowsOut, CircleHalf, Crosshair, Stack } from "@phosphor-icons/react";
import { assetUrl, getPixels } from "../api/client";
import { loadBitmap } from "../gl/core";
import { Viewer, type Layers, type ViewerBox } from "../gl/viewer";
import { attachViewerInput } from "../gl/viewerInput";
import { formatWindow } from "../gl/windowing";
import { boxOf, displayLabel, heatmapOf, leadMarker, maskOf, regionOf, strokeFor } from "../lib/findings";
import { gl as tokens } from "../design/tokens";
import { selectFinding, useSession } from "./store";

const bitmapCache = new Map<string, Promise<ImageBitmap>>();
const bitmap = (url: string) => {
  let p = bitmapCache.get(url);
  if (!p) {
    p = loadBitmap(url);
    p.catch(() => bitmapCache.delete(url));
    bitmapCache.set(url, p);
  }
  return p;
};

const LAYER_KEYS: { key: keyof Layers; label: string; hotkey: string }[] = [
  { key: "heatmap", label: "Heatmap", hotkey: "1" },
  { key: "mask", label: "Mask", hotkey: "2" },
  { key: "boxes", label: "Boxes", hotkey: "3" },
  { key: "anatomy", label: "Anatomy", hotkey: "4" },
];

export function Viewport({ viewerRef, onDropFile }: { viewerRef: MutableRefObject<Viewer | null>; onDropFile: (f: File) => void }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const labelRef = useRef<HTMLDivElement>(null);
  const readoutRef = useRef<HTMLSpanElement>(null);
  const [glError, setGlError] = useState<string | null>(null);
  const [layers, setLayers] = useState<Layers>({ heatmap: true, mask: true, boxes: true, anatomy: false });
  const [invert, setInvert] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [imageSize, setImageSize] = useState<[number, number] | null>(null);

  const mode = useSession((s) => s.mode);
  const image = useSession((s) => s.image);
  const modality = useSession((s) => s.modality);
  const studyId = useSession((s) => s.studyId);
  const findings = useSession((s) => s.findings);
  const selected = useSession((s) => s.selected);
  const hover = useSession((s) => s.hoverFinding);
  const active = hover ?? selected;
  const annotations = useSession((s) => s.annotations);
  const anatomyUrl = useSession((s) => s.anatomyUrl);
  const provenance = useSession((s) => s.provenance);
  const title = useSession((s) => s.title);
  const streaming = useSession((s) => s.streaming);

  // Viewer lifetime. StrictMode mounts twice; the viewer never calls loseContext, so the
  // second mount gets the same live context back.
  useEffect(() => {
    const canvas = canvasRef.current!;
    const v = Viewer.create(canvas);
    if (!v) {
      setGlError("This browser has no WebGL2, which the viewer needs to show images.");
      return;
    }
    viewerRef.current = v;
    const unsub = v.subscribe((s) => {
      setLayers(s.layers);
      setInvert(s.invert);
      if (readoutRef.current) {
        const w = formatWindow(s.windowing);
        readoutRef.current.textContent = `W ${w.W}  L ${w.L}  zoom ${(s.view.scale).toFixed(2)}×`;
      }
      positionLabel();
    });
    const detach = attachViewerInput(canvas, v, {
      onPick: (u, vv) => {
        const hit = pickFinding(u, vv);
        if (hit) selectFinding(hit);
      },
    });
    return () => {
      unsub();
      detach();
      v.destroy();
      viewerRef.current = null;
    };
    // positionLabel and pickFinding read refs/latest state; the viewer is created once.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Image: 8-bit first for an instant paint, then full-depth float16 for grayscale live studies.
  useEffect(() => {
    const v = viewerRef.current;
    if (!v || !image) return;
    let cancelled = false;
    const color = modality === "skin_dermoscopy";
    bitmap(image.url)
      .then((bmp) => {
        if (cancelled) return;
        const size: [number, number] = [image.width || bmp.width, image.height || bmp.height];
        v.setImage({ kind: "bitmap", bitmap: bmp, color }, size);
        setImageSize(size);
        if (studyId && !color) {
          getPixels(studyId)
            .then((px) => !cancelled && v.setImage({ kind: "r16f", data: px.data, width: px.width, height: px.height }, px.original))
            .catch(() => undefined); // the 8-bit image stays; full depth is an enhancement
        }
      })
      .catch((e) => !cancelled && setGlError(`Could not load the image: ${(e as Error).message}`));
    return () => {
      cancelled = true;
    };
  }, [image, modality, studyId, viewerRef]);

  // Overlays follow the selection.
  const sel = findings.find((f) => f.finding_id === selected) ?? null;
  useEffect(() => {
    const v = viewerRef.current;
    if (!v) return;
    let cancelled = false;
    const heat = sel ? heatmapOf(sel) : null;
    if (heat) bitmap(assetUrl(heat)).then((b) => !cancelled && v.setHeatmap(b)).catch(() => v.setHeatmap(null));
    else v.setHeatmap(null);

    const maskRef = (sel && maskOf(sel)) ?? annotations.find((a) => a.kind === "mask")?.ref ?? null;
    if (maskRef) bitmap(assetUrl(maskRef)).then((b) => !cancelled && v.setMask(b)).catch(() => v.setMask(null));
    else v.setMask(null);
    return () => {
      cancelled = true;
    };
  }, [sel, annotations, viewerRef]);

  useEffect(() => {
    const v = viewerRef.current;
    if (!v) return;
    if (anatomyUrl) bitmap(anatomyUrl).then((b) => v.setAnatomy(b)).catch(() => v.setAnatomy(null));
    else v.setAnatomy(null);
  }, [anatomyUrl, viewerRef]);

  useEffect(() => {
    const v = viewerRef.current;
    if (!v || !imageSize) return;
    const boxes: ViewerBox[] = [];
    let hi = -1;
    for (const f of findings) {
      const uv = boxOf(f, imageSize);
      if (!uv) continue;
      if (f.status === "rejected" && f.finding_id !== active) continue; // withheld boxes only when inspected
      const s = strokeFor(f.status);
      const isActive = f.finding_id === active;
      if (isActive) hi = boxes.length;
      boxes.push({ uv, color: s.color, dashed: s.dashed, alpha: isActive || !active ? 1 : 0.28 });
    }
    for (const a of annotations) {
      if (a.kind === "bbox" && a.bbox) boxes.push({ uv: a.bbox, color: tokens.ink, dashed: true, alpha: 0.9 });
    }
    v.setBoxes(boxes, hi);
    positionLabel();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [findings, active, annotations, imageSize, viewerRef]);

  function selectedBox(): [number, number, number, number] | null {
    const s = getSel();
    if (s && imageSize) return boxOf(s, imageSize);
    const ann = annotations.find((a) => a.kind === "bbox" && a.bbox);
    return ann?.bbox ?? null;
  }

  function getSel() {
    return findings.find((f) => f.finding_id === active) ?? null;
  }

  function positionLabel() {
    const v = viewerRef.current;
    const el = labelRef.current;
    if (!v || !el) return;
    const box = selectedBox();
    if (!box) {
      el.style.opacity = "0";
      return;
    }
    // Above the box's top-left corner, like a radiographer's marker beside the film edge.
    const [x, y] = v.uvToScreen(box[0], box[1]);
    el.style.opacity = "1";
    el.style.transform = `translate3d(${Math.round(x)}px, ${Math.round(y - 30)}px, 0)`;
  }

  function pickFinding(u: number, v: number): string | null {
    if (!imageSize) return null;
    let best: string | null = null;
    let bestArea = Infinity;
    for (const f of findings) {
      const b = boxOf(f, imageSize);
      if (!b) continue;
      if (u >= b[0] && u <= b[2] && v >= b[1] && v <= b[3]) {
        const area = (b[2] - b[0]) * (b[3] - b[1]);
        if (area < bestArea) {
          best = f.finding_id;
          bestArea = area;
        }
      }
    }
    return best;
  }

  const region = sel ? regionOf(sel) : null;
  const marker = leadMarker(region);
  const annotationOnly = !sel && annotations.find((a) => a.kind === "bbox");
  const summary = sel
    ? `${displayLabel(sel.label)}${region ? ` in the ${region}` : ""}, ${strokeFor(sel.status).label.toLowerCase()}`
    : mode === "empty"
      ? "No study open"
      : `${title}, no findings selected`;

  return (
    <section
      aria-label="Image viewport"
      className={`relative min-h-0 overflow-hidden bg-film-base ${dragOver ? "outline-2 outline-dashed outline-pencil-yellow -outline-offset-8" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragOver(false);
        const f = e.dataTransfer.files?.[0];
        if (f) onDropFile(f);
      }}
    >
      <canvas
        ref={canvasRef}
        role="img"
        aria-label={summary}
        tabIndex={0}
        className="absolute inset-0 h-full w-full touch-none select-none"
        style={{ viewTransitionName: "film" }}
      />

      {/* Label pinned to the selected box; moved by transform only. */}
      <div ref={labelRef} aria-hidden="true" className="pointer-events-none absolute left-0 top-0 opacity-0 transition-opacity duration-150">
        {(sel || annotationOnly) && (
          <div className="flex items-center gap-1.5 rounded-[var(--radius-control)] bg-film-base/85 px-1.5 py-1 text-[12px] leading-none text-ink">
            {marker && <span className="grid size-[18px] place-items-center rounded-[3px] border border-ink/70 font-mono text-[11px] font-medium">{marker}</span>}
            <span>{sel ? (region ?? displayLabel(sel.label)) : annotationOnly && annotationOnly.label}</span>
          </div>
        )}
      </div>

      {mode === "empty" && !glError && (
        <div className="pointer-events-none absolute inset-0 grid place-items-center px-6 text-center">
          <div>
            <p className="text-[21px] font-light text-ink">Drop a study here or pick a sample case.</p>
            <p className="mt-2 text-[14px] text-ink-dim">PNG, JPEG or DICOM. Images stay on this machine and the analysis server.</p>
          </div>
        </div>
      )}
      {glError && (
        <div className="absolute inset-0 grid place-items-center px-6 text-center">
          <p className="max-w-md text-ink">{glError}</p>
        </div>
      )}

      {/* Toolbar */}
      {mode !== "empty" && (
        <div className="absolute right-3 top-3 flex items-center gap-1 rounded-[var(--radius-panel)] bg-film-panel/90 p-1" role="toolbar" aria-label="Viewer layers">
          {LAYER_KEYS.map((l) => (
            <button
              key={l.key}
              type="button"
              aria-pressed={layers[l.key]}
              aria-keyshortcuts={l.hotkey}
              title={`${l.label} (${l.hotkey})`}
              onClick={() => viewerRef.current?.setLayers({ [l.key]: !layers[l.key] })}
              className={`flex h-7 items-center gap-1.5 rounded-[var(--radius-control)] px-2 text-[12px] transition-colors duration-150 ${
                layers[l.key] ? "bg-film-raised text-ink" : "text-ink-dim hover:text-ink"
              }`}
            >
              <kbd className="font-mono text-[10px] text-ink-dim">{l.hotkey}</kbd>
              {l.label}
            </button>
          ))}
          <span className="mx-1 h-4 w-px bg-film-line" aria-hidden="true" />
          <button type="button" aria-pressed={invert} title="Invert (I)" aria-label="Invert" onClick={() => viewerRef.current?.setInvert(!invert)} className={`grid size-7 place-items-center rounded-[var(--radius-control)] ${invert ? "bg-film-raised text-ink" : "text-ink-dim hover:text-ink"}`}>
            <CircleHalf size={16} />
          </button>
          <button type="button" title="Fit to view (0)" aria-label="Fit to view" onClick={() => viewerRef.current?.fit()} className="grid size-7 place-items-center rounded-[var(--radius-control)] text-ink-dim hover:text-ink">
            <ArrowsOut size={16} />
          </button>
          {sel && imageSize && (
            <button type="button" title="Centre on finding" aria-label="Centre on selected finding" onClick={() => { const b = boxOf(sel, imageSize); if (b) viewerRef.current?.centerOn(b); }} className="grid size-7 place-items-center rounded-[var(--radius-control)] text-ink-dim hover:text-ink">
              <Crosshair size={16} />
            </button>
          )}
        </div>
      )}

      {mode !== "empty" && (
        <div className="pointer-events-none absolute inset-x-3 bottom-3 flex items-end justify-between gap-4">
          <p className="max-w-[60ch] rounded-[var(--radius-control)] bg-film-base/80 px-1.5 py-0.5 text-[12px] leading-snug text-ink-dim">
            {streaming ? (
              <span className="inline-flex items-center gap-1.5"><Stack size={12} /> Analysis running</span>
            ) : (
              (sel ? provenance.findings : provenance.annotations) ?? provenance.findings
            )}
          </p>
          <span ref={readoutRef} className="shrink-0 whitespace-pre font-mono text-[12px] tabular text-ink-dim" aria-live="off" />
        </div>
      )}
    </section>
  );
}
