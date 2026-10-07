import vert from "./shaders/fullscreen.vert.glsl?raw";
import frag from "./shaders/viewer.frag.glsl?raw";
import {
  createGL,
  createProgram,
  fullscreenTriangle,
  resizeToDisplay,
  textureEmpty,
  textureFromBitmap,
  textureLUT,
  textureR16F,
  type GLHandle,
  type Program,
} from "./core";
import { fitView, imageToScreen, limitsFor, panBy, screenToImage, zoomAt, type View } from "./camera";
import { DEFAULT_WINDOW, type Windowing } from "./windowing";
import { cividisRGBA } from "./cividis";
import { gl as glTokens, type Rgb } from "../design/tokens";

export type ViewerImage =
  | { kind: "bitmap"; bitmap: ImageBitmap; color: boolean }
  | { kind: "r16f"; data: Uint16Array; width: number; height: number };

export interface ViewerBox {
  uv: [number, number, number, number];
  color: Rgb;
  dashed: boolean;
  /** 1 for the active finding; context boxes recede so one finding reads at a time. */
  alpha?: number;
}

export interface Layers {
  heatmap: boolean;
  mask: boolean;
  boxes: boolean;
  anatomy: boolean;
}

export interface ViewerState {
  view: View;
  windowing: Windowing;
  invert: boolean;
  layers: Layers;
  imageSize: [number, number] | null;
}

const UNIFORMS = [
  "u_canvas", "u_dpr", "u_offset", "u_scale", "u_imageSize", "u_image", "u_isColor", "u_level", "u_width",
  "u_invert", "u_heat", "u_lut", "u_heatOn", "u_mask", "u_maskOn", "u_anat", "u_anatOn", "u_boxCount",
  "u_boxes", "u_boxStyle", "u_boxAlpha", "u_boxHi", "u_boxesOn", "u_bg", "u_ink",
];

const MAX_DPR = 2;
/** Room for the layer toolbar above and the provenance caption and readout below. */
const FIT_PAD = { top: 56, right: 16, bottom: 44, left: 16 };

export class Viewer {
  private h: GLHandle;
  private prog!: Program;
  private vao!: WebGLVertexArrayObject;
  private tex: { image: WebGLTexture | null; heat: WebGLTexture | null; mask: WebGLTexture | null; anat: WebGLTexture | null; lut: WebGLTexture | null; empty: WebGLTexture | null } =
    { image: null, heat: null, mask: null, anat: null, lut: null, empty: null };
  private src: { image: ViewerImage | null; heat: ImageBitmap | null; mask: ImageBitmap | null; anat: ImageBitmap | null } =
    { image: null, heat: null, mask: null, anat: null };

  private view: View = { offsetX: 0, offsetY: 0, scale: 1 };
  private autoFit = true;
  private windowing: Windowing = { ...DEFAULT_WINDOW };
  private invert = false;
  private layers: Layers = { heatmap: true, mask: true, boxes: true, anatomy: false };
  private boxes: ViewerBox[] = [];
  private highlight = -1;
  private imageSize: [number, number] | null = null;
  private isColor = false;

  private raf = 0;
  private listeners = new Set<(s: ViewerState) => void>();
  private resizeObs: ResizeObserver;
  private disposed = false;
  /** Duration of the last GPU submission, for the perf HUD and benchmark. */
  lastFrameMs = 0;

  static create(canvas: HTMLCanvasElement): Viewer | null {
    const h = createGL(canvas);
    return h ? new Viewer(h) : null;
  }

  private constructor(h: GLHandle) {
    this.h = h;
    this.build();
    h.onRestore(() => {
      this.build();
      this.reuploadAll();
      this.requestRender();
    });
    this.resizeObs = new ResizeObserver(() => this.onResize());
    this.resizeObs.observe(h.canvas);
  }

  // -- GL resources ----------------------------------------------------------------------------
  private build() {
    const gl = this.h.gl;
    this.prog = createProgram(gl, vert, frag, UNIFORMS);
    this.vao = fullscreenTriangle(gl);
    this.tex.lut = textureLUT(gl, cividisRGBA());
    this.tex.empty = textureEmpty(gl);
    this.tex.image = this.tex.heat = this.tex.mask = this.tex.anat = null;
  }

  private reuploadAll() {
    if (this.src.image) this.uploadImage(this.src.image);
    if (this.src.heat) this.tex.heat = textureFromBitmap(this.h.gl, this.src.heat, "R8", { mips: false });
    if (this.src.mask) this.tex.mask = textureFromBitmap(this.h.gl, this.src.mask, "R8", { mips: false });
    if (this.src.anat) this.tex.anat = textureFromBitmap(this.h.gl, this.src.anat, "RGBA8", { mips: false, nearest: false });
  }

  private uploadImage(img: ViewerImage) {
    const gl = this.h.gl;
    if (this.tex.image) gl.deleteTexture(this.tex.image);
    if (img.kind === "bitmap") {
      this.tex.image = textureFromBitmap(gl, img.bitmap, img.color ? "RGBA8" : "R8");
      this.isColor = img.color;
    } else {
      const canMip = this.h.ext.colorBufferFloat || this.h.ext.colorBufferHalfFloat;
      this.tex.image = textureR16F(gl, img.data, img.width, img.height, canMip);
      this.isColor = false;
    }
  }

  // -- public API ------------------------------------------------------------------------------
  /** `originalSize` is the grid overlays live on; a downsampled 16-bit texture still maps 0..1. */
  setImage(img: ViewerImage, originalSize?: [number, number]) {
    this.src.image = img;
    const size: [number, number] = originalSize ?? (img.kind === "bitmap" ? [img.bitmap.width, img.bitmap.height] : [img.width, img.height]);
    const sizeChanged = !this.imageSize || this.imageSize[0] !== size[0] || this.imageSize[1] !== size[1];
    this.imageSize = size;
    this.uploadImage(img);
    if (sizeChanged) {
      this.autoFit = true;
      this.fit();
    }
    this.requestRender();
  }

  setHeatmap(bitmap: ImageBitmap | null) {
    this.replaceOverlay("heat", bitmap, "R8");
  }

  setMask(bitmap: ImageBitmap | null) {
    this.replaceOverlay("mask", bitmap, "R8");
  }

  setAnatomy(bitmap: ImageBitmap | null) {
    this.replaceOverlay("anat", bitmap, "RGBA8");
  }

  private replaceOverlay(key: "heat" | "mask" | "anat", bitmap: ImageBitmap | null, format: "R8" | "RGBA8") {
    const gl = this.h.gl;
    const old = this.tex[key];
    if (old) gl.deleteTexture(old);
    this.src[key] = bitmap;
    this.tex[key] = bitmap ? textureFromBitmap(gl, bitmap, format, { mips: false }) : null;
    this.requestRender();
  }

  setBoxes(boxes: ViewerBox[], highlight = -1) {
    this.boxes = boxes.slice(0, 8);
    this.highlight = highlight;
    this.requestRender();
  }

  setHighlight(index: number) {
    if (index === this.highlight) return;
    this.highlight = index;
    this.requestRender();
  }

  setLayers(layers: Partial<Layers>) {
    this.layers = { ...this.layers, ...layers };
    this.emitAndRender();
  }

  setWindow(w: Windowing) {
    this.windowing = w;
    this.emitAndRender();
  }

  setInvert(on: boolean) {
    this.invert = on;
    this.emitAndRender();
  }

  fit() {
    if (!this.imageSize) return;
    const c = this.h.canvas;
    this.view = fitView(c.clientWidth, c.clientHeight, this.imageSize[0], this.imageSize[1], FIT_PAD);
    this.autoFit = true;
    this.emitAndRender();
  }

  zoomAt(x: number, y: number, factor: number) {
    if (!this.imageSize) return;
    const c = this.h.canvas;
    const fit = fitView(c.clientWidth, c.clientHeight, this.imageSize[0], this.imageSize[1], FIT_PAD);
    this.view = zoomAt(this.view, x, y, factor, limitsFor(fit));
    this.autoFit = false;
    this.emitAndRender();
  }

  panBy(dx: number, dy: number) {
    this.view = panBy(this.view, dx, dy);
    this.autoFit = false;
    this.emitAndRender();
  }

  /** Centre a normalised image region in the viewport without changing zoom. */
  centerOn(uv: [number, number, number, number]) {
    if (!this.imageSize) return;
    const c = this.h.canvas;
    const cx = ((uv[0] + uv[2]) / 2) * this.imageSize[0];
    const cy = ((uv[1] + uv[3]) / 2) * this.imageSize[1];
    const [sx, sy] = imageToScreen(this.view, cx, cy);
    this.panBy(c.clientWidth / 2 - sx, c.clientHeight / 2 - sy);
  }

  uvToScreen(u: number, v: number): [number, number] {
    if (!this.imageSize) return [0, 0];
    return imageToScreen(this.view, u * this.imageSize[0], v * this.imageSize[1]);
  }

  screenToUv(x: number, y: number): [number, number] {
    if (!this.imageSize) return [0, 0];
    const [ix, iy] = screenToImage(this.view, x, y);
    return [ix / this.imageSize[0], iy / this.imageSize[1]];
  }

  getState(): ViewerState {
    return { view: this.view, windowing: this.windowing, invert: this.invert, layers: this.layers, imageSize: this.imageSize };
  }

  subscribe(fn: (s: ViewerState) => void): () => void {
    this.listeners.add(fn);
    fn(this.getState());
    return () => this.listeners.delete(fn);
  }

  destroy() {
    this.disposed = true;
    cancelAnimationFrame(this.raf);
    this.resizeObs.disconnect();
    this.listeners.clear();
    this.h.dispose(); // no loseContext(): a StrictMode remount must find a live context
  }

  // -- rendering -------------------------------------------------------------------------------
  private emitAndRender() {
    const s = this.getState();
    this.listeners.forEach((fn) => fn(s));
    this.requestRender();
  }

  private onResize() {
    if (this.autoFit) this.fit();
    else this.requestRender();
  }

  requestRender() {
    if (this.raf || this.disposed) return;
    this.raf = requestAnimationFrame(() => {
      this.raf = 0;
      this.draw();
    });
  }

  private draw() {
    if (this.h.isLost()) return;
    const t0 = performance.now();
    const gl = this.h.gl;
    const dpr = Math.min(window.devicePixelRatio || 1, MAX_DPR);
    resizeToDisplay(this.h.canvas, dpr);
    gl.viewport(0, 0, this.h.canvas.width, this.h.canvas.height);
    gl.useProgram(this.prog.program);
    const u = this.prog.u;
    const size = this.imageSize ?? [1, 1];

    gl.uniform2f(u.u_canvas, this.h.canvas.width, this.h.canvas.height);
    gl.uniform1f(u.u_dpr, dpr);
    gl.uniform2f(u.u_offset, this.view.offsetX, this.view.offsetY);
    gl.uniform1f(u.u_scale, this.view.scale);
    gl.uniform2f(u.u_imageSize, size[0], size[1]);
    gl.uniform1i(u.u_isColor, this.isColor ? 1 : 0);
    gl.uniform1f(u.u_level, this.windowing.level);
    gl.uniform1f(u.u_width, this.windowing.width);
    gl.uniform1i(u.u_invert, this.invert ? 1 : 0);
    gl.uniform3fv(u.u_bg, glTokens.filmBase);
    gl.uniform3fv(u.u_ink, glTokens.ink);

    const bind = (unit: number, tex: WebGLTexture | null, name: string) => {
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, tex ?? this.tex.empty);
      gl.uniform1i(u[name], unit);
    };
    bind(0, this.tex.image, "u_image");
    bind(1, this.tex.heat, "u_heat");
    bind(2, this.tex.lut, "u_lut");
    bind(3, this.tex.mask, "u_mask");
    bind(4, this.tex.anat, "u_anat");
    gl.uniform1f(u.u_heatOn, this.layers.heatmap && this.tex.heat ? 1 : 0);
    gl.uniform1f(u.u_maskOn, this.layers.mask && this.tex.mask ? 1 : 0);
    gl.uniform1f(u.u_anatOn, this.layers.anatomy && this.tex.anat ? 1 : 0);

    const boxes = new Float32Array(32);
    const styles = new Float32Array(32);
    const alphas = new Float32Array(8);
    this.boxes.forEach((b, i) => {
      boxes.set(b.uv, i * 4);
      styles.set([b.color[0], b.color[1], b.color[2], b.dashed ? 1 : 0], i * 4);
      alphas[i] = b.alpha ?? 1;
    });
    gl.uniform1i(u.u_boxCount, this.boxes.length);
    gl.uniform4fv(u.u_boxes, boxes);
    gl.uniform4fv(u.u_boxStyle, styles);
    gl.uniform1fv(u.u_boxAlpha, alphas);
    gl.uniform1i(u.u_boxHi, this.highlight);
    gl.uniform1f(u.u_boxesOn, this.layers.boxes ? 1 : 0);

    if (this.tex.image) {
      gl.bindVertexArray(this.vao);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      gl.bindVertexArray(null);
    } else {
      gl.clearColor(glTokens.filmBase[0], glTokens.filmBase[1], glTokens.filmBase[2], 1);
      gl.clear(gl.COLOR_BUFFER_BIT);
    }
    this.lastFrameMs = performance.now() - t0;
  }
}
