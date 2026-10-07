import filmVert from "./shaders/film.vert.glsl?raw";
import filmFrag from "./shaders/film.frag.glsl?raw";
import roomFrag from "./shaders/room.frag.glsl?raw";
import fullVert from "../gl/shaders/fullscreen.vert.glsl?raw";
import { createGL, createProgram, fullscreenTriangle, loadBitmap, textureEmpty, textureFromBitmap, textureLUT, type GLHandle, type Program } from "../gl/core";
import { cividisRGBA } from "../gl/cividis";
import { gl as tokens } from "../design/tokens";
import type { FilmDraw, Scene } from "./scene";

const FILM_U = [
  "u_view", "u_rect", "u_rot", "u_z", "u_pivot", "u_focal", "u_tex", "u_tex2", "u_cdf", "u_swap", "u_sub", "u_heat", "u_lut",
  "u_heatAmt", "u_anat", "u_anatAmt", "u_del", "u_delThresh", "u_delShift", "u_delRandom", "u_gamma", "u_bright", "u_alpha",
  "u_backlit", "u_layer", "u_fit", "u_box0", "u_box0Style", "u_box0Alpha", "u_box1", "u_box1Style", "u_box1Alpha", "u_sizePx", "u_loupe",
  "u_time", "u_grain", "u_ink",
];
const ROOM_U = ["u_canvas", "u_dpr", "u_bg", "u_panel", "u_on", "u_time"];

export interface StoryTextures {
  wrist: string;
  chest: string;
  atlas: string;
  heat: string;
  anatomy: string | null;
  cdf: number[];
}

type TexKey = "wrist" | "chest" | "atlas" | "heat" | "anatomy";

export class StoryRenderer {
  private h: GLHandle;
  private film!: Program;
  private room!: Program;
  private quad!: WebGLVertexArrayObject;
  private tri!: WebGLVertexArrayObject;
  private tex: Partial<Record<TexKey | "lut" | "cdf" | "empty", WebGLTexture>> = {};
  private bitmaps: Partial<Record<TexKey, ImageBitmap>> = {};
  private cdf: number[] = [];
  dpr = 1;

  static create(canvas: HTMLCanvasElement): StoryRenderer | null {
    const h = createGL(canvas);
    return h ? new StoryRenderer(h) : null;
  }

  private constructor(h: GLHandle) {
    this.h = h;
    this.build();
    h.onRestore(() => {
      this.build();
      this.uploadAll();
    });
  }

  get canvas() {
    return this.h.canvas;
  }

  isLost() {
    return this.h.isLost();
  }

  private build() {
    const gl = this.h.gl;
    this.film = createProgram(gl, filmVert, filmFrag, FILM_U);
    this.room = createProgram(gl, fullVert, roomFrag, ROOM_U);
    this.tri = fullscreenTriangle(gl);
    const vao = gl.createVertexArray()!;
    const buf = gl.createBuffer()!;
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
    this.quad = vao;
    this.tex = { lut: textureLUT(gl, cividisRGBA()), empty: textureEmpty(gl) };
  }

  async load(t: StoryTextures) {
    const entries: [TexKey, string | null][] = [["wrist", t.wrist], ["chest", t.chest], ["atlas", t.atlas], ["heat", t.heat], ["anatomy", t.anatomy]];
    await Promise.all(entries.map(async ([k, url]) => {
      if (url) this.bitmaps[k] = await loadBitmap(url);
    }));
    this.cdf = t.cdf;
    this.uploadAll();
    // Warm both programs once on a hidden frame so the first real frame never stalls.
    this.h.gl.finish();
  }

  private uploadAll() {
    const gl = this.h.gl;
    const film = (k: TexKey, fmt: "R8" | "RGBA8", mips: boolean) => {
      const b = this.bitmaps[k];
      if (b) this.tex[k] = textureFromBitmap(gl, b, fmt, { mips });
    };
    film("wrist", "R8", true);
    film("chest", "R8", true);
    film("atlas", "R8", true);
    film("heat", "R8", false);
    film("anatomy", "RGBA8", false);
    if (this.cdf.length) {
      const bytes = new Uint8Array(this.cdf.map((v) => Math.round(v * 255)));
      const t = gl.createTexture()!;
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, bytes.length, 1, 0, gl.RED, gl.UNSIGNED_BYTE, bytes);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      this.tex.cdf = t;
    }
  }

  resize(dpr: number) {
    this.dpr = dpr;
    const c = this.h.canvas;
    const w = Math.max(1, Math.round(c.clientWidth * dpr));
    const hh = Math.max(1, Math.round(c.clientHeight * dpr));
    if (c.width !== w || c.height !== hh) {
      c.width = w;
      c.height = hh;
    }
  }

  draw(scene: Scene) {
    if (this.h.isLost()) return;
    const gl = this.h.gl;
    const c = this.h.canvas;
    gl.viewport(0, 0, c.width, c.height);
    gl.disable(gl.DEPTH_TEST);

    // Room and lightbox.
    gl.disable(gl.BLEND);
    gl.useProgram(this.room.program);
    const ru = this.room.u;
    gl.uniform2f(ru.u_canvas, c.width, c.height);
    gl.uniform1f(ru.u_dpr, this.dpr);
    gl.uniform3fv(ru.u_bg, tokens.filmBase);
    const p = scene.panel;
    gl.uniform4f(ru.u_panel, p.rect[0], p.rect[1], p.rect[2], p.rect[3]);
    gl.uniform1f(ru.u_on, p.on);
    gl.uniform1f(ru.u_time, scene.time);
    gl.bindVertexArray(this.tri);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    // Films, back to front, premultiplied alpha.
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.useProgram(this.film.program);
    const u = this.film.u;
    const vw = c.width / this.dpr;
    const vh = c.height / this.dpr;
    gl.uniform2f(u.u_view, vw, vh);
    gl.uniform2f(u.u_pivot, vw / 2, vh / 2);
    gl.uniform1f(u.u_focal, 1600);
    gl.uniform1f(u.u_time, scene.time);
    gl.uniform3fv(u.u_ink, tokens.ink);
    this.bind(2, "lut", "u_lut");
    this.bind(3, "heat", "u_heat");
    this.bind(4, "anatomy", "u_anat");
    this.bind(5, "cdf", "u_cdf");
    gl.bindVertexArray(this.quad);

    for (const f of scene.films) {
      if (f.alpha <= 0.002 || f.rect[2] < 1 || f.rect[3] < 1) continue;
      if (f.layerSep > 0.001) {
        // Exploded view: image, heat and boxes on separate planes.
        this.drawFilm(f, 1, 0);
        this.drawFilm(f, 2, f.layerSep * 150);
        this.drawFilm(f, 3, f.layerSep * 300);
      } else {
        this.drawFilm(f, 0, 0);
      }
    }
    gl.bindVertexArray(null);
  }

  private bind(unit: number, key: keyof StoryRenderer["tex"], name: string) {
    const gl = this.h.gl;
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, this.tex[key] ?? this.tex.empty!);
    gl.uniform1i(this.film.u[name], unit);
  }

  private drawFilm(f: FilmDraw, layer: number, extraZ: number) {
    const gl = this.h.gl;
    const u = this.film.u;
    this.bind(0, f.tex, "u_tex");
    this.bind(1, "chest", "u_tex2");
    gl.uniform4f(u.u_rect, f.rect[0], f.rect[1], f.rect[2], f.rect[3]);
    gl.uniform3f(u.u_rot, f.rot[0], f.rot[1], f.rot[2]);
    gl.uniform1f(u.u_z, f.z + extraZ);
    gl.uniform1f(u.u_swap, f.swap);
    gl.uniform4f(u.u_sub, f.sub[0], f.sub[1], f.sub[2], f.sub[3]);
    // Contain-fit: the texture keeps its own aspect whatever the quad's shape is mid-flight.
    const bmp = this.bitmaps[f.tex];
    const texAspect = bmp ? (bmp.width * f.sub[2]) / (bmp.height * f.sub[3]) : 1;
    const quadAspect = f.rect[2] / Math.max(1, f.rect[3]);
    gl.uniform2f(u.u_fit, Math.max(1, quadAspect / texAspect), Math.max(1, texAspect / quadAspect));
    gl.uniform1f(u.u_heatAmt, f.heat);
    gl.uniform1f(u.u_anatAmt, f.anat);
    gl.uniform1f(u.u_del, f.del);
    gl.uniform1f(u.u_delThresh, f.delThresh);
    gl.uniform2f(u.u_delShift, f.delShift[0], f.delShift[1]);
    gl.uniform1f(u.u_delRandom, f.delRandom);
    gl.uniform1f(u.u_gamma, f.gamma);
    gl.uniform1f(u.u_bright, f.bright);
    gl.uniform1f(u.u_alpha, f.alpha);
    gl.uniform1f(u.u_backlit, f.backlit);
    gl.uniform1i(u.u_layer, layer);
    const [b0, b1] = [f.boxes[0], f.boxes[1]];
    gl.uniform4fv(u.u_box0, b0 ? b0.uv : [0, 0, 0, 0]);
    gl.uniform4fv(u.u_box0Style, b0 ? [...b0.color, b0.dashed ? 1 : 0] : [0, 0, 0, 0]);
    gl.uniform1f(u.u_box0Alpha, b0 ? b0.alpha : 0);
    gl.uniform4fv(u.u_box1, b1 ? b1.uv : [0, 0, 0, 0]);
    gl.uniform4fv(u.u_box1Style, b1 ? [...b1.color, b1.dashed ? 1 : 0] : [0, 0, 0, 0]);
    gl.uniform1f(u.u_box1Alpha, b1 ? b1.alpha : 0);
    gl.uniform2f(u.u_sizePx, f.rect[2], f.rect[3]);
    gl.uniform3f(u.u_loupe, f.loupe[0], f.loupe[1], f.loupe[2]);
    gl.uniform1f(u.u_grain, f.grain);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
  }
}
