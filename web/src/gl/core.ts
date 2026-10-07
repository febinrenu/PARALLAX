// Minimal WebGL2 helpers shared by the story renderer and the clinical viewer. Kept small and
// explicit on purpose: every texture format and pixel-store setting matters for medical images.

export interface GLHandle {
  gl: WebGL2RenderingContext;
  canvas: HTMLCanvasElement;
  isLost: () => boolean;
  /** Called after a lost context is restored; rebuild programs and textures here. */
  onRestore: (fn: () => void) => () => void;
  onLost: (fn: () => void) => () => void;
  ext: { colorBufferFloat: boolean; colorBufferHalfFloat: boolean; parallelCompile: KHRParallelShaderCompile | null };
  dispose: () => void;
}

interface KHRParallelShaderCompile {
  readonly COMPLETION_STATUS_KHR: number;
}

export function createGL(canvas: HTMLCanvasElement): GLHandle | null {
  const gl = canvas.getContext("webgl2", {
    alpha: false, // GL paints its own background, so compositing over the DOM is free
    antialias: false,
    depth: false,
    stencil: false,
    premultipliedAlpha: false,
    preserveDrawingBuffer: false,
    powerPreference: "high-performance",
  });
  if (!gl) return null;

  let lost = false;
  const restoreFns = new Set<() => void>();
  const lostFns = new Set<() => void>();
  const ext = {
    colorBufferFloat: false,
    colorBufferHalfFloat: false,
    parallelCompile: null as KHRParallelShaderCompile | null,
  };
  const readExtensions = () => {
    ext.colorBufferFloat = !!gl.getExtension("EXT_color_buffer_float");
    ext.colorBufferHalfFloat = !!gl.getExtension("EXT_color_buffer_half_float");
    ext.parallelCompile = gl.getExtension("KHR_parallel_shader_compile") as KHRParallelShaderCompile | null;
  };
  readExtensions();

  const onLost = (e: Event) => {
    e.preventDefault(); // required, or the browser never restores the context
    lost = true;
    lostFns.forEach((fn) => fn());
  };
  const onRestored = () => {
    lost = false;
    readExtensions();
    restoreFns.forEach((fn) => fn());
  };
  canvas.addEventListener("webglcontextlost", onLost, false);
  canvas.addEventListener("webglcontextrestored", onRestored, false);

  return {
    gl,
    canvas,
    ext,
    isLost: () => lost,
    onRestore: (fn) => {
      restoreFns.add(fn);
      return () => restoreFns.delete(fn);
    },
    onLost: (fn) => {
      lostFns.add(fn);
      return () => lostFns.delete(fn);
    },
    dispose: () => {
      canvas.removeEventListener("webglcontextlost", onLost);
      canvas.removeEventListener("webglcontextrestored", onRestored);
      restoreFns.clear();
      lostFns.clear();
    },
  };
}

export interface Program {
  program: WebGLProgram;
  u: Record<string, WebGLUniformLocation | null>;
}

function compile(gl: WebGL2RenderingContext, type: number, src: string): WebGLShader {
  const sh = gl.createShader(type);
  if (!sh) throw new Error("createShader failed");
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  return sh;
}

/** Compile and link. With KHR_parallel_shader_compile the status check is deferred to `ready`. */
export function createProgram(gl: WebGL2RenderingContext, vs: string, fs: string, uniforms: string[]): Program {
  const program = gl.createProgram();
  if (!program) throw new Error("createProgram failed");
  const v = compile(gl, gl.VERTEX_SHADER, vs);
  const f = compile(gl, gl.FRAGMENT_SHADER, fs);
  gl.attachShader(program, v);
  gl.attachShader(program, f);
  gl.bindAttribLocation(program, 0, "a_pos");
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const log = `${gl.getShaderInfoLog(v) ?? ""}\n${gl.getShaderInfoLog(f) ?? ""}\n${gl.getProgramInfoLog(program) ?? ""}`;
    throw new Error(`shader link failed:\n${log}`);
  }
  gl.deleteShader(v);
  gl.deleteShader(f);
  const u: Program["u"] = {};
  for (const name of uniforms) u[name] = gl.getUniformLocation(program, name);
  return { program, u };
}

/** A single triangle covering the viewport; cheaper than a quad (no diagonal seam). */
export function fullscreenTriangle(gl: WebGL2RenderingContext): WebGLVertexArrayObject {
  const vao = gl.createVertexArray();
  const buf = gl.createBuffer();
  if (!vao || !buf) throw new Error("VAO allocation failed");
  gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  gl.bindVertexArray(null);
  return vao;
}

export type TexFormat = "R8" | "RGBA8";

/**
 * Upload a decoded bitmap. Grayscale films go to R8 (luminance is read from red); colour
 * images stay RGBA8 and the shader computes luminance for window/level.
 */
export function textureFromBitmap(
  gl: WebGL2RenderingContext,
  bitmap: ImageBitmap,
  format: TexFormat,
  opts: { mips?: boolean; nearest?: boolean } = {},
): WebGLTexture {
  const tex = gl.createTexture();
  if (!tex) throw new Error("createTexture failed");
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
  gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
  gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.NONE);
  if (format === "R8") {
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, gl.RED, gl.UNSIGNED_BYTE, bitmap);
  } else {
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, bitmap);
  }
  setFiltering(gl, opts.mips ?? true, opts.nearest ?? false);
  return tex;
}

/**
 * Full-depth grayscale as R16F (half floats in a Uint16Array). R16F is filterable in core
 * WebGL2 but not colour-renderable, so generateMipmap is only legal with a colour-buffer
 * float extension; otherwise sample with LINEAR and no mips.
 */
export function textureR16F(
  gl: WebGL2RenderingContext,
  halfFloats: Uint16Array,
  width: number,
  height: number,
  canMip: boolean,
): WebGLTexture {
  const tex = gl.createTexture();
  if (!tex) throw new Error("createTexture failed");
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1); // odd widths would break the upload otherwise
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.R16F, width, height, 0, gl.RED, gl.HALF_FLOAT, halfFloats);
  setFiltering(gl, canMip, false);
  return tex;
}

/** Small lookup table (e.g. the cividis colormap) as a 256x1 RGBA8 texture. */
export function textureLUT(gl: WebGL2RenderingContext, rgba: Uint8Array): WebGLTexture {
  const tex = gl.createTexture();
  if (!tex) throw new Error("createTexture failed");
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, rgba.length / 4, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, rgba);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  return tex;
}

/** 1x1 black texture so samplers are always bound to something valid. */
export function textureEmpty(gl: WebGL2RenderingContext): WebGLTexture {
  const tex = gl.createTexture();
  if (!tex) throw new Error("createTexture failed");
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0, 0, 0, 0]));
  setFiltering(gl, false, true);
  return tex;
}

function setFiltering(gl: WebGL2RenderingContext, mips: boolean, nearest: boolean) {
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  if (nearest) {
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    return;
  }
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  if (mips) {
    gl.generateMipmap(gl.TEXTURE_2D);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
  } else {
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  }
}

/** Decode an image URL off the main thread without any colour management or premultiplication. */
export async function loadBitmap(url: string, signal?: AbortSignal): Promise<ImageBitmap> {
  const res = await fetch(url, { signal });
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const blob = await res.blob();
  return createImageBitmap(blob, {
    colorSpaceConversion: "none",
    premultiplyAlpha: "none",
    imageOrientation: "from-image",
  });
}

/** Resize the drawing buffer to the canvas's CSS size times a clamped DPR. Returns true if changed. */
export function resizeToDisplay(canvas: HTMLCanvasElement, dpr: number): boolean {
  const w = Math.max(1, Math.round(canvas.clientWidth * dpr));
  const h = Math.max(1, Math.round(canvas.clientHeight * dpr));
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
    return true;
  }
  return false;
}
