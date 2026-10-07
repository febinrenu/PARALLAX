#version 300 es
// The film. Every chapter effect is a uniform on this one program, so the protagonist never
// changes material between chapters. Output is premultiplied alpha.
precision highp float;

in vec2 v_uv;
out vec4 o;

uniform sampler2D u_tex;      // R8 film, mipmapped (deletion blur samples higher mips)
uniform sampler2D u_tex2;     // the film that develops in (u_swap)
uniform sampler2D u_cdf;      // luminance CDF of u_tex2, 64x1
uniform float u_swap;
uniform vec4 u_sub;           // texture sub-rect (atlas cell), uv x, y, w, h
uniform vec2 u_fit;           // contain-fit of u_tex inside the quad: uv scale (>= 1 letterboxes)

uniform sampler2D u_heat;
uniform sampler2D u_lut;
uniform float u_heatAmt;
uniform sampler2D u_anat;
uniform float u_anatAmt;

uniform float u_del;          // deletion strength 0..1
uniform float u_delThresh;    // heat >= threshold is the measured top-10% region
uniform vec2 u_delShift;      // np.roll offset of the random comparison region
uniform float u_delRandom;    // 0 = measured region, 1 = random region

uniform float u_gamma;        // > 1 darkens (the underexposed quality beat)
uniform float u_bright;
uniform float u_alpha;
uniform float u_backlit;      // 1 = on lit glass, 0 = self-lit in the dark room
uniform int u_layer;          // 0 full, 1 image only, 2 heat only, 3 boxes only

uniform vec4 u_box0;          // uv xyxy
uniform vec4 u_box0Style;     // rgb, dashed
uniform float u_box0Alpha;
uniform vec4 u_box1;
uniform vec4 u_box1Style;
uniform float u_box1Alpha;

uniform vec2 u_sizePx;        // on-screen size, CSS px
uniform vec3 u_loupe;         // centre uv, radius in px (<= 0 disables)
uniform float u_time;
uniform float u_grain;
uniform vec3 u_ink;

const vec3 GLASS = vec3(0.918, 0.945, 0.961);
const vec3 COLD = vec3(0.86, 0.9, 0.93);

float hash(vec2 p) {
  p = fract(p * vec2(123.34, 456.21));
  p += dot(p, p + 45.32);
  return fract(p.x * p.y);
}

float boxEdge(vec2 p, vec2 b0, vec2 b1) {
  vec2 c = 0.5 * (b0 + b1);
  vec2 h = 0.5 * abs(b1 - b0);
  vec2 d = abs(p - c) - h;
  return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}

float boxStroke(vec2 px, vec4 box, vec4 style) {
  vec2 b0 = box.xy * u_sizePx, b1 = box.zw * u_sizePx;
  float d = abs(boxEdge(px, b0, b1));
  float a = 1.0 - smoothstep(1.1, 2.2, d);
  if (style.w > 0.5) a *= step(0.45, fract(((px.x - b0.x) + (px.y - b0.y)) / 12.0));
  return a;
}

vec2 subUv(vec2 uv) { return u_sub.xy + uv * u_sub.zw; }
vec2 fitUv(vec2 uv) { return (uv - 0.5) * u_fit + 0.5; }

void main() {
  vec2 uv = v_uv;
  vec2 px = uv * u_sizePx;

  // Loupe: a 2x magnifier with a local contrast boost, like a radiologist leaning in.
  float inLoupe = 0.0;
  float ring = 0.0;
  if (u_loupe.z > 0.0) {
    vec2 c = u_loupe.xy * u_sizePx;
    float d = length(px - c);
    inLoupe = 1.0 - smoothstep(u_loupe.z - 1.0, u_loupe.z, d);
    ring = 1.0 - smoothstep(0.6, 1.8, abs(d - u_loupe.z));
    uv = mix(uv, u_loupe.xy + (uv - u_loupe.xy) * 0.5, inLoupe);
  }

  float heat = texture(u_heat, uv).r;
  vec2 delUv = fract(uv - u_delShift * u_delRandom);
  float region = smoothstep(u_delThresh - 0.04, u_delThresh + 0.01, texture(u_heat, delUv).r);
  float lod = region * u_del * 6.5;

  vec2 uv1 = fitUv(uv);
  float inside1 = step(0.0, uv1.x) * step(uv1.x, 1.0) * step(0.0, uv1.y) * step(uv1.y, 1.0);
  float l1 = textureLod(u_tex, subUv(clamp(uv1, 0.0, 1.0)), lod).r * inside1;
  float l = l1;
  if (u_swap > 0.0) {
    float l2 = textureLod(u_tex2, uv, lod).r;
    float c2 = texture(u_cdf, vec2(l2, 0.5)).r;
    // Develops like film in a tray: the densest tissue (bone) comes up first, everything
    // else stays a faint fog until its turn. Wide edges keep it photographic, not posterised.
    float reveal = smoothstep(1.0 - u_swap - 0.28, 1.0 - u_swap + 0.04, c2);
    float fog = l2 * 0.18 * smoothstep(0.0, 0.25, u_swap);
    float l2v = mix(fog, l2, reveal);
    l = max(l1 * (1.0 - smoothstep(0.0, 0.5, u_swap)), l2v);
  }
  l = pow(clamp(l, 0.0, 1.0), u_gamma);
  if (inLoupe > 0.0) l = mix(l, smoothstep(0.08, 0.92, l), inLoupe);

  vec3 col = l * mix(COLD, GLASS, u_backlit) * u_bright;
  // Outside a letterboxed film the quad is clear: a portrait sheet floats until the square
  // film that develops in covers the whole quad.
  float cover = max(inside1, smoothstep(0.0, 0.35, u_swap));
  float alpha = u_alpha * cover;

  // Faint edges for the floating planes of the exploded view, so they read as sheets.
  vec2 e2 = min(px, u_sizePx - px);
  float plane = 1.0 - smoothstep(0.6, 1.6, min(e2.x, e2.y));

  if (u_layer == 2) { // heat layer of the exploded view
    vec3 hc = texture(u_lut, vec2(heat, 0.5)).rgb;
    float a = smoothstep(0.12, 0.5, heat) * 0.85;
    vec4 c = vec4(hc * a, a);
    c = mix(c, vec4(u_ink * 0.5, 0.5), plane);
    o = c * u_alpha;
    return;
  }

  if (u_layer == 0 && u_heatAmt > 0.0) {
    vec3 hc = texture(u_lut, vec2(heat, 0.5)).rgb;
    col = mix(col, hc * max(u_bright, 0.6), u_heatAmt * 0.62 * smoothstep(0.18, 0.55, heat));
    float fw = max(fwidth(heat), 1e-4);
    col = mix(col, hc, u_heatAmt * 0.9 * (1.0 - smoothstep(0.0, 1.5 * fw, abs(heat - 0.5))));
  }

  if (u_layer == 0 && u_anatAmt > 0.0) {
    vec3 a = texture(u_anat, uv).rgb;
    vec3 fw = max(fwidth(a), vec3(1e-4));
    vec3 e = 1.0 - smoothstep(vec3(0.0), 1.5 * fw, abs(a - 0.5));
    col = mix(col, u_ink, u_anatAmt * 0.6 * max(max(e.r, e.g), e.b));
  }

  if (u_layer == 0 && u_del > 0.0) {
    // Mark the deleted region's outline so the eye knows what was removed.
    float fw = max(fwidth(region), 1e-4);
    float edge = 1.0 - smoothstep(0.0, 1.5 * fw, abs(region - 0.5));
    col = mix(col, u_ink, u_del * 0.55 * edge);
  }

  float b0 = boxStroke(px, u_box0, u_box0Style) * u_box0Alpha;
  float b1 = boxStroke(px, u_box1, u_box1Style) * u_box1Alpha;
  if (u_layer == 3) {
    float a = max(b0, b1);
    vec3 c = b0 >= b1 ? u_box0Style.rgb : u_box1Style.rgb;
    vec4 outc = vec4(c * a, a);
    outc = mix(outc, vec4(u_ink * 0.5, 0.5), plane * (1.0 - a));
    o = outc * u_alpha;
    return;
  }
  if (u_layer == 0) {
    col = mix(col, u_box0Style.rgb, b0);
    col = mix(col, u_box1Style.rgb, b1);
  }

  col += (hash(px + fract(u_time) * 97.0) - 0.5) * u_grain * 0.05;
  col = mix(col, u_ink, ring * 0.85);
  o = vec4(col * alpha, alpha);
}
