#version 300 es
// Clinical viewer: one pass over a fullscreen triangle.
// Window/level in highp (mediump bands visibly on mobile), heatmap through the cividis LUT with
// an iso-contour as the non-colour cue, hatched mask with a contour, anatomy contours, and up to
// eight boxes drawn as screen-space SDFs (solid = verified, dashed = uncertain).
precision highp float;

uniform vec2 u_canvas;      // drawing buffer size, device px
uniform float u_dpr;
uniform vec2 u_offset;      // image origin on screen, CSS px
uniform float u_scale;      // CSS px per image px
uniform vec2 u_imageSize;   // original image size, px (overlays live on this grid)

uniform sampler2D u_image;
uniform int u_isColor;
uniform float u_level;      // window centre, 0..1
uniform float u_width;      // window width, 0..1
uniform int u_invert;

uniform sampler2D u_heat;
uniform sampler2D u_lut;
uniform float u_heatOn;
uniform sampler2D u_mask;
uniform float u_maskOn;
uniform sampler2D u_anat;
uniform float u_anatOn;

uniform int u_boxCount;
uniform vec4 u_boxes[8];    // uv xyxy
uniform vec4 u_boxStyle[8]; // rgb + dashed (0/1)
uniform float u_boxAlpha[8];
uniform int u_boxHi;        // highlighted box index, -1 for none
uniform float u_boxesOn;

uniform vec3 u_bg;
uniform vec3 u_ink;

out vec4 outColor;

const vec3 LUMA = vec3(0.299, 0.587, 0.114);

float boxEdge(vec2 p, vec2 b0, vec2 b1) {
  vec2 c = 0.5 * (b0 + b1);
  vec2 h = 0.5 * abs(b1 - b0);
  vec2 d = abs(p - c) - h;
  return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}

void main() {
  vec2 css = vec2(gl_FragCoord.x, u_canvas.y - gl_FragCoord.y) / u_dpr;
  vec2 ipx = (css - u_offset) / u_scale;
  vec2 uv = ipx / u_imageSize;

  vec3 col = u_bg;
  bool inside = all(greaterThanEqual(uv, vec2(0.0))) && all(lessThanEqual(uv, vec2(1.0)));

  if (inside) {
    vec4 s = texture(u_image, uv);
    float v = u_isColor == 1 ? dot(s.rgb, LUMA) : s.r;
    float lo = u_level - 0.5 * u_width;
    float g = clamp((v - lo) / max(u_width, 1e-4), 0.0, 1.0);
    if (u_invert == 1) g = 1.0 - g;
    if (u_isColor == 1) {
      // Colour images keep their hue; window/level rescales their luminance.
      col = clamp(s.rgb * (g / max(v, 1e-3)), 0.0, 1.0);
    } else {
      col = vec3(g);
    }

    if (u_heatOn > 0.0) {
      float h = texture(u_heat, uv).r;
      vec3 hc = texture(u_lut, vec2(h, 0.5)).rgb;
      col = mix(col, hc, u_heatOn * 0.58 * smoothstep(0.06, 0.42, h));
      float fw = max(fwidth(h), 1e-4);
      float iso = 1.0 - smoothstep(0.0, 1.5 * fw, abs(h - 0.5));
      col = mix(col, hc, u_heatOn * iso * 0.9);
    }

    if (u_maskOn > 0.0) {
      float m = texture(u_mask, uv).r;
      float fw = max(fwidth(m), 1e-4);
      float edge = 1.0 - smoothstep(0.0, 1.5 * fw, abs(m - 0.5));
      float hatch = step(0.5, fract((css.x + css.y) / 9.0)) * step(0.5, m);
      vec3 mc = vec3(0.953, 0.784, 0.275);
      col = mix(col, mc, u_maskOn * (0.18 * hatch + 0.95 * edge));
    }

    if (u_anatOn > 0.0) {
      vec3 a = texture(u_anat, uv).rgb;
      vec3 fw = max(fwidth(a), vec3(1e-4));
      vec3 e = 1.0 - smoothstep(vec3(0.0), 1.5 * fw, abs(a - 0.5));
      float edge = max(max(e.r, e.g), e.b);
      col = mix(col, u_ink, u_anatOn * 0.55 * edge);
    }
  }

  if (u_boxesOn > 0.0) {
    for (int i = 0; i < 8; i++) {
      if (i >= u_boxCount) break;
      vec2 b0 = u_offset + u_boxes[i].xy * u_imageSize * u_scale;
      vec2 b1 = u_offset + u_boxes[i].zw * u_imageSize * u_scale;
      float d = abs(boxEdge(css, b0, b1));
      float w = i == u_boxHi ? 1.6 : 0.9; // half stroke width, CSS px
      float a = 1.0 - smoothstep(w, w + 1.0 / u_dpr + 0.5, d);
      if (u_boxStyle[i].w > 0.5) {
        float t = (css.x - b0.x) + (css.y - b0.y);
        a *= step(0.45, fract(t / 12.0));
      }
      col = mix(col, u_boxStyle[i].rgb, a * u_boxesOn * u_boxAlpha[i]);
    }
  }

  outColor = vec4(col, 1.0);
}
