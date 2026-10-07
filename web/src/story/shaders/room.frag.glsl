#version 300 es
// The reading room: dark film-base with a faint vignette, and the lightbox panel. The panel is
// emissive diffuser glass with three fluorescent tube bands; its light spills into the room as
// halation. u_on carries the flicker, so the tubes can stutter on and die out.
precision highp float;

uniform vec2 u_canvas;   // device px
uniform float u_dpr;
uniform vec3 u_bg;
uniform vec4 u_panel;    // CSS px rect
uniform float u_on;      // 0..1, flicker included
uniform float u_time;

out vec4 o;

float sdRect(vec2 p, vec4 r) {
  vec2 c = r.xy + 0.5 * r.zw;
  vec2 d = abs(p - c) - 0.5 * r.zw;
  return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}

void main() {
  vec2 css = vec2(gl_FragCoord.x, u_canvas.y - gl_FragCoord.y) / u_dpr;
  vec2 view = u_canvas / u_dpr;
  vec2 q = css / view - 0.5;
  vec3 col = u_bg * (1.0 - 0.28 * dot(q, q));

  if (u_on > 0.001 && u_panel.z > 0.0) {
    float d = sdRect(css, u_panel);
    vec3 glass = vec3(0.918, 0.945, 0.961);
    if (d <= 0.0) {
      float x = (css.x - u_panel.x) / u_panel.z;
      float tubes = 0.93 + 0.07 * (0.5 + 0.5 * cos((x * 3.0) * 6.2831853));
      float y = (css.y - u_panel.y) / u_panel.w;
      float fall = 1.0 - 0.06 * pow(abs(y - 0.5) * 2.0, 2.0);
      col = mix(col, glass * tubes * fall, u_on);
    } else {
      float halo = exp(-d / 110.0) * 0.20 + exp(-d / 18.0) * 0.10;
      col += glass * halo * u_on;
    }
  }
  o = vec4(col, 1.0);
}
