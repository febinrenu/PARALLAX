#version 300 es
// A film quad placed in CSS pixels, rotated in 3D about its centre and projected with a
// perspective around u_pivot. Writing w lets the GPU interpolate UVs perspective-correctly.
layout(location = 0) in vec2 a_pos; // 0..1

uniform vec2 u_view;   // viewport, CSS px
uniform vec4 u_rect;   // x, y, w, h, CSS px
uniform vec3 u_rot;    // radians: x, y, z
uniform float u_z;     // px towards the viewer
uniform vec2 u_pivot;  // perspective origin, CSS px
uniform float u_focal; // px

out vec2 v_uv;

void main() {
  vec2 c = u_rect.xy + 0.5 * u_rect.zw;
  vec3 q = vec3((a_pos - 0.5) * u_rect.zw, 0.0);

  float cx = cos(u_rot.x), sx = sin(u_rot.x);
  float cy = cos(u_rot.y), sy = sin(u_rot.y);
  float cz = cos(u_rot.z), sz = sin(u_rot.z);
  q = vec3(q.x * cz - q.y * sz, q.x * sz + q.y * cz, q.z);
  q = vec3(q.x * cy + q.z * sy, q.y, -q.x * sy + q.z * cy);
  q = vec3(q.x, q.y * cx - q.z * sx, q.y * sx + q.z * cx);
  q.z += u_z;

  float s = u_focal / max(u_focal - q.z, 1.0);
  vec2 sp = u_pivot + (c + q.xy - u_pivot) * s;
  vec2 ndc = vec2(sp.x / u_view.x * 2.0 - 1.0, 1.0 - sp.y / u_view.y * 2.0);
  float w = 1.0 / s;
  gl_Position = vec4(ndc * w, 0.0, w);
  v_uv = a_pos;
}
