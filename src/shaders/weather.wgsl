// Cloud weather map, generated once: r = coverage, g = cloud type (0 flat .. 1 towering).
// Periodic over WORLD_L (the map is scrolled by the wind at sampling time).
//#include "common.wgsl"
//#include "noise.wgsl"

@group(1) @binding(0) var outW: texture_storage_2d<rgba8unorm, write>;
const WN: f32 = 512.0;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let uv = (vec2f(id.xy) + 0.5) / WN;
  // domain warp for organic cloud streets and clusters
  let w = vec2f(fbm(uv * 4.0, vec2i(4), 3, 0.5), fbm(uv * 4.0 + vec2f(5.2, 1.3), vec2i(4), 3, 0.5));
  let q = uv + w * 0.06;
  let n1 = fbm(q * 16.0 + vec2f(0.37, 0.11), vec2i(16), 4, 0.55);
  let n2 = fbm(q * 4.0 + vec2f(2.1, 7.7), vec2i(4), 3, 0.5);
  let cov = sat(n1 * 2.6 + n2 * 2.0 + 0.16);
  let typ = sat(fbm(q * 6.0 + vec2f(9.1, 3.3), vec2i(6), 3, 0.5) * 2.5 + 0.6);
  textureStore(outW, id.xy, vec4f(cov, typ, 0.0, 1.0));
}
