// Tiling 128^3 cloud noise: r = Perlin-Worley, g/b/a = Worley fBm at increasing frequency.
//#include "common.wgsl"
//#include "noise.wgsl"

@group(1) @binding(0) var outN: texture_storage_3d<rgba8unorm, write>;

fn worleyFbm(p: vec3f, f: i32) -> f32 {
  let a = worley3(p * f32(f), vec3i(f));
  let b = worley3(p * f32(f * 2), vec3i(f * 2));
  let c = worley3(p * f32(f * 4), vec3i(f * 4));
  return sat(1.0 - (a * 0.625 + b * 0.25 + c * 0.125));
}

fn perlinFbm(p: vec3f, f: i32, oct: i32) -> f32 {
  var s = 0.0;
  var a = 0.5;
  var fr = f;
  for (var i = 0; i < oct; i++) {
    s += noise3(p * f32(fr), vec3i(fr)) * a;
    a *= 0.5;
    fr *= 2;
  }
  return s;
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let p = (vec3f(id) + 0.5) / 128.0;
  let pf = sat(perlinFbm(p, 4, 5) * 1.4 + 0.5);
  let w = worleyFbm(p, 4);
  let pw = sat(remap(pf, w - 1.0, 1.0, 0.0, 1.0));
  textureStore(outN, id, vec4f(pw, worleyFbm(p, 6), worleyFbm(p, 12), worleyFbm(p, 24)));
}
