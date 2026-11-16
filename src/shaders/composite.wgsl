// Composite: sky, aerial perspective, volumetric clouds and fog over the opaque scene.
//#include "common.wgsl"
//#include "noise.wgsl"
//#include "world.wgsl"
//#include "sky.wgsl"

@group(2) @binding(0) var sceneColor: texture_2d<f32>;
@group(2) @binding(1) var sceneDepth: texture_depth_2d;
@group(2) @binding(2) var cloudTex: texture_2d<f32>;
@group(2) @binding(3) var fogTex: texture_2d<f32>;
@group(2) @binding(4) var litOut: texture_storage_2d<rgba16float, write>;
@group(2) @binding(5) var volDepth: texture_2d<f32>;

// Depth-aware 2x2 upsample of the half-resolution fog.
fn fogUpsample(uv: vec2f, dist: f32) -> vec4f {
  let hp = uv * F.screen.xy * 0.5 - 0.5;
  let b = floor(hp);
  let f = hp - b;
  let bi = vec2i(b);
  let maxI = vec2i(ceil(F.screen.xy * 0.5)) - 1;
  let dref = min(dist, FOG_MAX);
  var acc = vec4f(0.0);
  var wsum = 0.0;
  for (var k = 0; k < 4; k++) {
    let o = vec2i(k & 1, k >> 1);
    let q = clamp(bi + o, vec2i(0), maxI);
    let wb = select(1.0 - f.x, f.x, o.x == 1) * select(1.0 - f.y, f.y, o.y == 1);
    let dk = textureLoad(volDepth, q, 0).r;
    let w = (wb + 1e-3) / (1e-3 + abs(dk - dref) / max(dref, 1.0));
    acc += textureLoad(fogTex, q, 0) * w;
    wsum += w;
  }
  return acc / wsum;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let px = vec2i(id.xy);
  if (px.x >= i32(F.screen.x) || px.y >= i32(F.screen.y)) { return; }
  let uv = (vec2f(px) + 0.5) * F.screen.zw;
  let depth = textureLoad(sceneDepth, px, 0);
  let dir = viewDir(uv);
  var c: vec3f;
  var dist = 1e9;
  if (depth <= 0.0) {
    c = skyFull(dir);
  } else {
    c = textureLoad(sceneColor, px, 0).rgb;
    let rel = relFromDepth(uv, depth);
    dist = length(rel);
    let ap = aerial(uv, dist);
    c = c * ap.a + ap.rgb;
  }
  // clouds only where this pixel's surface lies beyond the cloud-layer entry
  let span = cloudSpan(dir);
  if (span.x >= 0.0 && dist > span.x) {
    let tuv = uv * F.screen.xy * F.full.zw;
    let cl = textureSampleLevel(cloudTex, sLinClamp, tuv, 0.0);
    c = c * cl.a + cl.rgb;
  }
  let fg = fogUpsample(uv, dist);
  c = c * fg.a + fg.rgb;
  textureStore(litOut, px, vec4f(max(c, vec3f(0.0)), 1.0));
}
