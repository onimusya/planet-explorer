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

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let px = vec2i(id.xy);
  if (px.x >= i32(F.screen.x) || px.y >= i32(F.screen.y)) { return; }
  let uv = (vec2f(px) + 0.5) * F.screen.zw;
  let depth = textureLoad(sceneDepth, px, 0);
  var c: vec3f;
  if (depth <= 0.0) {
    c = skyFull(viewDir(uv));
  } else {
    c = textureLoad(sceneColor, px, 0).rgb;
    let rel = relFromDepth(uv, depth);
    let ap = aerial(uv, length(rel));
    c = c * ap.a + ap.rgb;
  }
  let tuv = uv * F.screen.xy * F.full.zw;
  let cl = textureSampleLevel(cloudTex, sLinClamp, tuv, 0.0);
  c = c * cl.a + cl.rgb;
  let fg = textureSampleLevel(fogTex, sLinClamp, tuv, 0.0);
  c = c * fg.a + fg.rgb;
  textureStore(litOut, px, vec4f(max(c, vec3f(0.0)), 1.0));
}
