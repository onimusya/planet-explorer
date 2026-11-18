// Physically-based bloom: 13-tap downsample chain (Karis average on first level) + tent upsample.
//#include "common.wgsl"

@group(1) @binding(0) var src: texture_2d<f32>;
@group(1) @binding(1) var dst: texture_storage_2d<rgba16float, write>;
@group(1) @binding(2) var addTex: texture_2d<f32>;
@group(1) @binding(3) var<uniform> BP: vec4f; // x = first level, y = source uv scale (viewport), z = radius

fn karis(c: vec3f) -> f32 { return 1.0 / (1.0 + luma(c)); }

@compute @workgroup_size(8, 8)
fn down(@builtin(global_invocation_id) id: vec3u) {
  let ds = vec2u(textureDimensions(dst));
  if (id.x >= ds.x || id.y >= ds.y) { return; }
  let ss = vec2f(textureDimensions(src, 0));
  let uv = (vec2f(id.xy) + 0.5) / vec2f(ds) * BP.y;
  let t = 1.0 / ss;
  let a = textureSampleLevel(src, sLinClamp, uv + t * vec2f(-2.0, -2.0), 0.0).rgb;
  let b = textureSampleLevel(src, sLinClamp, uv + t * vec2f(0.0, -2.0), 0.0).rgb;
  let c = textureSampleLevel(src, sLinClamp, uv + t * vec2f(2.0, -2.0), 0.0).rgb;
  let d = textureSampleLevel(src, sLinClamp, uv + t * vec2f(-2.0, 0.0), 0.0).rgb;
  let e = textureSampleLevel(src, sLinClamp, uv, 0.0).rgb;
  let f = textureSampleLevel(src, sLinClamp, uv + t * vec2f(2.0, 0.0), 0.0).rgb;
  let g = textureSampleLevel(src, sLinClamp, uv + t * vec2f(-2.0, 2.0), 0.0).rgb;
  let h = textureSampleLevel(src, sLinClamp, uv + t * vec2f(0.0, 2.0), 0.0).rgb;
  let i = textureSampleLevel(src, sLinClamp, uv + t * vec2f(2.0, 2.0), 0.0).rgb;
  let j = textureSampleLevel(src, sLinClamp, uv + t * vec2f(-1.0, -1.0), 0.0).rgb;
  let k = textureSampleLevel(src, sLinClamp, uv + t * vec2f(1.0, -1.0), 0.0).rgb;
  let l = textureSampleLevel(src, sLinClamp, uv + t * vec2f(-1.0, 1.0), 0.0).rgb;
  let m = textureSampleLevel(src, sLinClamp, uv + t * vec2f(1.0, 1.0), 0.0).rgb;
  var o: vec3f;
  if (BP.x > 0.5) {
    let g0 = (a + b + d + e) * 0.25;
    let g1 = (b + c + e + f) * 0.25;
    let g2 = (d + e + g + h) * 0.25;
    let g3 = (e + f + h + i) * 0.25;
    let g4 = (j + k + l + m) * 0.25;
    let w0 = karis(g0) * 0.125;
    let w1 = karis(g1) * 0.125;
    let w2 = karis(g2) * 0.125;
    let w3 = karis(g3) * 0.125;
    let w4 = karis(g4) * 0.5;
    o = (g0 * w0 + g1 * w1 + g2 * w2 + g3 * w3 + g4 * w4) / (w0 + w1 + w2 + w3 + w4);
    // airborne particles (their own layer, see fx.wgsl) join the first level
    o += (textureSampleLevel(addTex, sLinClamp, uv + t * vec2f(-1.0, -1.0), 0.0).rgb + textureSampleLevel(addTex, sLinClamp, uv + t * vec2f(1.0, -1.0), 0.0).rgb
        + textureSampleLevel(addTex, sLinClamp, uv + t * vec2f(-1.0, 1.0), 0.0).rgb + textureSampleLevel(addTex, sLinClamp, uv + t * vec2f(1.0, 1.0), 0.0).rgb) * 0.25;
  } else {
    o = e * 0.125 + (a + c + g + i) * 0.03125 + (b + d + f + h) * 0.0625 + (j + k + l + m) * 0.125;
  }
  textureStore(dst, id.xy, vec4f(min(o, vec3f(6e4)), 1.0));
}

@compute @workgroup_size(8, 8)
fn up(@builtin(global_invocation_id) id: vec3u) {
  let ds = vec2u(textureDimensions(dst));
  if (id.x >= ds.x || id.y >= ds.y) { return; }
  let ss = vec2f(textureDimensions(src, 0));
  let uv = (vec2f(id.xy) + 0.5) / vec2f(ds);
  let t = BP.z / ss;
  var s = textureSampleLevel(src, sLinClamp, uv, 0.0).rgb * 4.0;
  s += (textureSampleLevel(src, sLinClamp, uv + vec2f(-t.x, 0.0), 0.0).rgb + textureSampleLevel(src, sLinClamp, uv + vec2f(t.x, 0.0), 0.0).rgb
      + textureSampleLevel(src, sLinClamp, uv + vec2f(0.0, -t.y), 0.0).rgb + textureSampleLevel(src, sLinClamp, uv + vec2f(0.0, t.y), 0.0).rgb) * 2.0;
  s += textureSampleLevel(src, sLinClamp, uv + vec2f(-t.x, -t.y), 0.0).rgb + textureSampleLevel(src, sLinClamp, uv + vec2f(t.x, -t.y), 0.0).rgb
      + textureSampleLevel(src, sLinClamp, uv + vec2f(-t.x, t.y), 0.0).rgb + textureSampleLevel(src, sLinClamp, uv + vec2f(t.x, t.y), 0.0).rgb;
  s /= 16.0;
  let base = textureSampleLevel(addTex, sLinClamp, uv, 0.0).rgb;
  textureStore(dst, id.xy, vec4f(base + s, 1.0));
}
