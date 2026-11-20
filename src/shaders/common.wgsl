// Shared per-frame uniforms and samplers (bind group 0 for every pipeline).
// World constants (WORLD_L, PLANET_R, SEED_U, ...) are prepended at startup from world.ts.

struct Frame {
  viewProj: mat4x4f,      // camera-relative, jittered
  invViewProj: mat4x4f,   // inverse of viewProj
  prevViewProj: mat4x4f,  // previous frame, relative to previous camera, unjittered
  viewProjNJ: mat4x4f,    // current, unjittered
  camPos: vec4f,          // xyz world position (wrapped into [0, WORLD_L)), w = time (s)
  camDelta: vec4f,        // xyz = camPos - prevCamPos (continuous), w = dt
  camFwd: vec4f,          // w = tan(fovY/2)
  camRight: vec4f,        // w = aspect
  camUp: vec4f,           // w = near plane
  sunDir: vec4f,          // direction to sun, w = cos(sun angular radius)
  sunColor: vec4f,        // rgb = solar illuminance at top of atmosphere, w = exposure
  jitter: vec4f,          // xy = NDC jitter now, zw = previous
  screen: vec4f,          // internal render size (px) and reciprocal
  full: vec4f,            // allocated render target size and reciprocal
  outSize: vec4f,         // output (canvas) size and reciprocal
  misc: vec4f,            // x frame index, y render scale, z quality, w wind strength
  cascadeSplits: vec4f,   // far distance of each shadow cascade (m)
  shadowMats: array<mat4x4f, 4>,
  shadowInfo: vec4f,      // x = shadow map size, y = 1/size, z = cascade count
  extra: vec4f,           // x = camera height above terrain, y = -, z = glow boost, w = debug
  sky: vec4f,             // xyz = direction to the real sun (sunDir is the key light), w = night (star field)
  skyCol: vec4f,          // rgb = sunlight on the gas giant and the moon, w = sun disk
};

@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var sLinRep: sampler;
@group(0) @binding(2) var sLinClamp: sampler;
@group(0) @binding(3) var sPoint: sampler;
@group(0) @binding(4) var sShadow: sampler_comparison;

const PI: f32 = 3.14159265;
const TAU: f32 = 6.2831853;

fn sat(x: f32) -> f32 { return clamp(x, 0.0, 1.0); }
fn sat3(x: vec3f) -> vec3f { return clamp(x, vec3f(0.0), vec3f(1.0)); }
fn luma(c: vec3f) -> f32 { return dot(c, vec3f(0.2126, 0.7152, 0.0722)); }
fn remap(x: f32, a: f32, b: f32, c: f32, d: f32) -> f32 { return c + (x - a) / (b - a) * (d - c); }
fn sq(x: f32) -> f32 { return x * x; }

// Planet curvature: vertical drop of a point at camera-relative horizontal offset.
fn curveDrop(rel: vec2f) -> f32 { return dot(rel, rel) * (0.5 / PLANET_R); }

// World position (wrapped coordinates near the camera) to camera-relative curved position.
fn toRel(world: vec3f) -> vec3f {
  var r = world - F.camPos.xyz;
  r.y -= curveDrop(r.xz);
  return r;
}

fn time() -> f32 { return F.camPos.w; }

// Periodic world delta (shortest wrapped vector).
fn wrapD2(d: vec2f) -> vec2f { return d - WORLD_L * round(d / WORLD_L); }

// Linear view-space distance from reversed-Z infinite depth.
fn viewZ(depth: f32) -> f32 { return F.camUp.w / max(depth, 1e-7); }

// Reconstruct camera-relative position from uv (0..1 over internal viewport) and depth.
fn relFromDepth(uv: vec2f, depth: f32) -> vec3f {
  let ndc = vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, max(depth, 1e-7), 1.0);
  let p = F.invViewProj * ndc;
  return p.xyz / p.w;
}
fn viewDir(uv: vec2f) -> vec3f { return normalize(relFromDepth(uv, 1.0)); }
