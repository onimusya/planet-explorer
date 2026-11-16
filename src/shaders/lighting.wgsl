// Lighting: cascaded shadow maps (group 2) and BRDF helpers.

@group(2) @binding(0) var shadowMap: texture_depth_2d_array;

const POISSON = array<vec2f, 12>(
  vec2f(-0.326, -0.406), vec2f(-0.840, -0.074), vec2f(-0.696, 0.457), vec2f(-0.203, 0.621),
  vec2f(0.962, -0.195), vec2f(0.473, -0.480), vec2f(0.519, 0.767), vec2f(0.185, -0.893),
  vec2f(0.507, 0.064), vec2f(0.896, 0.412), vec2f(-0.322, -0.933), vec2f(-0.792, -0.598));

fn cascadeShadow(ci: i32, rel: vec3f, nrm: vec3f, rot: vec2f, softness: f32) -> f32 {
  let m = F.shadowMats[ci];
  let ts = 2.0 / (length(vec3f(m[0][0], m[1][0], m[2][0])) * F.shadowInfo.x);
  let p = rel + nrm * ts * 1.5;
  let lp = F.shadowMats[ci] * vec4f(p, 1.0);
  let uv = lp.xy * vec2f(0.5, -0.5) + 0.5;
  if (any(uv < vec2f(0.002)) || any(uv > vec2f(0.998)) || lp.z > 1.0) { return 1.0; }
  let z = lp.z - 0.0004;
  let rad = F.shadowInfo.y * softness;
  var s = 0.0;
  for (var i = 0; i < 12; i++) {
    let o = POISSON[i];
    let ro = vec2f(o.x * rot.x - o.y * rot.y, o.x * rot.y + o.y * rot.x);
    s += textureSampleCompareLevel(shadowMap, sShadow, uv + ro * rad, ci, z);
  }
  return s / 12.0;
}

// Cascaded shadow lookup with dithered cascade blending. dist = view distance (m).
fn csmShadow(rel: vec3f, nrm: vec3f, dist: f32, dither: f32) -> f32 {
  var ci = 0;
  if (dist > F.cascadeSplits.x) { ci = 1; }
  if (dist > F.cascadeSplits.y) { ci = 2; }
  if (dist > F.cascadeSplits.z) { ci = 3; }
  if (dist > F.cascadeSplits.w) { return 1.0; }
  // blend into the next cascade over the last 15% of the current one
  let far = F.cascadeSplits[ci];
  let blend = sat((dist / far - 0.85) / 0.15);
  if (dither < blend) { ci = min(ci + 1, 3); }
  if (ci == 3 && dist > F.cascadeSplits.w * 0.9) {
    let fade = sat((dist / F.cascadeSplits.w - 0.9) / 0.1);
    if (dither < fade) { return 1.0; }
  }
  let a = dither * TAU;
  return cascadeShadow(ci, rel, nrm, vec2f(cos(a), sin(a)), 1.6);
}

fn fresnelSchlick(f0: vec3f, c: f32) -> vec3f {
  return f0 + (vec3f(1.0) - f0) * pow(1.0 - sat(c), 5.0);
}

// GGX specular (returns BRDF * NdotL)
fn ggx(n: vec3f, v: vec3f, l: vec3f, rough: f32, f0: vec3f) -> vec3f {
  let h = normalize(v + l);
  let nl = sat(dot(n, l));
  let nv = max(dot(n, v), 1e-3);
  let nh = sat(dot(n, h));
  let a = max(rough * rough, 0.002);
  let a2 = a * a;
  let d = a2 / (PI * sq(nh * nh * (a2 - 1.0) + 1.0));
  let k = a * 0.5;
  let vis = 1.0 / ((nl * (1.0 - k) + k) * (nv * (1.0 - k) + k));
  let f = fresnelSchlick(f0, dot(v, h));
  return f * (d * vis * 0.25 * nl);
}

// Approximate split-sum environment specular weight (Karis mobile approximation).
fn envBRDF(f0: vec3f, rough: f32, nv: f32) -> vec3f {
  let c0 = vec4f(-1.0, -0.0275, -0.572, 0.022);
  let c1 = vec4f(1.0, 0.0425, 1.04, -0.04);
  let r = rough * c0 + c1;
  let a004 = min(r.x * r.x, exp2(-9.28 * nv)) * r.x + r.y;
  let ab = vec2f(-1.04, 1.04) * a004 + r.zw;
  return f0 * ab.x + ab.y;
}
