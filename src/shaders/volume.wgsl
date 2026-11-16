// Volumetric clouds (temporally accumulated) and height / valley fog with terrain- and
// cloud-shadowed light shafts. Both run at half the internal resolution (two entry points).
//#include "common.wgsl"
//#include "noise.wgsl"
//#include "world.wgsl"

@group(2) @binding(0) var depthTex: texture_depth_2d;
@group(2) @binding(1) var cloudHist: texture_2d<f32>;
@group(2) @binding(2) var cloudOut: texture_storage_2d<rgba16float, write>;
@group(2) @binding(3) var fogOut: texture_storage_2d<rgba16float, write>;
@group(2) @binding(4) var depthOut: texture_storage_2d<r32float, write>;
@group(2) @binding(5) var<uniform> VU: vec4f; // x = reset history, yz = previous internal size, w = debug mask

const CLOUD_SIGMA: f32 = 0.055;  // extinction (1/m) at density 1
const CLOUD_STEPS: i32 = 64;
const CLOUD_FAR: f32 = 60000.0;
const FOG_STEPS: i32 = 20;

fn windOff() -> vec3f { let w = cloudWind(); return vec3f(w.x, 0.0, w.y); }

// Cloud density at a camera-relative position (alt = curved altitude). full adds detail erosion.
fn cloudDensity(rel: vec3f, alt: f32, full: bool) -> f32 {
  let h01 = (alt - CLOUD_BOT) / (CLOUD_TOP - CLOUD_BOT);
  if (h01 <= 0.0 || h01 >= 1.0) { return 0.0; }
  // wind shear: upper parts drift further (breaks vertical extrusion of the 2D coverage)
  let wp = F.camPos.xz + rel.xz - vec2f(140.0, 60.0) * h01;
  let wt = cloudWeather(wp);
  let cov = wt.x * 0.85;
  if (cov < 0.03) { return 0.0; }
  // coverage-dependent height: dense cores tower, cloud edges stay low (dome-shaped cumulus)
  let top = mix(0.3, 1.0, wt.y) * sat(cov * 1.5 + 0.15);
  if (h01 >= top) { return 0.0; }
  let sp = vec3f(wp.x, alt, wp.y) + windOff();
  let n = textureSampleLevel(noiseTex, sLinRep, sp * (1.0 / 2048.0), 0.0);
  let lowF = n.g * 0.625 + n.b * 0.25 + n.a * 0.125;
  let base = sat(remap(n.r, lowF - 1.0, 1.0, 0.0, 1.0));
  // flat base, dome-shaped top: higher altitudes need stronger noise to stay cloudy
  let hn = h01 / top;
  let thr = (1.0 - cov) + sq(hn) * 0.9 * cov;
  var d = sat(remap(base * smoothstep(0.0, 0.06, h01), thr, 1.0, 0.0, 1.0)) * cov;
  if (!full || d <= 0.0) { return d; }
  let dn = textureSampleLevel(noiseTex, sLinRep, sp * (1.0 / 512.0) + vec3f(0.0, time() * 0.006, 0.0), 0.0);
  let hf = dn.g * 0.625 + dn.b * 0.25 + dn.a * 0.125;
  let hfm = mix(hf, 1.0 - hf, sat(hn * 3.0));
  return sat(remap(d, hfm * 0.55, 1.0, 0.0, 1.0)) * mix(0.6, 1.0, sat(hn * 4.0));
}

// Optical depth toward the sun (base shape only, growing steps).
fn sunOD(rel: vec3f, sun: vec3f, jit: f32) -> f32 {
  var od = 0.0;
  var st = 60.0;
  var t = -st * 0.5 * jit;
  for (var i = 0; i < 5; i++) {
    let q = rel + sun * (t + st * 0.5);
    od += cloudDensity(q, altOf(q), false) * st;
    t += st;
    st *= 1.9;
  }
  return od * CLOUD_SIGMA;
}

struct CloudRes { s: vec3f, t: f32, d: f32 };

fn marchClouds(dir: vec3f, tScene: f32, jit: f32, uv: vec2f) -> CloudRes {
  var r: CloudRes;
  r.s = vec3f(0.0);
  r.t = 1.0;
  let span = cloudSpan(dir);
  r.d = select(30000.0, span.x, span.x >= 0.0);
  if (span.x < 0.0) { return r; }
  let t0 = span.x;
  let t1 = min(min(span.y, tScene), CLOUD_FAR);
  if (t1 <= t0) { return r; }
  let sun = F.sunDir.xyz;
  let cosT = dot(dir, sun);
  // multiple-scattering octaves (Wrenninge-style): phase anisotropy and extinction both decay
  let ph0 = mix(hgPhase(cosT, 0.8), hgPhase(cosT, -0.3), 0.2);
  let ph1 = mix(hgPhase(cosT, 0.4), hgPhase(cosT, -0.15), 0.2);
  let ph2 = hgPhase(cosT, 0.2);
  let ph3 = 1.0 / (4.0 * PI);
  let sunC = sunLightAt(0.5 * (CLOUD_BOT + CLOUD_TOP));
  let ambT = ambientSH(vec3f(0.0, 1.0, 0.0));
  // cloud bases: sky light from the sides plus sunlight bounced off the ground
  let ambB = ambientSH(vec3f(0.0, -1.0, 0.0)) * 0.7 + ambT * 0.3 + sunC * (GROUND_ALBEDO * (F.sunDir.y / PI));
  let powderAmt = 0.6 * sat(0.6 - 0.6 * cosT);
  var t = t0 + clamp(t0 * 0.011, 45.0, 420.0) * jit;
  var T = 1.0;
  var S = vec3f(0.0);
  var ws = 0.0;
  var wd = 0.0;
  for (var i = 0; i < CLOUD_STEPS; i++) {
    if (t >= t1) { break; }
    let dt = clamp(t * 0.011, 45.0, 420.0);
    let tt = t;
    let rel = dir * tt;
    let alt = altOf(rel);
    let dens = cloudDensity(rel, alt, true);
    if (dens > 0.003) {
      let od = sunOD(rel, sun, fract(jit + f32(i) * 0.618034));
      let h01 = sat((alt - CLOUD_BOT) / (CLOUD_TOP - CLOUD_BOT));
      let powder = 1.0 - powderAmt * exp(-dens * CLOUD_SIGMA * 180.0);
      let Ls = sunC * powder * (ph0 * exp(-od) + 0.7 * ph1 * exp(-od * 0.5) + 0.49 * ph2 * exp(-od * 0.25) + 0.34 * ph3 * exp(-od * 0.125));
      let La = mix(ambB, ambT, h01);
      let Ts = exp(-dens * CLOUD_SIGMA * dt);
      let w = T * (1.0 - Ts);
      S += (Ls + La) * w;
      ws += w;
      wd += w * tt;
      T *= Ts;
      if (T < 0.01) { break; }
    }
    t += dt;
  }
  if (ws > 1e-4) { r.d = wd / ws; }
  let ap = aerial(uv, r.d);
  r.s = S * ap.a + ap.rgb * (1.0 - T);
  r.t = T;
  return r;
}

fn fogMarchDensity(wp: vec2f, alt: f32, th: f32) -> f32 {
  let hA = max(alt - th, 0.0);
  let a = max(alt, 0.0);
  // low ground fog + thin regional haze, faded out well below the cloud base (the clouds and
  // the aerial-perspective LUT own the upper air; fog there showed cloud-shadow columns)
  let cap = 1.0 - smoothstep(CLOUD_BOT - 1100.0, CLOUD_BOT - 400.0, alt);
  let base = (0.00011 * exp(-a / 170.0) + 0.000007 * exp(-a / 1000.0)) * cap;
  let valley = valleyMask(wp) * 0.0022 * exp(-hA / 35.0);
  let n = textureSampleLevel(noiseTex, sLinRep, (vec3f(wp.x, alt * 2.0, wp.y) + windOff() * 0.5) * (1.0 / 1024.0), 0.0).g;
  return (base + valley) * sq(mix(0.15, 1.45, n));
}

fn marchFog(dir: vec3f, tScene: f32, jit: f32) -> vec4f {
  let tmax = min(tScene, FOG_MAX);
  let sun = F.sunDir.xyz;
  let cosT = dot(dir, sun);
  let ph = mix(hgPhase(cosT, 0.55), 1.0 / (4.0 * PI), 0.3);
  let sunC = sunLightAt(clamp(F.camPos.y, 0.0, 600.0));
  let amb = ambientSH(vec3f(0.0, 1.0, 0.0)) * 0.8;
  var T = 1.0;
  var S = vec3f(0.0);
  var tp = 0.0;
  for (var i = 0; i < FOG_STEPS; i++) {
    let f = (f32(i) + jit) / f32(FOG_STEPS);
    let t = tmax * f * f;
    let tn = tmax * sq((f32(i) + 1.0) / f32(FOG_STEPS));
    let dt = max(tn - tp, 0.0);
    tp = tn;
    let rel = dir * t;
    let alt = altOf(rel);
    let wp = F.camPos.xz + rel.xz;
    let fld = fieldAt(wp, log2(max(dt, 16.0) / 16.0));
    let dens = fogMarchDensity(wp, alt, fld.x);
    if (dens > 1e-7) {
      let sh = fieldShadow(fld, alt) * cloudShadow(vec3f(wp.x, alt, wp.y));
      let Ts = exp(-dens * dt);
      S += T * (sunC * ph * sh + amb) * (1.0 - Ts);
      T *= Ts;
    }
  }
  return vec4f(S, T);
}

fn distOf(uv: vec2f, d: f32) -> f32 { return select(length(relFromDepth(uv, d)), 1e9, d <= 0.0); }

fn halfSize() -> vec2i { return vec2i(ceil(F.screen.xy * 0.5)); }

@compute @workgroup_size(8, 8)
fn clouds(@builtin(global_invocation_id) id: vec3u) {
  let hp = vec2i(id.xy);
  if (any(hp >= halfSize())) { return; }
  let fmax = vec2i(F.screen.xy) - 1;
  let d0 = textureLoad(depthTex, min(hp * 2, fmax), 0);
  let d1 = textureLoad(depthTex, min(hp * 2 + vec2i(1, 0), fmax), 0);
  let d2 = textureLoad(depthTex, min(hp * 2 + vec2i(0, 1), fmax), 0);
  let d3 = textureLoad(depthTex, min(hp * 2 + vec2i(1, 1), fmax), 0);
  let dFar = min(min(d0, d1), min(d2, d3)); // reversed-Z: smallest depth = farthest
  let uv = (vec2f(hp) * 2.0 + 1.0) * F.screen.zw;
  let dir = viewDir(uv);
  var cr: CloudRes;
  cr.s = vec3f(0.0); cr.t = 1.0; cr.d = 30000.0;
  if ((u32(VU.w) & 1u) == 0u) { cr = marchClouds(dir, distOf(uv, dFar), ign(vec2f(hp), F.misc.x), uv); }
  var cl = vec4f(cr.s, cr.t);
  if (VU.x < 0.5) {
    let pc = F.prevViewProj * vec4f(dir * cr.d + F.camDelta.xyz, 1.0);
    if (pc.w > 0.0) {
      let puv = pc.xy / pc.w * vec2f(0.5, -0.5) + 0.5;
      if (all(puv >= vec2f(0.0)) && all(puv <= vec2f(1.0))) {
        let huv = puv * VU.yz * 0.5 / vec2f(textureDimensions(cloudHist));
        let h = textureSampleLevel(cloudHist, sLinClamp, huv, 0.0);
        cl = mix(cl, h, 0.85);
      }
    }
  }
  textureStore(cloudOut, hp, cl);
}

@compute @workgroup_size(8, 8)
fn fog(@builtin(global_invocation_id) id: vec3u) {
  let hp = vec2i(id.xy);
  if (any(hp >= halfSize())) { return; }
  let fmax = vec2i(F.screen.xy) - 1;
  // rotating sub-sample of the 2x2 block; depth-aware upsample in the composite
  let fi = u32(F.misc.x) & 3u;
  let ps = min(hp * 2 + vec2i(i32(fi & 1u), i32(fi >> 1u)), fmax);
  let suv = (vec2f(ps) + 0.5) * F.screen.zw;
  let sd = distOf(suv, textureLoad(depthTex, ps, 0));
  var fg = vec4f(0.0, 0.0, 0.0, 1.0);
  if ((u32(VU.w) & 2u) == 0u) { fg = marchFog(viewDir(suv), sd, ign(vec2f(hp) + 17.0, F.misc.x)); }
  textureStore(fogOut, hp, fg);
  textureStore(depthOut, hp, vec4f(min(sd, FOG_MAX), 0.0, 0.0, 0.0));
}
