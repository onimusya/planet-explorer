//#include "common.wgsl"
//#include "atmosphere.wgsl"

@group(1) @binding(0) var transTex: texture_2d<f32>;
@group(1) @binding(1) var msTex: texture_2d<f32>;
@group(1) @binding(2) var skyTex: texture_2d<f32>;
@group(1) @binding(3) var transOut: texture_storage_2d<rgba16float, write>;
@group(1) @binding(4) var msOut: texture_storage_2d<rgba16float, write>;
@group(1) @binding(5) var skyOut: texture_storage_2d<rgba16float, write>;
@group(1) @binding(6) var apOut: texture_storage_3d<rgba16float, write>;
@group(1) @binding(7) var<storage, read_write> shOut: array<vec4f, 9>;

fn transmittanceTo(p: vec3f, dir: vec3f) -> vec3f {
  let r = length(p);
  let mu = dot(p / r, dir);
  return sampleTrans(transTex, sLinClamp, r, mu);
}
fn msLookup(p: vec3f, sunDir: vec3f) -> vec3f {
  let r = length(p);
  let mu = dot(p / r, sunDir);
  return textureSampleLevel(msTex, sLinClamp, msUV(r, mu), 0.0).rgb;
}
fn planetShadow(p: vec3f, sunDir: vec3f) -> f32 {
  let h = raySphere(p, sunDir, ATM_BOTTOM);
  return select(1.0, 0.0, h.x > 0.0);
}

@compute @workgroup_size(8, 8)
fn transmittanceLut(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= u32(TRANS_W) || id.y >= u32(TRANS_H)) { return; }
  let uv = (vec2f(id.xy) + 0.5) / vec2f(TRANS_W, TRANS_H);
  let rm = transParams(uv);
  let ro = vec3f(0.0, rm.x, 0.0);
  let rd = vec3f(sqrt(max(1.0 - rm.y * rm.y, 0.0)), rm.y, 0.0);
  let tTop = raySphere(ro, rd, ATM_TOP).y;
  let N = 48;
  let dt = max(tTop, 0.0) / f32(N);
  var od = vec3f(0.0);
  for (var i = 0; i < N; i++) {
    let p = ro + rd * ((f32(i) + 0.5) * dt);
    od += medium(length(p) - ATM_BOTTOM).ext * dt;
  }
  textureStore(transOut, id.xy, vec4f(exp(-od), 1.0));
}

var<workgroup> wsL: array<vec3f, 64>;
var<workgroup> wsF: array<vec3f, 64>;

@compute @workgroup_size(64)
fn multiScatterLut(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let uv = (vec2f(wg.xy) + 0.5) / MS_RES;
  let muS = uv.x * 2.0 - 1.0;
  let r = ATM_BOTTOM + 0.01 + uv.y * (ATM_TOP - ATM_BOTTOM - 0.02);
  let sunDir = vec3f(sqrt(max(1.0 - muS * muS, 0.0)), muS, 0.0);
  let i = f32(li % 8u);
  let j = f32(li / 8u);
  let th = TAU * (i + 0.5) / 8.0;
  let ph = acos(1.0 - 2.0 * (j + 0.5) / 8.0);
  let rd = vec3f(cos(th) * sin(ph), cos(ph), sin(th) * sin(ph));
  let ro = vec3f(0.0, r, 0.0);
  let tb = raySphere(ro, rd, ATM_BOTTOM);
  let tt = raySphere(ro, rd, ATM_TOP);
  let hitGround = tb.x > 0.0;
  let tMax = select(tt.y, tb.x, hitGround);
  let N = 24;
  let dt = tMax / f32(N);
  var L = vec3f(0.0);
  var fms = vec3f(0.0);
  var thr = vec3f(1.0);
  let iso = 1.0 / (4.0 * PI);
  for (var k = 0; k < N; k++) {
    let p = ro + rd * ((f32(k) + 0.5) * dt);
    let m = medium(length(p) - ATM_BOTTOM);
    let st = exp(-m.ext * dt);
    let sunT = transmittanceTo(p, sunDir) * planetShadow(p, sunDir);
    let S = m.scat * iso * sunT;
    let ext = max(m.ext, vec3f(1e-7));
    L += thr * (S - S * st) / ext;
    fms += thr * (m.scat - m.scat * st) / ext;
    thr *= st;
  }
  if (hitGround) {
    let gp = ro + rd * tMax;
    let n = normalize(gp);
    L += thr * transmittanceTo(gp, sunDir) * sat(dot(n, sunDir)) * GROUND_ALBEDO / PI;
  }
  wsL[li] = L;
  wsF[li] = fms;
  workgroupBarrier();
  var s = 32u;
  loop {
    if (li < s) {
      wsL[li] += wsL[li + s];
      wsF[li] += wsF[li + s];
    }
    workgroupBarrier();
    if (s == 1u) { break; }
    s = s / 2u;
  }
  if (li == 0u) {
    let l2 = wsL[0] / 64.0;
    let f = wsF[0] / 64.0;
    textureStore(msOut, wg.xy, vec4f(l2 / (1.0 - f), 1.0));
  }
}

// Integrates single + multiple scattering along a ray. Returns (luminance, transmittance).
struct Scatter { L: vec3f, T: vec3f };
fn integrate(ro: vec3f, rd: vec3f, tMax: f32, sunDir: vec3f, N: i32) -> Scatter {
  let c = dot(rd, sunDir);
  let pr = rayleighPhase(c);
  let pm = miePhase(c);
  var L = vec3f(0.0);
  var thr = vec3f(1.0);
  var tPrev = 0.0;
  for (var k = 0; k < N; k++) {
    // quadratic step distribution: more samples near the camera
    let t1 = tMax * sq((f32(k) + 1.0) / f32(N));
    let t = 0.5 * (tPrev + t1);
    let dt = t1 - tPrev;
    tPrev = t1;
    let p = ro + rd * t;
    let m = medium(length(p) - ATM_BOTTOM);
    let st = exp(-m.ext * dt);
    let sunT = transmittanceTo(p, sunDir) * planetShadow(p, sunDir);
    let ms = msLookup(p, sunDir);
    let S = (m.rayS * pr + m.mieS * pm) * sunT + ms * m.scat;
    let ext = max(m.ext, vec3f(1e-7));
    L += thr * (S - S * st) / ext;
    thr *= st;
  }
  var o: Scatter;
  o.L = L;
  o.T = thr;
  return o;
}

fn camRadius() -> f32 { return ATM_BOTTOM + max(F.camPos.y, 1.0) * 0.001; }

@compute @workgroup_size(8, 8)
fn skyViewLut(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= u32(SKY_W) || id.y >= u32(SKY_H)) { return; }
  var uv = (vec2f(id.xy) + 0.5) / vec2f(SKY_W, SKY_H);
  uv.x = (uv.x - 0.5 / SKY_W) * (SKY_W / (SKY_W - 1.0));
  uv.y = (uv.y - 0.5 / SKY_H) * (SKY_H / (SKY_H - 1.0));
  let r = camRadius();
  let vh = sqrt(max(r * r - ATM_BOTTOM * ATM_BOTTOM, 0.0));
  let beta = acos(clamp(vh / r, -1.0, 1.0));
  let zha = PI - beta;
  var vzc: f32;
  if (uv.y < 0.5) {
    var c = 2.0 * uv.y;
    c = 1.0 - c;
    c = c * c;
    c = 1.0 - c;
    vzc = cos(zha * c);
  } else {
    var c = uv.y * 2.0 - 1.0;
    c = c * c;
    vzc = cos(zha + beta * c);
  }
  let cx = uv.x * uv.x;
  let lvc = -(cx * 2.0 - 1.0);
  let sy = F.sunDir.y;
  let sunDir = vec3f(sqrt(max(1.0 - sy * sy, 0.0)), sy, 0.0);
  let svz = sqrt(max(1.0 - vzc * vzc, 0.0));
  let rd = vec3f(svz * lvc, vzc, svz * sqrt(max(1.0 - lvc * lvc, 0.0)));
  let ro = vec3f(0.0, r, 0.0);
  let tb = raySphere(ro, rd, ATM_BOTTOM);
  let tt = raySphere(ro, rd, ATM_TOP);
  var tMax = tt.y;
  if (tb.x > 0.0) { tMax = tb.x; }
  let s = integrate(ro, rd, max(tMax, 0.0), sunDir, 40);
  textureStore(skyOut, id.xy, vec4f(s.L, 1.0));
}

@compute @workgroup_size(4, 4, 1)
fn aerialLut(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= u32(AP_RES) || id.y >= u32(AP_RES)) { return; }
  let uv = (vec2f(id.xy) + 0.5) / AP_RES;
  let rd = viewDir(uv);
  let ro = vec3f(0.0, camRadius(), 0.0);
  let sunDir = F.sunDir.xyz;
  let c = dot(rd, sunDir);
  let pr = rayleighPhase(c);
  let pm = miePhase(c);
  var L = vec3f(0.0);
  var thr = vec3f(1.0);
  var tPrev = 0.0;
  for (var z = 0u; z < u32(AP_RES); z++) {
    let tEnd = AP_MAX_KM * sq((f32(z) + 1.0) / AP_RES);
    // 2 sub-steps per slice
    for (var k = 0; k < 2; k++) {
      let t1 = mix(tPrev, tEnd, (f32(k) + 1.0) / 2.0);
      let t0 = mix(tPrev, tEnd, f32(k) / 2.0);
      let dt = t1 - t0;
      let p = ro + rd * (0.5 * (t0 + t1));
      let m = medium(length(p) - ATM_BOTTOM);
      let st = exp(-m.ext * dt);
      let sunT = transmittanceTo(p, sunDir) * planetShadow(p, sunDir);
      let ms = msLookup(p, sunDir);
      let S = (m.rayS * pr + m.mieS * pm) * sunT + ms * m.scat;
      let ext = max(m.ext, vec3f(1e-7));
      L += thr * (S - S * st) / ext;
      thr *= st;
    }
    tPrev = tEnd;
    textureStore(apOut, vec3u(id.xy, z), vec4f(L, dot(thr, vec3f(1.0 / 3.0))));
  }
}

// Sky irradiance as pre-convolved L2 spherical harmonics (9 coefficients, rgb).
var<workgroup> wsSH: array<array<vec3f, 9>, 64>;

fn shBasis(d: vec3f) -> array<f32, 9> {
  return array<f32, 9>(
    0.282095,
    0.488603 * d.y, 0.488603 * d.z, 0.488603 * d.x,
    1.092548 * d.x * d.y, 1.092548 * d.y * d.z, 0.315392 * (3.0 * d.z * d.z - 1.0),
    1.092548 * d.x * d.z, 0.546274 * (d.x * d.x - d.y * d.y));
}

@compute @workgroup_size(64)
fn ambientSH(@builtin(local_invocation_index) li: u32) {
  var acc: array<vec3f, 9>;
  for (var k = 0; k < 9; k++) { acc[k] = vec3f(0.0); }
  let sun = F.sunColor.rgb;
  let alt = clamp(F.camPos.y, 0.0, 4000.0);
  let sunT = sunTransmittance(transTex, sLinClamp, 0.0, F.sunDir.y);
  // irradiance on flat ground (direct sun + rough sky term) used for the ground bounce
  let groundIrr = sun * sunT * max(F.sunDir.y, 0.0) + sun * 0.06;
  // night: starlight, airglow and the glow around the gas giant add to the dim sky's fill, so
  // faces turned away from the giant still read
  let skyGain = 1.0 + F.sky.w * 3.0;
  let NS = 16u;
  for (var s = 0u; s < NS; s++) {
    let idx = f32(li * NS + s) + 0.5;
    let total = f32(64u * NS);
    let z = 1.0 - 2.0 * idx / total;
    let rr = sqrt(max(1.0 - z * z, 0.0));
    let phi = idx * 2.39996323;
    let d = vec3f(rr * cos(phi), z, rr * sin(phi));
    // our SH basis uses (x, y, z) with y up: remap so basis "z" axis is world up
    var Lr: vec3f;
    if (d.y > -0.03) {
      Lr = skyRadiance(skyTex, sLinClamp, alt, vec3f(d.x, max(d.y, 0.0), d.z), F.sunDir.xyz) * sun * skyGain;
    } else {
      Lr = GROUND_ALBEDO * groundIrr / PI;
    }
    let b = shBasis(vec3f(d.x, d.z, d.y));
    for (var k = 0; k < 9; k++) { acc[k] += Lr * b[k]; }
  }
  for (var k = 0; k < 9; k++) { wsSH[li][k] = acc[k]; }
  workgroupBarrier();
  if (li < 9u) {
    var sum = vec3f(0.0);
    for (var t = 0u; t < 64u; t++) { sum += wsSH[t][li]; }
    let w = 4.0 * PI / f32(64u * NS);
    // cosine-lobe convolution constants (A0 = pi, A1 = 2pi/3, A2 = pi/4), divided by pi -> irradiance/pi
    var band = 1.0;
    if (li >= 1u && li <= 3u) { band = 2.0 / 3.0; }
    if (li >= 4u) { band = 0.25; }
    shOut[li] = vec4f(sum * w * band, 0.0);
  }
}
