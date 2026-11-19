// Instanced flora / rock rendering (main + shadow), wind animation, per-part alien materials.
//#include "common.wgsl"
//#include "noise.wgsl"
//#include "world.wgsl"
//#include "lighting.wgsl"
//#include "materials.wgsl"

struct VIn {
  @location(0) pos: vec3f,
  @location(1) nrm: vec4f,
  @location(2) uv: vec2f,
  @location(3) attr: vec4f,
  @location(4) ia: vec4f, // world xyz, scale
  @location(5) ib: vec4f, // yaw, seed, species, yStretch
};

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) world: vec3f,
  @location(1) rel: vec3f,
  @location(2) nrm: vec3f,
  @location(3) uv: vec2f,
  @location(4) @interpolate(flat) info: vec4f, // part, species, seed, scale
  @location(5) ag: vec2f, // ao, grad
  @location(6) lpos: vec3f, // local (object) position for procedural patterns
};

@group(3) @binding(0) var<uniform> cascadeIdx: vec4u;

fn windOffset(wp: vec3f, flex: f32, part: f32, scale: f32, seed: f32) -> vec3f {
  let t = time();
  let ph = dot(wp.xz, vec2f(0.021, 0.034)) + seed * 6.283;
  let gust = 0.6 + 0.4 * sin(t * 0.37 + wp.x * 0.004 + wp.z * 0.003);
  let sway = (sin(t * 1.15 + ph) * 0.7 + sin(t * 2.3 + ph * 1.7) * 0.3) * gust;
  let wd = vec2f(0.86, 0.5);
  var o = vec3f(wd.x, 0.0, wd.y) * sway * flex * flex * 0.32 * scale;
  if (part == 1.0 || part == 8.0 || part == 4.0) {
    let fl = sin(t * 6.5 + wp.x * 2.1 + wp.z * 1.7 + seed * 40.0) * 0.045 * flex * scale;
    o += vec3f(fl, fl * 0.6, -fl * 0.5);
  }
  return o;
}

fn transformV(v: VIn) -> VOut {
  let yaw = v.ib.x;
  let c = cos(yaw);
  let s = sin(yaw);
  let sc = v.ia.w;
  var lp = v.pos;
  lp.y *= v.ib.w;
  let rp = vec3f(lp.x * c + lp.z * s, lp.y, -lp.x * s + lp.z * c) * sc;
  let rn = vec3f(v.nrm.x * c + v.nrm.z * s, v.nrm.y / v.ib.w, -v.nrm.x * s + v.nrm.z * c);
  let part = round(v.attr.x * 255.0);
  var wp = v.ia.xyz + rp;
  wp += windOffset(wp, v.attr.y, part, sc, v.ib.y);
  var o: VOut;
  o.world = wp;
  o.rel = toRel(wp);
  o.nrm = normalize(rn);
  o.uv = vec2f(v.uv.x, v.uv.y * 64.0);
  o.info = vec4f(part, v.ib.z, v.ib.y, sc);
  o.ag = vec2f(v.attr.z, v.attr.w);
  o.lpos = v.pos;
  return o;
}

@vertex
fn vs(v: VIn) -> VOut {
  var o = transformV(v);
  o.pos = F.viewProj * vec4f(o.rel, 1.0);
  return o;
}

@vertex
fn vsShadow(v: VIn) -> VOut {
  var o = transformV(v);
  o.pos = F.shadowMats[cascadeIdx.x] * vec4f(o.rel, 1.0);
  return o;
}

// ---------------------------------------------------------------------------------------------
// Fauna (fauna.ts): rest-pose meshes posed per vertex from the pose buffer that fauna.wgsl writes
// each frame (5 vec4 per animal). attr = (part, bone / 255, ao, blend weight), uv.y = along / 64.
@group(3) @binding(1) var<storage, read> faunaPose: array<vec4f>;
@group(3) @binding(2) var<uniform> rig: array<vec4f, 96>;

struct FIn {
  @location(0) pos: vec3f,
  @location(1) nrm: vec4f,
  @location(2) uv: vec2f,
  @location(3) attr: vec4f,
};

fn rotX(p: vec3f, a: f32) -> vec3f { let c = cos(a); let s = sin(a); return vec3f(p.x, p.y * c - p.z * s, p.y * s + p.z * c); }
fn rotY(p: vec3f, a: f32) -> vec3f { let c = cos(a); let s = sin(a); return vec3f(p.x * c + p.z * s, p.y, -p.x * s + p.z * c); }
fn rotZ(p: vec3f, a: f32) -> vec3f { let c = cos(a); let s = sin(a); return vec3f(p.x * c - p.y * s, p.x * s + p.y * c, p.z); }
// angle of a (y, z) direction from straight down, positive toward +z (forward)
fn downAng(d: vec2f) -> f32 { return atan2(d.y, -d.x); }

struct Posed { p: vec3f, n: vec3f };

// Herd animal skeleton (model space, z forward): 0 body, 1 neck, 2 head, 3 tail, 4-7 upper legs,
// 8-11 lower legs (LF, RF, LH, RH). Rig: 0-3 hips (xyz, upper length), 4-7 rest knees (xyz, lower
// length), 8 neck pivot, 9 head pivot, 10 tail pivot, 11 gait (stride, lift, duty), 12/13 rest
// upper/lower leg angles, 14 knee bend sign, 15/16 rest foot z/y.
fn poseCreature(v: FIn, base: u32, rb: u32) -> Posed {
  var o: Posed;
  o.p = v.pos;
  o.n = v.nrm.xyz;
  let bone = u32(round(v.attr.y * 255.0));
  let w = v.attr.w;
  let P1 = faunaPose[base + 1u];
  let P2 = faunaPose[base + 2u];
  if (bone >= 4u) {
    // legs: planted-foot gait cycle, solved by two-bone IK in the leg's sagittal plane
    let L = (bone - 4u) & 3u;
    let hip = rig[rb + L];
    let knee0 = rig[rb + 4u + L];
    let gp = rig[rb + 11u];
    let ph = fract(P1.z + f32(L & 1u) * 0.5 + select(0.0, 0.25, L < 2u));
    var fz = 0.0;
    var fy = 0.0;
    if (ph < gp.z) {
      fz = gp.x * (0.5 - ph / gp.z);
    } else {
      let q = (ph - gp.z) / (1.0 - gp.z);
      fz = gp.x * (q * q * (3.0 - 2.0 * q) - 0.5);
      fy = gp.y * P1.w * sin(PI * q);
    }
    let foot = vec2f(rig[rb + 16u][L] + fy + faunaPose[base + 3u][L], rig[rb + 15u][L] + fz);
    let H = hip.yz;
    let dv = foot - H;
    let l1 = hip.w;
    let l2 = knee0.w;
    let d = clamp(length(dv), abs(l1 - l2) + 0.02, (l1 + l2) * 0.995);
    let cosA = clamp((l1 * l1 + d * d - l2 * l2) / (2.0 * l1 * d), -1.0, 1.0);
    let aU = downAng(dv) + rig[rb + 14u][L] * acos(cosA);
    let K = H + l1 * vec2f(-cos(aU), sin(aU));
    // the thigh eases out of the body near the hip; the shin blends with the thigh at the knee
    let kU = select(1.0, w, bone < 8u);
    let rU = (rig[rb + 12u][L] - aU) * kU;
    let pU = vec3f(v.pos.x, H) + rotX(vec3f(0.0, v.pos.yz - H), rU);
    let nU = rotX(o.n, rU);
    if (bone < 8u) {
      o.p = pU;
      o.n = nU;
    } else {
      let rL = rig[rb + 13u][L] - downAng(foot - K);
      let pL = vec3f(v.pos.x, K) + rotX(vec3f(0.0, v.pos.yz - knee0.yz), rL);
      o.p = mix(pU, pL, w);
      o.n = mix(nU, rotX(o.n, rL), w);
    }
  } else if (bone == 3u) {
    // tail: swish about its root, the angle growing toward the tip
    let tp = rig[rb + 10u].xyz;
    o.p = tp + rotY(v.pos - tp, P2.w * w);
    o.n = rotY(o.n, P2.w * w);
  } else if (bone >= 1u) {
    // neck bends along its length (angle grows toward the head); the head pitches on top of it
    var p = v.pos;
    var n = o.n;
    var k = w;
    if (bone == 2u) {
      let hp = rig[rb + 9u].xyz;
      p = hp + rotX(p - hp, P2.z);
      n = rotX(n, P2.z);
      k = 1.0;
    }
    let np = rig[rb + 8u].xyz;
    o.p = np + rotY(rotX(p - np, P2.x * k), P2.y * k);
    o.n = rotY(rotX(n, P2.x * k), P2.y * k);
  }
  return o;
}

fn faunaVOut(wp: vec3f, n: vec3f, v: FIn, info: vec4f) -> VOut {
  var o: VOut;
  o.world = wp;
  o.rel = toRel(wp);
  o.nrm = normalize(n);
  o.uv = vec2f(v.uv.x, v.uv.y * 64.0);
  o.info = info;
  o.ag = vec2f(v.attr.z, v.uv.y * 64.0);
  o.lpos = v.pos;
  return o;
}

// Quadrupeds: the herd animal (rig base 0, species 20) and the titans (rig base 32, species 24).
fn creatureV(v: FIn, ii: u32, rb: u32, sp: f32) -> VOut {
  let base = ii * 5u;
  let P0 = faunaPose[base];
  let P1 = faunaPose[base + 1u];
  let P4 = faunaPose[base + 4u];
  let ps = poseCreature(v, base, rb);
  // body: roll, pitch (nose up), heading; then scale and place, with the walking bob
  let wp = rotY(rotX(rotZ(ps.p, P1.y), -P1.x), P0.w) * P4.x + P0.xyz + vec3f(0.0, P4.z, 0.0);
  let n = rotY(rotX(rotZ(ps.n, P1.y), -P1.x), P0.w);
  return faunaVOut(wp, n, v, vec4f(round(v.attr.x * 255.0), sp, P4.y, P4.x));
}

// Humanoid skeleton (travellers: rig base 52, species 25; colossi: rig base 64, species 26):
// 30 body, 31 head and neck, 32/33 upper arms, 34/35 forearms, 36/37 thighs, 38/39 shins (left,
// right). Rig: +0 neck pivot, +1/+2 shoulders, +3/+4 elbows, +5/+6 hips, +7/+8 knees, +9 gait
// (hip swing, knee lift, arm swing, elbow rest), +10 (pelvis height). Limbs swing in their
// sagittal planes: hips and shoulders forward (rotX < 0), knees back, elbows forward.
fn bipedV(v: FIn, ii: u32, rb: u32, sp: f32) -> VOut {
  let base = ii * 5u;
  let P0 = faunaPose[base];
  let P1 = faunaPose[base + 1u];
  let P2 = faunaPose[base + 2u];
  let P4 = faunaPose[base + 4u];
  let bone = u32(round(v.attr.y * 255.0));
  let w = v.attr.w;
  let g = rig[rb + 9u];
  let ph = P1.x * TAU;
  let walk = P1.y;
  var p = v.pos;
  var n = v.nrm.xyz;
  if (bone >= 36u) {
    let s = (bone - 36u) & 1u;
    let a = ph + f32(s) * PI;
    let hip = rig[rb + 5u + s].xyz;
    let hf = walk * g.x * sin(a);
    // the knee folds through the swing, when the foot comes forward
    let kf = walk * (g.y * pow(max(cos(a), 0.0), 1.5) + 0.06);
    if (bone < 38u) {
      p = hip + rotX(p - hip, -hf * w);
      n = rotX(n, -hf * w);
    } else {
      let knee = rig[rb + 7u + s].xyz;
      p = hip + rotX(knee + rotX(p - knee, kf * w) - hip, -hf);
      n = rotX(rotX(n, kf * w), -hf);
    }
  } else if (bone >= 32u) {
    let s = (bone - 32u) & 1u;
    let a = ph + f32(1u - s) * PI;
    let sh = rig[rb + 1u + s].xyz;
    // arms swing against the legs; the second arm also rises forward to scan or to reach up
    let raise = select(0.0, P2.z, s == 1u);
    let sf = walk * g.z * sin(a) + raise * 1.45;
    let ef = g.w + walk * 0.22 * max(sin(a), 0.0) + raise * 0.35;
    if (bone < 34u) {
      p = sh + rotX(p - sh, -sf * w);
      n = rotX(n, -sf * w);
    } else {
      let el = rig[rb + 3u + s].xyz;
      p = sh + rotX(el + rotX(p - el, -ef * w) - sh, -sf);
      n = rotX(rotX(n, -ef * w), -sf);
    }
  } else if (bone == 31u) {
    let nk = rig[rb].xyz;
    p = nk + rotY(rotX(p - nk, P2.y * w), P2.x * w);
    n = rotY(rotX(n, P2.y * w), P2.x * w);
  }
  if (bone < 36u) {
    // the upper body leans forward from the pelvis
    let pel = vec3f(0.0, rig[rb + 10u].x, 0.0);
    p = pel + rotX(p - pel, P1.z);
    n = rotX(n, P1.z);
  }
  let wp = rotY(p, P0.w) * P4.x + P0.xyz + vec3f(0.0, P4.z, 0.0);
  var o = faunaVOut(wp, rotY(n, P0.w), v, vec4f(round(v.attr.x * 255.0), sp, P4.y, P4.x));
  o.ag.y = P1.w; // jet thrust, for the nozzle glow
  return o;
}

// Sky jelly: 0 bell (w: 0 at the apex to 1 at the rim), 1 oral arms and tentacles (w: 0 at the
// root to 1 at the tip). Each pulse squeezes the rim in and lifts the tentacles, which also stream
// behind the drift. Pose: (pos, spin), (pulse phase, drift x, drift z, -), -, -, (scale, seed, ...).
fn jellyV(v: FIn, ii: u32) -> VOut {
  let base = ii * 5u;
  let P0 = faunaPose[base];
  let P1 = faunaPose[base + 1u];
  let P4 = faunaPose[base + 4u];
  let bone = u32(round(v.attr.y * 255.0));
  let w = v.attr.w;
  let ph = fract(P1.x);
  let c = smoothstep(0.0, 0.2, ph) * (1.0 - smoothstep(0.2, 0.8, ph));
  var p = v.pos;
  var n = v.nrm.xyz;
  var drag = vec3f(0.0);
  if (bone == 0u) {
    let k = 1.0 - 0.2 * c * w * w;
    p = vec3f(p.x * k, p.y * (1.0 + 0.1 * c * (1.0 - w)) - 0.06 * c * w, p.z * k);
    n = normalize(vec3f(n.x, n.y * k, n.z));
  } else {
    let tt = time();
    let sw = w * w;
    let k = 1.0 - 0.2 * c * (1.0 - w);
    p = vec3f(p.x * k, p.y + sw * 0.3 * c, p.z * k);
    p += vec3f(sin(tt * 0.9 + p.y * 1.3 + P4.y * 20.0), 0.0, cos(tt * 0.7 + p.y * 1.1 + P4.y * 31.0)) * (0.3 * sw);
    drag = vec3f(-P1.y, 0.0, -P1.z) * sw;
  }
  let wp = rotY(p, P0.w) * P4.x + drag * P4.x + P0.xyz;
  return faunaVOut(wp, rotY(n, P0.w), v, vec4f(round(v.attr.x * 255.0), 27.0, P4.y, P4.x));
}

// Rigid fliers posed from (forward, bank) like the birds: right, up, forward axes.
fn flyAxes(P1: vec4f) -> mat3x3f {
  let f = P1.xyz;
  let r0 = normalize(cross(vec3f(0.0, 1.0, 0.0), f) + vec3f(1e-4, 0.0, 0.0));
  let u0 = cross(f, r0);
  return mat3x3f(r0 * cos(P1.w) - u0 * sin(P1.w), u0 * cos(P1.w) + r0 * sin(P1.w), f);
}

// Sky ray: 0 body and wings, undulating in a wave that runs from the front edge to the back and
// grows toward the tips; 1 tail (w along it), swaying. Pose as the birds' (veg.wgsl birdV).
fn rayV(v: FIn, ii: u32) -> VOut {
  let base = ii * 5u;
  let P0 = faunaPose[base];
  let P1 = faunaPose[base + 1u];
  let P2 = faunaPose[base + 2u];
  let P3 = faunaPose[base + 3u];
  let bone = u32(round(v.attr.y * 255.0));
  var p = v.pos;
  var n = v.nrm.xyz;
  if (bone == 0u) {
    let ax = abs(p.x);
    let arg = P2.x - p.z * 2.4 - ax * 0.8;
    p.y += P2.y * pow(ax, 1.6) * sin(arg);
    let slope = P2.y * 1.6 * pow(max(ax, 1e-3), 0.6) * sin(arg) * sign(p.x);
    n = normalize(n + vec3f(-slope, 0.0, 0.0) * sign(n.y));
  } else {
    let w = v.attr.w;
    let a = (P3.x + 0.25 * sin(P2.x * 0.7 - w * 4.0)) * w;
    let root = vec3f(0.0, 0.0, -0.42);
    p = root + rotY(p - root, a);
    n = rotY(n, a);
  }
  let m = flyAxes(P1);
  return faunaVOut(P0.xyz + m * p * P0.w, m * n, v, vec4f(round(v.attr.x * 255.0), 28.0, P2.w, P0.w));
}

// Starship: rigid. Pose (pos, scale: 0 between passes), (forward, bank), (thrust, variant, seed, -).
// The engine thrust rides in the grad slot for the nozzle glow.
fn shipV(v: FIn, ii: u32) -> VOut {
  let base = ii * 5u;
  let P0 = faunaPose[base];
  let P2 = faunaPose[base + 2u];
  let m = flyAxes(faunaPose[base + 1u]);
  var o = faunaVOut(P0.xyz + m * v.pos * P0.w, m * v.nrm.xyz, v, vec4f(round(v.attr.x * 255.0), 29.0, P2.z, P0.w));
  o.ag.y = P2.x;
  return o;
}

// Flyer skeleton: 20 body, 21/22 inner wing L/R, 23/24 outer wing L/R, 25 tail. Rig: 20/21 shoulder
// L/R, 22/23 elbow L/R, 24 tail pivot. Pose: (pos, scale), (forward, bank), (flap phase, flap
// amplitude, glide lift, seed), (tail pitch, -, flock, -).
fn birdV(v: FIn, ii: u32) -> VOut {
  let base = (FAUNA_NC + ii) * 5u;
  let P0 = faunaPose[base];
  let P1 = faunaPose[base + 1u];
  let P2 = faunaPose[base + 2u];
  let P3 = faunaPose[base + 3u];
  let bone = u32(round(v.attr.y * 255.0));
  let w = v.attr.w;
  var p = v.pos;
  var n = v.nrm.xyz;
  if (bone >= 21u && bone <= 24u) {
    let right = (bone & 1u) == 0u;
    let sg = select(-1.0, 1.0, right);
    let S = rig[select(20u, 21u, right)].xyz;
    let a1 = sg * (P2.y * sin(P2.x) + 0.1 + P2.z);
    if (bone <= 22u) {
      p = S + rotZ(p - S, a1 * w);
      n = rotZ(n, a1 * w);
    } else {
      // the outer wing lags the stroke, folding slightly on the upstroke
      let E = rig[select(22u, 23u, right)].xyz;
      let a2 = sg * (P2.y * 0.55 * sin(P2.x - 0.7) + P2.z * 0.6);
      let pIn = S + rotZ(p - S, a1);
      let pOut = S + rotZ(E + rotZ(p - E, a2) - S, a1);
      p = mix(pIn, pOut, w);
      n = mix(rotZ(n, a1), rotZ(rotZ(n, a2), a1), w);
    }
  } else if (bone == 25u) {
    let tp = rig[24].xyz;
    let a = (P3.x + 0.08 * sin(P2.x)) * w;
    p = tp + rotX(p - tp, a);
    n = rotX(n, a);
  }
  let f = P1.xyz;
  let r0 = normalize(cross(vec3f(0.0, 1.0, 0.0), f));
  let u0 = cross(f, r0);
  let r = r0 * cos(P1.w) - u0 * sin(P1.w);
  let u = u0 * cos(P1.w) + r0 * sin(P1.w);
  let wp = P0.xyz + (r * p.x + u * p.y + f * p.z) * P0.w;
  return faunaVOut(wp, r * n.x + u * n.y + f * n.z, v, vec4f(round(v.attr.x * 255.0), 21.0 + P3.z, P2.w, P0.w));
}

fn toClip(o: VOut) -> VOut { var r = o; r.pos = F.viewProj * vec4f(o.rel, 1.0); return r; }
fn toShadow(o: VOut) -> VOut { var r = o; r.pos = F.shadowMats[cascadeIdx.x] * vec4f(o.rel, 1.0); return r; }

@vertex fn vsCreature(v: FIn, @builtin(instance_index) ii: u32) -> VOut { return toClip(creatureV(v, ii, 0u, 20.0)); }
@vertex fn vsCreatureShadow(v: FIn, @builtin(instance_index) ii: u32) -> VOut { return toShadow(creatureV(v, ii, 0u, 20.0)); }
@vertex fn vsTitan(v: FIn, @builtin(instance_index) ii: u32) -> VOut { return toClip(creatureV(v, ii, 32u, 24.0)); }
@vertex fn vsTitanShadow(v: FIn, @builtin(instance_index) ii: u32) -> VOut { return toShadow(creatureV(v, ii, 32u, 24.0)); }
@vertex fn vsTraveller(v: FIn, @builtin(instance_index) ii: u32) -> VOut { return toClip(bipedV(v, ii, 52u, 25.0)); }
@vertex fn vsTravellerShadow(v: FIn, @builtin(instance_index) ii: u32) -> VOut { return toShadow(bipedV(v, ii, 52u, 25.0)); }
@vertex fn vsColossus(v: FIn, @builtin(instance_index) ii: u32) -> VOut { return toClip(bipedV(v, ii, 64u, 26.0)); }
@vertex fn vsColossusShadow(v: FIn, @builtin(instance_index) ii: u32) -> VOut { return toShadow(bipedV(v, ii, 64u, 26.0)); }
@vertex fn vsJelly(v: FIn, @builtin(instance_index) ii: u32) -> VOut { return toClip(jellyV(v, ii)); }
@vertex fn vsRay(v: FIn, @builtin(instance_index) ii: u32) -> VOut { return toClip(rayV(v, ii)); }
@vertex fn vsRayShadow(v: FIn, @builtin(instance_index) ii: u32) -> VOut { return toShadow(rayV(v, ii)); }
@vertex fn vsShip(v: FIn, @builtin(instance_index) ii: u32) -> VOut { return toClip(shipV(v, ii)); }

@vertex
fn vsBird(v: FIn, @builtin(instance_index) ii: u32) -> VOut {
  var o = birdV(v, ii);
  o.pos = F.viewProj * vec4f(o.rel, 1.0);
  return o;
}

@vertex
fn vsBirdShadow(v: FIn, @builtin(instance_index) ii: u32) -> VOut {
  var o = birdV(v, ii);
  o.pos = F.shadowMats[cascadeIdx.x] * vec4f(o.rel, 1.0);
  return o;
}

// ---------------------------------------------------------------------------------------------
fn leafAlpha(uv: vec2f, seed: f32) -> f32 {
  let d = uv - 0.5;
  let r = length(d) * 2.0;
  let th = atan2(d.y, d.x);
  let lobes = 0.55 + 0.45 * pow(abs(cos(th * 2.5 + seed * 6.0)), 0.6);
  return step(r, lobes * 0.98);
}
fn frondAlpha(uv: vec2f) -> f32 {
  let w = 1.0 - abs(uv.x - 0.5) * 2.0;
  let lobe = abs(fract(uv.y * 3.0) - 0.5) * 2.0;
  return step(0.08 + 0.55 * lobe * lobe, w);
}

struct Mat { alb: vec3f, rough: f32, emit: vec3f, trans: f32, spec: f32 };

fn hue(seed: f32, a: vec3f, b: vec3f, c: vec3f) -> vec3f {
  let t = fract(seed * 7.13);
  if (t < 0.5) { return mix(a, b, t * 2.0); }
  return mix(b, c, (t - 0.5) * 2.0);
}

const GILL_RIBS: f32 = 96.0;

// Fauna surfaces. Herd animals (sp 20): indigo saddle with striped edges over pale teal or lilac
// flanks, cream belly, dark socks and rows of luminous spots. Flyers (sp 21-23, one palette per
// flock). lp is the rest-pose position, so markings stay on the skin as it moves.
// Titans (24), travellers (25), colossi (26), sky jellies (27), sky rays (28) and starships (29).
// lp is the rest-pose position in model units; grad carries a ship's engine thrust.
fn pick4(i: f32, a: vec3f, b: vec3f, c: vec3f, d: vec3f) -> vec3f {
  if (i < 1.0) { return a; }
  if (i < 2.0) { return b; }
  if (i < 3.0) { return c; }
  return d;
}

fn lifeMat(part: f32, sp: f32, seed: f32, uv: vec2f, grad: f32, lp: vec3f, glow: f32) -> Mat {
  var m: Mat;
  m.alb = vec3f(0.4);
  m.rough = 0.55;
  m.emit = vec3f(0.0);
  m.trans = 0.0;
  m.spec = 0.04;
  let n = noise3(lp * 4.0 + seed * 9.0, BIGP3);
  if (part == 12.0) { // eyes: glossy; the colossi's burn
    m.alb = vec3f(0.01);
    m.rough = 0.08;
    m.spec = 0.08;
    m.emit = select(vec3f(0.04, 0.1, 0.12) * glow, vec3f(0.35, 0.95, 1.0) * 6.0, sp > 25.5 && sp < 26.5);
    return m;
  }
  if (part == 13.0) {
    m.alb = vec3f(0.08, 0.07, 0.07);
    m.rough = 0.45;
    return m;
  }
  if (sp < 24.5) { // titans: sandstone flanks, dark saddle stripes, pale belly, amber back plates
    let tint = fract(seed * 3.17);
    if (part == 11.0) {
      let tip = smoothstep(0.35, 1.0, uv.y);
      m.alb = mix(vec3f(0.45, 0.22, 0.08), vec3f(0.95, 0.62, 0.25), tip * 0.6);
      m.trans = 0.6;
      m.rough = 0.45;
      m.emit = vec3f(1.0, 0.5, 0.12) * tip * tip * 0.5 * glow;
      return m;
    }
    let flank = mix(vec3f(0.6, 0.4, 0.27), vec3f(0.42, 0.43, 0.5), tint);
    let back = mix(vec3f(0.2, 0.12, 0.1), vec3f(0.13, 0.12, 0.17), tint);
    let stripe = smoothstep(0.35, 0.65, 0.5 + 0.5 * sin(lp.z * 7.0 + n * 2.2 + sin(lp.x * 6.0)));
    let top = smoothstep(1.55, 1.95, lp.y + stripe * 0.35 + n * 0.08) + smoothstep(2.0, 2.4, lp.y) * stripe * 0.6;
    var c = mix(flank, back, sat(top));
    c = mix(c, vec3f(0.8, 0.72, 0.6), (1.0 - smoothstep(1.05, 1.3, lp.y + n * 0.06)) * smoothstep(0.55, 0.85, lp.y));
    c = mix(c, back * 0.7, 1.0 - smoothstep(0.15, 0.5, lp.y));
    m.alb = c * (0.85 + 0.3 * n);
    m.rough = 0.78;
    m.trans = 0.1;
    return m;
  }
  if (sp < 25.5) { // travellers' exosuits: four liveries, light strips, glossy visors, jets
    let li = floor(fract(seed * 4.13) * 4.0);
    let mainC = pick4(li, vec3f(0.8, 0.33, 0.05), vec3f(0.84, 0.84, 0.8), vec3f(0.06, 0.4, 0.43), vec3f(0.58, 0.07, 0.05));
    let secC = pick4(li, vec3f(0.8, 0.8, 0.76), vec3f(0.17, 0.21, 0.25), vec3f(0.8, 0.75, 0.64), vec3f(0.14, 0.14, 0.16));
    let acc = pick4(li, vec3f(0.2, 0.9, 1.0), vec3f(1.0, 0.6, 0.15), vec3f(1.0, 0.3, 0.8), vec3f(0.6, 0.95, 1.0));
    if (part == 20.0) { // visor
      m.alb = vec3f(0.02, 0.025, 0.03);
      m.rough = 0.04;
      m.spec = 0.09;
      m.emit = acc * 0.16;
      return m;
    }
    if (part == 21.0) { // jet nozzles
      m.alb = vec3f(0.07);
      m.rough = 0.35;
      m.emit = mix(acc, vec3f(1.0), 0.4) * (0.25 + 6.0 * grad);
      return m;
    }
    let ax = abs(lp.x);
    let dark = vec3f(0.05, 0.05, 0.06);
    var c = secC;
    var r = 0.6;
    if (lp.y > 1.53 && ax < 0.16) { c = mainC; r = 0.38; }                               // helmet
    else if (lp.z < -0.125 && lp.y > 1.0 && lp.y < 1.47) { c = select(secC, mainC, ax > 0.1); } // backpack
    else if (ax < 0.2 && lp.y > 0.86) {                                                   // torso
      c = select(secC, mainC, lp.z > 0.03 && lp.y > 1.15);
      if (lp.y < 0.97) { c = dark; r = 0.7; }                                             // belt
    } else if (lp.y > 0.8 && ax >= 0.2) {                                                 // arms
      c = select(secC, mainC, lp.y > 1.36);
      if (lp.y < 0.87) { c = dark; r = 0.7; }                                             // gloves
    } else {                                                                              // legs
      c = select(secC, mainC, lp.y > 0.45 && lp.y < 0.58);
      if (lp.y < 0.17) { c = dark; r = 0.72; }                                            // boots
    }
    m.alb = c * (0.9 + 0.15 * n);
    m.rough = r;
    m.spec = 0.05;
    // light strips: across the chest, along the backpack's top edge, at the helmet's temples
    let chest = step(0.08, lp.z) * (1.0 - smoothstep(0.008, 0.016, abs(lp.y - 1.3))) * (1.0 - smoothstep(0.07, 0.09, ax));
    let pack = step(lp.z, -0.2) * smoothstep(1.39, 1.41, lp.y) * (1.0 - smoothstep(0.09, 0.11, ax)) * step(lp.y, 1.45);
    let temple = step(1.6, lp.y) * step(lp.y, 1.72) * smoothstep(0.118, 0.126, ax) * step(-0.02, lp.z);
    m.emit = acc * (chest * 3.0 + pack * 2.0 + temple * 1.5);
    return m;
  }
  if (sp < 26.5) { // colossi: basalt cut by glowing veins, crystal shards on the shoulders
    let vc = mix(vec3f(0.25, 0.85, 1.0), vec3f(0.85, 0.35, 1.0), fract(seed * 2.7));
    if (part == 11.0) {
      let tip = smoothstep(0.2, 1.0, uv.y);
      m.alb = vc * 0.25;
      m.rough = 0.12;
      m.spec = 0.08;
      m.trans = 0.6;
      m.emit = vc * (0.25 + 1.4 * tip * tip) * glow;
      return m;
    }
    let n2 = noise3(lp * 13.0 + seed * 5.0, BIGP3);
    var c = mix(vec3f(0.1, 0.09, 0.12), vec3f(0.25, 0.22, 0.23), smoothstep(-0.4, 0.5, n + 0.3 * n2));
    c = mix(c, vec3f(0.2, 0.27, 0.18), smoothstep(0.25, 0.55, n2) * smoothstep(1.0, 1.4, lp.y) * 0.6); // lichen
    m.alb = c;
    m.rough = 0.88;
    m.spec = 0.03;
    let vein = 1.0 - smoothstep(0.0, 0.05, abs(noise3(lp * vec3f(5.0, 3.0, 5.0) + seed * 3.0, BIGP3)));
    m.emit = vc * vein * smoothstep(0.2, 0.6, lp.y) * 1.3 * glow;
    return m;
  }
  if (sp < 27.5) { // sky jellies: translucent bells, luminous rims and ribs, glowing tentacle tips
    let c = mix(vec3f(0.95, 0.38, 0.95), vec3f(0.3, 0.85, 1.0), smoothstep(0.3, 0.7, fract(seed * 3.3)));
    m.trans = 1.0;
    m.rough = 0.3;
    m.spec = 0.06;
    if (part == 23.0) {
      let rim = smoothstep(0.78, 1.0, uv.y);
      let ribs = smoothstep(0.75, 1.0, cos(uv.x * TAU * 8.0)) * smoothstep(0.2, 0.8, uv.y);
      m.alb = c * 0.35 + 0.1;
      m.emit = c * (0.12 + 1.1 * rim + 0.5 * ribs) * glow;
    } else {
      m.alb = c * 0.3;
      m.emit = c * (0.15 + 1.2 * smoothstep(0.45, 1.0, uv.y)) * 0.8 * glow;
    }
    return m;
  }
  if (sp < 28.5) { // sky rays: indigo backs with pale spots, cream bellies, glowing wing margins
    let below = lp.y < -0.002;
    let g = vec2f(lp.x * 11.0, lp.z * 11.0);
    let sd = length(fract(g) - 0.5 - (hash22(vec2i(floor(g))) - 0.5) * 0.5);
    let spot = 1.0 - smoothstep(0.1, 0.18, sd);
    var c = mix(vec3f(0.06, 0.08, 0.17), vec3f(0.6, 0.66, 0.74), spot * 0.7);
    c = select(c, vec3f(0.78, 0.74, 0.68) * (0.9 + 0.1 * n), below);
    m.alb = c;
    m.rough = 0.5;
    m.trans = 0.35;
    let edge = smoothstep(0.8, 0.97, max(abs(uv.y - 0.5) * 2.0, uv.x));
    m.emit = vec3f(0.2, 0.85, 1.0) * edge * spot * 1.6 * glow;
    return m;
  }
  // starships: two-tone liveries, panel seams, glossy canopies, engine glow with the thrust
  let li = floor(fract(seed * 5.31) * 4.0);
  let base = pick4(li, vec3f(0.82, 0.8, 0.76), vec3f(0.46, 0.5, 0.56), vec3f(0.62, 0.1, 0.07), vec3f(0.12, 0.42, 0.45));
  let stripeC = pick4(li, vec3f(0.78, 0.25, 0.06), vec3f(0.92, 0.58, 0.1), vec3f(0.85, 0.85, 0.8), vec3f(0.92, 0.9, 0.84));
  let engC = pick4(li, vec3f(0.3, 0.8, 1.0), vec3f(1.0, 0.55, 0.2), vec3f(1.0, 0.35, 0.75), vec3f(0.45, 1.0, 0.85));
  if (part == 17.0) {
    m.alb = vec3f(0.015, 0.02, 0.03);
    m.rough = 0.04;
    m.spec = 0.1;
    m.emit = vec3f(0.04, 0.09, 0.11);
    return m;
  }
  if (part == 18.0) {
    m.alb = vec3f(0.05);
    m.emit = engC * (1.5 + 7.0 * grad) * (0.92 + 0.08 * sin(time() * 37.0 + seed * 50.0));
    return m;
  }
  let ax = abs(lp.x);
  let stripe = (1.0 - smoothstep(0.22, 0.3, ax)) * step(0.4, lp.y) + smoothstep(5.0, 5.2, ax) * (1.0 - smoothstep(6.0, 6.2, ax));
  let seam = (1.0 - smoothstep(0.0, 0.05, abs(fract(lp.z / 1.6) - 0.5) - 0.44)) + (1.0 - smoothstep(0.0, 0.05, abs(fract(lp.x / 1.3) - 0.5) - 0.44));
  // pale countershaded bellies keep ships readable from the ground, where only the shaded
  // underside faces the viewer
  let belly = 1.0 - smoothstep(-0.3, 0.12, lp.y + n * 0.08);
  let hullC = mix(mix(base, stripeC, sat(stripe)), vec3f(0.7, 0.7, 0.68), belly * 0.7);
  m.alb = hullC * (1.0 - 0.18 * sat(seam)) * (0.86 + 0.22 * n);
  m.rough = mix(0.32, 0.5, smoothstep(-0.2, 0.6, n));
  m.spec = 0.06;
  // running lights on the wingtips, red to port and green to starboard, slowly blinking
  let tipL = smoothstep(6.95, 7.1, ax) * (0.4 + 0.6 * step(0.5, fract(time() * 0.7 + seed)));
  m.emit = select(vec3f(0.1, 1.0, 0.3), vec3f(1.0, 0.1, 0.05), lp.x < 0.0) * tipL * 3.0;
  return m;
}

fn faunaMat(part: f32, sp: f32, seed: f32, uv: vec2f, grad: f32, lp: vec3f, glow: f32) -> Mat {
  if (sp > 23.5) { return lifeMat(part, sp, seed, uv, grad, lp, glow); }
  var m: Mat;
  m.alb = vec3f(0.4);
  m.rough = 0.55;
  m.emit = vec3f(0.0);
  m.trans = 0.0;
  m.spec = 0.035;
  if (part == 12.0) { // eyes: dark and glossy, with a faint luminous iris
    m.alb = vec3f(0.012);
    m.rough = 0.08;
    m.spec = 0.08;
    m.emit = vec3f(0.03, 0.16, 0.18) * glow * smoothstep(0.55, 0.9, uv.y);
    return m;
  }
  if (part == 13.0) { // hooves, beaks
    m.alb = vec3f(0.07, 0.055, 0.06);
    m.rough = 0.42;
    m.spec = 0.05;
    return m;
  }
  let n = noise3(lp * 7.0 + seed * 9.0, BIGP3);
  if (sp < 20.5) {
    let tint = fract(seed * 3.17);
    let back = mix(vec3f(0.08, 0.10, 0.26), vec3f(0.15, 0.07, 0.24), tint);
    let flank = mix(vec3f(0.30, 0.58, 0.62), vec3f(0.50, 0.44, 0.68), tint);
    if (part == 11.0) { // dorsal sails and antennae: translucent, luminous toward the tips
      let tip = smoothstep(0.3, 1.0, uv.y);
      m.alb = mix(back * 1.6, vec3f(0.45, 0.85, 0.95), tip * 0.5);
      m.trans = 0.8;
      m.rough = 0.4;
      m.emit = vec3f(0.15, 0.85, 1.0) * tip * tip * 1.4 * glow;
      return m;
    }
    let stripe = 0.5 + 0.5 * sin(lp.z * 11.0 + n * 2.5 + sin(lp.x * 9.0) * 0.8);
    let top = smoothstep(1.3, 1.6, lp.y + stripe * 0.18 + n * 0.05);
    let under = 1.0 - smoothstep(1.0, 1.2, lp.y + n * 0.04);
    var c = mix(flank, back, top);
    c = mix(c, vec3f(0.78, 0.74, 0.66), under * smoothstep(0.6, 0.85, lp.y));
    c = mix(c, back * 0.8, 1.0 - smoothstep(0.25, 0.55, lp.y));
    m.alb = c * (0.88 + 0.24 * n);
    m.trans = 0.2;
    // luminous spots on a jittered grid along the flanks
    let g = vec2f(lp.z * 6.5, lp.y * 8.0);
    let gi = vec2i(floor(g));
    let jit = hash22(gi + vec2i(i32(seed * 97.0), select(0, 31, lp.x > 0.0))) - 0.5;
    let sd = length(fract(g) - 0.5 - jit * 0.5);
    let band = smoothstep(1.05, 1.15, lp.y) * (1.0 - smoothstep(1.42, 1.52, lp.y)) * smoothstep(0.12, 0.25, abs(lp.x))
      * step(-0.95, lp.z) * step(lp.z, 0.75);
    let spot = (1.0 - smoothstep(0.12, 0.2, sd)) * band;
    m.alb = mix(m.alb, vec3f(0.6, 0.9, 0.95), spot * 0.5);
    // slow waves of light running from the shoulders to the tail
    let wave = 0.5 + 0.5 * sin(time() * 1.3 + lp.z * 2.4 + seed * 40.0);
    m.emit = vec3f(0.12, 0.8, 1.0) * spot * (0.8 + 1.2 * wave * wave) * glow;
    return m;
  }
  // flyers: 21 vista (amber and cream), 22 valley (dusk teal with luminous bars), 23 highlands (rust and white)
  let f = sp - 21.0;
  var top = vec3f(0.62, 0.30, 0.08);
  var pale = vec3f(0.86, 0.78, 0.62);
  var tipC = vec3f(0.05, 0.04, 0.05);
  if (f > 0.5 && f < 1.5) { top = vec3f(0.05, 0.16, 0.20); pale = vec3f(0.30, 0.52, 0.55); tipC = vec3f(0.02, 0.05, 0.08); }
  if (f > 1.5) { top = vec3f(0.48, 0.16, 0.08); pale = vec3f(0.88, 0.86, 0.80); tipC = vec3f(0.10, 0.06, 0.05); }
  if (part == 15.0) { // wings and tail fan (uv.y: root to tip, uv.x: leading to trailing edge)
    var c = mix(top, pale, smoothstep(0.2, 0.7, uv.x) * 0.6);
    c = mix(c, tipC, smoothstep(0.62, 0.8, uv.y + uv.x * 0.15));
    let bars = smoothstep(0.6, 0.9, sin(uv.y * 38.0)) * smoothstep(0.3, 0.6, uv.x);
    m.alb = c * (1.0 - 0.25 * bars);
    m.trans = 0.55;
    m.rough = 0.6;
    if (f > 0.5 && f < 1.5) { m.emit = vec3f(0.1, 0.8, 1.0) * bars * smoothstep(0.3, 0.6, uv.y) * 0.8 * glow; }
    return m;
  }
  let belly = 1.0 - smoothstep(-0.03, 0.04, lp.y + n * 0.02);
  m.alb = mix(top, pale, belly) * (0.9 + 0.2 * n);
  m.rough = 0.65;
  m.trans = 0.1;
  return m;
}

fn material(part: f32, sp: f32, seed: f32, uv: vec2f, grad: f32, wp: vec3f, lp: vec3f, valley: f32, ribAA: f32) -> Mat {
  var m: Mat;
  m.alb = vec3f(0.3);
  m.rough = 0.7;
  m.emit = vec3f(0.0);
  m.trans = 0.0;
  m.spec = 0.04;
  let glow = 1.0 + valley * 1.5 + F.extra.z;
  if (part >= 10.0) { return faunaMat(part, sp, seed, uv, grad, lp, glow); }
  let gn = fbm(vec2f(uv.x * 8.0, uv.y * 2.0), vec2i(8, 1 << 20), 3, 0.5);
  if (part == 0.0) { // bark
    let groove = abs(sin(uv.x * TAU * 5.0 + gn * 3.0));
    m.alb = mix(vec3f(0.12, 0.07, 0.08), vec3f(0.26, 0.15, 0.14), groove) * (0.85 + 0.3 * gn);
    m.rough = 0.85;
  } else if (part == 9.0) { // smooth stem
    if (sp == 0.0) { // coral spire / tree stems: coral pink
      m.alb = hue(seed, vec3f(0.62, 0.20, 0.22), vec3f(0.72, 0.36, 0.20), vec3f(0.48, 0.16, 0.36)) * (0.8 + 0.3 * grad);
      m.trans = 0.3;
    } else if (sp == 1.0) { // mushroom stalk: pale, faintly luminous veins
      m.alb = vec3f(0.62, 0.62, 0.66) * (0.85 + 0.2 * gn);
      m.trans = 0.5;
      let vein = smoothstep(0.85, 1.0, abs(sin(uv.x * TAU * 6.0 + gn * 4.0)));
      m.emit = vec3f(0.1, 0.7, 0.9) * vein * 0.6 * glow * smoothstep(0.2, 0.9, grad);
    } else if (sp == 3.0) {
      m.alb = vec3f(0.12, 0.26, 0.16);
      m.trans = 0.4;
    } else {
      m.alb = vec3f(0.10, 0.12, 0.16);
      m.trans = 0.3;
    }
    m.rough = 0.6;
  } else if (part == 1.0) { // tree leaf cards
    let base = select(
      hue(seed, vec3f(0.30, 0.06, 0.40), vec3f(0.56, 0.10, 0.26), vec3f(0.70, 0.34, 0.08)),
      hue(seed, vec3f(0.04, 0.26, 0.34), vec3f(0.10, 0.38, 0.30), vec3f(0.22, 0.10, 0.42)), valley > 0.4);
    m.alb = base * (0.7 + 0.5 * grad);
    m.trans = 0.8;
    m.rough = 0.65;
  } else if (part == 8.0) { // fronds
    var c = hue(seed, vec3f(0.42, 0.06, 0.16), vec3f(0.55, 0.12, 0.24), vec3f(0.52, 0.30, 0.08));
    if (sp == 0.0) { c = hue(seed, vec3f(0.05, 0.30, 0.26), vec3f(0.10, 0.40, 0.34), vec3f(0.30, 0.42, 0.12)); }
    if (sp == 8.0) { c = hue(seed, vec3f(0.05, 0.13, 0.08), vec3f(0.10, 0.17, 0.06), vec3f(0.14, 0.08, 0.13)); } // island vines
    c = mix(c, vec3f(0.04, 0.24, 0.30), valley * 0.8);
    let mid = 1.0 - abs(uv.x - 0.5) * 2.0;
    m.alb = c * (0.55 + 0.5 * grad) * (0.8 + 0.3 * mid);
    m.trans = 0.75;
    m.rough = 0.6;
    if (valley > 0.3) { m.emit = vec3f(0.1, 0.8, 0.9) * smoothstep(0.92, 1.0, grad) * 0.5 * glow; }
  } else if (part == 2.0) { // mushroom cap with luminous spots
    let capC = hue(seed, vec3f(0.14, 0.08, 0.36), vec3f(0.36, 0.06, 0.32), vec3f(0.04, 0.24, 0.34));
    let ang = uv.x * TAU;
    let sp2 = vec2f(cos(ang), sin(ang)) * (grad * 6.0 + 1.0);
    let cell = worley3(vec3f(sp2 * 1.3, seed * 11.0), BIGP3);
    let spot = 1.0 - smoothstep(0.18, 0.3, cell);
    m.alb = mix(capC * (0.7 + 0.5 * (1.0 - grad)), vec3f(0.8, 0.85, 0.9), spot * 0.3);
    m.emit = mix(vec3f(0.2, 0.9, 1.0), vec3f(0.9, 0.4, 1.0), fract(seed * 3.7)) * spot * 1.6 * glow;
    m.rough = 0.45;
    m.trans = 0.25;
  } else if (part == 3.0) { // gills: bioluminescent underside, ribs anti-aliased, brightest at the rim
    // lamella spacing wanders (phase-modulated ribs) so the underside never reads as a ribbed umbrella
    let wob = sin(uv.x * TAU * 7.0 + seed * 40.0) * 1.7 + sin(uv.x * TAU * 17.0 + seed * 13.0) * 0.6;
    let rib = 0.5 + 0.5 * cos(uv.x * TAU * GILL_RIBS + wob) * ribAA;
    let c = mix(vec3f(0.0, 0.42, 1.0), vec3f(0.30, 0.05, 1.0), fract(seed * 5.1));
    m.alb = vec3f(0.2, 0.17, 0.22) * (0.7 + 0.3 * rib);
    let band = smoothstep(0.4, 1.0, grad);
    m.emit = c * (0.25 + 0.75 * rib) * (0.03 + 0.97 * band * band) * 0.36 * glow;
    m.rough = 0.8;
  } else if (part == 4.0) { // petals
    let tip = smoothstep(0.1, 1.0, grad);
    var a = vec3f(0.55, 0.05, 0.12);
    var b = vec3f(0.98, 0.60, 0.12);
    if (fract(seed * 2.3) > 0.6) { a = vec3f(0.35, 0.05, 0.45); b = vec3f(0.95, 0.55, 0.85); }
    let vein = smoothstep(0.75, 1.0, abs(sin(uv.x * 30.0))) * 0.25;
    m.alb = mix(a, b, tip) * (1.0 - vein);
    m.trans = 0.9;
    m.rough = 0.5;
    m.emit = b * smoothstep(0.85, 1.0, tip) * 0.25 * glow;
  } else if (part == 5.0) { // crystal: dark glossy gem body, luminous tips
    let c = hue(seed, vec3f(0.012, 0.045, 1.0), vec3f(0.25, 0.024, 0.76), vec3f(0.02, 0.38, 0.62));
    m.alb = c * 0.3;
    m.rough = 0.05;
    m.spec = 0.06;
    m.trans = 0.5;
    let tip = smoothstep(0.62, 1.0, grad);
    m.emit = c * (tip * tip * 1.6 + 0.04) * mix(0.3, 1.0, valley) * glow;
  } else if (part == 6.0) { // glowing bulbs
    let c = hue(seed, vec3f(0.2, 1.0, 0.9), vec3f(1.0, 0.75, 0.25), vec3f(0.9, 0.3, 1.0));
    let pulse = 0.75 + 0.25 * sin(time() * (1.0 + fract(seed * 9.0)) + seed * 30.0);
    m.alb = c * 0.3;
    m.emit = c * 3.2 * pulse * glow;
    m.rough = 0.3;
    m.trans = 1.0;
  } else { // rock: local strata geology, worn bright edges, dark crevices (grad = convexity)
    let n1 = fbm3(wp * 0.7 + seed * 13.0, 3);
    var r = strata(wp.y, wp.xz) * (0.78 + 0.45 * n1) * (0.86 + 0.3 * fbm3(wp * 0.035 + seed * 3.0, 2));
    r *= mix(0.55, 1.18, smoothstep(0.15, 0.85, grad));
    m.alb = r;
    m.rough = 0.78;
    m.spec = 0.035;
  }
  return m;
}

fn shade(in: VOut, front: bool, cut: bool) -> vec4f {
  let part = in.info.x;
  let sp = in.info.y;
  let seed = in.info.z;
  // derivative-based rib anti-aliasing (uniform control flow)
  let ribAA = sat(1.5 - fwidth(in.uv.x) * GILL_RIBS * 1.5);
  if (cut) {
    var a = 1.0;
    if (part == 1.0) { a = leafAlpha(in.uv, seed + floor(in.ag.y * 10.0)); }
    else if (part == 8.0) { a = frondAlpha(in.uv); }
    if (a < 0.5) { discard; }
  }
  let dist = length(in.rel);
  let V = -in.rel / max(dist, 1e-3);
  let L = F.sunDir.xyz;
  var n = normalize(in.nrm);
  // closed fauna surfaces keep their normals; fins, membranes and flora are two-sided
  let closed = part == 10.0 || part == 12.0 || part == 13.0 || part == 14.0 || (part >= 16.0 && part <= 22.0) || part == 25.0;
  if (!front && !closed) { n = -n; }
  let valley = valleyMask(in.world.xz);
  var m = material(part, sp, seed, in.uv, in.ag.y, in.world, in.lpos, valley, ribAA);
  if (part == 7.0) {
    // chipped relief from 3D noise gradient (world space, fades with distance)
    let bf = 1.0 - smoothstep(25.0, 70.0, dist);
    if (bf > 0.0) {
      let q = in.world * 1.6;
      let e = 0.07;
      let h0 = noise3(q, BIGP3) + 0.5 * noise3(q * 2.7, BIGP3);
      let hx = noise3(q + vec3f(e, 0.0, 0.0), BIGP3) + 0.5 * noise3((q + vec3f(e, 0.0, 0.0)) * 2.7, BIGP3);
      let hy = noise3(q + vec3f(0.0, e, 0.0), BIGP3) + 0.5 * noise3((q + vec3f(0.0, e, 0.0)) * 2.7, BIGP3);
      let hz = noise3(q + vec3f(0.0, 0.0, e), BIGP3) + 0.5 * noise3((q + vec3f(0.0, 0.0, e)) * 2.7, BIGP3);
      let gr = vec3f(hx - h0, hy - h0, hz - h0) / e;
      n = normalize(n - (gr - n * dot(gr, n)) * 0.22 * bf);
    }
    // dust and ground cover settle on up-facing surfaces, tying rocks into the local terrain
    let a = attrAt(in.world.xz);
    let dustN = fbm3(in.world * 0.9, 2);
    let up = smoothstep(0.6, 0.95, n.y + dustN * 0.35);
    // floating-island plateaus are always turfed, whatever lies on the ground below them
    let lushF = select(smoothstep(0.2, 0.5, a.y), 1.0, sp == 8.0);
    let soil = mix(vec3f(0.36, 0.22, 0.16), meadowColor(in.world.xz, 0.5 + dustN, valley, highlandMask(in.world.xz)) * 0.55, lushF);
    m.alb = mix(m.alb, soil, up * 0.75);
    m.rough = mix(m.rough, 0.92, up);
  }
  let sunC = sunLightAt(in.world.y);
  var sh = terrainShadow(in.world.xz, in.world.y + 0.5) * cloudShadow(in.world);
  if (sh > 0.001) { sh *= csmShadow(in.rel, n, dist, ign(in.pos.xy, F.misc.x)); }
  let ndl = dot(n, L);
  // wrapped diffuse for foliage softness
  let wrap = select(0.0, 0.35, m.trans > 0.0);
  let dif = sat((ndl + wrap) / (1.0 + wrap));
  var col = m.alb * (1.0 / PI) * dif * sunC * sh;
  col += ggx(n, V, L, m.rough, vec3f(m.spec)) * sunC * sh;
  // translucency when backlit
  let bt = pow(sat(dot(-V, L)), 3.0) * m.trans;
  col += m.alb * sunC * sh * bt * 0.9;
  let ao = in.ag.x;
  col += m.alb * ambientSH(n) * ao;
  let R = reflect(-V, n);
  let nv = sat(dot(n, V));
  if (part == 5.0 || part == 17.0 || part == 20.0) {
    // gems, canopies, visors: full Fresnel sky reflection (lower hemisphere mirrored and darkened as ground)
    let fr = m.spec + (1.0 - m.spec) * pow(1.0 - nv, 5.0);
    let rs = skyColor(normalize(vec3f(R.x, abs(R.y), R.z))) * select(0.35, 1.0, R.y > 0.0);
    col += rs * fr * mix(0.4, 1.0, sh);
  } else {
    // rough environment specular (keeps shaded surfaces from going flat)
    col += ambientSH(R) * envBRDF(vec3f(m.spec), m.rough, nv) * ao;
  }
  if (part == 6.0) {
    // glowing bulbs are translucent and lit from within: face-on the view crosses the most glowing
    // flesh, so the middle burns hot while the limb falls off to the saturated hue and the faceted
    // outline melts into its halo instead of reading as a flat, clipped polygon
    m.emit *= 0.15 + 0.85 * nv * nv;
  }
  if (part >= 10.0 && part != 12.0 && !(part >= 16.0 && part <= 18.0) && part != 20.0 && part != 21.0) {
    // creature sheen: a velvet rim at grazing angles that lifts silhouettes off dark backdrops
    let rim = pow(1.0 - nv, 4.0);
    col += (m.alb * 0.7 + 0.03) * rim * (ambientSH(n) * 1.6 + sunC * sh * sat(ndl + 0.6) * 0.5) * ao;
  }
  col += m.emit;
  return vec4f(col, 1.0);
}

@fragment
fn fsOpaque(in: VOut, @builtin(front_facing) front: bool) -> @location(0) vec4f {
  return shade(in, front, false);
}

@fragment
fn fsCutout(in: VOut, @builtin(front_facing) front: bool) -> @location(0) vec4f {
  return shade(in, front, true);
}

@fragment
fn fsShadowCut(in: VOut) {
  var a = 1.0;
  if (in.info.x == 1.0) { a = leafAlpha(in.uv, in.info.z + floor(in.ag.y * 10.0)); }
  else if (in.info.x == 8.0) { a = frondAlpha(in.uv); }
  if (a < 0.5) { discard; }
}
