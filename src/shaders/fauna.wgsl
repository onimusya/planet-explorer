// Fauna animation: one thread per animal, traveller or ship writes this frame's pose into a buffer
// that the vertex shaders in veg.wgsl read (5 vec4 each, slots in world.ts: herd, flocks, titans,
// travellers, colossi, jellies, rays, ships). Stateless: every pose follows from the time, so the
// walkers keep their walk-and-pause rhythm along their loops, and the flyers and ships their
// circuits, without any stored simulation state.
//#include "common.wgsl"
//#include "noise.wgsl"
//#include "world.wgsl"
//#include "valley.wgsl"
//#include "ships.wgsl"

@group(2) @binding(0) var<storage, read_write> pose: array<vec4f>;
// World positions on and beside the herd trail (trailTable, once at startup).
@group(2) @binding(1) var<storage, read_write> trailTab: array<vec2f>;

// The herd trail mapped to the world, tabulated: HERD_TAB_NS samples along the loop, each with
// HERD_TAB_NL points across it, every 2 m from 6 m right (-) to 6 m left (+) of the trail.
// valleyWorld is a fixed-point iteration over five evaluations of the meander warp, a long chain
// of noise evaluations per call. Called from the pose update, it made the herd, the travellers
// and the valley flock the slowest groups to update (0.04-0.14 ms of GPU time each per frame)
// and to compile.
@compute @workgroup_size(64)
fn trailTable(@builtin(global_invocation_id) id: vec3u) {
  let n = id.x;
  if (n >= HERD_TAB_NS * HERD_TAB_NL) { return; }
  let i = n / HERD_TAB_NL;
  let lat = -6.0 + 2.0 * f32(n - i * HERD_TAB_NL);
  let pa = herdPath(f32(i) * trailLength() / f32(HERD_TAB_NS));
  trailTab[n] = valleyWorld(pa.xy + vec2f(-pa.w, pa.z) * lat);
}

// World position at arc length s along the herd trail, lat metres to its left: bilinear in the
// table, a few centimetres from the exact mapping between its samples.
fn trailWorld(s: f32, lat: f32) -> vec2f {
  let L = trailLength();
  let x = (s - L * floor(s / L)) / L * f32(HERD_TAB_NS);
  let i0 = min(u32(x), HERD_TAB_NS - 1u);
  let i1 = (i0 + 1u) % HERD_TAB_NS;
  let fx = x - f32(i0);
  let y = clamp((lat + 6.0) * 0.5, 0.0, f32(HERD_TAB_NL - 1u) - 0.001);
  let j = u32(y);
  let fy = y - f32(j);
  let a = mix(trailTab[i0 * HERD_TAB_NL + j], trailTab[i0 * HERD_TAB_NL + j + 1u], fy);
  let b = mix(trailTab[i1 * HERD_TAB_NL + j], trailTab[i1 * HERD_TAB_NL + j + 1u], fy);
  return mix(a, b, fx);
}

// Herd rhythm (HERD_WALK, HERD_RAMP, HERD_CRUISE, HERD_GRAZE, HERD_S0) comes from world.ts, which
// replays it to frame the herd from the valley destination camera.
const STRIDE_W: f32 = 1.35; // metres per gait cycle at scale 1
// Feet of a scale-1 animal in model space (x, z), for ground sampling: LF, RF, LH, RH.
const HIP_X = vec4f(-0.24, 0.24, -0.26, 0.26);
const HIP_Z = vec4f(0.64, 0.64, -0.78, -0.78);

fn rnd(i: u32, k: u32) -> f32 { return u2f(hash3u(i, k, 0x6d2bu)); }

// Walk-and-pause rhythm: bouts of wd metres at cruise m/s with smooth starts and stops (ramp of
// each bout), then pause seconds standing. Distance travelled and speed at time t.
fn travel(t: f32, wd: f32, ramp: f32, cruise: f32, pause: f32) -> vec2f {
  let tw = wd / (cruise * (1.0 - ramp));
  let period = tw + pause;
  let k = floor(t / period);
  let tc = t - k * period;
  var d = 1.0;
  var v = 0.0;
  if (tc < tw) {
    let x = tc / tw;
    if (x < ramp) {
      let y = x / ramp;
      d = ramp * (y * y * y - 0.5 * y * y * y * y);
      v = y * y * (3.0 - 2.0 * y);
    } else if (x < 1.0 - ramp) {
      d = 0.5 * ramp + (x - ramp);
      v = 1.0;
    } else {
      let y = (1.0 - x) / ramp;
      d = 1.0 - ramp - ramp * (y * y * y - 0.5 * y * y * y * y);
      v = y * y * (3.0 - 2.0 * y);
    }
    d /= 1.0 - ramp;
  }
  return vec2f((k + d) * wd, v * cruise);
}

// The herd's distance along its trail and speed at time t (world.ts valleyFraming replays it).
fn herdTravel(t: f32) -> vec2f { return travel(t, HERD_WALK, HERD_RAMP, HERD_CRUISE, HERD_GRAZE); }

// World position (xy) and direction (zw) at arc length s on a stadium loop L (centre x, z, half
// length, turn radius) whose straight lanes run along ax, off metres outward of the loop.
fn loopAt(s: f32, L: vec4f, ax: vec2f, off: f32) -> vec4f {
  let st = stadium(s, L.z, L.w);
  let dp = vec2f(ax.y, -ax.x);
  let dir = ax * st.z + dp * st.w;
  return vec4f(L.xy + ax * st.x + dp * st.y + vec2f(-dir.y, dir.x) * off, dir);
}

// Ground height as the terrain mesh draws it (bicubic near the camera, plus the near-field relief).
fn groundAt(p: vec2f, camD: f32) -> f32 {
  var h = mix(heightCubic(p), heightAt(p), smoothstep(160.0, 320.0, camD));
  let fade = 1.0 - smoothstep(160.0, 420.0, camD);
  if (fade > 0.0) { h += fbmd(p * 0.125, vec2i(2048), 4, 0.5).x * mix(0.35, 1.9, attrAt(p).w) * fade; }
  return h;
}

// A quadruped standing at pw facing fwd: body height, pitch and roll from the ground under it, and
// the ground under each foot relative to the tilted body (model units, for the leg IK in veg.wgsl).
// k scales the hip layout (HIP_X, HIP_Z are the herd animal's).
struct Quad { y: f32, pitch: f32, roll: f32, gl: vec4f };
fn quadBody(pw: vec2f, fwd: vec2f, sc: f32, camD: f32, k: f32) -> Quad {
  let rgt = vec2f(fwd.y, -fwd.x);
  let s = sc * k;
  // ground under the body's centre (0) and the four feet (1-4), in one loop: the shader compiler
  // inlines every groundAt call site whole, and each call is a chain of dependent texture reads
  var h: array<f32, 5>;
  for (var j = 0; j < 5; j++) {
    var o = vec2f(0.0);
    if (j > 0) { o = vec2f(HIP_X[j - 1], HIP_Z[j - 1]); }
    h[j] = groundAt(pw + rgt * (o.x * s) + fwd * (o.y * s), camD);
  }
  // pitch from the front feet against the back ones, roll from the right feet against the left
  let hF = 0.5 * (h[1] + h[2]);
  let hB = 0.5 * (h[3] + h[4]);
  var q: Quad;
  q.pitch = atan2(hF - hB, (HIP_Z[0] - HIP_Z[2]) * s);
  q.roll = 0.5 * atan2(0.5 * (h[2] + h[4] - h[1] - h[3]), 0.5 * (HIP_X[1] - HIP_X[0] + HIP_X[3] - HIP_X[2]) * s);
  q.y = 0.5 * h[0] + 0.25 * (hF + hB);
  for (var f = 0; f < 4; f++) {
    let plane = q.y + sin(q.pitch) * HIP_Z[f] * s + sin(q.roll) * HIP_X[f] * s;
    q.gl[f] = clamp((h[1 + f] - plane) / sc, -0.25, 0.25);
  }
  return q;
}

fn herd(i: u32) {
  let t = time();
  let fi = f32(i);
  let young = i >= 7u;
  let r1 = rnd(i, 1u);
  let r2 = rnd(i, 2u);
  let r3 = rnd(i, 3u);
  let r4 = rnd(i, 4u);
  let r5 = rnd(i, 5u);
  let sc = select(1.45 + 0.3 * r1, 0.75 + 0.12 * r1, young);
  // loose formation: adults alternate between two files, each young one walks beside an adult
  var behind = fi * 5.0 + r3 * 2.0;
  var lat = select(-3.0, 3.0, (i & 1u) == 1u) + (r2 - 0.5) * 1.6;
  if (young) {
    let a = f32(i - 7u) * 2.0 + 1.0;
    behind = a * 5.0 + 1.2 + r3 * 1.0;
    lat = 3.0 - 2.2 + (r2 - 0.5) * 0.4;
  }
  let lag = (r4 - 0.5) * 2.5; // each animal starts and stops a little apart from the others
  let tv = herdTravel(t + lag);
  let s = HERD_S0 + tv.x - behind;
  let walk = sat(tv.y / (0.55 * HERD_CRUISE));
  // lateral wander follows the distance walked, so standing animals do not drift
  let la = lat + 1.2 * sin(s * 0.045 + r5 * TAU);
  let lb = lat + 1.2 * sin((s + 0.8) * 0.045 + r5 * TAU);
  let pw = trailWorld(s, la);
  let pw2 = trailWorld(s + 0.8, lb);
  let fwd = normalize(wrapD2(pw2 - pw));
  let rgt = vec2f(fwd.y, -fwd.x);
  let camD = length(wrapD2(pw - F.camPos.xz));
  let q = quadBody(pw, fwd, sc, camD, 1.0);
  let y = q.y;
  let gait = s / (STRIDE_W * sc);
  // head: up and bobbing while walking; down and nibbling while grazing, looking up now and then
  let graze = 1.0 - smoothstep(0.1, 0.5, walk);
  let look = smoothstep(0.06, 0.2, fbm(vec2f(t * 0.07, fi * 3.7), vec2i(1 << 16), 2, 0.5));
  let down = graze * (1.0 - look);
  var neckP = mix(-0.12 + 0.05 * sin(TAU * 2.0 * gait), 1.62 + 0.06 * sin(t * 6.3 + r1 * 20.0), down);
  var neckY = 0.5 * fbm(vec2f(t * 0.05 + 13.0, fi * 2.9), vec2i(1 << 16), 2, 0.5) * (1.0 - 0.6 * down);
  let headP = mix(0.12 - 0.05 * sin(TAU * 2.0 * gait + 1.0), 0.5, down);
  // turn to watch a camera that comes close
  let toCam = wrapD2(F.camPos.xz - pw);
  let dCam = length(vec3f(toCam.x, F.camPos.y - y - 1.6 * sc, toCam.y));
  let notice = 1.0 - smoothstep(24.0, 42.0, dCam);
  if (notice > 0.0) {
    let ang = atan2(dot(toCam, rgt), dot(toCam, fwd));
    neckY = mix(neckY, clamp(ang, -1.15, 1.15), notice);
    let up = atan2(F.camPos.y - y - 1.8 * sc, length(toCam));
    neckP = mix(neckP, clamp(-0.6 * up - 0.15, -0.7, 0.4), notice);
  }
  let tail = 0.22 * sin(t * 1.25 + r2 * 9.0) + walk * 0.12 * sin(TAU * gait);
  let bob = walk * 0.035 * sc * cos(TAU * 2.0 * gait);
  let o = i * 5u;
  pose[o] = vec4f(pw.x, y, pw.y, atan2(fwd.x, fwd.y));
  pose[o + 1u] = vec4f(q.pitch, q.roll, gait, walk);
  pose[o + 2u] = vec4f(neckP, neckY, headP, tail);
  pose[o + 3u] = q.gl;
  pose[o + 4u] = vec4f(sc, fract(r1 * 7.3 + fi * 0.137), bob, 0.0);
}

// ---------------------------------------------------------------------------------------------
// Flocks: each is a kettle of gliders circling in a thermal in view of its destination (FLOCKn:
// centre x, z, altitude, kettle radius). Every bird keeps its own circle radius, speed and height
// layer (radii follow a low-discrepancy sequence, layers the index, so birds on similar circles fly
// at different heights); the kettle drifts and breathes. Velocity and banking come from finite
// differences, so tighter circles bank harder.
fn flockOf(f: u32) -> vec4f {
  if (f == 0u) { return FLOCK0; }
  if (f == 1u) { return FLOCK1; }
  return FLOCK2;
}

// anchor: the valley kettle's point on the herd trail (flock 1 only, see bird)
fn birdAt(f: u32, k: u32, t: f32, anchor: vec2f) -> vec3f {
  let fl = flockOf(f);
  let ff = f32(f);
  let id = k + f * 97u;
  let h1 = rnd(id, 11u);
  let h2 = rnd(id, 12u);
  let h3 = rnd(id, 13u);
  let dir = select(1.0, -1.0, f == 1u);
  let r0 = fl.w * (0.45 + 0.55 * fract(f32(k) * 0.618034 + 0.3));
  let th = dir * t * (9.0 + 4.0 * h1) / r0 + h3 * TAU;
  let r = r0 * (1.0 + 0.12 * sin(t * 0.05 + h2 * TAU));
  var c = fl.xy + vec2f(sin(t * 0.021 + ff), cos(t * 0.017 + ff * 1.7)) * (fl.w * 0.35);
  if (f == 1u) {
    // the valley kettle trails the herd the way scavengers follow grazers: it circles over the
    // trail ~40 m behind the animals, which puts it beyond them as the valley destination camera
    // looks back along the trail (main.ts valleyPose)
    c = anchor + vec2f(sin(t * 0.021), cos(t * 0.017)) * (fl.w * 0.2);
  }
  let layer = (f32(k) / f32(FAUNA_BPF - 1u) - 0.5) * fl.w * 0.55;
  let y = fl.z + layer + 3.0 * sin(t * 0.11 + h2 * TAU) + 4.0 * sin(t * 0.031 + ff);
  return vec3f(c.x + r * cos(th), y, c.y + r * sin(th));
}

fn bird(k: u32, i: u32) {
  let t = time();
  let f = k / FAUNA_BPF;
  let m = k - f * FAUNA_BPF;
  // the valley kettle's anchor moves at the herd's walking pace: one sample serves all three
  // positions below
  var anchor = vec2f(0.0);
  if (f == 1u) { anchor = trailWorld(HERD_S0 + herdTravel(t).x - 55.0, 0.0); }
  let p0 = birdAt(f, m, t, anchor);
  let pm = birdAt(f, m, t - 0.12, anchor);
  let pp = birdAt(f, m, t + 0.12, anchor);
  let vel = (pp - pm) / 0.24;
  let acc = (pp - 2.0 * p0 + pm) / (0.12 * 0.12);
  let fwd = normalize(vel);
  let r0 = normalize(cross(vec3f(0.0, 1.0, 0.0), fwd));
  let bank = clamp(atan(dot(acc, r0) / 9.81), -0.9, 0.9);
  let climb = vel.y / max(length(vel), 0.1);
  let h = rnd(m + f * 97u, 15u);
  // flapping in bursts between glides, and whenever climbing
  let burst = smoothstep(-0.05, 0.2, fbm(vec2f(t * 0.11 + f32(m) * 1.37, f32(f) * 5.1), vec2i(1 << 16), 2, 0.5));
  let flap = max(burst, sat(climb * 5.0));
  let sc = FLOCK_SCALE[f] * (0.85 + 0.3 * h);
  let freq = 1.9 / sqrt(sc);
  let o = i * 5u;
  pose[o] = vec4f(p0, sc);
  pose[o + 1u] = vec4f(fwd, bank);
  pose[o + 2u] = vec4f(t * TAU * freq + h * TAU, mix(0.07, 0.8, flap), (1.0 - flap) * 0.1, fract(h * 13.7));
  pose[o + 3u] = vec4f(-0.5 * climb, 0.0, f32(f), 0.0);
}

// ---------------------------------------------------------------------------------------------
// Titans: long-necked giants grazing the Highlands mesa. They walk its loop in a loose file with
// long pauses to browse, low or reaching up, on the herd animal's leg rig (rig base 32, veg.wgsl).
const TITAN_STRIDE: f32 = 1.4; // m per gait cycle at scale 1 (genTitan in fauna.ts)

fn titan(k: u32) {
  let i = FAUNA_OT + k;
  let t = time();
  let fk = f32(k);
  let r1 = rnd(i, 1u);
  let r2 = rnd(i, 2u);
  let r3 = rnd(i, 3u);
  let r4 = rnd(i, 4u);
  let young = k == FAUNA_NT - 1u;
  let sc = select(9.0 + 1.6 * r1, 6.2 + 0.6 * r1, young);
  let tv = travel(t + r4 * 7.0, 80.0, 0.3, 2.2, 34.0);
  let s = tv.x - fk * 32.0 - r3 * 6.0 + 200.0;
  let walk = sat(tv.y / (0.5 * 2.2));
  // the young one keeps to the inside of the file
  let lat = (r2 - 0.5) * 14.0 + select(0.0, -10.0, young);
  let pa = loopAt(s, TITAN_LOOP, TITAN_AX, lat + 4.0 * sin(s * 0.012 + r1 * TAU));
  let pb = loopAt(s + 3.0, TITAN_LOOP, TITAN_AX, lat + 4.0 * sin((s + 3.0) * 0.012 + r1 * TAU));
  let pw = pa.xy;
  let fwd = normalize(pb.xy - pa.xy);
  let q = quadBody(pw, fwd, sc, length(wrapD2(pw - F.camPos.xz)), 1.0);
  let gait = s / (TITAN_STRIDE * sc);
  // neck high and swaying while walking; paused, it browses low or reaches up to look around
  let pause = 1.0 - smoothstep(0.1, 0.6, walk);
  let reach = smoothstep(-0.08, 0.12, fbm(vec2f(t * 0.03, fk * 4.1), vec2i(1 << 16), 2, 0.5));
  let neckP = mix(0.06 + 0.04 * sin(TAU * 2.0 * gait), mix(0.8, -0.3, reach), pause);
  let neckY = 0.4 * fbm(vec2f(t * 0.025 + 5.0, fk * 2.3), vec2i(1 << 16), 2, 0.5);
  let headP = mix(0.1, mix(0.5, 0.05, reach), pause) + 0.05 * sin(t * 0.7 + r2 * 9.0);
  let tail = 0.16 * sin(t * 0.45 + r2 * 9.0) + walk * 0.08 * sin(TAU * gait);
  let bob = walk * 0.025 * sc * cos(TAU * 2.0 * gait);
  let o = i * 5u;
  pose[o] = vec4f(pw.x, q.y, pw.y, atan2(fwd.x, fwd.y));
  pose[o + 1u] = vec4f(q.pitch, q.roll, gait, walk);
  pose[o + 2u] = vec4f(neckP, neckY, headP, tail);
  pose[o + 3u] = q.gl;
  pose[o + 4u] = vec4f(sc, fract(r1 * 5.1 + fk * 0.31), bob, 0.0);
}

// ---------------------------------------------------------------------------------------------
// Humanoids (travellers in exosuits, colossi). Pose: (pos, heading), (gait phase in cycles, walk,
// lean, jet thrust), (head yaw, head pitch, arm raise, -), -, (scale, seed, bob, -).
const BIPED_STRIDE: f32 = 1.45; // m per gait cycle at scale 1 (genBiped in fauna.ts)

fn writeBiped(i: u32, pw: vec2f, heading: f32, cyc: f32, walk: f32, lean: f32, hy: f32, hp: f32,
              arm: f32, sc: f32, seed: f32) {
  let o = i * 5u;
  let y = groundAt(pw, length(wrapD2(pw - F.camPos.xz)));
  let bob = walk * 0.03 * sc * (cos(TAU * 2.0 * cyc) - 1.0);
  pose[o] = vec4f(pw.x, y, pw.y, heading);
  pose[o + 1u] = vec4f(cyc, walk, lean, 0.0);
  pose[o + 2u] = vec4f(hy, hp, arm, 0.0);
  pose[o + 3u] = vec4f(0.0);
  pose[o + 4u] = vec4f(sc, seed, bob, 0.0);
}

fn angDiff(a: f32, b: f32) -> f32 { return atan2(sin(b - a), cos(b - a)); }
fn headingTo(d: vec2f) -> f32 { return atan2(d.x, d.y); }

// Travellers: 0 stands near the edge of the Vista shelf gazing out, 1 paces along the edge,
// 2 and 3 follow the herd along its trail and stop to scan the animals while they graze.
fn traveller(k: u32) {
  let i = FAUNA_OH + k;
  let t = time();
  let fk = f32(k);
  let r1 = rnd(i, 1u);
  let r2 = rnd(i, 2u);
  let sc = 0.96 + 0.08 * r1;
  // slow look-around; now and then a scan with the raised arm, a few seconds in every 29
  var hy = 0.85 * fbm(vec2f(t * 0.06 + fk * 7.0, 3.1), vec2i(1 << 16), 2, 0.5);
  let hp = 0.06 + 0.2 * fbm(vec2f(t * 0.05, fk * 1.9 + 9.0), vec2i(1 << 16), 2, 0.5);
  let sp = fract(t / 29.0 + r2);
  var arm = smoothstep(0.62, 0.68, sp) * (1.0 - smoothstep(0.86, 0.92, sp));
  let hView = headingTo(vec2f(sin(0.38), -cos(0.38))); // out over the lake from the shelf
  var pw = TRAVELLER_A.xy;
  var heading = hView - 0.12;
  var cyc = 0.0;
  var walk = 0.0;
  if (k == 1u) {
    let A = TRAVELLER_A.zw;
    let B = TRAVELLER_B;
    let d = length(B - A);
    let tv = travel(t + r2 * 40.0, d, 0.3, 1.0, 8.0);
    let leg = floor(tv.x / d);
    let f = select(fract(tv.x / d), 1.0 - fract(tv.x / d), (u32(leg) & 1u) == 1u);
    pw = mix(A, B, f);
    walk = sat(tv.y / 0.5);
    let hWalk = headingTo(select(B - A, A - B, (u32(leg) & 1u) == 1u));
    heading = hView + angDiff(hView, hWalk) * smoothstep(0.0, 0.5, walk);
    cyc = tv.x / (BIPED_STRIDE * sc);
  } else if (k >= 2u) {
    let behind = 41.0 + 5.0 * (fk - 2.0);
    let tv = herdTravel(t - 1.5 - 0.8 * (fk - 2.0));
    let s = HERD_S0 + tv.x - behind;
    walk = sat(tv.y / (0.55 * HERD_CRUISE));
    let lat = select(-1.6, 1.4, k == 3u);
    pw = trailWorld(s, lat);
    let hWalk = headingTo(wrapD2(trailWorld(s + 0.8, lat) - pw));
    // standing, they face the middle of the grazing herd
    let hMid = headingTo(wrapD2(trailWorld(HERD_S0 + tv.x - 15.0, 0.0) - pw));
    heading = hMid + angDiff(hMid, hWalk) * smoothstep(0.0, 0.4, walk);
    cyc = s / (BIPED_STRIDE * sc);
    hy *= 0.6;
  }
  arm *= 1.0 - smoothstep(0.0, 0.25, walk);
  writeBiped(i, pw, heading, cyc, walk, 0.06 * walk, hy, hp, arm, sc, fract(r1 * 3.7 + fk * 0.29));
}

// Colossi: giant humanoid walkers on the Highlands mesa. Long strides around their loop, pauses to
// look about, an arm raised to the sky now and then.
fn colossus(k: u32) {
  let i = FAUNA_OK + k;
  let t = time();
  let fk = f32(k);
  let r1 = rnd(i, 1u);
  let r2 = rnd(i, 2u);
  let sc = 15.0 + 2.0 * r1;
  let tv = travel(t + r2 * 13.0, 150.0, 0.25, 4.0, 24.0);
  let s = tv.x - fk * 75.0 + 40.0;
  let walk = sat(tv.y / (0.5 * 4.0));
  let off = (r2 - 0.5) * 10.0;
  let pa = loopAt(s, COLOSSUS_LOOP, COLOSSUS_AX, off);
  let pb = loopAt(s + 4.0, COLOSSUS_LOOP, COLOSSUS_AX, off);
  let still = 1.0 - smoothstep(0.0, 0.3, walk);
  let hy = 0.7 * fbm(vec2f(t * 0.03 + fk * 5.0, 1.7), vec2i(1 << 16), 2, 0.5) * (0.4 + 0.6 * still);
  let hp = -0.05 + 0.15 * fbm(vec2f(t * 0.02, fk * 3.3), vec2i(1 << 16), 2, 0.5);
  let arm = still * smoothstep(0.05, 0.2, fbm(vec2f(t * 0.04, fk * 6.1 + 2.0), vec2i(1 << 16), 2, 0.5));
  writeBiped(i, pa.xy, headingTo(pb.xy - pa.xy), s / (BIPED_STRIDE * sc), walk, 0.1 * walk, hy, hp, arm,
    sc, fract(r1 * 4.3 + fk * 0.37));
}

// ---------------------------------------------------------------------------------------------
// Sky jellies drifting over the herd's clearing: each wanders around its own spot (spread on a
// sunflower spiral), pulses to rise a little and sinks slowly in between. Pose: (pos, spin),
// (pulse phase in cycles, drift x, drift z, -), -, -, (scale, seed, -, -).
fn jelly(k: u32) {
  let i = FAUNA_OJ + k;
  let t = time();
  let h1 = rnd(i, 21u);
  let h2 = rnd(i, 22u);
  let h3 = rnd(i, 23u);
  let h4 = rnd(i, 24u);
  let h5 = rnd(i, 25u);
  let a0 = (f32(k) + 0.6 * h1) * 2.39996;
  let r0 = sqrt((f32(k) + 0.5) / f32(FAUNA_NJ)) * 62.0;
  let ws = 0.012 + 0.012 * h3;
  let w = t * ws + h4 * TAU;
  let wr = 7.0 + 7.0 * h5;
  let c = JELLY_C + vec2f(cos(a0), sin(a0)) * r0 + vec2f(sin(w), cos(w * 0.83 + 1.3)) * wr;
  let v = vec2f(cos(w), -0.83 * sin(w * 0.83 + 1.3)) * wr * ws;
  let g = groundAt(c, length(wrapD2(c - F.camPos.xz)));
  let ph = t * (0.36 + 0.2 * h3) + h1 * 7.0;
  let cy = fract(ph);
  let y = g + 3.5 + 10.0 * h5 + 1.4 * sin(t * 0.11 + h2 * TAU) + 0.4 * (smoothstep(0.0, 0.3, cy) - cy);
  let o = i * 5u;
  pose[o] = vec4f(c.x, y, c.y, t * (0.04 + 0.05 * h4) + h1 * TAU);
  pose[o + 1u] = vec4f(ph, v * 6.0, 0.0);
  pose[o + 2u] = vec4f(0.0);
  pose[o + 3u] = vec4f(0.0);
  pose[o + 4u] = vec4f(0.55 + 0.6 * h2, fract(h1 * 9.7 + h3), 0.0, 0.0);
}

// Great sky rays soaring in wide, wandering loops over the plain below the Vista. Pose as the
// birds: (pos, scale), (forward, bank), (wave phase, wave amplitude, -, seed), (tail sway, ...).
fn rayAt(k: u32, t: f32) -> vec3f {
  let i = FAUNA_OR + k;
  let h1 = rnd(i, 31u);
  let h2 = rnd(i, 32u);
  let h3 = rnd(i, 33u);
  let dir = select(1.0, -1.0, (k & 1u) == 1u);
  let R = RAY_C.w * (0.75 + 0.35 * h1);
  let th = dir * t * (11.0 + 2.0 * h2) / R + h3 * TAU;
  let c = RAY_C.xy + vec2f(sin(t * 0.013 + h1 * 9.0), cos(t * 0.011 + h2 * 7.0)) * 70.0;
  let y = RAY_C.z + 22.0 * f32(k) + 20.0 * sin(t * 0.07 + h3 * TAU) + 9.0 * sin(th * 2.0 + h1 * 5.0);
  return vec3f(c.x + R * cos(th), y, c.y + R * 0.6 * sin(th));
}

fn skyRay(k: u32) {
  let i = FAUNA_OR + k;
  let t = time();
  let p0 = rayAt(k, t);
  let pm = rayAt(k, t - 0.25);
  let pp = rayAt(k, t + 0.25);
  let vel = (pp - pm) / 0.5;
  let acc = (pp - 2.0 * p0 + pm) / (0.25 * 0.25);
  let fwd = normalize(vel);
  let r0 = normalize(cross(vec3f(0.0, 1.0, 0.0), fwd));
  let climb = vel.y / max(length(vel), 0.1);
  let h = rnd(i, 35u);
  let o = i * 5u;
  pose[o] = vec4f(p0, 16.0 + 4.0 * h);
  pose[o + 1u] = vec4f(fwd, clamp(1.4 * atan(dot(acc, r0) / 9.81), -0.6, 0.6));
  pose[o + 2u] = vec4f(t * TAU * (0.15 + 0.03 * h) + h * TAU, 0.17 + 0.07 * sat(climb * 4.0), 0.0, fract(h * 7.3));
  pose[o + 3u] = vec4f(0.3 * sin(t * 0.45 + h * 6.0), 0.0, 0.0, 0.0);
}

// Starships (ships.wgsl). Pose: (pos, scale; 0 between passes), (forward, bank),
// (engine thrust, variant, seed, -).
fn ship(j: u32) {
  let i = FAUNA_OS + j;
  let ps = shipPass(j, time());
  let o = i * 5u;
  if (ps.y > SHIP_PASS) {
    pose[o] = vec4f(0.0, -2000.0, 0.0, 0.0);
    pose[o + 1u] = vec4f(0.0, 0.0, 1.0, 0.0);
    return;
  }
  pose[o] = vec4f(shipPos(j, ps.x, ps.y), 1.0);
  pose[o + 1u] = shipOrient(j, ps.x, ps.y);
  pose[o + 2u] = vec4f(shipThrust(ps.y), select(0.0, 1.0, j >= SHIP_FIGHTERS), fract(f32(j) * 0.618 + 0.13), 0.0);
}

// One workgroup per group (fauna.ts dispatches all 8 at once), so the groups run side by side.
// Each pose is a short chain of dependent reads, so a frame pays for the slowest group rather
// than their sum (one dispatch per group cost 2.6 times as much, measured before the trail table).
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) k: u32) {
  switch (wg.x) {
    case 0u: { if (k < FAUNA_NC) { herd(k); } }
    case 1u: { if (k < FAUNA_NB) { bird(k, FAUNA_NC + k); } }
    case 2u: { if (k < FAUNA_NT) { titan(k); } }
    case 3u: { if (k < FAUNA_NH) { traveller(k); } }
    case 4u: { if (k < FAUNA_NK) { colossus(k); } }
    case 5u: { if (k < FAUNA_NJ) { jelly(k); } }
    case 6u: { if (k < FAUNA_NR) { skyRay(k); } }
    default: { if (k < FAUNA_NS) { ship(k); } }
  }
}
