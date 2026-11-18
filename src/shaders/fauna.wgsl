// Fauna animation: one thread per herd animal or bird writes this frame's pose into a buffer that
// the creature and bird vertex shaders in veg.wgsl read (5 vec4 per animal, layout in fauna.ts).
// Stateless: every pose follows from the time, so the herd keeps its walk-and-graze rhythm along
// its trail and the flocks their circuits without any stored simulation state.
//#include "common.wgsl"
//#include "noise.wgsl"
//#include "world.wgsl"
//#include "valley.wgsl"

@group(2) @binding(0) var<storage, read_write> pose: array<vec4f>;

// Herd rhythm (HERD_WALK, HERD_RAMP, HERD_CRUISE, HERD_GRAZE, HERD_S0) comes from world.ts, which
// replays it to frame the herd from the valley destination camera.
const STRIDE_W: f32 = 1.35; // metres per gait cycle at scale 1
// Feet of a scale-1 animal in model space (x, z), for ground sampling: LF, RF, LH, RH.
const HIP_X = vec4f(-0.24, 0.24, -0.26, 0.26);
const HIP_Z = vec4f(0.64, 0.64, -0.78, -0.78);

fn rnd(i: u32, k: u32) -> f32 { return u2f(hash3u(i, k, 0x6d2bu)); }

// Distance travelled along the trail and speed at time t.
fn herdTravel(t: f32) -> vec2f {
  let tw = HERD_WALK / (HERD_CRUISE * (1.0 - HERD_RAMP));
  let period = tw + HERD_GRAZE;
  let k = floor(t / period);
  let tc = t - k * period;
  var d = 1.0;
  var v = 0.0;
  if (tc < tw) {
    let x = tc / tw;
    if (x < HERD_RAMP) {
      let y = x / HERD_RAMP;
      d = HERD_RAMP * (y * y * y - 0.5 * y * y * y * y);
      v = y * y * (3.0 - 2.0 * y);
    } else if (x < 1.0 - HERD_RAMP) {
      d = 0.5 * HERD_RAMP + (x - HERD_RAMP);
      v = 1.0;
    } else {
      let y = (1.0 - x) / HERD_RAMP;
      d = 1.0 - HERD_RAMP - HERD_RAMP * (y * y * y - 0.5 * y * y * y * y);
      v = y * y * (3.0 - 2.0 * y);
    }
    d /= 1.0 - HERD_RAMP;
  }
  return vec2f((k + d) * HERD_WALK, v * HERD_CRUISE);
}

// Ground height as the terrain mesh draws it (bicubic near the camera, plus the near-field relief).
fn groundAt(p: vec2f, camD: f32) -> f32 {
  var h = mix(heightCubic(p), heightAt(p), smoothstep(160.0, 320.0, camD));
  let fade = 1.0 - smoothstep(160.0, 420.0, camD);
  if (fade > 0.0) { h += fbmd(p * 0.125, vec2i(2048), 4, 0.5).x * mix(0.35, 1.9, attrAt(p).w) * fade; }
  return h;
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
  let pa = herdPath(s);
  let pb = herdPath(s + 0.8);
  let la = lat + 1.2 * sin(s * 0.045 + r5 * TAU);
  let lb = lat + 1.2 * sin((s + 0.8) * 0.045 + r5 * TAU);
  let pw = valleyWorld(pa.xy + vec2f(-pa.w, pa.z) * la);
  let pw2 = valleyWorld(pb.xy + vec2f(-pb.w, pb.z) * lb);
  let fwd = normalize(wrapD2(pw2 - pw));
  let rgt = vec2f(fwd.y, -fwd.x);
  let camD = length(wrapD2(pw - F.camPos.xz));
  let h0 = groundAt(pw, camD);
  let hF = groundAt(pw + fwd * (0.72 * sc), camD);
  let hB = groundAt(pw - fwd * (0.8 * sc), camD);
  let hR = groundAt(pw + rgt * (0.3 * sc), camD);
  let hL = groundAt(pw - rgt * (0.3 * sc), camD);
  let pitch = atan2(hF - hB, 1.52 * sc);
  let roll = 0.5 * atan2(hR - hL, 0.6 * sc);
  let y = 0.5 * h0 + 0.25 * (hF + hB);
  let gait = s / (STRIDE_W * sc);
  // ground under each foot relative to the tilted body (model units)
  var gl = vec4f(0.0);
  for (var k = 0; k < 4; k++) {
    let fp = pw + rgt * (HIP_X[k] * sc) + fwd * (HIP_Z[k] * sc);
    let plane = y + sin(pitch) * HIP_Z[k] * sc + sin(roll) * HIP_X[k] * sc;
    gl[k] = clamp((groundAt(fp, camD) - plane) / sc, -0.25, 0.25);
  }
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
  pose[o + 1u] = vec4f(pitch, roll, gait, walk);
  pose[o + 2u] = vec4f(neckP, neckY, headP, tail);
  pose[o + 3u] = gl;
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

fn birdAt(f: u32, k: u32, t: f32) -> vec3f {
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
    let hp = herdPath(HERD_S0 + herdTravel(t).x - 55.0);
    c = valleyWorld(hp.xy) + vec2f(sin(t * 0.021), cos(t * 0.017)) * (fl.w * 0.2);
  }
  let layer = (f32(k) / f32(FAUNA_BPF - 1u) - 0.5) * fl.w * 0.55;
  let y = fl.z + layer + 3.0 * sin(t * 0.11 + h2 * TAU) + 4.0 * sin(t * 0.031 + ff);
  return vec3f(c.x + r * cos(th), y, c.y + r * sin(th));
}

fn bird(k: u32, i: u32) {
  let t = time();
  let f = k / FAUNA_BPF;
  let m = k - f * FAUNA_BPF;
  let p0 = birdAt(f, m, t);
  let pm = birdAt(f, m, t - 0.12);
  let pp = birdAt(f, m, t + 0.12);
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

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x < FAUNA_NC) { herd(id.x); }
  else if (id.x < FAUNA_NC + FAUNA_NB) { bird(id.x - FAUNA_NC, id.x); }
}
