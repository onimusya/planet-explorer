// Starship traffic, shared by the pose pass (fauna.wgsl) and the engine trails (fx.wgsl). Each ship
// flies passes over one destination (its lane: camera x, y, z and view azimuth in degrees). A pass
// comes down from high up, flies across or over the camera's view, then climbs back out of sight;
// between passes the ship is gone. Positions follow from the time alone, so the trail renderer
// re-evaluates the same path at earlier times. Ships 0-5 are fighters, 6-8 haulers; lane = j % 3.
// Needs common.wgsl and noise.wgsl.

const SHIP_PASS: f32 = 40.0;  // s of flight per pass
const SHIP_TRAIL: f32 = 7.5;  // s of engine trail behind a ship
const SHIP_FIGHTERS: u32 = 6u;

fn shipLane(l: u32) -> vec4f {
  if (l == 0u) { return SHIP_LANE0; }
  if (l == 1u) { return SHIP_LANE1; }
  return SHIP_LANE2;
}

// (pass index, time into the pass) for ship j at time t; passes repeat every 56-80 s.
fn shipPass(j: u32, t: f32) -> vec2f {
  let h = u2f(hash3u(j, 7u, 0x5a17u));
  let period = SHIP_PASS + 16.0 + 24.0 * h;
  let x = t + h * 97.0 + f32(j) * 23.0;
  let k = floor(x / period);
  return vec2f(k, x - k * period);
}

// Position of ship j on pass k at pass time tau (0..SHIP_PASS).
fn shipPos(j: u32, k: f32, tau: f32) -> vec3f {
  let lane = j % 3u;
  let L = shipLane(lane);
  let ki = i32(k - 4096.0 * floor(k / 4096.0));
  let h = hash33(vec3i(i32(j), ki, 4117));
  let h2 = hash33(vec3i(i32(j), ki, 9133));
  let az = L.w * (PI / 180.0);
  let f = vec2f(sin(az), -cos(az)); // camera forward
  let r = vec2f(cos(az), sin(az));  // camera right
  let hauler = j >= SHIP_FIGHTERS;
  var dir: vec2f;
  var P: vec2f;
  var alt = L.y + mix(80.0, 230.0, h2.z);
  if (lane == 1u) {
    // the Bioluminescent Valley: down the canyon in either direction, 100-190 m over the floor
    dir = f * select(-1.0, 1.0, h.y < 0.5);
    P = L.xz + f * (h2.x - 0.5) * 240.0 + r * (h2.y - 0.5) * 150.0;
    alt = L.y + mix(100.0, 190.0, h2.z);
  } else if (h.x < 0.62 || hauler) {
    // across the view, 0.4-1.1 km out
    let yaw = (h.z - 0.5) * 0.8;
    dir = normalize(r * select(-1.0, 1.0, h.y < 0.5) * cos(yaw) + f * sin(yaw));
    P = L.xz + f * mix(420.0, 1100.0, h2.x) + r * (h2.y - 0.5) * 320.0;
  } else {
    // over the camera from behind (or toward it from far out), then away into the view
    let yaw = (h.z - 0.5) * 0.6;
    dir = normalize(f * cos(yaw) + r * sin(yaw)) * select(-1.0, 1.0, h.y < 0.7);
    P = L.xz + f * mix(-30.0, 140.0, h2.x) + r * (h2.y - 0.5) * 180.0;
  }
  // slowest at the closest approach, accelerating away from it in both directions
  let u = tau - 0.5 * SHIP_PASS;
  let v0 = select(mix(70.0, 105.0, h.y), mix(45.0, 60.0, h.y), hauler);
  let s = v0 * u + select(0.3, 0.17, hauler) * u * u * u;
  // gentle S-turns (the ships bank through them) and a climb out at both ends
  let side = vec2f(-dir.y, dir.x);
  let wig = 70.0 * sin(s / 700.0 + h.z * 6.3) * smoothstep(0.0, 400.0, abs(s) + 200.0);
  let xz = P + dir * s + side * wig;
  let y = alt + 1350.0 * pow(smoothstep(250.0, 3300.0, abs(s)), 1.3) - 25.0 * exp(-sq(s / 500.0));
  return vec3f(xz.x, y, xz.y);
}

// Engine thrust (0..1): hottest while accelerating out at the ends of a pass.
fn shipThrust(tau: f32) -> f32 {
  let u = abs(tau - 0.5 * SHIP_PASS) / (0.5 * SHIP_PASS);
  return 0.35 + 0.65 * smoothstep(0.15, 0.8, u);
}

// Engine nozzle offsets in ship space (x, y, z) at scale 1, per variant.
fn shipEngine(j: u32) -> vec3f {
  return select(vec3f(1.75, -0.05, -8.1), vec3f(3.9, 0.2, -8.9), j >= SHIP_FIGHTERS);
}

// Right, up and forward axes from a forward direction (xyz) and a bank angle (w), as in birdV.
fn bankFrame(fb: vec4f) -> mat3x3f {
  let f = fb.xyz;
  let r0 = normalize(cross(vec3f(0.0, 1.0, 0.0), f) + vec3f(1e-4, 0.0, 0.0));
  let u0 = cross(f, r0);
  return mat3x3f(r0 * cos(fb.w) - u0 * sin(fb.w), u0 * cos(fb.w) + r0 * sin(fb.w), f);
}

// Forward direction and bank angle of ship j at pass time tau: banked into its turns.
fn shipOrient(j: u32, k: f32, tau: f32) -> vec4f {
  let dt = 0.15;
  let p0 = shipPos(j, k, tau);
  let pm = shipPos(j, k, tau - dt);
  let pp = shipPos(j, k, tau + dt);
  let fwd = normalize(pp - pm);
  let acc = (pp - 2.0 * p0 + pm) / (dt * dt);
  let r0 = normalize(cross(vec3f(0.0, 1.0, 0.0), fwd) + vec3f(1e-4, 0.0, 0.0));
  return vec4f(fwd, clamp(1.6 * atan(dot(acc, r0) / 9.81), -1.1, 1.1));
}
