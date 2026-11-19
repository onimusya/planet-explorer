// Bioluminescent Valley geometry shared by terrain generation and the fauna: the meander warp,
// valley coordinates (u = metres along VALLEY_A -> VALLEY_B, v = metres to the right of the
// meandering centreline) and the herd's trail. Needs common.wgsl and noise.wgsl.

// Domain warp that makes the valley meander (terrain_gen.wgsl carves the valley with it).
fn valleyWarp(p: vec2f) -> vec2f {
  let uv = p / WORLD_L;
  let wq = uv * 4.0;
  let warp = vec2f(fbm(wq, vec2i(4), 4, 0.5), fbm(wq + vec2f(5.2, 1.3), vec2i(4), 4, 0.5));
  let q = uv + warp * 0.04;
  return vec2f(fbm(q * 16.0 + vec2f(2.1, 0.0), vec2i(16), 3, 0.5), fbm(q * 16.0 + vec2f(0.0, 4.7), vec2i(16), 3, 0.5)) * 260.0;
}

fn valleyAxis() -> vec2f {
  let ab = wrapD2(VALLEY_B - VALLEY_A);
  return ab / length(ab);
}

// World position -> valley coordinates (u, v).
fn valleyUV(p: vec2f) -> vec2f {
  let ax = valleyAxis();
  let ap = wrapD2(p + valleyWarp(p) - VALLEY_A);
  return vec2f(dot(ap, ax), dot(ap, vec2f(-ax.y, ax.x)));
}

// Valley coordinates -> world position: fixed-point inverse of the warp, which varies slowly
// enough (wavelengths of 256 m and more) for a few iterations to converge.
fn valleyWorld(uv: vec2f) -> vec2f {
  let ax = valleyAxis();
  let w = VALLEY_A + ax * uv.x + vec2f(-ax.y, ax.x) * uv.y;
  var p = w - valleyWarp(w);
  for (var i = 0; i < 4; i++) { p = w - valleyWarp(p); }
  return p;
}


// The herd trail is a stadium-shaped loop in valley coordinates: two lanes HERD.w either side of
// v = HERD.y, from u = HERD.x - HERD.z to HERD.x + HERD.z, joined by U-turns of radius HERD.w.
// Signed distance to the trail line, negative inside the loop.
fn trailSD(uv: vec2f) -> f32 {
  let pa = uv - vec2f(HERD.x - HERD.z, HERD.y);
  let h = clamp(pa.x / (2.0 * HERD.z), 0.0, 1.0);
  return length(pa - vec2f(2.0 * HERD.z * h, 0.0)) - HERD.w;
}
fn trailDist(uv: vec2f) -> f32 { return abs(trailSD(uv)); }

fn trailLength() -> f32 { return 4.0 * HERD.z + TAU * HERD.w; }

// Point (xy) and unit direction (zw) at arc length s along a stadium loop of half length hl and
// turn radius r, in coordinates centred on the loop: the first straight runs at y = -r toward +x.
fn stadium(s: f32, hl: f32, r: f32) -> vec4f {
  let len = 4.0 * hl + TAU * r;
  var x = s - len * floor(s / len);
  let lane = 2.0 * hl;
  let turn = PI * r;
  if (x < lane) { return vec4f(x - hl, -r, 1.0, 0.0); }
  x -= lane;
  if (x < turn) {
    let a = x / r;
    return vec4f(hl + r * sin(a), -r * cos(a), cos(a), sin(a));
  }
  x -= turn;
  if (x < lane) { return vec4f(hl - x, r, -1.0, 0.0); }
  x -= lane;
  let a = x / r;
  return vec4f(-hl - r * sin(a), r * cos(a), -cos(a), -sin(a));
}

// Point (xy) and unit direction (zw) in valley coordinates at arc length s along the herd's loop.
fn herdPath(s: f32) -> vec4f { return stadium(s, HERD.z, HERD.w) + vec4f(HERD.xy, 0.0, 0.0); }
