// Hashing and procedural noise. All lattice noises take an integer period (in lattice cells)
// per axis so the world can tile seamlessly every WORLD_L metres.

fn pcg(n: u32) -> u32 {
  var h = n * 747796405u + 2891336453u;
  h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
  return (h >> 22u) ^ h;
}
fn hash2u(x: u32, y: u32) -> u32 { return pcg(x ^ pcg(y ^ SEED_U)); }
fn hash3u(x: u32, y: u32, z: u32) -> u32 { return pcg(x ^ pcg(y ^ pcg(z ^ SEED_U))); }
fn u2f(h: u32) -> f32 { return f32(h >> 8u) * (1.0 / 16777216.0); }
fn hash21(p: vec2i) -> f32 { return u2f(hash2u(bitcast<u32>(p.x), bitcast<u32>(p.y))); }
fn hash22(p: vec2i) -> vec2f {
  let h = hash2u(bitcast<u32>(p.x), bitcast<u32>(p.y));
  return vec2f(u2f(h), u2f(pcg(h)));
}
fn hash31(p: vec3i) -> f32 { return u2f(hash3u(bitcast<u32>(p.x), bitcast<u32>(p.y), bitcast<u32>(p.z))); }
fn hash33(p: vec3i) -> vec3f {
  let h = hash3u(bitcast<u32>(p.x), bitcast<u32>(p.y), bitcast<u32>(p.z));
  let h2 = pcg(h);
  return vec3f(u2f(h), u2f(h2), u2f(pcg(h2)));
}
fn hashf(x: f32) -> f32 { return u2f(pcg(bitcast<u32>(x))); }

fn wrapi2(i: vec2i, P: vec2i) -> vec2i {
  let f = vec2f(i);
  let pf = vec2f(P);
  return vec2i(f - pf * floor(f / pf));
}
fn wrapi3(i: vec3i, P: vec3i) -> vec3i {
  let f = vec3f(i);
  let pf = vec3f(P);
  return vec3i(f - pf * floor(f / pf));
}

// 2D gradient noise with analytic derivatives (value, d/dx, d/dy). Range approx [-0.7, 0.7].
fn noised(p: vec2f, P: vec2i) -> vec3f {
  let fl = floor(p);
  let f = p - fl;
  let pf = vec2f(P);
  let w0 = fl - pf * floor(fl / pf);
  let w1 = select(w0 + 1.0, vec2f(0.0), w0 + 1.0 >= pf);
  let i0 = vec2i(w0);
  let i1 = vec2i(w1);
  let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  let du = 30.0 * f * f * (f * (f - 2.0) + 1.0);
  let ga = hash22(i0) * 2.0 - 1.0;
  let gb = hash22(vec2i(i1.x, i0.y)) * 2.0 - 1.0;
  let gc = hash22(vec2i(i0.x, i1.y)) * 2.0 - 1.0;
  let gd = hash22(i1) * 2.0 - 1.0;
  let va = dot(ga, f);
  let vb = dot(gb, f - vec2f(1.0, 0.0));
  let vc = dot(gc, f - vec2f(0.0, 1.0));
  let vd = dot(gd, f - vec2f(1.0, 1.0));
  let k = va - vb - vc + vd;
  let v = va + u.x * (vb - va) + u.y * (vc - va) + u.x * u.y * k;
  let d = ga + u.x * (gb - ga) + u.y * (gc - ga) + u.x * u.y * (ga - gb - gc + gd) + du * (u.yx * k + vec2f(vb, vc) - va);
  return vec3f(v, d);
}

// Periodic fBm with derivatives. p in lattice units of the first octave; P is first-octave period.
fn fbmd(p: vec2f, P: vec2i, oct: i32, gain: f32) -> vec3f {
  var s = vec3f(0.0);
  var a = 0.5;
  var f = 1.0;
  var per = P;
  // rotate-free: lacunarity 2 keeps integer periods
  for (var i = 0; i < oct; i++) {
    let n = noised(p * f + vec2f(0.371, 0.713) * f32(i), per);
    s += vec3f(n.x * a, n.yz * a * f);
    a *= gain;
    f *= 2.0;
    per *= 2;
  }
  return s;
}

fn fbm(p: vec2f, P: vec2i, oct: i32, gain: f32) -> f32 {
  var s = 0.0;
  var a = 0.5;
  var f = 1.0;
  var per = P;
  for (var i = 0; i < oct; i++) {
    s += noised(p * f + vec2f(0.371, 0.713) * f32(i), per).x * a;
    a *= gain;
    f *= 2.0;
    per *= 2;
  }
  return s;
}

// Ridged multifractal (0..1), sharp crests, with octave weighting by previous signal.
fn ridged(p: vec2f, P: vec2i, oct: i32) -> f32 {
  var s = 0.0;
  var a = 0.5;
  var f = 1.0;
  var per = P;
  var w = 1.0;
  var norm = 0.0;
  for (var i = 0; i < oct; i++) {
    var n = 1.0 - abs(noised(p * f + vec2f(0.519, 0.283) * f32(i), per).x) * 1.6;
    n = n * n;
    let c = n * a * w;
    s += c;
    norm += a;
    w = clamp(n * 1.6, 0.0, 1.0);
    a *= 0.5;
    f *= 2.0;
    per *= 2;
  }
  return s / norm;
}

// 3D gradient noise (value only), periodic per axis.
fn grad3(i: vec3i) -> vec3f { return hash33(i) * 2.0 - 1.0; }
fn noise3(p: vec3f, P: vec3i) -> f32 {
  let fl = floor(p);
  let i = vec3i(fl);
  let f = p - fl;
  let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  let n000 = dot(grad3(wrapi3(i, P)), f);
  let n100 = dot(grad3(wrapi3(i + vec3i(1, 0, 0), P)), f - vec3f(1.0, 0.0, 0.0));
  let n010 = dot(grad3(wrapi3(i + vec3i(0, 1, 0), P)), f - vec3f(0.0, 1.0, 0.0));
  let n110 = dot(grad3(wrapi3(i + vec3i(1, 1, 0), P)), f - vec3f(1.0, 1.0, 0.0));
  let n001 = dot(grad3(wrapi3(i + vec3i(0, 0, 1), P)), f - vec3f(0.0, 0.0, 1.0));
  let n101 = dot(grad3(wrapi3(i + vec3i(1, 0, 1), P)), f - vec3f(1.0, 0.0, 1.0));
  let n011 = dot(grad3(wrapi3(i + vec3i(0, 1, 1), P)), f - vec3f(0.0, 1.0, 1.0));
  let n111 = dot(grad3(wrapi3(i + vec3i(1, 1, 1), P)), f - vec3f(1.0, 1.0, 1.0));
  let x00 = mix(n000, n100, u.x);
  let x10 = mix(n010, n110, u.x);
  let x01 = mix(n001, n101, u.x);
  let x11 = mix(n011, n111, u.x);
  return mix(mix(x00, x10, u.y), mix(x01, x11, u.y), u.z);
}

const BIGP3 = vec3i(1 << 20, 1 << 20, 1 << 20);
fn fbm3(p: vec3f, oct: i32) -> f32 {
  var s = 0.0;
  var a = 0.5;
  var q = p;
  for (var i = 0; i < oct; i++) {
    s += noise3(q, BIGP3) * a;
    a *= 0.5;
    q = q * 2.03 + vec3f(1.7, 9.2, 3.1);
  }
  return s;
}

// Cellular (Worley) distance in 3D, periodic. Returns F1 distance.
fn worley3(p: vec3f, P: vec3i) -> f32 {
  let fl = floor(p);
  let i = vec3i(fl);
  let f = p - fl;
  var d = 1e9;
  for (var z = -1; z <= 1; z++) {
    for (var y = -1; y <= 1; y++) {
      for (var x = -1; x <= 1; x++) {
        let o = vec3i(x, y, z);
        let c = hash33(wrapi3(i + o, P));
        let r = vec3f(o) + c - f;
        d = min(d, dot(r, r));
      }
    }
  }
  return sqrt(d);
}

// Interleaved gradient noise (per-pixel dither), animated by frame.
fn ign(px: vec2f, frame: f32) -> f32 {
  let q = px + 5.588238 * (frame % 64.0);
  return fract(52.9829189 * fract(dot(q, vec2f(0.06711056, 0.00583715))));
}
