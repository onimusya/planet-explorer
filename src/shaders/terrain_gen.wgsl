// One-time terrain field generation (heights, erosion, attributes, AO, sun shadows, min/max).
//#include "common.wgsl"
//#include "noise.wgsl"

@group(1) @binding(0) var inA: texture_2d<f32>;
@group(1) @binding(1) var outR32: texture_storage_2d<r32float, write>;
@group(1) @binding(2) var outRGBA8: texture_storage_2d<rgba8unorm, write>;
@group(1) @binding(3) var inB: texture_2d<f32>;
@group(1) @binding(4) var outRGBA16: texture_storage_2d<rgba16float, write>;
@group(1) @binding(5) var inMax: texture_2d<f32>;
@group(1) @binding(6) var<storage, read_write> minmax: array<vec2f>;
@group(1) @binding(7) var<uniform> G: vec4u; // x = row offset for sliced dispatch
@group(1) @binding(8) var<storage, read_write> cpuGrid: array<f32>;

const N: i32 = HMAP_N;
const TX: f32 = WORLD_L / f32(HMAP_N);

fn ld(t: texture_2d<f32>, i: vec2i) -> vec4f {
  let w = ((i % vec2i(N)) + vec2i(N)) % vec2i(N);
  return textureLoad(t, w, 0);
}
fn bilin(t: texture_2d<f32>, p: vec2f) -> f32 {
  let q = p / TX;
  let fl = floor(q);
  let i = vec2i(fl);
  let f = q - fl;
  return mix(mix(ld(t, i).r, ld(t, i + vec2i(1, 0)).r, f.x), mix(ld(t, i + vec2i(0, 1)).r, ld(t, i + vec2i(1, 1)).r, f.x), f.y);
}

fn terrace(h: f32, stp: f32, sharp: f32) -> f32 {
  let k = h / stp;
  let fl = floor(k);
  let f = k - fl;
  let s = smoothstep(0.5 - sharp * 0.5, 0.5 + sharp * 0.5, f);
  return (fl + s) * stp;
}
fn smax(a: f32, b: f32, k: f32) -> f32 {
  let h = sat(0.5 + 0.5 * (a - b) / k);
  return mix(b, a, h) + k * h * (1.0 - h);
}
fn smin(a: f32, b: f32, k: f32) -> f32 { return -smax(-a, -b, k); }
fn regionDist(p: vec2f, c: vec2f) -> f32 { return length(wrapD2(p - c)); }

fn segInfo(p: vec2f, a: vec2f, b: vec2f) -> vec3f {
  let ab = wrapD2(b - a);
  let ap = wrapD2(p - a);
  let t = clamp(dot(ap, ab) / dot(ab, ab), 0.0, 1.0);
  let off = ap - ab * t;
  return vec3f(length(off), t, dot(off, normalize(vec2f(-ab.y, ab.x))));
}

// Designed macro landscape. Returns height in metres.
fn baseHeight(p: vec2f) -> f32 {
  let uv = p / WORLD_L;
  let wq = uv * 4.0;
  let warp = vec2f(fbm(wq, vec2i(4), 4, 0.5), fbm(wq + vec2f(5.2, 1.3), vec2i(4), 4, 0.5));
  let q = uv + warp * 0.04;
  let qm = q * WORLD_L;

  // Rolling base terrain
  var h = 40.0 + fbm(q * 8.0, vec2i(8), 5, 0.5) * 150.0 + fbm(q * 32.0, vec2i(32), 3, 0.5) * 22.0;

  // Mountains: ridged multifractal masked by low-frequency noise plus a designed northern range
  var mb = fbm(q * 2.0 + vec2f(0.3, 0.7), vec2i(2), 3, 0.5) * 1.3 - 0.05;
  let rn = segInfo(p, RANGE_A, RANGE_B);
  mb += 1.15 * exp(-sq(rn.x / 1500.0));
  mb -= 1.4 * exp(-sq(regionDist(p, LAKE_C) / 2300.0));
  mb -= 1.0 * exp(-sq(regionDist(p, VISTA_P) / 1800.0));
  mb -= 0.9 * exp(-sq(regionDist(p, HIGHLANDS_C) / 2400.0));
  let mMask = smoothstep(0.0, 0.65, mb);
  let r = ridged(q * 6.0, vec2i(6), 8);
  let r2 = ridged(q * 12.0 + vec2f(0.21, 0.63), vec2i(12), 6);
  h += mMask * (pow(r, 1.7) * 1650.0 + r2 * 160.0);

  // Lake basin in front of the vista
  let dl = length(wrapD2(p - LAKE_C)) + fbm(q * 48.0, vec2i(48), 2, 0.5) * 180.0;
  let basin = 1.0 - smoothstep(520.0, 1350.0, dl);
  h = mix(h, -22.0 + fbm(q * 64.0, vec2i(64), 2, 0.5) * 10.0 + smoothstep(0.0, 520.0, dl) * 18.0, basin);

  // Bioluminescent valley: deep, meandering canyon-valley with high terraced walls
  let wv = vec2f(fbm(q * 16.0 + vec2f(2.1, 0.0), vec2i(16), 3, 0.5), fbm(q * 16.0 + vec2f(0.0, 4.7), vec2i(16), 3, 0.5)) * 260.0;
  let vs = segInfo(p + wv, VALLEY_A, VALLEY_B);
  let vEnd = smoothstep(0.0, 0.12, vs.y) * (1.0 - smoothstep(0.86, 1.0, vs.y));
  let floorH = 26.0 + fbm(q * 96.0, vec2i(96), 2, 0.5) * 8.0;
  let wallT = smoothstep(230.0, 620.0, vs.x);
  let sideHi = mix(360.0, 640.0, smoothstep(-200.0, 200.0, -vs.z));
  var wall = mix(floorH, sideHi + r2 * 120.0, pow(wallT, 1.25));
  wall = mix(wall, terrace(wall, 42.0, 0.55), 0.55 * smoothstep(0.1, 0.9, wallT));
  let vW = (1.0 - smoothstep(700.0, 1300.0, vs.x)) * mix(0.35, 1.0, vEnd);
  h = mix(h, wall, vW);
  // valley floor stream channel (dips just below water level)
  let stream = 1.0 - smoothstep(10.0, 34.0, abs(vs.z + fbm(q * 128.0, vec2i(128), 2, 0.5) * 40.0));
  h -= stream * vEnd * (1.0 - smoothstep(200.0, 260.0, vs.x)) * 30.0;

  // Arrival vista promontory: cliff-edged plateau the camera starts on
  let vd = wrapD2(p - VISTA_P);
  let edge = dot(vd, VISTA_DIR) - 70.0 + fbm(q * 40.0, vec2i(40), 3, 0.5) * 160.0;
  let disc = length(vd + VISTA_DIR * 1400.0);
  let plat = (1.0 - smoothstep(-150.0, 20.0, edge)) * (1.0 - smoothstep(1500.0, 2300.0, disc));
  let platH = 420.0 + fbm(q * 64.0, vec2i(64), 3, 0.5) * 26.0 - smoothstep(600.0, 2000.0, disc) * 70.0;
  var hp = mix(h, platH, plat);
  // strata ledges only on the cliff band, not on the flat top
  hp = mix(hp, terrace(hp, 26.0, 0.5), 4.0 * plat * (1.0 - plat));
  h = smax(h, hp, 30.0);

  // Monumental highlands: mesa plateau cut by a canyon network with rivers at the bottom
  let hd = length(wrapD2(p - HIGHLANDS_C)) + fbm(q * 24.0, vec2i(24), 3, 0.5) * 420.0;
  let hm = 1.0 - smoothstep(1500.0, 2100.0, hd);
  let mesa = 470.0 + fbm(q * 12.0, vec2i(12), 3, 0.5) * 90.0;
  let cn = abs(fbm(q * 18.0 + vec2f(0.37, 0.11), vec2i(18), 3, 0.42));
  let canyon = 1.0 - smoothstep(0.022, 0.085, cn);
  var hh = mix(mesa, -14.0, pow(canyon, 0.8));
  // ledges on canyon walls
  hh = mix(hh, terrace(hh, 44.0, 0.6), smoothstep(0.02, 0.2, canyon) * (1.0 - smoothstep(0.85, 1.0, canyon)));
  // outlying buttes
  let bt = fbm(q * 36.0 + vec2f(0.9, 0.2), vec2i(36), 3, 0.5);
  let butte = smoothstep(0.15, 0.21, bt) * (1.0 - hm) * (1.0 - smoothstep(2300.0, 3600.0, hd)) * (1.0 - basin);
  h = mix(h, hh, hm);
  let bH = 300.0 + fbm(q * 6.0, vec2i(6), 2, 0.5) * 160.0;
  var hb = mix(h, bH, butte);
  hb = mix(hb, terrace(hb, 36.0, 0.6), 4.0 * butte * (1.0 - butte));
  h = max(h, hb);
  return h;
}

@compute @workgroup_size(8, 8)
fn genBase(@builtin(global_invocation_id) id: vec3u) {
  let px = vec2i(i32(id.x), i32(id.y + G.x));
  if (px.x >= N || px.y >= N) { return; }
  let p = vec2f(px) * TX;
  textureStore(outR32, px, vec4f(baseHeight(p), 0.0, 0.0, 1.0));
}

// Analytic erosion (gabor-like gullies aligned with the slope), periodic.
fn erosion(p: vec2f, dir: vec2f, P: i32) -> vec3f {
  let ip = floor(p);
  let fp = p - ip;
  var va = vec3f(0.0);
  var wt = 0.0;
  for (var i = -2; i <= 1; i++) {
    for (var j = -2; j <= 1; j++) {
      let o = vec2f(f32(i), f32(j));
      let cell = wrapi2(vec2i(ip - o), vec2i(P));
      let hsh = hash22(cell) * 0.5;
      let pp = fp + o - hsh;
      let d = dot(pp, pp);
      let w = exp(-d * 2.0);
      wt += w;
      let mag = dot(pp, dir);
      va += vec3f(cos(mag * TAU), -sin(mag * TAU) * dir) * w;
    }
  }
  return va / wt;
}

@compute @workgroup_size(8, 8)
fn genFinal(@builtin(global_invocation_id) id: vec3u) {
  let px = vec2i(i32(id.x), i32(id.y + G.x));
  if (px.x >= N || px.y >= N) { return; }
  let p = vec2f(px) * TX;
  let h0 = ld(inA, px).r;
  let gx = (ld(inA, px + vec2i(2, 0)).r - ld(inA, px - vec2i(2, 0)).r) / (4.0 * TX);
  let gz = (ld(inA, px + vec2i(0, 2)).r - ld(inA, px - vec2i(0, 2)).r) / (4.0 * TX);
  let slope = length(vec2f(gx, gz));
  let dir = vec2f(gz, -gx) * 1.6;
  let cells = 40;
  let pc = p / WORLD_L * f32(cells);
  var e = vec3f(0.0);
  var a = 0.5;
  var f = 1.0;
  var per = cells;
  for (var i = 0; i < 5; i++) {
    e += erosion(pc * f, dir + e.zy * vec2f(1.0, -1.0) * 2.2, per) * a * vec3f(1.0, f, f);
    a *= 0.5;
    f *= 2.0;
    per *= 2;
  }
  let amp = clamp((h0 - 80.0) * 0.065, 0.0, 85.0);
  let amt = smoothstep(0.08, 0.55, slope);
  var h = h0 + (e.x - 0.2) * amp * amt;
  // dune ripples / soft rolling detail on gentle lowlands
  let uv = p / WORLD_L;
  h += fbm(uv * 512.0, vec2i(512), 3, 0.5) * 2.5 * (1.0 - smoothstep(0.1, 0.35, slope));
  textureStore(outR32, px, vec4f(h, 0.0, 0.0, 1.0));
  // crease (0 gully .. 1 ridge), slope-based rockiness
  let crease = sat(e.x * 0.6 + 0.5);
  textureStore(outRGBA8, px, vec4f(crease, amt, sat(slope), 1.0));
}

// Max-reduction mip chain for conservative shadow marching.
@compute @workgroup_size(8, 8)
fn genMaxMip(@builtin(global_invocation_id) id: vec3u) {
  let dims = vec2i(textureDimensions(inMax, 0));
  let px = vec2i(id.xy);
  if (px.x * 2 >= dims.x || px.y * 2 >= dims.y) { return; }
  let a = textureLoad(inMax, px * 2, 0).r;
  let b = textureLoad(inMax, px * 2 + vec2i(1, 0), 0).r;
  let c = textureLoad(inMax, px * 2 + vec2i(0, 1), 0).r;
  let d = textureLoad(inMax, px * 2 + vec2i(1, 1), 0).r;
  textureStore(outR32, px, vec4f(max(max(a, b), max(c, d)), 0.0, 0.0, 1.0));
}

// Gradient + attributes (crease, lush, AO, rock).
@compute @workgroup_size(8, 8)
fn genAttr(@builtin(global_invocation_id) id: vec3u) {
  let px = vec2i(i32(id.x), i32(id.y + G.x));
  if (px.x >= N || px.y >= N) { return; }
  let p = vec2f(px) * TX;
  let h0 = ld(inA, px).r;
  let gx = (ld(inA, px + vec2i(1, 0)).r - ld(inA, px - vec2i(1, 0)).r) / (2.0 * TX);
  let gz = (ld(inA, px + vec2i(0, 1)).r - ld(inA, px - vec2i(0, 1)).r) / (2.0 * TX);
  let shd = sunShadowHeight(p);
  textureStore(outRGBA16, px, vec4f(gx, gz, shd.x, shd.y));
  // horizon-based ambient occlusion: 8 directions, 10 steps out to ~600 m
  var occ = 0.0;
  for (var d = 0; d < 8; d++) {
    let ang = (f32(d) + 0.5) * (TAU / 8.0);
    let dv = vec2f(cos(ang), sin(ang));
    var mt = 0.0;
    var t = 6.0;
    for (var s = 0; s < 10; s++) {
      let hh = bilin(inA, p + dv * t);
      mt = max(mt, (hh - h0) / t);
      t *= 1.62;
    }
    occ += mt / sqrt(1.0 + mt * mt);
  }
  let ao = 1.0 - occ / 8.0;
  let at = ld(inB, px);
  let slope = length(vec2f(gx, gz));
  let crease = at.r;
  // lushness: lowlands, gullies, gentle slopes, and close to water
  var lush = (1.0 - smoothstep(60.0, 520.0, h0)) * 0.7 + (1.0 - crease) * 0.3;
  lush *= 1.0 - smoothstep(0.35, 0.8, slope);
  lush = sat(lush + (1.0 - smoothstep(0.0, 14.0, abs(h0 - 3.0))) * 0.3);
  let rock = sat(smoothstep(0.42, 0.85, slope) + at.g * 0.4 * smoothstep(0.3, 0.6, slope));
  textureStore(outRGBA8, px, vec4f(crease, lush, sat(ao), rock));
}

// Sun shadow height: the minimum height at a point that still sees the sun (+ occluder distance).
fn sunShadowHeight(p: vec2f) -> vec2f {
  let sd = F.sunDir.xyz;
  let hl = length(sd.xz);
  let dir = sd.xz / hl;
  let tanE = sd.y / hl;
  let topMip = i32(textureNumLevels(inMax)) - 1;
  let gmax = textureLoad(inMax, vec2i(0), topMip).r;
  var hs = -2000.0;
  var occ = 0.0;
  var t = TX;
  for (var s = 0; s < 240; s++) {
    let stp = max(TX, t * 0.035);
    let q = p + dir * t;
    var hq: f32;
    if (t < 300.0) {
      hq = bilin(inA, q);
    } else {
      // bilinear over a max-mip whose texels are at most half a step wide (no blocky edges)
      let lv = clamp(i32(floor(log2(stp / TX))) - 1, 0, 6);
      let dims = vec2i(textureDimensions(inMax, lv));
      let g = q / (TX * exp2(f32(lv))) - 0.5;
      let g0 = floor(g);
      let fr = g - g0;
      let c0 = vec2i(g0);
      let a = textureLoad(inMax, ((c0 % dims) + dims) % dims, lv).r;
      let b = textureLoad(inMax, (((c0 + vec2i(1, 0)) % dims) + dims) % dims, lv).r;
      let cc = textureLoad(inMax, (((c0 + vec2i(0, 1)) % dims) + dims) % dims, lv).r;
      let d = textureLoad(inMax, (((c0 + vec2i(1, 1)) % dims) + dims) % dims, lv).r;
      hq = mix(mix(a, b, fr.x), mix(cc, d, fr.x), fr.y);
    }
    let cand = hq - t * tanE;
    if (cand > hs) { hs = cand; occ = t; }
    if (gmax - t * tanE < hs) { break; }
    t += stp;
  }
  return vec2f(hs, occ);
}

// Downsampled height grid for CPU queries (camera clearance, placement).
@compute @workgroup_size(8, 8)
fn genCpuGrid(@builtin(global_invocation_id) id: vec3u) {
  let n = u32(N / 4);
  if (id.x >= n || id.y >= n) { return; }
  cpuGrid[id.y * n + id.x] = ld(inA, vec2i(id.xy) * 4).r;
}
// Per-block (64x64 texels) min/max heights for CPU-side terrain node culling.
@compute @workgroup_size(8, 8)
fn genMinMax(@builtin(global_invocation_id) id: vec3u) {
  let B = N / 64;
  if (i32(id.x) >= B || i32(id.y) >= B) { return; }
  var mn = 1e9;
  var mx = -1e9;
  for (var y = 0; y <= 64; y += 2) {
    for (var x = 0; x <= 64; x += 2) {
      let h = ld(inA, vec2i(i32(id.x) * 64 + x, i32(id.y) * 64 + y)).r;
      mn = min(mn, h);
      mx = max(mx, h);
    }
  }
  minmax[id.y * u32(B) + id.x] = vec2f(mn, mx);
}
