// Dense GPU grass: tile culling/LOD (compute) + procedural blades (vertex shader from indices).
//#include "common.wgsl"
//#include "noise.wgsl"
//#include "world.wgsl"
//#include "lighting.wgsl"
//#include "materials.wgsl"

const TILE: f32 = 2.0;
const GRID: u32 = 96u;           // tiles across (192 m)
const BLADES = array<u32, 3>(160u, 56u, 18u);
const LOD_D = array<f32, 3>(18.0, 42.0, 92.0);

struct GU { planes: array<vec4f, 5>, info: vec4f }; // info.x = radius scale, y = density scale
@group(3) @binding(0) var<uniform> GC: GU;
@group(3) @binding(1) var<storage, read_write> gcount: array<atomic<u32>, 4>;
@group(3) @binding(2) var<storage, read_write> tiles: array<vec4f>;
@group(3) @binding(3) var<storage, read_write> gind: array<u32>;

const CAP: u32 = 4096u;

// Grass coverage (0..1) and region tint parameters at a point.
fn grassDensity(p: vec2f) -> f32 {
  let h = heightAt(p);
  if (h < WATER_Y + 0.35) { return 0.0; }
  return meadowCover(p, attrAt(p), length(gradAt(p)), h);
}

@compute @workgroup_size(8, 8)
fn cull(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= GRID || id.y >= GRID) { return; }
  let c0 = floor(F.camPos.xz / TILE) - f32(GRID / 2u);
  let o = (c0 + vec2f(id.xy)) * TILE;
  let ctr = o + TILE * 0.5;
  let h = heightAt(ctr);
  let rel = toRel(vec3f(ctr.x, h, ctr.y));
  let dist = length(rel);
  let R = LOD_D[2] * GC.info.x;
  if (dist > R + 3.0) { return; }
  if (grassDensity(ctr) < 0.02 && grassDensity(o) < 0.02 && grassDensity(o + TILE) < 0.02) { return; }
  let rad = TILE * 0.75 + 1.0;
  for (var k = 0; k < 5; k++) {
    let pl = GC.planes[k];
    if (dot(pl.xyz, rel) + pl.w < -rad * length(pl.xyz)) { return; }
  }
  var lod = 2u;
  if (dist < LOD_D[0] * GC.info.x) { lod = 0u; } else if (dist < LOD_D[1] * GC.info.x) { lod = 1u; }
  let i = atomicAdd(&gcount[lod], 1u);
  if (i < CAP) { tiles[lod * CAP + i] = vec4f(o, f32(lod), 0.0); }
}

@compute @workgroup_size(1)
fn finalize() {
  for (var l = 0u; l < 3u; l++) {
    gind[l * 5u + 1u] = min(atomicLoad(&gcount[l]), CAP);
    atomicStore(&gcount[l], 0u);
  }
}

// ---------------------------------------------------------------------------------------------
struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) world: vec3f,
  @location(1) rel: vec3f,
  @location(2) nrm: vec3f,
  @location(3) t: f32,
  @location(4) @interpolate(flat) col: vec3f,
  @location(5) @interpolate(flat) misc: vec2f, // glow, ao
};

fn terrainDetailG(xz: vec2f, rock: f32) -> f32 {
  let n = fbmd(xz * 0.125, vec2i(2048), 4, 0.5);
  return n.x * mix(0.35, 1.9, rock);
}

@vertex
fn vs(@builtin(vertex_index) vid: u32, @location(0) tile: vec4f) -> VOut {
  let blade = vid / 7u;
  let k = vid % 7u;
  let lod = u32(tile.z);
  let tc = vec2i(floor(tile.xy / TILE + 0.5));
  let h0 = hash33(vec3i(tc, i32(blade) * 7 + 3));
  let h1 = hash33(vec3i(tc, i32(blade) * 13 + 11));
  let p = tile.xy + h0.xy * TILE;
  var o: VOut;
  let dens = grassDensity(p) * GC.info.y;
  // distance-based thinning keeps a constant visual density across LODs
  var gy = heightAt(p);
  let dcam = distance(vec3f(p.x, gy, p.y), F.camPos.xyz);
  if (dcam < 320.0) {
    gy = mix(heightCubic(p), gy, smoothstep(160.0, 320.0, dcam));
    let fade = 1.0 - smoothstep(160.0, 420.0, dcam);
    gy += terrainDetailG(p, attrAt(p).w) * fade;
  }
  let R = LOD_D[2] * GC.info.x;
  let edge = 1.0 - smoothstep(R * 0.75, R, dcam);
  if (h0.z > dens || edge <= 0.0) {
    o.pos = vec4f(0.0, 0.0, -1.0, 1.0);
    return o;
  }
  let valley = valleyMask(p);
  let high = highlandMask(p);
  var lw = array<f32, 3>(1.0, 1.7, 2.9);
  let lodW = lw[lod];
  var H = (0.45 + h1.x * 0.6) * (0.7 + dens * 0.5) * mix(1.0, 1.5, high) * mix(1.0, 0.7, valley) * edge;
  let W = 0.035 * lodW * (0.8 + h1.y * 0.5);
  let yaw = h1.z * TAU;
  var face = vec2f(cos(yaw), sin(yaw));
  // far blades turn toward the camera to read as full coverage
  let toCam = normalize(F.camPos.xz - p + vec2f(1e-3));
  face = normalize(mix(face, vec2f(-toCam.y, toCam.x), select(0.0, 0.65, lod > 0u)));
  let side = vec2f(-face.y, face.x);
  // vertex layout: 0,1 base; 2,3 mid-low; 4,5 mid-high; 6 tip
  let seg = f32(k / 2u);
  var t = seg / 3.0;
  var sideOff = select(-1.0, 1.0, (k & 1u) == 1u);
  if (k == 6u) { t = 1.0; sideOff = 0.0; }
  let width = W * (1.0 - t * 0.85);
  // bend + wind
  let tm = time();
  let wph = dot(p, vec2f(0.11, 0.07)) + h1.x * 6.0;
  let gust = 0.55 + 0.45 * sin(tm * 0.45 + p.x * 0.013 + p.y * 0.009);
  let wind = vec2f(0.86, 0.5) * (sin(tm * 1.8 + wph) * 0.5 + 0.6) * gust;
  let bend = face * (0.18 + h1.y * 0.25) + wind * 0.45;
  let tt = t * t;
  let off = bend * tt * H;
  let yv = H * (t - 0.18 * tt * length(bend));
  let xz = p + off + side * width * sideOff;
  o.world = vec3f(xz.x, gy + yv, xz.y);
  o.rel = toRel(o.world);
  o.pos = F.viewProj * vec4f(o.rel, 1.0);
  // rounded normal across the blade, blended to terrain-up with distance
  let bn = normalize(vec3f(face.x, 0.4, face.y) + vec3f(side.x, 0.0, side.y) * sideOff * 0.6);
  o.nrm = normalize(mix(bn, normalAt(p), sat(dcam / 70.0) * 0.6 + 0.25));
  o.t = t;
  // colour by region with per-blade variation
  o.col = meadowColorP(meadowPatch(p), h1.x, valley, high) * (1.0 + 0.35 * meadowWave(p));
  o.misc = vec2f(valley * step(0.82, h1.y), 0.0);
  return o;
}

@fragment
fn fs(in: VOut, @builtin(front_facing) front: bool) -> @location(0) vec4f {
  let dist = length(in.rel);
  let V = -in.rel / max(dist, 1e-3);
  let L = F.sunDir.xyz;
  var n = normalize(in.nrm);
  if (!front) { n = normalize(vec3f(-n.x, n.y, -n.z)); }
  let t = in.t;
  let alb = in.col * mix(0.32, 1.12, t);
  let sunC = sunLightAt(in.world.y);
  var sh = terrainShadow(in.world.xz, in.world.y + 0.3) * cloudShadow(in.world);
  if (sh > 0.001) { sh *= csmShadow(in.rel, n, dist, ign(in.pos.xy, F.misc.x)); }
  let ndl = sat((dot(n, L) + 0.3) / 1.3);
  var col = alb * (1.0 / PI) * ndl * sunC * sh;
  let bt = pow(sat(dot(-V, L)), 2.5);
  col += alb * sunC * sh * bt * (0.25 + 0.75 * t) * 0.8;
  col += ggx(n, V, L, 0.45, vec3f(0.03)) * sunC * sh * t;
  let ao = mix(0.25, 1.0, t);
  col += alb * ambientSH(n) * ao;
  // bioluminescent tips in the valley
  col += vec3f(0.15, 0.9, 1.0) * in.misc.x * smoothstep(0.75, 1.0, t) * (1.6 + F.extra.z);
  return vec4f(col, 1.0);
}
