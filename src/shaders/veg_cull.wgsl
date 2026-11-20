// GPU-driven vegetation scatter + culling + LOD selection into per-(species,variant,lod) lists.
//#include "common.wgsl"
//#include "noise.wgsl"
//#include "world.wgsl"
//#include "ecology.wgsl"
//#include "scatter.wgsl"

@group(2) @binding(0) var<uniform> CU: CullU;
@group(2) @binding(1) var<storage, read> lists: array<vec2u>;
@group(2) @binding(2) var<storage, read_write> counts: array<atomic<u32>>;
@group(2) @binding(3) var<storage, read_write> inst: array<vec4f>;
@group(2) @binding(4) var<storage, read> drawMap: array<u32>;
@group(2) @binding(5) var<storage, read_write> indirect: array<u32>;
// explicitly placed instances (landmarks, hero flora): (x, z, y offset over ground, scale), (yaw, seed, species, variant)
@group(2) @binding(6) var<storage, read> marks: array<vec4f>;

fn emit(list: u32, a: vec4f, b: vec4f) {
  let i = atomicAdd(&counts[list], 1u);
  let L = lists[list];
  if (i < L.y) {
    let o = (L.x + i) * 2u;
    inst[o] = a;
    inst[o + 1u] = b;
  }
}

fn inLight(m: mat4x4f, c: vec3f, r: f32) -> bool {
  let lp = m * vec4f(c, 1.0);
  let rs = r * length(vec3f(m[0][0], m[1][0], m[2][0]));
  return abs(lp.x) <= 1.0 + rs && abs(lp.y) <= 1.0 + rs;
}

@compute @workgroup_size(8, 8, 1)
fn cull(@builtin(global_invocation_id) id: vec3u) {
  let s = id.z;
  if (f32(s) >= CU.info.x) { return; }
  let S = CU.sp[s];
  let grid = u32(S.a.y);
  if (id.x >= grid || id.y >= grid) { return; }
  let c0 = vec2i(floor(F.camPos.xz / S.a.x)) - vec2i(i32(grid / 2u));
  let c = scatterCell(S, s, c0 + vec2i(id.xy));
  let maxD = S.a.z * CU.info.z;
  if (length(c.p - F.camPos.xz) > maxD) { return; }
  let gy = heightAt(c.p);
  let eco = ecology(s, c.p, gy);
  if (c.h.z >= eco.dens) { return; }
  var o = scatterInst(S, s, c, gy, eco.scale);
  let rel = toRel(o.a.xyz);
  let dist = length(rel);
  if (dist > maxD) { return; }
  // shrink into the ground near the far limit (no popping)
  o.a.w *= smoothstep(1.0, 0.86, dist / maxD);
  place(s, rel, dist, o.a, o.b, o.variant, false);
}

fn inFrustum(ctr: vec3f, rad: f32) -> bool {
  var vis = true;
  for (var k = 0; k < 5; k++) {
    let pl = CU.planes[k];
    if (dot(pl.xyz, ctr) + pl.w < -rad * length(pl.xyz)) { vis = false; }
  }
  return vis;
}

// Visibility, LOD and shadow-list emission for one instance. The near shadow list feeds cascades
// 0-1 and the far list cascades 2-3; instances in the band around the cascade-1 split go to both
// so their shadows never fall into a cascade that skipped them. Landmarks (tall casters with
// long low-sun shadows) are tested against each cascade footprint instead of by distance.
fn place(s: u32, rel: vec3f, dist: f32, a: vec4f, b: vec4f, variant: u32, landmark: bool) {
  let S = CU.sp[s];
  let rad = S.c.z * a.w;
  let ctr = rel + vec3f(0.0, rad * S.d.w, 0.0);
  if (inFrustum(ctr, rad)) {
    var lod = 2u;
    let ld = dist / CU.info.z;
    if (ld < S.b.x) { lod = 0u; } else if (ld < S.b.y) { lod = 1u; }
    emit(u32(S.c.w) + variant * 3u + lod, a, b);
  }
  let nearD = S.b.z * CU.info.z;
  let farD = S.b.w * CU.info.z;
  if (dist < nearD && inLight(F.shadowMats[1], ctr, rad)) { emit(u32(S.d.x) + variant * 2u, a, b); }
  if (dist < farD && (landmark || dist >= F.cascadeSplits.y * 0.7) && inLight(F.shadowMats[3], ctr, rad)) {
    emit(u32(S.d.x) + variant * 2u + 1u, a, b);
  }
}

@compute @workgroup_size(64)
fn cullMarks(@builtin(global_invocation_id) id: vec3u) {
  let n = arrayLength(&marks) / 2u;
  if (id.x >= n) { return; }
  let ma = marks[id.x * 2u];
  let mb = marks[id.x * 2u + 1u];
  let s = u32(mb.z);
  let S = CU.sp[s];
  let p = F.camPos.xz + wrapD2(ma.xy - F.camPos.xz);
  let y = heightAt(p) + ma.z;
  let rel = toRel(vec3f(p.x, y, p.y));
  let dist = length(rel);
  let maxD = S.a.z * max(CU.info.z, 0.8);
  if (dist > maxD) { return; }
  let scale = ma.w * smoothstep(1.0, 0.9, dist / maxD);
  place(s, rel, dist, vec4f(p.x, y, p.y, scale), vec4f(mb.x, mb.y, mb.z, 1.0), u32(mb.w), true);
}

@compute @workgroup_size(64)
fn finalize(@builtin(global_invocation_id) id: vec3u) {
  let d = id.x;
  if (f32(d) >= CU.info.y) { return; }
  let list = drawMap[d];
  indirect[d * 5u + 1u] = min(atomicLoad(&counts[list]), lists[list].y);
}

@compute @workgroup_size(64)
fn reset(@builtin(global_invocation_id) id: vec3u) {
  if (id.x < arrayLength(&counts)) { atomicStore(&counts[id.x], 0u); }
}

// ---------------------------------------------------------------------------------------------
// Collision gather (collide.ts): the solid things near a probe point around the camera, read back by
// the CPU, which keeps the camera out of them. Flora come from the same scatter as the culling
// above, so they are exactly the drawn instances; then the landmarks, the animals and a window of
// exact terrain heights. Every entry is relative to the probe centre. (Its entry points live in
// this module so the includes are not bundled twice.)
const COL_CAP: u32 = 256u;   // instances per gather (collide.ts COL_CAP)
const COL_WIN: i32 = 32;     // terrain window, texels per side (collide.ts COL_WIN)
const COL_GRID: u32 = 24u;   // flora cells per side searched for each species

struct ColOut {
  count: atomic<u32>,
  pad: array<u32, 3>,
  head: array<f32, 8>,                         // count, seq, probe x, z, radius, window texel x, z, -
  heights: array<f32, 1024>,                   // COL_WIN^2 exact heights
  inst: array<vec4f, 512>,                     // COL_CAP x 2: (dx, y, dz, scale), (yaw, y stretch, kind, variant)
};
struct Probe { c: vec4f, k: vec4f }; // c = probe x, z (wrapped like camPos), radius, seq; k = flora species list

// (CU and marks are the culling pass's bindings 0 and 6; the gather's layout adds 7-9)
@group(2) @binding(7) var<storage, read> pose: array<vec4f>;
@group(2) @binding(8) var<storage, read_write> co: ColOut;
@group(2) @binding(9) var<uniform> PR: Probe;

fn colPut(a: vec4f, b: vec4f) {
  let i = atomicAdd(&co.count, 1u);
  if (i < COL_CAP) {
    co.inst[i * 2u] = a;
    co.inst[i * 2u + 1u] = b;
  }
}

@compute @workgroup_size(1)
fn colBegin() {
  atomicStore(&co.count, 0u);
}

// Flora: COL_GRID^2 cells of each species listed in PR.k around the probe.
@compute @workgroup_size(8, 8, 1)
fn colFlora(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= COL_GRID || id.y >= COL_GRID || id.z > 3u) { return; }
  let s = u32(PR.k[id.z]);
  let S = CU.sp[s];
  let pc = PR.c.xy;
  let c0 = vec2i(floor(pc / S.a.x)) - vec2i(i32(COL_GRID / 2u));
  let c = scatterCell(S, s, c0 + vec2i(id.xy));
  // the bounding radius at the largest scale the species reaches (ecology scales up to ~2.3x)
  let reach = PR.c.z + S.c.z * S.c.y * 2.3;
  let d = c.p - pc;
  if (dot(d, d) > reach * reach) { return; }
  let gy = heightAt(c.p);
  let eco = ecology(s, c.p, gy);
  if (c.h.z >= eco.dens) { return; }
  let o = scatterInst(S, s, c, gy, eco.scale);
  colPut(vec4f(d.x, o.a.y, d.y, o.a.w), vec4f(o.b.x, o.b.w, f32(s), f32(o.variant)));
}

// Landmarks (arches, floating islands, pillars), grounded as cullMarks does.
@compute @workgroup_size(64)
fn colMarks(@builtin(global_invocation_id) id: vec3u) {
  let n = arrayLength(&marks) / 2u;
  if (id.x >= n) { return; }
  let ma = marks[id.x * 2u];
  let mb = marks[id.x * 2u + 1u];
  let s = u32(mb.z);
  let d = wrapD2(ma.xy - PR.c.xy);
  let reach = PR.c.z + CU.sp[s].c.z * ma.w;
  if (dot(d, d) > reach * reach) { return; }
  let p = PR.c.xy + d;
  colPut(vec4f(d.x, heightAt(p) + ma.z, d.y, ma.w), vec4f(mb.x, 1.0, f32(s), mb.w));
}

// Animals that stand or walk (herd, titans, travellers, colossi) and the sky jellies:
// kind 100 + group, scale from the pose (fauna.wgsl).
@compute @workgroup_size(64)
fn colAnimals(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  var kind = -1.0;
  if (i < FAUNA_NC) { kind = 100.0; }
  else if (i >= FAUNA_OT && i < FAUNA_OT + FAUNA_NT) { kind = 102.0; }
  else if (i >= FAUNA_OH && i < FAUNA_OH + FAUNA_NH) { kind = 103.0; }
  else if (i >= FAUNA_OK && i < FAUNA_OK + FAUNA_NK) { kind = 104.0; }
  else if (i >= FAUNA_OJ && i < FAUNA_OJ + FAUNA_NJ) { kind = 105.0; }
  if (kind < 0.0) { return; }
  let p0 = pose[i * 5u];
  let sc = pose[i * 5u + 4u].x;
  let d = wrapD2(p0.xz - PR.c.xy);
  let reach = PR.c.z + sc * 4.0;
  if (dot(d, d) > reach * reach) { return; }
  colPut(vec4f(d.x, p0.y, d.y, sc), vec4f(p0.w, 1.0, kind, 0.0));
}

// Exact heights (texel convention of hLoad) of a COL_WIN^2 window centred on the probe, and the
// header that tells the CPU which probe this answers.
@compute @workgroup_size(8, 8)
fn colTerrain(@builtin(global_invocation_id) id: vec3u) {
  let o = vec2i(floor(PR.c.xy / TEXEL)) - vec2i(COL_WIN / 2);
  if (id.x == 0u && id.y == 0u) {
    co.head[0] = f32(min(atomicLoad(&co.count), COL_CAP));
    co.head[1] = PR.c.w;
    co.head[2] = PR.c.x;
    co.head[3] = PR.c.y;
    co.head[4] = PR.c.z;
    co.head[5] = f32(o.x);
    co.head[6] = f32(o.y);
  }
  if (i32(id.x) >= COL_WIN || i32(id.y) >= COL_WIN) { return; }
  co.heights[id.y * u32(COL_WIN) + id.x] = hLoad(o + vec2i(id.xy));
}
