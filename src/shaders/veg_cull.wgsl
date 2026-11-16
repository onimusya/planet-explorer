// GPU-driven vegetation scatter + culling + LOD selection into per-(species,variant,lod) lists.
//#include "common.wgsl"
//#include "noise.wgsl"
//#include "world.wgsl"
//#include "ecology.wgsl"

struct Species {
  a: vec4f, // cell, grid, maxDist, variants
  b: vec4f, // lod0, lod1, shadowNear, shadowFar
  c: vec4f, // scaleMin, scaleMax, radius, listBase
  d: vec4f, // shadowBase, sink, yStretch, 0
};
struct CullU {
  planes: array<vec4f, 5>,
  sp: array<Species, 12>,
  info: vec4f, // x = species count, y = draw count, z = distance scale
};

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
  let cell = S.a.x;
  let per = i32(WORLD_L / cell);
  let c0 = vec2i(floor(F.camPos.xz / cell)) - vec2i(i32(grid / 2u));
  let ci = c0 + vec2i(id.xy);
  let cw = wrapi2(ci, vec2i(per));
  let h = hash33(vec3i(cw, i32(s) * 7919 + 13));
  let p = (vec2f(ci) + 0.1 + h.xy * 0.8) * cell;
  let maxD = S.a.z * CU.info.z;
  let d2 = length(p - F.camPos.xz);
  if (d2 > maxD) { return; }
  let gy = heightAt(p);
  let eco = ecology(s, p, gy);
  if (h.z >= eco.dens) { return; }
  let y = gy - S.d.y;
  let rel = toRel(vec3f(p.x, y, p.y));
  let dist = length(rel);
  if (dist > maxD) { return; }
  let h2 = hash33(vec3i(cw, i32(s) * 31 + 7));
  var scale = mix(S.c.x, S.c.y, h2.x * h2.x) * eco.scale;
  // shrink into the ground near the far limit (no popping)
  scale *= smoothstep(1.0, 0.86, dist / maxD);
  let nv = u32(S.a.w);
  let variant = min(u32(h2.y * f32(nv)), nv - 1u);
  place(s, rel, dist, vec4f(p.x, y, p.y, scale), vec4f(h2.z * TAU, h.x, f32(s), mix(1.0, S.d.z, h.y)), variant, false);
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
