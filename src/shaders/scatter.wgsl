// Flora scatter, shared by the culling pass and the collision gather (both in veg_cull.wgsl):
// the gather must find exactly the instances that are drawn. Needs common, noise,
// world and ecology.

struct Species {
  a: vec4f, // cell, grid, maxDist, variants
  b: vec4f, // lod0, lod1, shadowNear, shadowFar
  c: vec4f, // scaleMin, scaleMax, radius, listBase
  d: vec4f, // shadowBase, sink, yStretch, bounding-sphere centre height (fraction of the radius)
};
struct CullU {
  planes: array<vec4f, 5>,
  sp: array<Species, 12>,
  info: vec4f, // x = species count, y = draw count, z = distance scale
};

// Grid cell ci (unwrapped index, around the camera) of species s: its wrapped index, its hash
// (z decides against the ecology density whether it holds an instance, x is the seed, y the
// y stretch) and the candidate position.
struct Cell { cw: vec2i, h: vec3f, p: vec2f };
fn scatterCell(S: Species, s: u32, ci: vec2i) -> Cell {
  var c: Cell;
  let cell = S.a.x;
  c.cw = wrapi2(ci, vec2i(i32(WORLD_L / cell)));
  c.h = hash33(vec3i(c.cw, i32(s) * 7919 + 13));
  c.p = (vec2f(ci) + 0.1 + c.h.xy * 0.8) * cell;
  return c;
}

// The instance of a cell that passed the density test, on ground height gy:
// a = (x, y, z, scale), before the culling pass shrinks it near its far limit;
// b = (yaw, seed, species, y stretch).
struct Inst { a: vec4f, b: vec4f, variant: u32 };
fn scatterInst(S: Species, s: u32, c: Cell, gy: f32, ecoScale: f32) -> Inst {
  var o: Inst;
  let h2 = hash33(vec3i(c.cw, i32(s) * 31 + 7));
  o.a = vec4f(c.p.x, gy - S.d.y, c.p.y, mix(S.c.x, S.c.y, h2.x * h2.x) * ecoScale);
  o.b = vec4f(h2.z * TAU, c.h.x, f32(s), mix(1.0, S.d.z, c.h.y));
  let nv = u32(S.a.w);
  o.variant = min(u32(h2.y * f32(nv)), nv - 1u);
  return o;
}
