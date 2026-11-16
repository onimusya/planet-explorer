// CDLOD terrain: geomorphed grid patches displaced by the heightfield + procedural detail.
//#include "common.wgsl"
//#include "noise.wgsl"
//#include "world.wgsl"
//#include "lighting.wgsl"
//#include "materials.wgsl"

struct TerrainU { morph: array<vec4f, 16> };
@group(3) @binding(0) var<uniform> TU: TerrainU;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) world: vec3f,
  @location(1) rel: vec3f,
};

const GRID: u32 = 64u;

// Near-field detail displacement (value, d/dx, d/dz) in metres. Periodic over WORLD_L.
fn terrainDetail(xz: vec2f, rock: f32) -> vec3f {
  let n = fbmd(xz * 0.125, vec2i(2048), 4, 0.5);
  let amp = mix(0.35, 1.9, rock);
  return vec3f(n.x * amp, n.yz * (amp * 0.125));
}

@vertex
fn vs(@builtin(vertex_index) vid: u32, @location(0) node: vec4f) -> VOut {
  let gx = vid % (GRID + 1u);
  let gy = vid / (GRID + 1u);
  let cell = node.z / f32(GRID);
  var xz = node.xy + vec2f(f32(gx), f32(gy)) * cell;
  let lod = i32(node.w);
  let h0 = heightAt(xz);
  let d0 = distance(vec3f(xz.x, h0, xz.y), F.camPos.xyz);
  let mr = TU.morph[lod];
  let k = sat((d0 - mr.x) / (mr.y - mr.x));
  let odd = vec2f(f32(gx & 1u), f32(gy & 1u));
  xz -= odd * cell * k;
  var h = heightAt(xz);
  let dist = distance(vec3f(xz.x, h, xz.y), F.camPos.xyz);
  // C1-smooth bicubic only near the camera (blended by distance: identical for shared vertices)
  if (dist < 320.0) {
    h = mix(heightCubic(xz), h, smoothstep(160.0, 320.0, dist));
  }
  let fade = 1.0 - smoothstep(160.0, 420.0, dist);
  if (fade > 0.0) {
    let rock = attrAt(xz).w * (1.0 - ledgeGrass(xz));
    h += terrainDetail(xz, rock).x * fade;
  }
  var o: VOut;
  o.world = vec3f(xz.x, h, xz.y);
  o.rel = toRel(o.world);
  o.pos = F.viewProj * vec4f(o.rel, 1.0);
  return o;
}

struct Surface {
  albedo: vec3f,
  rough: f32,
  n: vec3f,
  ao: f32,
  emit: vec3f,
  sheen: f32,
};

fn terrainSurface(wp: vec3f, dist: f32, fw: f32) -> Surface {
  let p = wp.xz;
  let y = wp.y;
  let attr = attrAt(p);
  let g = gradAt(p);
  var dg = vec2f(0.0);
  let lg = ledgeGrass(p);
  let dfade = 1.0 - smoothstep(160.0, 420.0, dist);
  if (dfade > 0.0) {
    dg = terrainDetail(p, attr.w * (1.0 - lg)).yz * dfade;
  }
  // micro bump (fades with pixel footprint)
  let mfade = 1.0 - smoothstep(0.05, 0.6, fw);
  if (mfade > 0.0) {
    let m = fbmd(p * 1.0, vec2i(16384), 3, 0.55);
    dg += m.yz * 0.22 * mfade;
  }
  let nGeo = normalize(vec3f(-g.x, 1.0, -g.y));
  var n = normalize(vec3f(-(g.x + dg.x), 1.0, -(g.y + dg.y)));
  let slope = 1.0 - nGeo.y;
  let nv = fbm(p / 16.0, vec2i(1024), 3, 0.5) + 0.5;
  let nb = fbm(p / 256.0, vec2i(64), 3, 0.5);
  let valley = valleyMask(p);
  let highland = highlandMask(p);

  var rockW = smoothstep(0.26, 0.46, slope + (nv - 0.5) * 0.22 + (attr.w - 0.4) * 0.25);
  rockW = max(rockW, smoothstep(0.62, 0.9, attr.w));
  rockW = max(rockW, highland * smoothstep(0.12, 0.3, slope + (nv - 0.5) * 0.3)) * (1.0 - lg);
  let snowLine = 1180.0 + nb * 320.0;
  let snowW = smoothstep(snowLine, snowLine + 140.0, y) * (1.0 - smoothstep(0.32, 0.6, slope));
  let shoreW = (1.0 - smoothstep(0.8, 4.0 + nv * 3.0, y)) * (1.0 - rockW);
  let grassW = (1.0 - rockW) * (1.0 - shoreW) * meadowCover(p, attr, length(g), y);

  // rock: strata with weathering. On steep faces the variation noise runs in wall-plane coordinates
  // (and the xz-only crease term is damped) so nothing streaks vertically down the cliffs.
  var rock = strata(y, p);
  let crease = attr.x;
  let steep = smoothstep(0.3, 0.65, slope);
  let hz = normalize(vec2f(nGeo.x, nGeo.z) + vec2f(1e-5, 0.0));
  let wallUV = vec2f(dot(p, vec2f(-hz.y, hz.x)), y * 1.4);
  let nvW = fbm(mix(p, wallUV, steep) / 8.0, vec2i(2048), 3, 0.55) + 0.5;
  rock *= mix(mix(0.62, 1.12, crease), 0.92, steep * 0.75) * mix(0.8, 1.15, nvW);
  rock = mix(rock, rock * vec3f(0.80, 0.86, 1.05), highland * 0.0 + (1.0 - crease) * 0.25);
  // soil / dirt
  let dirt = mix(vec3f(0.30, 0.17, 0.13), vec3f(0.44, 0.27, 0.18), nv);
  // meadow ground: near camera it is the soil between blades, far it reads as the grass canopy
  let canopy = meadowColor(p, nv, valley, highland) * (0.9 + 0.3 * meadowWave(p));
  let under = canopy * 0.42 + dirt * 0.25;
  let grassC = mix(under, canopy, smoothstep(25.0, 120.0, dist));
  let sand = mix(vec3f(0.62, 0.50, 0.40), vec3f(0.76, 0.62, 0.50), nv);
  let snow = vec3f(0.86, 0.88, 0.95);

  var s: Surface;
  var alb = mix(dirt, rock, rockW);
  alb = mix(alb, grassC, grassW);
  alb = mix(alb, sand * mix(0.55, 1.0, smoothstep(0.0, 1.2, y)), shoreW);
  alb = mix(alb, snow, snowW);
  s.albedo = alb;
  s.rough = mix(mix(0.82, 0.9, grassW), 0.45, snowW);
  s.rough = mix(s.rough, 0.3, shoreW * (1.0 - smoothstep(0.0, 1.0, y)));
  n = rockBump(wp, n, rockW * (1.0 - smoothstep(60.0, 260.0, dist)) * 0.9);
  s.n = normalize(mix(n, nGeo, snowW * 0.6));
  s.ao = attr.z * mix(0.7, 1.0, crease);
  s.sheen = grassW;
  // bioluminescent moss specks on the valley floor
  var emit = vec3f(0.0);
  if (valley > 0.01 && dist < 900.0) {
    let sp = hash21(vec2i(floor(p * 1.5)));
    let spk = smoothstep(0.985, 1.0, sp) * (1.0 - rockW) * valley;
    let tw = 0.6 + 0.4 * sin(time() * 1.7 + sp * 60.0);
    emit = mix(vec3f(0.1, 0.9, 1.0), vec3f(0.8, 0.3, 1.0), fract(sp * 13.0)) * spk * tw * 2.5 * (1.0 - smoothstep(300.0, 900.0, dist));
  }
  s.emit = emit;
  return s;
}

@fragment
fn fs(in: VOut) -> @location(0) vec4f {
  let fw = length(fwidth(in.world));
  let p = in.world.xz;
  let y = in.world.y;
  let dist = length(in.rel);
  let V = -in.rel / max(dist, 1e-3);
  let L = F.sunDir.xyz;
  let dither = ign(in.pos.xy, F.misc.x);
  let s = terrainSurface(in.world, dist, fw);
  let n = s.n;
  let sunC = sunLightAt(y);
  var sh = terrainShadow(p, y + 1.5) * cloudShadow(in.world);
  if (sh > 0.001) { sh *= csmShadow(in.rel, n, dist, dither); }
  let ndl = dot(n, L);
  let diff = s.albedo * (1.0 / PI) * sat(ndl) * sunC * sh;
  let spec = ggx(n, V, L, s.rough, vec3f(0.035)) * sunC * sh;
  // grass canopy forward scatter (glowing meadows against the sun)
  let fwd = pow(sat(dot(-V, L)), 6.0) * s.sheen * 0.35;
  let trans = s.albedo * sunC * sh * fwd;
  var amb = s.albedo * ambientSH(n) * s.ao;
  // warm bounce from lit surroundings on steep faces
  amb += s.albedo * sunC * 0.02 * (1.0 - n.y) * s.ao;
  let col = diff + spec + trans + amb + s.emit;
  return vec4f(col, 1.0);
}
