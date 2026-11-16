// World data bind group (group 1): terrain fields, atmosphere LUTs, ambient SH, 3D noise.
//#include "atmosphere.wgsl"

@group(1) @binding(0) var hTex: texture_2d<f32>;     // r32float heights (vertex convention: texel i = position i*TEXEL)
@group(1) @binding(1) var gTex: texture_2d<f32>;     // rgba16float: dh/dx, dh/dz, sun-shadow height, occluder distance
@group(1) @binding(2) var aTex: texture_2d<f32>;     // rgba8 attributes: crease, lush, ao, rock
@group(1) @binding(3) var transLut: texture_2d<f32>;
@group(1) @binding(4) var skyLut: texture_2d<f32>;
@group(1) @binding(5) var apLut: texture_3d<f32>;
@group(1) @binding(6) var<storage, read> ambSH: array<vec4f, 9>;
@group(1) @binding(7) var noiseTex: texture_3d<f32>; // 128^3 tiling noise (r: perlin-worley, gba: worley fbm)
@group(1) @binding(8) var weatherTex: texture_2d<f32>; // cloud weather map over WORLD_L: r coverage, g type
@group(1) @binding(9) var fTex: texture_2d<f32>;       // coarse mipmapped (height, shadow height, occluder dist)

const TEXEL: f32 = WORLD_L / f32(HMAP_N);

fn hLoad(i: vec2i) -> f32 {
  return textureLoad(hTex, i & vec2i(HMAP_N - 1), 0).r;
}

fn heightAt(p: vec2f) -> f32 {
#if HFILTER
  return textureSampleLevel(hTex, sLinRep, (p / TEXEL + 0.5) / f32(HMAP_N), 0.0).r;
#else
  return heightAtManual(p);
#endif
}

fn heightAtManual(p: vec2f) -> f32 {
  let t = p / TEXEL;
  let fl = floor(t);
  let i = vec2i(fl);
  let f = t - fl;
  let a = hLoad(i);
  let b = hLoad(i + vec2i(1, 0));
  let c = hLoad(i + vec2i(0, 1));
  let d = hLoad(i + vec2i(1, 1));
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

fn crWeights(t: f32) -> vec4f {
  let t2 = t * t;
  let t3 = t2 * t;
  return vec4f(-0.5 * t3 + t2 - 0.5 * t, 1.5 * t3 - 2.5 * t2 + 1.0, -1.5 * t3 + 2.0 * t2 + 0.5 * t, 0.5 * t3 - 0.5 * t2);
}

// Catmull-Rom bicubic height (C1 smooth, interpolating) for near-field terrain vertices.
fn heightCubic(p: vec2f) -> f32 {
  let t = p / TEXEL;
  let fl = floor(t);
  let i = vec2i(fl);
  let f = t - fl;
  let wx = crWeights(f.x);
  let wy = crWeights(f.y);
  var s = 0.0;
  for (var y = 0; y < 4; y++) {
    let r = vec4f(hLoad(i + vec2i(-1, y - 1)), hLoad(i + vec2i(0, y - 1)), hLoad(i + vec2i(1, y - 1)), hLoad(i + vec2i(2, y - 1)));
    s += dot(r, wx) * wy[y];
  }
  return s;
}

fn wuv(p: vec2f) -> vec2f { return (p / TEXEL + 0.5) / f32(HMAP_N); }
fn gradAt(p: vec2f) -> vec2f { return textureSampleLevel(gTex, sLinRep, wuv(p), 0.0).xy; }
fn attrAt(p: vec2f) -> vec4f { return textureSampleLevel(aTex, sLinRep, wuv(p), 0.0); }
// Coarse (16 m, mipmapped) height / sun-shadow height / occluder distance for volumetrics.
fn fieldAt(p: vec2f, lod: f32) -> vec3f { return textureSampleLevel(fTex, sLinRep, wuv(p), lod).xyz; }
fn fieldShadow(f: vec3f, y: f32) -> f32 {
  let w = 2.0 + f.z * 0.008;
  return smoothstep(-w, w, y - f.y);
}
fn normalAt(p: vec2f) -> vec3f {
  let g = gradAt(p);
  return normalize(vec3f(-g.x, 1.0, -g.y));
}

// Soft terrain shadow from the precomputed sun-shadow height field.
fn terrainShadow(p: vec2f, y: f32) -> f32 {
  let s = textureSampleLevel(gTex, sLinRep, wuv(p), 0.0).zw;
  let w = 1.0 + s.y * 0.006;
  return smoothstep(-w, w, y - s.x);
}

fn sunLightAt(altM: f32) -> vec3f {
  return F.sunColor.rgb * sunTransmittance(transLut, sLinClamp, altM, F.sunDir.y);
}

// Pre-convolved SH sky irradiance (returns irradiance / pi).
fn ambientSH(n: vec3f) -> vec3f {
  let d = vec3f(n.x, n.z, n.y);
  var r = ambSH[0].rgb * 0.282095;
  r += ambSH[1].rgb * (0.488603 * d.y) + ambSH[2].rgb * (0.488603 * d.z) + ambSH[3].rgb * (0.488603 * d.x);
  r += ambSH[4].rgb * (1.092548 * d.x * d.y) + ambSH[5].rgb * (1.092548 * d.y * d.z);
  r += ambSH[6].rgb * (0.315392 * (3.0 * d.z * d.z - 1.0)) + ambSH[7].rgb * (1.092548 * d.x * d.z);
  r += ambSH[8].rgb * (0.546274 * (d.x * d.x - d.y * d.y));
  return max(r, vec3f(0.0));
}

// Aerial perspective from the froxel LUT. uv = screen uv (0..1 over viewport).
fn aerial(uv: vec2f, distM: f32) -> vec4f {
  let s = sqrt(clamp(distM * 0.001 / AP_MAX_KM, 0.0, 1.0)) * AP_RES - 1.0;
  let w = (max(s, 0.0) + 0.5) / AP_RES;
  var ap = textureSampleLevel(apLut, sLinClamp, vec3f(uv, w), 0.0);
  if (s < 0.0) { ap = mix(vec4f(0.0, 0.0, 0.0, 1.0), ap, s + 1.0); }
  return vec4f(ap.rgb * F.sunColor.rgb, ap.a);
}

fn skyColor(dir: vec3f) -> vec3f {
  return skyRadiance(skyLut, sLinClamp, F.camPos.y, dir, F.sunDir.xyz) * F.sunColor.rgb;
}

// ---------------------------------------------------------------------------------------------
// Region masks for the three designed destinations (analytic, periodic).
fn regionDist(p: vec2f, c: vec2f) -> f32 { return length(wrapD2(p - c)); }
fn valleyMask(p: vec2f) -> f32 {
  // distance to the valley axis segment
  let a = VALLEY_A;
  let ab = wrapD2(VALLEY_B - a);
  let ap = wrapD2(p - a);
  let t = clamp(dot(ap, ab) / dot(ab, ab), 0.0, 1.0);
  let d = length(ap - ab * t);
  return 1.0 - smoothstep(500.0, 1300.0, d);
}
fn highlandMask(p: vec2f) -> f32 { return 1.0 - smoothstep(2000.0, 3600.0, regionDist(p, HIGHLANDS_C)); }

// ---------------------------------------------------------------------------------------------
// Cloud coverage (shared by volumetric clouds and terrain cloud shadows). Periodic over WORLD_L.
fn cloudWind() -> vec2f { return vec2f(6.0, 2.5) * time(); }
// (coverage, type) of the cloud layer above world xz; the vista's sun path is kept clear.
fn cloudWeather(xz: vec2f) -> vec2f {
  let w = textureSampleLevel(weatherTex, sLinRep, (xz + cloudWind()) / WORLD_L, 0.0).rg;
  let clear = smoothstep(1500.0, 4000.0, length(wrapD2(xz - VISTA_SUNCLEAR)));
  return vec2f(sat(w.x + CLOUD_COVER) * clear, w.y);
}
fn cloudCoverage(xz: vec2f) -> f32 { return cloudWeather(xz).x; }
// Approximate cloud shadow for a world point (projects along the sun ray to the cloud layer).
fn cloudShadow(p: vec3f) -> f32 {
  let mid = 0.5 * (CLOUD_BOT + CLOUD_TOP);
  let t = max(mid - p.y, 0.0) / max(F.sunDir.y, 0.05);
  let q = p.xz + F.sunDir.xz * t;
  let c = cloudCoverage(q);
  return 1.0 - 0.8 * smoothstep(0.25, 0.7, c);
}

// Roots of alt(t) = H along a camera-relative ray in the curved-world convention
// alt(t) = camY + t*d.y + t^2*|d.xz|^2/(2R). Returns (t0, t1), t0 <= t1, or (-1, -1).
fn shellRoots(d: vec3f, H: f32) -> vec2f {
  let a = dot(d.xz, d.xz) * (0.5 / PLANET_R);
  let b = d.y;
  let c = F.camPos.y - H;
  if (a < 1e-12) {
    let t = -c / select(b, 1e-6, abs(b) < 1e-6);
    return vec2f(t, t);
  }
  let disc = b * b - 4.0 * a * c;
  if (disc < 0.0) { return vec2f(-1.0); }
  let q = -0.5 * (b + select(-1.0, 1.0, b >= 0.0) * sqrt(disc));
  let r1 = q / a;
  let r2 = c / select(q, -1e-9, abs(q) < 1e-9);
  return vec2f(min(r1, r2), max(r1, r2));
}

// Distance interval of a view ray inside the cloud slab (x < 0: no intersection).
fn cloudSpan(d: vec3f) -> vec2f {
  let y = F.camPos.y;
  let rt = shellRoots(d, CLOUD_TOP);
  if (y < CLOUD_BOT) {
    let rb = shellRoots(d, CLOUD_BOT);
    return vec2f(rb.y, rt.y);
  }
  if (y <= CLOUD_TOP) { return vec2f(0.0, rt.y); }
  if (rt.y <= 0.0) { return vec2f(-1.0); }
  return vec2f(max(rt.x, 0.0), rt.y);
}

// Altitude (heightmap convention) of a camera-relative point on the curved world.
fn altOf(rel: vec3f) -> f32 { return F.camPos.y + rel.y + curveDrop(rel.xz); }
