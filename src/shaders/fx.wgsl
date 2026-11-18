// Airborne particles: luminous spores drifting up from the Bioluminescent Valley floor and sunlit
// pollen motes over the rest of the planet. Stateless: a particle's position follows from its
// index, the time and the wind, inside a box that travels with the camera (positions wrap at the
// box edges, where particles fade out). Drawn after TAA into an additive HDR layer at output
// resolution, so small moving lights are never smeared by the TAA history; depth-tested by hand
// against the scene depth. Bloom and the final pass add the layer.
//#include "common.wgsl"
//#include "noise.wgsl"
//#include "world.wgsl"

@group(2) @binding(0) var sceneDepth: texture_depth_2d;

const SPORE_BOX: f32 = 200.0;
const MOTE_BOX: f32 = 110.0;
const WIND: vec2f = vec2f(0.86, 0.5); // the foliage sway direction (veg.wgsl)

struct PO {
  @builtin(position) pos: vec4f,
  @location(0) q: vec2f,                      // sprite coordinate, -1..1
  @location(1) @interpolate(flat) col: vec3f, // radiance at the sprite centre
  @location(2) @interpolate(flat) vz: f32,    // view depth (m)
};

fn hg(c: f32, g: f32) -> f32 { return (1.0 - g * g) / (4.0 * PI * pow(1.0 + g * g - 2.0 * g * c, 1.5)); }

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> PO {
  var o: PO;
  o.pos = vec4f(0.0, 0.0, -2.0, 1.0); // outside the clip volume unless the particle is shown
  o.q = vec2f(0.0);
  o.col = vec3f(0.0);
  o.vz = 1.0;
  let spore = ii < FX_SPORES;
  let h = hash33(vec3i(i32(ii), select(5, 9, spore), 433));
  let h2 = hash33(vec3i(i32(ii), 17, 977));
  let t = time();
  let B = select(MOTE_BOX, SPORE_BOX, spore);
  let drift = WIND * (t * select(0.5, 0.12, spore) * (0.7 + 0.6 * h2.x))
    + vec2f(sin(t * 0.21 + h.z * 40.0), cos(t * 0.17 + h.x * 40.0)) * select(0.5, 1.2, spore);
  var d = h.xy * B + drift - F.camPos.xz;
  d -= B * round(d / B);
  let xz = F.camPos.xz + d;
  let edge = 1.0 - smoothstep(0.36 * B, 0.5 * B, max(abs(d.x), abs(d.y)));
  let ground = max(heightAt(xz), WATER_Y);
  var y: f32;
  var col: vec3f;
  var r: f32;
  if (spore) {
    // rising from the valley floor, thinning with height, twinkling
    let H = 26.0;
    let hgt = H * fract(h.z * 7.31 + t * (0.008 + 0.01 * h2.y));
    y = ground + 0.2 + hgt;
    let dens = smoothstep(0.45, 0.9, valleyMask(xz)) * exp(-hgt / 8.0) * smoothstep(0.0, 1.2, hgt) * (1.0 - smoothstep(0.8 * H, H, hgt));
    if (dens * edge < 0.004) { return o; }
    let tw = 0.5 + 0.5 * sin(t * (1.1 + 2.3 * h2.z) + h.x * 50.0);
    let c = select(vec3f(0.22, 0.95, 1.0), vec3f(0.72, 0.42, 1.0), h2.y > 0.62);
    col = c * (6.0 * dens * (0.35 + 0.65 * tw) * edge);
    r = 0.025 + 0.035 * h2.x;
  } else {
    // pollen and dust lit by the sun: bright only when backlit (strong forward scattering)
    y = ground + 0.4 + 10.0 * h.z * h.z + sin(t * 0.3 + h.x * 30.0) * 0.4;
    let dens = (1.0 - smoothstep(0.3, 0.6, valleyMask(xz))) * (1.0 - smoothstep(900.0, 1100.0, ground));
    if (dens * edge < 0.004) { return o; }
    let wp0 = vec3f(xz.x, y, xz.y);
    let V = normalize(toRel(wp0));
    let sh = terrainShadow(xz, y) * cloudShadow(wp0);
    col = sunLightAt(y) * sh * (hg(dot(V, F.sunDir.xyz), 0.78) * 0.06 * dens * edge);
    r = 0.012 + 0.014 * h2.x;
  }
  let rel = toRel(vec3f(xz.x, y, xz.y));
  col *= smoothstep(0.8, 3.0, length(rel));
  let c = F.viewProjNJ * vec4f(rel, 1.0);
  if (c.w < 0.5 || max(col.x, max(col.y, col.z)) < 1e-4) { return o; }
  // screen radius kept at 1.2 px or more, with the light spread over the larger sprite
  let px = r / (c.w * F.camFwd.w) * F.outSize.y * 0.5;
  let pr = clamp(px, 1.2, 24.0);
  col *= min(sq(px / pr), 1.0);
  let corner = vec2f(f32(vi & 1u), f32(vi >> 1u)) * 2.0 - 1.0;
  o.pos = c + vec4f(corner * pr * 2.0 * F.outSize.zw * c.w, 0.0, 0.0);
  o.q = corner;
  o.col = col;
  o.vz = c.w;
  return o;
}

@fragment
fn fs(in: PO) -> @location(0) vec4f {
  let r2 = dot(in.q, in.q);
  if (r2 >= 1.0) { discard; }
  // scene depth at internal resolution, with a soft edge so particles fade into what they touch
  let ip = vec2i(in.pos.xy * F.outSize.zw * F.screen.xy);
  let sz = viewZ(textureLoad(sceneDepth, ip, 0));
  let soft = sat((sz - in.vz) / 0.5);
  let fall = (exp(-r2 * 3.5) - 0.0302) / 0.9698;
  return vec4f(in.col * fall * soft, 0.0);
}
