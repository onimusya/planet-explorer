// Instanced flora / rock rendering (main + shadow), wind animation, per-part alien materials.
//#include "common.wgsl"
//#include "noise.wgsl"
//#include "world.wgsl"
//#include "lighting.wgsl"
//#include "materials.wgsl"

struct VIn {
  @location(0) pos: vec3f,
  @location(1) nrm: vec4f,
  @location(2) uv: vec2f,
  @location(3) attr: vec4f,
  @location(4) ia: vec4f, // world xyz, scale
  @location(5) ib: vec4f, // yaw, seed, species, yStretch
};

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) world: vec3f,
  @location(1) rel: vec3f,
  @location(2) nrm: vec3f,
  @location(3) uv: vec2f,
  @location(4) @interpolate(flat) info: vec4f, // part, species, seed, scale
  @location(5) ag: vec2f, // ao, grad
  @location(6) lpos: vec3f, // local (object) position for procedural patterns
};

@group(3) @binding(0) var<uniform> cascadeIdx: vec4u;

fn windOffset(wp: vec3f, flex: f32, part: f32, scale: f32, seed: f32) -> vec3f {
  let t = time();
  let ph = dot(wp.xz, vec2f(0.021, 0.034)) + seed * 6.283;
  let gust = 0.6 + 0.4 * sin(t * 0.37 + wp.x * 0.004 + wp.z * 0.003);
  let sway = (sin(t * 1.15 + ph) * 0.7 + sin(t * 2.3 + ph * 1.7) * 0.3) * gust;
  let wd = vec2f(0.86, 0.5);
  var o = vec3f(wd.x, 0.0, wd.y) * sway * flex * flex * 0.32 * scale;
  if (part == 1.0 || part == 8.0 || part == 4.0) {
    let fl = sin(t * 6.5 + wp.x * 2.1 + wp.z * 1.7 + seed * 40.0) * 0.045 * flex * scale;
    o += vec3f(fl, fl * 0.6, -fl * 0.5);
  }
  return o;
}

fn transformV(v: VIn) -> VOut {
  let yaw = v.ib.x;
  let c = cos(yaw);
  let s = sin(yaw);
  let sc = v.ia.w;
  var lp = v.pos;
  lp.y *= v.ib.w;
  let rp = vec3f(lp.x * c + lp.z * s, lp.y, -lp.x * s + lp.z * c) * sc;
  let rn = vec3f(v.nrm.x * c + v.nrm.z * s, v.nrm.y / v.ib.w, -v.nrm.x * s + v.nrm.z * c);
  let part = round(v.attr.x * 255.0);
  var wp = v.ia.xyz + rp;
  wp += windOffset(wp, v.attr.y, part, sc, v.ib.y);
  var o: VOut;
  o.world = wp;
  o.rel = toRel(wp);
  o.nrm = normalize(rn);
  o.uv = vec2f(v.uv.x, v.uv.y * 64.0);
  o.info = vec4f(part, v.ib.z, v.ib.y, sc);
  o.ag = vec2f(v.attr.z, v.attr.w);
  o.lpos = v.pos;
  return o;
}

@vertex
fn vs(v: VIn) -> VOut {
  var o = transformV(v);
  o.pos = F.viewProj * vec4f(o.rel, 1.0);
  return o;
}

@vertex
fn vsShadow(v: VIn) -> VOut {
  var o = transformV(v);
  o.pos = F.shadowMats[cascadeIdx.x] * vec4f(o.rel, 1.0);
  return o;
}

// ---------------------------------------------------------------------------------------------
fn leafAlpha(uv: vec2f, seed: f32) -> f32 {
  let d = uv - 0.5;
  let r = length(d) * 2.0;
  let th = atan2(d.y, d.x);
  let lobes = 0.55 + 0.45 * pow(abs(cos(th * 2.5 + seed * 6.0)), 0.6);
  return step(r, lobes * 0.98);
}
fn frondAlpha(uv: vec2f) -> f32 {
  let w = 1.0 - abs(uv.x - 0.5) * 2.0;
  let lobe = abs(fract(uv.y * 3.0) - 0.5) * 2.0;
  return step(0.08 + 0.55 * lobe * lobe, w);
}

struct Mat { alb: vec3f, rough: f32, emit: vec3f, trans: f32, spec: f32 };

fn hue(seed: f32, a: vec3f, b: vec3f, c: vec3f) -> vec3f {
  let t = fract(seed * 7.13);
  if (t < 0.5) { return mix(a, b, t * 2.0); }
  return mix(b, c, (t - 0.5) * 2.0);
}

const GILL_RIBS: f32 = 96.0;

fn material(part: f32, sp: f32, seed: f32, uv: vec2f, grad: f32, wp: vec3f, lp: vec3f, valley: f32, ribAA: f32) -> Mat {
  var m: Mat;
  m.alb = vec3f(0.3);
  m.rough = 0.7;
  m.emit = vec3f(0.0);
  m.trans = 0.0;
  m.spec = 0.04;
  let glow = 1.0 + valley * 1.5 + F.extra.z;
  let gn = fbm(vec2f(uv.x * 8.0, uv.y * 2.0), vec2i(8, 1 << 20), 3, 0.5);
  if (part == 0.0) { // bark
    let groove = abs(sin(uv.x * TAU * 5.0 + gn * 3.0));
    m.alb = mix(vec3f(0.12, 0.07, 0.08), vec3f(0.26, 0.15, 0.14), groove) * (0.85 + 0.3 * gn);
    m.rough = 0.85;
  } else if (part == 9.0) { // smooth stem
    if (sp == 0.0) { // coral spire / tree stems: coral pink
      m.alb = hue(seed, vec3f(0.62, 0.20, 0.22), vec3f(0.72, 0.36, 0.20), vec3f(0.48, 0.16, 0.36)) * (0.8 + 0.3 * grad);
      m.trans = 0.3;
    } else if (sp == 1.0) { // mushroom stalk: pale, faintly luminous veins
      m.alb = vec3f(0.62, 0.62, 0.66) * (0.85 + 0.2 * gn);
      m.trans = 0.5;
      let vein = smoothstep(0.85, 1.0, abs(sin(uv.x * TAU * 6.0 + gn * 4.0)));
      m.emit = vec3f(0.1, 0.7, 0.9) * vein * 0.6 * glow * smoothstep(0.2, 0.9, grad);
    } else if (sp == 3.0) {
      m.alb = vec3f(0.12, 0.26, 0.16);
      m.trans = 0.4;
    } else {
      m.alb = vec3f(0.10, 0.12, 0.16);
      m.trans = 0.3;
    }
    m.rough = 0.6;
  } else if (part == 1.0) { // tree leaf cards
    let base = select(
      hue(seed, vec3f(0.30, 0.06, 0.40), vec3f(0.56, 0.10, 0.26), vec3f(0.70, 0.34, 0.08)),
      hue(seed, vec3f(0.04, 0.26, 0.34), vec3f(0.10, 0.38, 0.30), vec3f(0.22, 0.10, 0.42)), valley > 0.4);
    m.alb = base * (0.7 + 0.5 * grad);
    m.trans = 0.8;
    m.rough = 0.65;
  } else if (part == 8.0) { // fronds
    var c = hue(seed, vec3f(0.42, 0.06, 0.16), vec3f(0.55, 0.12, 0.24), vec3f(0.52, 0.30, 0.08));
    if (sp == 0.0) { c = hue(seed, vec3f(0.05, 0.30, 0.26), vec3f(0.10, 0.40, 0.34), vec3f(0.30, 0.42, 0.12)); }
    c = mix(c, vec3f(0.04, 0.24, 0.30), valley * 0.8);
    let mid = 1.0 - abs(uv.x - 0.5) * 2.0;
    m.alb = c * (0.55 + 0.5 * grad) * (0.8 + 0.3 * mid);
    m.trans = 0.75;
    m.rough = 0.6;
    if (valley > 0.3) { m.emit = vec3f(0.1, 0.8, 0.9) * smoothstep(0.92, 1.0, grad) * 0.5 * glow; }
  } else if (part == 2.0) { // mushroom cap with luminous spots
    let capC = hue(seed, vec3f(0.14, 0.08, 0.36), vec3f(0.36, 0.06, 0.32), vec3f(0.04, 0.24, 0.34));
    let ang = uv.x * TAU;
    let sp2 = vec2f(cos(ang), sin(ang)) * (grad * 6.0 + 1.0);
    let cell = worley3(vec3f(sp2 * 1.3, seed * 11.0), BIGP3);
    let spot = 1.0 - smoothstep(0.18, 0.3, cell);
    m.alb = mix(capC * (0.7 + 0.5 * (1.0 - grad)), vec3f(0.8, 0.85, 0.9), spot * 0.3);
    m.emit = mix(vec3f(0.2, 0.9, 1.0), vec3f(0.9, 0.4, 1.0), fract(seed * 3.7)) * spot * 1.6 * glow;
    m.rough = 0.45;
    m.trans = 0.25;
  } else if (part == 3.0) { // gills: bioluminescent underside, ribs anti-aliased, brightest at the rim
    let rib = 0.5 + 0.5 * cos(uv.x * TAU * GILL_RIBS) * ribAA;
    let c = mix(vec3f(0.0, 0.42, 1.0), vec3f(0.30, 0.05, 1.0), fract(seed * 5.1));
    m.alb = vec3f(0.16, 0.18, 0.24);
    m.emit = c * (0.4 + 0.6 * rib) * (0.12 + 0.88 * grad * grad) * 0.48 * glow;
    m.rough = 0.8;
  } else if (part == 4.0) { // petals
    let tip = smoothstep(0.1, 1.0, grad);
    var a = vec3f(0.55, 0.05, 0.12);
    var b = vec3f(0.98, 0.60, 0.12);
    if (fract(seed * 2.3) > 0.6) { a = vec3f(0.35, 0.05, 0.45); b = vec3f(0.95, 0.55, 0.85); }
    let vein = smoothstep(0.75, 1.0, abs(sin(uv.x * 30.0))) * 0.25;
    m.alb = mix(a, b, tip) * (1.0 - vein);
    m.trans = 0.9;
    m.rough = 0.5;
    m.emit = b * smoothstep(0.85, 1.0, tip) * 0.25 * glow;
  } else if (part == 5.0) { // crystal: dark glossy gem body, luminous tips
    let c = hue(seed, vec3f(0.012, 0.045, 1.0), vec3f(0.25, 0.024, 0.76), vec3f(0.02, 0.38, 0.62));
    m.alb = c * 0.3;
    m.rough = 0.05;
    m.spec = 0.06;
    m.trans = 0.5;
    let tip = smoothstep(0.62, 1.0, grad);
    m.emit = c * (tip * tip * 1.6 + 0.04) * mix(0.3, 1.0, valley) * glow;
  } else if (part == 6.0) { // glowing bulbs
    let c = hue(seed, vec3f(0.2, 1.0, 0.9), vec3f(1.0, 0.75, 0.25), vec3f(0.9, 0.3, 1.0));
    let pulse = 0.75 + 0.25 * sin(time() * (1.0 + fract(seed * 9.0)) + seed * 30.0);
    m.alb = c * 0.3;
    m.emit = c * 3.2 * pulse * glow;
    m.rough = 0.3;
    m.trans = 1.0;
  } else { // rock (luminance floor so small boulders never read as holes)
    var r = strata(wp.y, wp.xz) * 0.85;
    r *= 0.75 + 0.35 * fbm3(lp * 1.7, 3) + 0.15;
    m.alb = r * max(1.0, 0.33 / max(luma(r), 0.02));
    m.rough = 0.8;
  }
  return m;
}

fn shade(in: VOut, front: bool, cut: bool) -> vec4f {
  let part = in.info.x;
  let sp = in.info.y;
  let seed = in.info.z;
  // derivative-based rib anti-aliasing (uniform control flow)
  let ribAA = sat(1.5 - fwidth(in.uv.x) * GILL_RIBS * 1.5);
  if (cut) {
    var a = 1.0;
    if (part == 1.0) { a = leafAlpha(in.uv, seed + floor(in.ag.y * 10.0)); }
    else if (part == 8.0) { a = frondAlpha(in.uv); }
    if (a < 0.5) { discard; }
  }
  let dist = length(in.rel);
  let V = -in.rel / max(dist, 1e-3);
  let L = F.sunDir.xyz;
  var n = normalize(in.nrm);
  if (!front) { n = -n; }
  let valley = valleyMask(in.world.xz);
  let m = material(part, sp, seed, in.uv, in.ag.y, in.world, in.lpos, valley, ribAA);
  let sunC = sunLightAt(in.world.y);
  var sh = terrainShadow(in.world.xz, in.world.y + 0.5) * cloudShadow(in.world);
  if (sh > 0.001) { sh *= csmShadow(in.rel, n, dist, ign(in.pos.xy, F.misc.x)); }
  let ndl = dot(n, L);
  // wrapped diffuse for foliage softness
  let wrap = select(0.0, 0.35, m.trans > 0.0);
  let dif = sat((ndl + wrap) / (1.0 + wrap));
  var col = m.alb * (1.0 / PI) * dif * sunC * sh;
  col += ggx(n, V, L, m.rough, vec3f(m.spec)) * sunC * sh;
  // translucency when backlit
  let bt = pow(sat(dot(-V, L)), 3.0) * m.trans;
  col += m.alb * sunC * sh * bt * 0.9;
  let ao = in.ag.x;
  col += m.alb * ambientSH(n) * ao;
  let R = reflect(-V, n);
  let nv = sat(dot(n, V));
  if (part == 5.0) {
    // gems: full Fresnel sky reflection (lower hemisphere mirrored and darkened as ground)
    let fr = m.spec + (1.0 - m.spec) * pow(1.0 - nv, 5.0);
    let rs = skyColor(normalize(vec3f(R.x, abs(R.y), R.z))) * select(0.35, 1.0, R.y > 0.0);
    col += rs * fr * mix(0.4, 1.0, sh);
  } else {
    // rough environment specular (keeps shaded surfaces from going flat)
    col += ambientSH(R) * envBRDF(vec3f(m.spec), m.rough, nv) * ao;
  }
  col += m.emit;
  return vec4f(col, 1.0);
}

@fragment
fn fsOpaque(in: VOut, @builtin(front_facing) front: bool) -> @location(0) vec4f {
  return shade(in, front, false);
}

@fragment
fn fsCutout(in: VOut, @builtin(front_facing) front: bool) -> @location(0) vec4f {
  return shade(in, front, true);
}

@fragment
fn fsShadowCut(in: VOut) {
  var a = 1.0;
  if (in.info.x == 1.0) { a = leafAlpha(in.uv, in.info.z + floor(in.ag.y * 10.0)); }
  else if (in.info.x == 8.0) { a = frondAlpha(in.uv); }
  if (a < 0.5) { discard; }
}
