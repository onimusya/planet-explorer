// Sky: atmosphere + sun disk + ringed gas giant + moon. Needs common/noise/world.

struct Body { col: vec3f, a: f32 };

// (GIANT_DIR comes from world.ts: the night's light comes from it)
const GIANT_RAD: f32 = 0.165;    // angular radius (rad)
const GIANT_AXIS: vec3f = vec3f(0.30, 0.94, 0.16);
const MOON_DIR: vec3f = vec3f(0.9623, 0.2588, 0.0842);
const MOON_RAD: f32 = 0.05;

fn sphereHit(d: vec3f, c: vec3f, rad: f32) -> vec2f {
  // unit-distance sphere: centre c (unit), radius sin(rad)
  let r = sin(rad);
  let b = dot(c, d);
  let h = b * b - (1.0 - r * r);
  if (h < 0.0) { return vec2f(-1.0, h); }
  return vec2f(b - sqrt(h), h);
}

fn gasGiant(d: vec3f, sun: vec3f) -> Body {
  var o: Body;
  o.col = vec3f(0.0);
  o.a = 0.0;
  let c = normalize(GIANT_DIR);
  let A = normalize(GIANT_AXIS);
  let r = sin(GIANT_RAD);
  let hit = sphereHit(d, c, GIANT_RAD);
  // ring plane intersection
  let tr = dot(c, A) / dot(d, A);
  let rp = d * tr - c;
  let rr = length(rp) / r;
  var ringA = 0.0;
  var ringC = vec3f(0.0);
  if (tr > 0.0 && rr > 1.28 && rr < 2.35) {
    let band = fbm(vec2f(rr * 40.0, 0.5), vec2i(4096), 4, 0.6) + 0.5;
    let gap = smoothstep(0.02, 0.06, abs(rr - 1.86)) * smoothstep(0.01, 0.03, abs(rr - 2.12));
    let fine = fbm(vec2f(rr * 160.0, 0.5), vec2i(16384), 3, 0.6) + 0.5;
    ringA = sat(band * 1.2 - 0.2) * mix(0.6, 1.0, fine) * gap * smoothstep(1.28, 1.40, rr) * (1.0 - smoothstep(2.2, 2.35, rr)) * 0.9;
    // planet shadow on the rings
    let rpW = c + rp;
    let toC = c - rpW;
    let b = dot(toC, sun);
    let perp = length(toC - sun * b);
    let shadow = select(1.0, smoothstep(r * 0.96, r * 1.02, perp), b > 0.0);
    let ph = 0.6 + 0.4 * sat(dot(-d, sun) * 0.5 + 0.5);
    ringC = mix(vec3f(0.62, 0.50, 0.42), vec3f(0.50, 0.44, 0.52), band) * mix(0.7, 1.0, fine) * shadow * ph;
  }
  if (hit.x > 0.0) {
    let p = d * hit.x;
    let n = normalize(p - c);
    let lat = dot(n, A);
    let e = normalize(cross(A, vec3f(0.0, 0.0, 1.0)));
    let lon = atan2(dot(n, cross(A, e)), dot(n, e));
    let turb = fbm3(n * 3.0, 4) * 0.35 + fbm3(n * 9.0 + vec3f(2.0), 3) * 0.12;
    let bl = lat * 9.0 + turb;
    let b1 = sin(bl * 3.1) * 0.5 + 0.5;
    let b2 = sin(bl * 7.3 + 1.7) * 0.5 + 0.5;
    var col = mix(vec3f(0.807, 0.456, 0.254), vec3f(0.905, 0.791, 0.631), b1);
    col = mix(col, vec3f(0.323, 0.212, 0.431), b2 * 0.5);
    col = mix(col, vec3f(0.78, 0.38, 0.30), smoothstep(0.55, 0.9, sin(bl * 1.3 + 0.4)) * 0.35);
    // great storm
    let sp = vec2f(lon - 0.9, (lat + 0.32) * 3.2);
    let storm = 1.0 - smoothstep(0.12, 0.3, length(sp * vec2f(1.0, 1.6)) + turb * 0.1);
    col = mix(col, vec3f(0.85, 0.45, 0.32), storm * 0.8);
    let ndl = dot(n, sun);
    let lit = smoothstep(-0.08, 0.35, ndl) * (0.25 + 0.75 * sat(ndl));
    // ring shadow on the planet
    let tp = dot(c - p, A) / dot(sun, A);
    let rq = length(p + sun * tp - c) / r;
    var rs = 1.0;
    if (tp > 0.0 && rq > 1.28 && rq < 2.35) { rs = 1.0 - 0.6 * smoothstep(1.3, 1.45, rq) * (1.0 - smoothstep(2.2, 2.35, rq)); }
    let rim = pow(1.0 - sat(dot(n, -d)), 3.0);
    col = col * lit * rs + vec3f(0.55, 0.70, 0.90) * rim * sat(ndl + 0.3) * 0.25;
    // night side faint glow (ring-shine)
    col += vec3f(0.02, 0.025, 0.04) * (1.0 - lit);
    o.col = col;
    o.a = 1.0;
    let ringFront = tr > 0.0 && tr < hit.x;
    if (ringFront) {
      o.col = mix(o.col, ringC, ringA);
    }
  } else {
    o.col = ringC * ringA;
    o.a = ringA;
    // atmospheric halo just outside the limb
    let cosd = dot(d, c);
    let ang = acos(clamp(cosd, -1.0, 1.0)) - GIANT_RAD;
    let halo = exp(-max(ang, 0.0) * 160.0) * sat(dot(normalize(d - c * cosd), sun) + 0.4);
    o.col += vec3f(0.5, 0.7, 1.0) * halo * 0.35;
    o.a = max(o.a, halo * 0.5);
  }
  return o;
}

fn moonBody(d: vec3f, sun: vec3f) -> Body {
  var o: Body;
  o.col = vec3f(0.0);
  o.a = 0.0;
  let c = normalize(MOON_DIR);
  let hit = sphereHit(d, c, MOON_RAD);
  if (hit.x > 0.0) {
    let n = normalize(d * hit.x - c);
    let cr = worley3(n * 6.0, BIGP3);
    let cr2 = worley3(n * 15.0 + vec3f(3.0), BIGP3);
    var alb = vec3f(0.62, 0.64, 0.70) * (0.75 + 0.25 * fbm3(n * 5.0, 3));
    alb *= mix(0.7, 1.0, smoothstep(0.15, 0.35, cr)) * mix(0.85, 1.0, smoothstep(0.1, 0.3, cr2));
    let ndl = dot(n, sun);
    o.col = alb * smoothstep(-0.05, 0.25, ndl) * (0.3 + 0.7 * sat(ndl));
    o.a = 1.0;
  }
  return o;
}

// Night sky: stars on a grid of cells over the six faces of a cube (at most one star per cell,
// drawn as a soft dot about an output pixel wide so the jittered TAA samples agree on it), denser
// along a band of glowing gas and dust that crosses the sky.
const STAR_N: f32 = 420.0;                              // cells along a cube face
const BAND_N: vec3f = vec3f(0.404, -0.807, -0.431);     // normal of the galactic band's plane: it rises from the
                                                        // Vista's left horizon and arches over the gas giant

fn starField(d: vec3f) -> vec3f {
  let a = abs(d);
  var uv: vec2f;
  var face: i32;
  if (a.x >= a.y && a.x >= a.z) { uv = d.zy / a.x; face = select(1, 0, d.x > 0.0); }
  else if (a.y >= a.z) { uv = d.xz / a.y; face = select(3, 2, d.y > 0.0); }
  else { uv = d.xy / a.z; face = select(5, 4, d.z > 0.0); }
  let bx = dot(d, BAND_N);
  let band = exp(-bx * bx * 22.0);
  let g = (uv * 0.5 + 0.5) * STAR_N;
  let base = floor(g - 0.5);
  // grid units to radians (radial stretch of the cube map), and radians per output pixel
  let toRad = (2.0 / STAR_N) / (1.0 + dot(uv, uv));
  let pix = 2.0 * F.camFwd.w * F.outSize.w;
  let tw = time();
  var col = vec3f(0.0);
  for (var j = 0; j < 2; j++) {
    for (var i = 0; i < 2; i++) {
      let cell = base + vec2f(f32(i), f32(j));
      let h = hash33(vec3i(vec2i(cell), face * 7919 + 101));
      if (h.z > 0.1 + 0.3 * band) { continue; }
      let h2 = hash33(vec3i(vec2i(cell), face * 7919 + 202));
      let ang = length(g - (cell + 0.15 + 0.7 * h.xy)) * toRad;
      let rad = (0.55 + 0.6 * h2.x * h2.x * h2.x) * pix;
      // most stars faint, a few bright; a slow shimmer that grows toward the horizon
      var b = 0.015 + 3.2 * pow(h2.y, 40.0);
      b *= 1.0 + 0.3 * sin(tw * (2.0 + 5.0 * h.x) + h.y * 40.0) * (1.0 - sat(d.y * 2.5));
      let tint = mix(mix(vec3f(1.0, 0.72, 0.48), vec3f(1.0, 0.93, 0.8), sat(h2.z * 3.0)), vec3f(0.72, 0.84, 1.0), sat(h2.z * 2.2 - 1.0));
      col += tint * b * exp(-ang * ang / (rad * rad) * 2.0);
    }
  }
  // the band: clouds of glowing gas, teal to magenta, cut by darker lanes of dust (two lookups in
  // the tiling cloud noise volume, at scales that hide its repeats)
  if (band > 0.02) {
    let n1 = textureSampleLevel(noiseTex, sLinRep, d * 0.83 + vec3f(0.31, 0.17, 0.53), 0.0).r;
    let n2 = textureSampleLevel(noiseTex, sLinRep, d * 2.37 + vec3f(0.71, 0.29, 0.11), 0.0).g;
    let gas = band * smoothstep(0.25, 0.85, n1) * (1.0 - 0.75 * smoothstep(0.5, 0.75, n2));
    col += mix(vec3f(0.10, 0.34, 0.42), vec3f(0.42, 0.13, 0.40), smoothstep(0.3, 0.7, n2)) * gas * 0.45;
  }
  return col * 0.3;
}

// Full sky radiance for a view direction (no clouds).
fn skyFull(d: vec3f) -> vec3f {
  // F.sunDir is the light that shades the world; the gas giant, the moon and the sun disk follow
  // the real sun (F.sky), which differ at night.
  let sun = F.sky.xyz;
  let body = F.skyCol.rgb;
  var c = skyColor(d);
  let r = ATM_BOTTOM + max(F.camPos.y, 1.0) * 0.001;
  let gnd = raySphere(vec3f(0.0, r, 0.0), d, ATM_BOTTOM).x > 0.0;
  if (!gnd) {
    let T = sampleTrans(transLut, sLinClamp, r, d.y);
    // planets read darker than the open sky behind the in-scattered light (artistic contrast)
    var space = vec3f(0.0);
    if (F.sky.w > 0.0) { space = starField(d) * F.sky.w; }
    let g = gasGiant(d, sun);
    let m = moonBody(d, sun);
    space = mix(space, m.col * body * 0.14, m.a);
    space = mix(space, g.col * body * 0.42, g.a);
    c *= 1.0 - max(0.72 * g.a, 0.3 * m.a);
    // sun disk with limb darkening and soft corona
    let cs = dot(d, sun);
    let sunR = 0.0125;
    let ang = acos(clamp(cs, -1.0, 1.0));
    let disk = 1.0 - smoothstep(sunR * 0.9, sunR, ang);
    let limb = 1.0 - 0.6 * sq(sat(ang / sunR));
    space += body * (disk * limb * 260.0 + exp(-ang * 60.0) * 2.0) * (1.0 - g.a) * F.skyCol.w;
    c += T * space;
  }
  return c;
}
