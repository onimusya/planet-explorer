// Water: flat camera-centred radial grid at WATER_Y; per-pixel waves, refraction with
// depth-based absorption, screen-space reflections with sky/planet fallback, sun glint, foam.
//#include "common.wgsl"
//#include "noise.wgsl"
//#include "world.wgsl"
//#include "lighting.wgsl"
//#include "sky.wgsl"

@group(3) @binding(0) var colorCopy: texture_2d<f32>;
@group(3) @binding(1) var depthCopy: texture_depth_2d;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) world: vec3f,
  @location(1) rel: vec3f,
};

const SEGS: u32 = 128u;
const RINGS: u32 = 72u;

fn ringRadius(r: u32) -> f32 {
  return 2.0 * (pow(1.115, f32(r)) - 1.0) / 0.115;
}

@vertex
fn vs(@builtin(vertex_index) vid: u32) -> VOut {
  let quad = vid / 6u;
  let k = vid % 6u;
  let r = quad / SEGS;
  let s = quad % SEGS;
  var dr = 0u;
  var ds = 0u;
  // two triangles: (0,0) (1,0) (0,1) / (0,1) (1,0) (1,1)
  if (k == 1u || k == 4u) { dr = 1u; }
  if (k == 2u || k == 3u) { ds = 1u; }
  if (k == 5u) { dr = 1u; ds = 1u; }
  let rad = ringRadius(r + dr);
  let a = f32(s + ds) / f32(SEGS) * TAU;
  // snap the grid centre to 4 m to keep vertices stable
  let c = floor(F.camPos.xz / 4.0) * 4.0;
  let xz = c + vec2f(cos(a), sin(a)) * rad;
  var o: VOut;
  o.world = vec3f(xz.x, WATER_Y, xz.y);
  o.rel = toRel(o.world);
  o.pos = F.viewProj * vec4f(o.rel, 1.0);
  return o;
}

// Sum of directional waves: returns height gradient (d/dx, d/dz).
fn waveGrad(p: vec2f, t: f32, fade: f32) -> vec2f {
  var g = vec2f(0.0);
  var amp = 0.06;
  var k = 0.22;
  var ang = 0.6;
  for (var i = 0; i < 6; i++) {
    let d = vec2f(cos(ang), sin(ang));
    let w = sqrt(9.81 * k);
    let ph = dot(p, d) * k + t * w + f32(i) * 1.7;
    g += d * (k * amp * cos(ph));
    amp *= 0.62;
    k *= 1.72;
    ang += 2.1;
  }
  let rp = p * 0.5 + vec2f(t * 0.35, t * 0.21);
  let rn = fbmd(rp, vec2i(8192), 3, 0.5);
  g += rn.yz * 0.5 * 0.05;
  return g * fade;
}

fn ssr(rel: vec3f, R: vec3f, dist: f32) -> vec4f {
  var t = 1.5 + dist * 0.02;
  var prevT = 0.0;
  for (var i = 0; i < 28; i++) {
    let q = rel + R * t;
    let clip = F.viewProj * vec4f(q, 1.0);
    if (clip.w <= 0.05) { break; }
    let uv = clip.xy / clip.w * vec2f(0.5, -0.5) + 0.5;
    if (any(uv < vec2f(0.0)) || any(uv > vec2f(1.0))) { break; }
    let sp = vec2i(uv * F.screen.xy);
    let sd = textureLoad(depthCopy, sp, 0);
    if (sd > 0.0) {
      let sceneZ = viewZ(sd);
      let qZ = clip.w;
      if (qZ > sceneZ + 0.2 && qZ < sceneZ + (t - prevT) * 1.5 + 4.0) {
        let edge = sat(min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y)) * 8.0);
        let c = textureSampleLevel(colorCopy, sLinClamp, uv * F.screen.xy * F.full.zw, 0.0).rgb;
        let ap = aerial(uv, t);
        return vec4f(c * ap.a + ap.rgb, edge);
      }
    }
    prevT = t;
    t *= 1.32;
  }
  return vec4f(0.0);
}

const ABSORB: vec3f = vec3f(0.42, 0.10, 0.065);
const SCATTER: vec3f = vec3f(0.004, 0.022, 0.028);

@fragment
fn fs(in: VOut) -> @location(0) vec4f {
  let px = vec2i(in.pos.xy);
  let uv = in.pos.xy * F.screen.zw;
  let dist = length(in.rel);
  let V = -in.rel / max(dist, 1e-3);
  let p = in.world.xz;
  let t = time();
  let L = F.sunDir.xyz;
  let fade = 1.0 - smoothstep(300.0, 3500.0, dist);
  let g = waveGrad(p, t, fade);
  let n = normalize(vec3f(-g.x, 1.0, -g.y));
  // depth of the scene behind the water surface
  let sd = textureLoad(depthCopy, px, 0);
  var sceneDist = 1e6;
  if (sd > 0.0) { sceneDist = length(relFromDepth(uv, sd)); }
  let thick = max(sceneDist - dist, 0.0);
  let vDepth = max(WATER_Y - heightAt(p), 0.0);
  // refraction
  let ro = n.xz * 0.06 * sat(vDepth / 4.0) * (1.0 - smoothstep(50.0, 600.0, dist));
  var ruv = uv + ro;
  let rsd = textureLoad(depthCopy, vec2i(clamp(ruv, vec2f(0.0), vec2f(0.999)) * F.screen.xy), 0);
  if (rsd > 0.0 && viewZ(rsd) < dist * dot(-V, F.camFwd.xyz)) { ruv = uv; }
  let under = textureSampleLevel(colorCopy, sLinClamp, clamp(ruv, vec2f(0.0), vec2f(0.999)) * F.screen.xy * F.full.zw, 0.0).rgb;
  let path = min(thick, 60.0) * 0.5 + vDepth * 0.7;
  let T = exp(-ABSORB * path);
  let sunC = sunLightAt(0.0);
  var sh = terrainShadow(p, WATER_Y + 0.5) * cloudShadow(in.world);
  if (sh > 0.001) { sh *= csmShadow(in.rel, vec3f(0.0, 1.0, 0.0), dist, ign(in.pos.xy, F.misc.x)); }
  let inscat = SCATTER * (sunC * (0.25 + 0.75 * sh) * 0.35 + ambientSH(vec3f(0.0, 1.0, 0.0)) * PI) / (ABSORB + 0.05);
  let refr = under * T + inscat * (1.0 - T);
  // reflection
  var R = reflect(-V, n);
  R.y = abs(R.y) + 0.002;
  R = normalize(R);
  var refl = skyFull(R);
  if (dist < 6000.0) {
    let s = ssr(in.rel, R, dist);
    refl = mix(refl, s.rgb, s.a);
  }
  let nv = max(dot(n, V), 0.0);
  let fr = 0.02 + 0.98 * pow(1.0 - nv, 5.0);
  var col = mix(refr, refl, fr);
  // sun glint
  col += ggx(n, V, L, mix(0.05, 0.12, 1.0 - fade), vec3f(0.02)) * sunC * sh * 1.6;
  // shoreline foam
  let foamN = fbm(p * 0.35 + vec2f(t * 0.05, 0.0), vec2i(5734), 3, 0.5) + 0.5;
  let foam = (1.0 - smoothstep(0.0, 0.35 + foamN * 0.5, vDepth)) * smoothstep(0.35, 0.7, foamN + 0.25 * sin(t * 1.3 + vDepth * 9.0));
  col = mix(col, vec3f(0.75, 0.8, 0.82) * (sunC * sh * 0.12 + ambientSH(vec3f(0.0, 1.0, 0.0)) * 1.2), foam * 0.65 * (1.0 - smoothstep(80.0, 400.0, dist)));
  // soft shoreline edge (fade into refraction where very shallow)
  let edge = smoothstep(0.0, 0.25, vDepth);
  col = mix(under, col, edge);
  return vec4f(col, 1.0);
}
