// Shared surface materials: sedimentary rock strata, meadow ground, palettes.

fn hash1(x: f32) -> f32 { return fract(sin(x * 127.1 + 311.7) * 43758.5453); }

// Sedimentary banding: folded major beds (~24 m) with thin sub-layers. Returns linear albedo.
fn strata(y: f32, p: vec2f) -> vec3f {
  let w = fbm(p / 1024.0, vec2i(16), 2, 0.5) * 140.0 + fbm(p / 128.0, vec2i(128), 2, 0.5) * 9.0;
  let yy = y + w;
  let t1 = yy / 24.0;
  let id1 = floor(t1);
  let f1 = t1 - id1;
  let t2 = yy / 3.7;
  let id2 = floor(t2);
  let f2 = t2 - id2;
  let r2 = 0.84 + 0.28 * hash1(id2 * 1.37 + 3.0);
  var c = mix(strataPalette(hash1(id1 + 17.0)), strataPalette(hash1(id1 + 18.0)), smoothstep(0.86, 1.0, f1)) * r2;
  c *= 1.0 - 0.16 * (1.0 - smoothstep(0.0, 0.1, f2));
  return c;
}

fn strataPalette(r: f32) -> vec3f {
  if (r < 0.2) { return vec3f(0.50, 0.35, 0.25); }  // warm tan
  if (r < 0.38) { return vec3f(0.44, 0.21, 0.14); } // rust
  if (r < 0.55) { return vec3f(0.56, 0.38, 0.22); } // ochre
  if (r < 0.68) { return vec3f(0.32, 0.24, 0.28); } // mauve shale
  if (r < 0.84) { return vec3f(0.60, 0.45, 0.38); } // pale coral
  return vec3f(0.26, 0.13, 0.10);                    // oxide red
}

// Triplanar bump for steep rock faces (periodic in x/z). Returns perturbed normal.
fn rockBump(w: vec3f, n: vec3f, strength: f32) -> vec3f {
  if (strength <= 0.001) { return n; }
  let P = vec2i(2048, 1 << 20);
  let gx = fbmd(vec2f(w.z, w.y * 1.6) * 0.125, P, 4, 0.55);
  let gz = fbmd(vec2f(w.x, w.y * 1.6) * 0.125, P, 4, 0.55);
  let wx = n.x * n.x;
  let wz = n.z * n.z;
  let d = vec3f(0.0, gx.z * 1.6, gx.y) * wx + vec3f(gz.y, gz.z * 1.6, 0.0) * wz;
  return normalize(n - d * strength);
}
// Meadow colour patches: x = broad (~64 m) species patches, y = small (~16 m) clumps. 0..1.
fn meadowPatch(p: vec2f) -> vec2f {
  return vec2f(sat(fbm(p / 64.0, vec2i(256), 2, 0.5) * 1.4 + 0.5), sat(fbm(p / 16.0 + 7.3, vec2i(1024), 2, 0.5) * 1.2 + 0.5));
}
// Meadow / ground-cover tones by region and patch. v = per-blade variation (0..1).
fn meadowColorP(pt: vec2f, v: f32, valley: f32, highland: f32) -> vec3f {
  let a = smoothstep(0.3, 0.75, pt.x);
  let b = smoothstep(0.55, 0.85, pt.y);
  let rose = mix(vec3f(0.60, 0.13, 0.22), vec3f(0.84, 0.32, 0.31), v);
  let mag = mix(vec3f(0.44, 0.08, 0.34), vec3f(0.70, 0.22, 0.50), v);
  let apr = mix(vec3f(0.74, 0.34, 0.16), vec3f(0.90, 0.54, 0.26), v);
  let pink = mix(mix(rose, mag, a), apr, b * 0.65);
  let gold = mix(vec3f(0.58, 0.37, 0.10), vec3f(0.82, 0.60, 0.22), v);
  let olive = mix(vec3f(0.34, 0.33, 0.10), vec3f(0.52, 0.50, 0.17), v);
  let rust = mix(vec3f(0.55, 0.20, 0.09), vec3f(0.72, 0.32, 0.14), v);
  let gc = mix(mix(gold, olive, a * 0.75), rust, b * 0.55);
  let teal = mix(vec3f(0.06, 0.20, 0.30), vec3f(0.12, 0.30, 0.34), v);
  let vio = mix(vec3f(0.16, 0.08, 0.30), vec3f(0.26, 0.12, 0.40), v);
  let moss = mix(teal, vio, a);
  return mix(mix(pink, gc, highland), moss, valley);
}
fn meadowColor(p: vec2f, v: f32, valley: f32, highland: f32) -> vec3f {
  return meadowColorP(meadowPatch(p), v, valley, highland);
}
// Arrival-vista shelf (vistaLedge in terrain_gen): ground cover runs over its rounded lip and the rock
// detail is suppressed there, so the first frame's foreground reads as a grassy cliff edge. 0..1.
fn ledgeGrass(p: vec2f) -> f32 {
  let lc = wrapD2(p - VISTA_CAM);
  if (dot(lc, lc) > 14400.0) { return 0.0; }
  let la = dot(lc, VISTA_DIR);
  let ls = dot(lc, vec2f(-VISTA_DIR.y, VISTA_DIR.x));
  let lipA = 19.0 + fbm(p / WORLD_L * 512.0, vec2i(512), 2, 0.5) * (2.5 + abs(ls) * 0.25) - ls * ls * 0.003;
  return (1.0 - smoothstep(lipA - 7.0, lipA - 4.0, la)) * smoothstep(-40.0, -25.0, la) * (1.0 - smoothstep(45.0, 62.0, abs(ls)));
}
// Grass coverage 0..1, shared by the blade field and the terrain's far canopy so the two always agree.
// a = attrAt(p) (crease, lush, ao, rock), slope = |grad|, h = height.
fn meadowCover(p: vec2f, a: vec4f, slope: f32, h: f32) -> f32 {
  let n = fbm(p / 32.0, vec2i(512), 2, 0.5);
  let lush = a.y + n * 0.35 + 0.22 + valleyMask(p) * 0.3 + highlandMask(p) * 0.15;
  let steep = mix(1.0 - smoothstep(0.42, 0.62, slope + a.w * 0.3), 1.0, ledgeGrass(p));
  return smoothstep(0.25, 0.5, lush) * steep * (1.0 - smoothstep(1100.0, 1250.0, h));
}
// Travelling wind-gust brightness waves over meadows (shared by blades and the far canopy).
fn meadowWave(p: vec2f) -> f32 {
  let t = time();
  let w = sin(dot(p, vec2f(0.043, 0.025)) - t * 1.9 + fbm(p / 64.0 + 3.1, vec2i(256), 2, 0.5) * 5.0);
  return smoothstep(0.2, 1.0, w) * (0.6 + 0.4 * sin(t * 0.45 + p.x * 0.013 + p.y * 0.009));
}
