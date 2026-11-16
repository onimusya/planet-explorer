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
// Meadow / ground-cover tones by region. v = variation noise (0..1).
fn meadowColor(p: vec2f, v: f32, valley: f32, highland: f32) -> vec3f {
  let pink = mix(vec3f(0.62, 0.15, 0.24), vec3f(0.80, 0.32, 0.30), v);
  let gold = mix(vec3f(0.62, 0.40, 0.12), vec3f(0.78, 0.56, 0.20), v);
  let moss = mix(vec3f(0.16, 0.10, 0.26), vec3f(0.10, 0.22, 0.30), v);
  var c = mix(pink, gold, highland);
  c = mix(c, moss, valley);
  return c;
}
