// Signed distance fields of meshes on voxel grids, for camera collision (collide.ts). Built once
// at startup in a worker (sdf.worker.ts), so the main thread keeps generating the world.

export interface SdfData {
  nx: number; ny: number; nz: number; vs: number; // grid size, voxel size (local units)
  ox: number; oy: number; oz: number; // grid origin (local units)
  d: Float32Array; // signed distance at voxel centres (local units, negative inside)
  br: number; // bounding radius around the local origin
}

// Felzenszwalb-Huttenlocher squared distance transform of f (length n, stride st, offset o) in place.
function edt1(f: Float32Array, o: number, st: number, n: number, v: Int32Array, z: Float64Array, g: Float64Array) {
  for (let q = 0; q < n; q++) g[q] = f[o + q * st];
  let k = 0;
  v[0] = 0; z[0] = -1e20; z[1] = 1e20;
  for (let q = 1; q < n; q++) {
    let s = ((g[q] + q * q) - (g[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
    while (s <= z[k]) { k--; s = ((g[q] + q * q) - (g[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]); }
    k++; v[k] = q; z[k] = s; z[k + 1] = 1e20;
  }
  k = 0;
  for (let q = 0; q < n; q++) {
    while (z[k + 1] < q) k++;
    const d = q - v[k];
    f[o + q * st] = d * d + g[v[k]];
  }
}
function edt3(f: Float32Array, nx: number, ny: number, nz: number) {
  const m = Math.max(nx, ny, nz);
  const v = new Int32Array(m), z = new Float64Array(m + 1), g = new Float64Array(m);
  for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) edt1(f, (k * ny + j) * nx, 1, nx, v, z, g);
  for (let k = 0; k < nz; k++) for (let i = 0; i < nx; i++) edt1(f, k * ny * nx + i, nx, ny, v, z, g);
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) edt1(f, j * nx + i, nx * ny, nz, v, z, g);
}

// Voxelises triangles (positions P, xyz per vertex; tris, 3 indices each) with res voxels along
// the longest side of their bounds, closes them by closeR voxels and stores the signed distance
// to the surface.
export function buildSdf(P: Float32Array, tris: Uint32Array, res: number, closeR: number): SdfData | null {
  if (tris.length === 0) return null;
  let x0 = 1e9, y0 = 1e9, z0 = 1e9, x1 = -1e9, y1 = -1e9, z1 = -1e9;
  for (let i = 0; i < tris.length; i++) {
    const t = tris[i] * 3;
    x0 = Math.min(x0, P[t]); x1 = Math.max(x1, P[t]);
    y0 = Math.min(y0, P[t + 1]); y1 = Math.max(y1, P[t + 1]);
    z0 = Math.min(z0, P[t + 2]); z1 = Math.max(z1, P[t + 2]);
  }
  const vs = Math.max(x1 - x0, y1 - y0, z1 - z0) / res;
  const pad = closeR + 2;
  const nx = Math.ceil((x1 - x0) / vs) + 2 * pad, ny = Math.ceil((y1 - y0) / vs) + 2 * pad, nz = Math.ceil((z1 - z0) / vs) + 2 * pad;
  const ox = x0 - pad * vs, oy = y0 - pad * vs, oz = z0 - pad * vs;
  const N = nx * ny * nz;
  const occ = new Uint8Array(N);
  for (let t = 0; t < tris.length; t += 3) {
    const a = tris[t] * 3, b = tris[t + 1] * 3, c = tris[t + 2] * 3;
    const e1 = Math.hypot(P[b] - P[a], P[b + 1] - P[a + 1], P[b + 2] - P[a + 2]);
    const e2 = Math.hypot(P[c] - P[a], P[c + 1] - P[a + 1], P[c + 2] - P[a + 2]);
    const e3 = Math.hypot(P[c] - P[b], P[c + 1] - P[b + 1], P[c + 2] - P[b + 2]);
    const n = Math.max(1, Math.ceil(Math.max(e1, e2, e3) / (0.45 * vs)));
    for (let i = 0; i <= n; i++) for (let j = 0; j <= n - i; j++) {
      const u = i / n, w = j / n;
      const x = P[a] + (P[b] - P[a]) * u + (P[c] - P[a]) * w;
      const y = P[a + 1] + (P[b + 1] - P[a + 1]) * u + (P[c + 1] - P[a + 1]) * w;
      const z = P[a + 2] + (P[b + 2] - P[a + 2]) * u + (P[c + 2] - P[a + 2]) * w;
      occ[(Math.floor((z - oz) / vs) * ny + Math.floor((y - oy) / vs)) * nx + Math.floor((x - ox) / vs)] = 1;
    }
  }
  // Morphological closing by closeR voxels, so tubes with open ends (stems, trunks) and thin
  // shells come out solid: the surface is dilated, the outside is flood-filled from the grid's
  // border around the dilated shape, and what lies more than closeR from that outside is solid.
  // Openings wider than about 2 closeR voxels (arches, the space under a cap) stay open.
  const dS = new Float32Array(N);
  for (let i = 0; i < N; i++) dS[i] = occ[i] ? 0 : 1e20;
  edt3(dS, nx, ny, nz);
  const r2 = closeR * closeR;
  const out = new Uint8Array(N);
  const queue = new Int32Array(N);
  let qh = 0, qt = 0;
  const seed = (i: number) => { if (dS[i] > r2 && !out[i]) { out[i] = 1; queue[qt++] = i; } };
  for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    if (i === 0 || j === 0 || k === 0 || i === nx - 1 || j === ny - 1 || k === nz - 1) seed((k * ny + j) * nx + i);
  }
  while (qh < qt) {
    const c = queue[qh++];
    const i = c % nx, j = Math.floor(c / nx) % ny, k = Math.floor(c / (nx * ny));
    if (i > 0) seed(c - 1);
    if (i < nx - 1) seed(c + 1);
    if (j > 0) seed(c - nx);
    if (j < ny - 1) seed(c + nx);
    if (k > 0) seed(c - nx * ny);
    if (k < nz - 1) seed(c + nx * ny);
  }
  // distance to that outside, then the solid and the signed distance to its surface
  const dO = dS; // reused
  for (let i = 0; i < N; i++) dO[i] = out[i] ? 0 : 1e20;
  edt3(dO, nx, ny, nz);
  const solid = occ; // reused
  for (let i = 0; i < N; i++) solid[i] = dO[i] > r2 ? 1 : 0;
  const dOut = new Float32Array(N), dIn = dO; // reused
  for (let i = 0; i < N; i++) { dOut[i] = solid[i] ? 0 : 1e20; dIn[i] = solid[i] ? 1e20 : 0; }
  edt3(dOut, nx, ny, nz);
  edt3(dIn, nx, ny, nz);
  const d = dOut; // reused: the result
  for (let i = 0; i < N; i++) d[i] = solid[i] ? -(Math.sqrt(dIn[i]) - 0.5) * vs : (Math.sqrt(dOut[i]) - 0.5) * vs;
  let br = 0;
  for (const cx of [ox, ox + nx * vs]) for (const cy of [oy, oy + ny * vs]) for (const cz of [oz, oz + nz * vs]) br = Math.max(br, Math.hypot(cx, cy, cz));
  return { nx, ny, nz, vs, ox, oy, oz, d, br };
}
