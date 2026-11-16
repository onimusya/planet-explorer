// Explicit placements: landmarks (arches, floating islands, rock pillars) and hero flora that frame
// the destinations. Built once on the CPU after terrain generation from the CPU height grid; the GPU
// cull re-grounds each instance on the exact heightmap (y offset is relative to the ground).
import { rng } from './math';
import { SEED, VISTA_P, VISTA_DIR, HIGHLANDS_C, VALLEY_A, VALLEY_B, WATER_Y, WORLD_L } from './world';
import { SP } from './vegetation';

type H = (x: number, z: number) => number;
const D2R = Math.PI / 180;
const ARCH_W = [62, 34, 120];

export function buildMarks(h: H): Float32Array {
  const out: number[] = [];
  const placed: [number, number, number][] = []; // x, z, keep-out radius
  const wrap = (v: number) => ((v % WORLD_L) + WORLD_L) % WORLD_L;
  const dist2 = (ax: number, az: number, bx: number, bz: number) => {
    let dx = Math.abs(ax - bx) % WORLD_L, dz = Math.abs(az - bz) % WORLD_L;
    dx = Math.min(dx, WORLD_L - dx); dz = Math.min(dz, WORLD_L - dz);
    return Math.hypot(dx, dz);
  };
  const free = (x: number, z: number, r: number) => placed.every(([px, pz, pr]) => dist2(x, z, px, pz) > r + pr);
  const slope = (x: number, z: number, d = 16) => Math.hypot(h(x + d, z) - h(x - d, z), h(x, z + d) - h(x, z - d)) / (2 * d);
  // local relief: how far the point sits below the highest ground within r (canyon floors > 0)
  const relief = (x: number, z: number, r: number) => {
    let m = -1e9;
    for (let i = 0; i < 12; i++) { const a = (i / 12) * Math.PI * 2; m = Math.max(m, h(x + Math.cos(a) * r, z + Math.sin(a) * r)); }
    return m - h(x, z);
  };
  const push = (x: number, z: number, yOff: number, scale: number, yaw: number, seed: number, sp: number, variant: number, keep: number) => {
    out.push(wrap(x), wrap(z), yOff, scale, yaw, seed, sp, variant);
    placed.push([x, z, keep]);
  };
  // yaw so that local +Z faces direction (dx, dz) (matches the vertex shader rotation)
  const yawTo = (dx: number, dz: number) => Math.atan2(dx, dz);
  const arch = (x: number, z: number, yaw: number, scale: number, variant: number, seed: number, maxStep = 0.45) => {
    const hw = (ARCH_W[variant] / 2) * scale, c = Math.cos(yaw), s = Math.sin(yaw);
    const hc = h(x, z), ha = h(x - hw * c, z + hw * s), hb = h(x + hw * c, z - hw * s);
    if (Math.abs(ha - hb) > 14 * scale * maxStep * 2 || Math.min(ha, hb) < WATER_Y + 1) return false;
    push(x, z, Math.min(ha, hb, hc) - hc - 1.5 * scale, scale, yaw, seed, SP.ARCH, variant, hw + 40);
    return true;
  };
  const pillar = (x: number, z: number, scale: number, variant: number, seed: number, yaw: number) => {
    let m = h(x, z);
    for (let i = 0; i < 6; i++) { const a = (i / 6) * Math.PI * 2; m = Math.min(m, h(x + Math.cos(a) * 9 * scale, z + Math.sin(a) * 9 * scale)); }
    push(x, z, m - h(x, z) - 1.2 * scale, scale, yaw, seed, SP.PILLAR, variant, 30 * scale + 25);
  };
  const island = (x: number, z: number, y: number, scale: number, variant: number, seed: number, yaw: number) => {
    push(x, z, y - h(x, z), scale, yaw, seed, SP.ISLAND, variant, 75 * scale + 60);
  };
  const at = (ox: number, oz: number, azDeg: number, d: number): [number, number] =>
    [ox + Math.sin(azDeg * D2R) * d, oz - Math.cos(azDeg * D2R) * d];

  // ---- A. Arrival Vista: islands hanging over the mesas, an arch past the lake, framing trees ----
  {
    const cx = VISTA_P[0] + VISTA_DIR[0] * 30, cz = VISTA_P[1] + VISTA_DIR[1] * 30;
    const cy = h(cx, cz) + 12;
    const isl = (az: number, d: number, el: number, sc: number, v: number, seed: number) => {
      const [x, z] = at(cx, cz, az, d);
      island(x, z, cy + Math.tan(el * D2R) * d, sc, v, seed, az * D2R * 1.7 + seed);
    };
    isl(2, 1400, 12.5, 2.0, 0, 1);
    isl(-15, 2600, 9.0, 1.8, 1, 2);
    isl(10, 4200, 8.5, 1.4, 2, 3);
    isl(47, 3100, 13.0, 1.4, 0, 4);
    isl(27, 5200, 9.5, 1.2, 2, 5);
    {
      const [x, z] = at(cx, cz, 37, 1350);
      arch(x, z, yawTo(cx - x, cz - z) + 0.35, 2.3, 0, 11, 1);
    }
    // framing flora on the cliff lip
    const tree = (az: number, d: number, sc: number, v: number, seed: number) => {
      const [x, z] = at(cx, cz, az, d);
      push(x, z, -0.5, sc, seed * 2.1, seed, SP.TREE, v, 6);
    };
    // the camera stands on a cliff lip: these grow on the slope just below it, crowns at eye level
    tree(-40, 13, 2.6, 1, 21);
    tree(66, 15, 3.0, 2, 22);
  }

  // ---- B. Bioluminescent Valley: a rock bridge across the floor, islands trailing vines overhead ----
  {
    const ax = VALLEY_B[0] - VALLEY_A[0], az = VALLEY_B[1] - VALLEY_A[1];
    const L = Math.hypot(ax, az), ux = ax / L, uz = az / L;
    for (const [t, sc, v, seed] of [[0.33, 1.6, 2, 31], [0.62, 1.2, 0, 32]] as const) {
      const x = VALLEY_A[0] + ax * t, z = VALLEY_A[1] + az * t;
      arch(x, z, yawTo(ux, uz), sc, v, seed, 1);
    }
    const vy = (t: number) => h(VALLEY_A[0] + ax * t, VALLEY_A[1] + az * t);
    island(VALLEY_A[0] + ax * 0.45 - uz * 140, VALLEY_A[1] + az * 0.45 + ux * 140, vy(0.45) + 260, 1.3, 1, 33, 0.4);
    island(VALLEY_A[0] + ax * 0.7 + uz * 220, VALLEY_A[1] + az * 0.7 - ux * 220, vy(0.7) + 330, 1.8, 0, 34, 2.1);
  }

  // ---- C. Highlands + world scatter ----
  const r = rng(SEED * 3 + 101);
  // pillars on canyon floors and basin margins of the highlands
  let n = 0;
  for (let k = 0; k < 4000 && n < 46; k++) {
    const a = r() * Math.PI * 2, d = Math.sqrt(r()) * 3000;
    const x = HIGHLANDS_C[0] + Math.cos(a) * d, z = HIGHLANDS_C[1] + Math.sin(a) * d;
    const y = h(x, z);
    if (y < WATER_Y + 3 || slope(x, z) > 0.22 || relief(x, z, 220) < 70) continue;
    const sc = 0.8 + r() * 0.9;
    if (!free(x, z, 30 * sc + 25)) continue;
    pillar(x, z, sc, Math.floor(r() * 3), 200 + k, r() * 6.283);
    n++;
  }
  // arches on flat mesa tops and plains
  for (let k = 0, m = 0; k < 3000 && m < 12; k++) {
    const a = r() * Math.PI * 2, d = 600 + Math.sqrt(r()) * 4200;
    const x = HIGHLANDS_C[0] + Math.cos(a) * d, z = HIGHLANDS_C[1] + Math.sin(a) * d;
    if (slope(x, z, 24) > 0.1) continue;
    const v = r() < 0.25 ? 2 : r() < 0.5 ? 1 : 0, sc = 0.9 + r() * 0.8;
    if (!free(x, z, ARCH_W[v] * sc * 0.5 + 300)) continue;
    if (arch(x, z, r() * 6.283, sc, v, 500 + k, 0.3)) m++;
  }
  // floating islands drifting over the highlands and along the way to the vista
  // (kept out of the composed vista frame, where only the hero islands above hang)
  const vx = VISTA_P[0], vz = VISTA_P[1];
  for (let k = 0, m = 0; k < 2000 && m < 16; k++) {
    const a = r() * Math.PI * 2, d = Math.sqrt(r()) * 5200;
    const x = HIGHLANDS_C[0] + Math.cos(a) * d, z = HIGHLANDS_C[1] + Math.sin(a) * d;
    const sc = 1.0 + r() * r() * 1.4;
    const vaz = Math.atan2(x - vx, -(z - vz)) / D2R;
    if (dist2(x, z, vx, vz) < 7500 && vaz > -40 && vaz < 75) continue;
    if (!free(x, z, 75 * sc + 420)) continue;
    island(x, z, Math.max(h(x, z), WATER_Y) + 220 + r() * 520 + sc * 60, sc, Math.floor(r() * 3), 600 + k, r() * 6.283);
    m++;
  }
  return new Float32Array(out);
}
