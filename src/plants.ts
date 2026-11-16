// Procedural flora / rock meshes, generated once at startup (deterministic per seed).
// Vertex: pos f32x3 | normal snorm8x4 | uv unorm16x2 | attr unorm8x4 (part, flex, ao, grad) = 24 bytes.
import { rng } from './math';

export const PART = { BARK: 0, LEAF: 1, CAP: 2, GILL: 3, PETAL: 4, CRYSTAL: 5, BULB: 6, ROCK: 7, FROND: 8, STEM: 9 } as const;
export const STRIDE = 24;

type R = () => number;
type V3 = [number, number, number];

const add = (a: V3, b: V3): V3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const mul = (a: V3, s: number): V3 => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const len = (a: V3) => Math.hypot(a[0], a[1], a[2]);
const norm = (a: V3): V3 => { const l = len(a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
const lerp3 = (a: V3, b: V3, t: number): V3 => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];

export class MeshBuilder {
  p: number[] = []; n: number[] = []; uv: number[] = []; at: number[] = [];
  opaque: number[] = []; cutout: number[] = [];
  get count() { return this.p.length / 3; }
  v(pos: V3, nrm: V3, u: number, w: number, part: number, flex: number, ao: number, grad: number): number {
    const nn = norm(nrm);
    this.p.push(pos[0], pos[1], pos[2]);
    this.n.push(nn[0], nn[1], nn[2]);
    this.uv.push(u, w);
    this.at.push(part, flex, ao, grad);
    return this.count - 1;
  }
  tri(a: number, b: number, c: number, cut = false) { (cut ? this.cutout : this.opaque).push(a, b, c); }
  quad(a: number, b: number, c: number, d: number, cut = false) { this.tri(a, b, c, cut); this.tri(a, c, d, cut); }

  // Generalised cylinder along a polyline with per-ring radius. Returns ring vertex start.
  tube(path: V3[], radius: number[], sides: number, part: number, flexOf: (t: number) => number, aoOf: (t: number) => number, cap = false, vScale = 1) {
    const n = path.length;
    let tan = norm(sub(path[1], path[0]));
    let ref: V3 = Math.abs(tan[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
    let nx = norm(cross(ref, tan));
    let ny = cross(tan, nx);
    const start = this.count;
    let vacc = 0;
    for (let i = 0; i < n; i++) {
      if (i > 0) {
        const t2 = norm(sub(path[Math.min(i + 1, n - 1)], path[i - 1]));
        // parallel transport
        const ax = cross(tan, t2);
        const s = len(ax);
        if (s > 1e-5) {
          const k = norm(ax), c = dot(tan, t2), ang = Math.atan2(s, c);
          const rot = (v: V3): V3 => {
            const cs = Math.cos(ang), sn = Math.sin(ang);
            return add(add(mul(v, cs), mul(cross(k, v), sn)), mul(k, dot(k, v) * (1 - cs)));
          };
          nx = rot(nx); ny = rot(ny);
        }
        tan = t2;
        vacc += len(sub(path[i], path[i - 1]));
      }
      const t = i / (n - 1);
      for (let j = 0; j <= sides; j++) {
        const a = (j / sides) * Math.PI * 2;
        const dir = add(mul(nx, Math.cos(a)), mul(ny, Math.sin(a)));
        this.v(add(path[i], mul(dir, radius[i])), dir, j / sides, (vacc * vScale) % 64, part, flexOf(t), aoOf(t), t);
      }
    }
    for (let i = 0; i < n - 1; i++) for (let j = 0; j < sides; j++) {
      const a = start + i * (sides + 1) + j, b = a + 1, c = a + sides + 1, d = c + 1;
      this.tri(a, c, b); this.tri(b, c, d);
    }
    if (cap) {
      const tip = this.v(add(path[n - 1], mul(tan, radius[n - 1] * 0.6)), tan, 0.5, 0, part, flexOf(1), aoOf(1), 1);
      const base = start + (n - 1) * (sides + 1);
      for (let j = 0; j < sides; j++) this.tri(base + j, tip, base + j + 1);
    }
    return start;
  }

  // Surface of revolution around +Y: profile [r, y] pairs, with explicit normals derived from the profile.
  lathe(profile: [number, number][], sides: number, center: V3, part: number, flexOf: (t: number) => number, ao: (t: number) => number, flip = false, tilt: V3 = [0, 1, 0]) {
    const start = this.count;
    const up = norm(tilt);
    const ref: V3 = Math.abs(up[1]) < 0.95 ? [0, 1, 0] : [1, 0, 0];
    const ex = norm(cross(ref, up)), ez = cross(up, ex);
    const m = profile.length;
    for (let i = 0; i < m; i++) {
      const [r, y] = profile[i];
      const pr = profile[Math.max(i - 1, 0)], nx = profile[Math.min(i + 1, m - 1)];
      const dr = nx[0] - pr[0], dy = nx[1] - pr[1];
      // profile normal (outward): (dy, -dr)
      let pn: [number, number] = [dy, -dr];
      const l = Math.hypot(pn[0], pn[1]) || 1;
      pn = [pn[0] / l, pn[1] / l];
      if (flip) pn = [-pn[0], -pn[1]];
      const t = i / (m - 1);
      for (let j = 0; j <= sides; j++) {
        const a = (j / sides) * Math.PI * 2;
        const ca = Math.cos(a), sa = Math.sin(a);
        const radial = add(mul(ex, ca), mul(ez, sa));
        const pos = add(center, add(mul(radial, r), mul(up, y)));
        const nrm = add(mul(radial, pn[0]), mul(up, pn[1]));
        this.v(pos, nrm, j / sides, t, part, flexOf(t), ao(t), t);
      }
    }
    for (let i = 0; i < m - 1; i++) for (let j = 0; j < sides; j++) {
      const a = start + i * (sides + 1) + j, b = a + 1, c = a + sides + 1, d = c + 1;
      if (flip) { this.tri(a, b, c); this.tri(b, d, c); } else { this.tri(a, c, b); this.tri(b, c, d); }
    }
  }

  // Curved strip (leaf / petal / frond) along a spine with width profile. Double-sided via cull none.
  strip(spine: V3[], side: V3, width: (t: number) => number, part: number, flexOf: (t: number) => number, cut: boolean, bend: (t: number) => V3, uvScale = 1) {
    const n = spine.length;
    const start = this.count;
    for (let i = 0; i < n; i++) {
      const t = i / (n - 1);
      const tan = norm(sub(spine[Math.min(i + 1, n - 1)], spine[Math.max(i - 1, 0)]));
      const w = width(t);
      const s = norm(side);
      const nrm = norm(add(cross(s, tan), bend(t)));
      const cup = mul(nrm, w * 0.18);
      this.v(add(add(spine[i], mul(s, -w)), cup), nrm, 0, t * uvScale, part, flexOf(t), 0.6 + 0.4 * t, t);
      this.v(spine[i], nrm, 0.5, t * uvScale, part, flexOf(t), 0.75 + 0.25 * t, t);
      this.v(add(add(spine[i], mul(s, w)), cup), nrm, 1, t * uvScale, part, flexOf(t), 0.6 + 0.4 * t, t);
    }
    for (let i = 0; i < n - 1; i++) {
      const a = start + i * 3;
      this.quad(a, a + 3, a + 4, a + 1, cut);
      this.quad(a + 1, a + 4, a + 5, a + 2, cut);
    }
  }
}

// -------------------------------------------------------------------------------------------
// noise for rock displacement (startup only)
function vnoise3(x: number, y: number, z: number, seed: number): number {
  const h = (i: number, j: number, k: number) => {
    let n = (i * 374761393 + j * 668265263 + k * 2147483647 + seed * 144665) | 0;
    n = (n ^ (n >>> 13)) * 1274126177;
    return ((n ^ (n >>> 16)) >>> 0) / 4294967296;
  };
  const xi = Math.floor(x), yi = Math.floor(y), zi = Math.floor(z);
  const fx = x - xi, fy = y - yi, fz = z - zi;
  const s = (t: number) => t * t * (3 - 2 * t);
  const ux = s(fx), uy = s(fy), uz = s(fz);
  let r = 0;
  for (let c = 0; c < 8; c++) {
    const dx = c & 1, dy = (c >> 1) & 1, dz = (c >> 2) & 1;
    r += h(xi + dx, yi + dy, zi + dz) * (dx ? ux : 1 - ux) * (dy ? uy : 1 - uy) * (dz ? uz : 1 - uz);
  }
  return r * 2 - 1;
}
function fbm3(x: number, y: number, z: number, seed: number, oct: number) {
  let s = 0, a = 0.5, f = 1;
  for (let i = 0; i < oct; i++) { s += vnoise3(x * f, y * f, z * f, seed + i * 31) * a; a *= 0.5; f *= 2.03; }
  return s;
}

function icosphere(subdiv: number): { v: V3[]; f: number[] } {
  const t = (1 + Math.sqrt(5)) / 2;
  let v: V3[] = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]].map((p) => norm(p as V3));
  let f = [0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11, 1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1, 8, 3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9, 4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1];
  for (let s = 0; s < subdiv; s++) {
    const mid = new Map<number, number>();
    const m = (a: number, b: number) => {
      const k = Math.min(a, b) * 100000 + Math.max(a, b);
      let i = mid.get(k);
      if (i === undefined) { i = v.length; v.push(norm(lerp3(v[a], v[b], 0.5))); mid.set(k, i); }
      return i;
    };
    const nf: number[] = [];
    for (let i = 0; i < f.length; i += 3) {
      const a = f[i], b = f[i + 1], c = f[i + 2];
      const ab = m(a, b), bc = m(b, c), ca = m(c, a);
      nf.push(a, ab, ca, b, bc, ab, c, ca, bc, ab, bc, ca);
    }
    f = nf;
  }
  return { v, f };
}

// Displaced sphere (boulders, foliage blobs). Flat-ish bottom for rocks.
function blob(mb: MeshBuilder, center: V3, radius: V3, subdiv: number, seed: number, part: number, rough: number, flatBottom: boolean, flex: number, ao: number) {
  const { v, f } = icosphere(subdiv);
  const pos: V3[] = v.map((d) => {
    const n = fbm3(d[0] * 1.6 + seed, d[1] * 1.6, d[2] * 1.6, seed, 4);
    let r = 1 + n * rough;
    let p: V3 = [d[0] * radius[0] * r, d[1] * radius[1] * r, d[2] * radius[2] * r];
    if (flatBottom && p[1] < -radius[1] * 0.35) p = [p[0], -radius[1] * 0.35 + (p[1] + radius[1] * 0.35) * 0.15, p[2]];
    return add(center, p);
  });
  // smooth vertex normals from faces
  const nr: V3[] = v.map(() => [0, 0, 0]);
  for (let i = 0; i < f.length; i += 3) {
    const a = pos[f[i]], b = pos[f[i + 1]], c = pos[f[i + 2]];
    const fn = cross(sub(b, a), sub(c, a));
    for (let k = 0; k < 3; k++) { const q = nr[f[i + k]]; q[0] += fn[0]; q[1] += fn[1]; q[2] += fn[2]; }
  }
  const base = mb.count;
  for (let i = 0; i < v.length; i++) {
    const d = v[i];
    const u = Math.atan2(d[2], d[0]) / (2 * Math.PI) + 0.5;
    const occl = ao * (0.55 + 0.45 * (d[1] * 0.5 + 0.5));
    mb.v(pos[i], nr[i], u, d[1] * 0.5 + 0.5, part, flex, occl, d[1] * 0.5 + 0.5);
  }
  for (let i = 0; i < f.length; i += 3) mb.tri(base + f[i], base + f[i + 1], base + f[i + 2]);
}

// -------------------------------------------------------------------------------------------
// Species generators. lod: 0 near, 1 mid, 2 far.

function curvePath(base: V3, dir: V3, length: number, segs: number, bend: V3, r: R, wiggle: number): V3[] {
  const pts: V3[] = [base];
  let d = norm(dir);
  let p = base;
  const step = length / segs;
  for (let i = 1; i <= segs; i++) {
    d = norm(add(d, add(mul(bend, step / length), [(r() - 0.5) * wiggle, (r() - 0.5) * wiggle * 0.3, (r() - 0.5) * wiggle])));
    p = add(p, mul(d, step));
    pts.push(p);
  }
  return pts;
}

function foliageClump(mb: MeshBuilder, c: V3, rad: number, cards: number, cardSize: number, r: R, lod: number) {
  if (lod >= 2) { blob(mb, c, [rad * 0.95, rad * 0.75, rad * 0.95], 1, Math.floor(r() * 1000), PART.LEAF, 0.35, false, 1, 0.9); return; }
  for (let i = 0; i < cards; i++) {
    // points biased to the outer shell, flattened underside
    const th = r() * Math.PI * 2, ph = Math.acos(1 - 2 * Math.pow(r(), 0.8));
    const rr = rad * (0.45 + 0.55 * Math.sqrt(r()));
    const off: V3 = [Math.sin(ph) * Math.cos(th) * rr, Math.cos(ph) * rr * 0.72, Math.sin(ph) * Math.sin(th) * rr];
    const p = add(c, off);
    const sn = norm(add(off, [0, rad * 0.35, 0]));
    // random card orientation, loosely facing outward
    const t1 = norm(cross(sn, [r() - 0.5, r() - 0.5, r() - 0.5]));
    const t2 = norm(cross(sn, t1));
    const s = cardSize * (0.75 + r() * 0.5);
    const a = add(p, add(mul(t1, -s), mul(t2, -s)));
    const b = add(p, add(mul(t1, s), mul(t2, -s)));
    const cc = add(p, add(mul(t1, s), mul(t2, s)));
    const d = add(p, add(mul(t1, -s), mul(t2, s)));
    const shell = rr / rad;
    const ao = 0.45 + 0.55 * shell * (0.6 + 0.4 * (off[1] / rad * 0.5 + 0.5));
    const i0 = mb.v(a, sn, 0, 0, PART.LEAF, 1, ao, shell);
    const i1 = mb.v(b, sn, 1, 0, PART.LEAF, 1, ao, shell);
    const i2 = mb.v(cc, sn, 1, 1, PART.LEAF, 1, ao, shell);
    const i3 = mb.v(d, sn, 0, 1, PART.LEAF, 1, ao, shell);
    mb.quad(i0, i1, i2, i3, true);
  }
}

export function genTree(seed: number, variant: number, lod: number): MeshBuilder {
  const r = rng(seed * 7919 + variant * 104729);
  const mb = new MeshBuilder();
  const sides = lod === 0 ? 9 : lod === 1 ? 6 : 4;
  const segs = lod === 0 ? 12 : lod === 1 ? 7 : 4;
  if (variant === 0) {
    // "Puffcrown": twisting trunk, upswept branches carrying clustered leaf crowns
    const H = 11 + r() * 3;
    const trunk = curvePath([0, -0.6, 0], [0.08, 1, 0.03], H, segs, [0.5, 0.0, -0.3], r, 0.08);
    mb.tube(trunk, trunk.map((_, i) => 0.55 * Math.pow(1 - i / trunk.length, 1.3) + 0.12 + (i === 0 ? 0.25 : 0)), sides, PART.BARK, (t) => t * t * 0.5, (t) => 0.55 + 0.45 * t);
    const nb = lod === 2 ? 4 : 6;
    for (let b = 0; b < nb; b++) {
      const t0 = 0.5 + 0.45 * (b / nb) + r() * 0.05;
      const bi = Math.floor(t0 * (trunk.length - 1));
      const base = trunk[bi];
      const ang = b * 2.4 + r();
      const dir: V3 = [Math.cos(ang), 0.75 + r() * 0.5, Math.sin(ang)];
      const L = 3.2 + r() * 2.2;
      const br = curvePath(base, dir, L, Math.max(3, segs / 2), [0, 0.8, 0], r, 0.15);
      mb.tube(br, br.map((_, i) => 0.17 * (1 - i / br.length) + 0.05), Math.max(3, sides - 3), PART.BARK, (t) => 0.4 + t * 0.6, () => 0.7);
      foliageClump(mb, add(br[br.length - 1], [0, 0.4, 0]), 1.9 + r() * 0.8, lod === 0 ? 34 : 12, lod === 0 ? 0.62 : 0.95, r, lod);
    }
    foliageClump(mb, add(trunk[trunk.length - 1], [0, 0.8, 0]), 2.4, lod === 0 ? 42 : 14, lod === 0 ? 0.66 : 1.0, r, lod);
  } else if (variant === 1) {
    // "Lantern palm": tall arched trunk with drooping fronds and glowing fruit bulbs
    const H = 13 + r() * 4;
    const trunk = curvePath([0, -0.6, 0], [0.2, 1, 0], H, segs, [1.6, -0.2, 0.4], r, 0.04);
    mb.tube(trunk, trunk.map((_, i) => 0.42 * Math.pow(1 - i / trunk.length, 0.8) + 0.1 + (i === 0 ? 0.2 : 0)), sides, PART.BARK, (t) => t * t * 0.7, (t) => 0.6 + 0.4 * t);
    const top = trunk[trunk.length - 1];
    const nf = lod === 0 ? 9 : lod === 1 ? 7 : 5;
    for (let i = 0; i < nf; i++) {
      const ang = (i / nf) * Math.PI * 2 + r() * 0.3;
      const out: V3 = [Math.cos(ang), 0.55, Math.sin(ang)];
      const L = 5 + r() * 2;
      const spine = curvePath(top, out, L, lod === 0 ? 8 : 5, [0, -2.4, 0], r, 0.02);
      const side = norm(cross(out, [0, 1, 0]));
      mb.strip(spine, side, (t) => (0.15 + Math.sin(Math.PI * Math.min(t * 1.15, 1)) * 0.75) * (1 - t * 0.3), PART.FROND, (t) => 0.6 + t * 0.4, true, () => [0, 0.4, 0], 3);
    }
    if (lod < 2) {
      const nbulb = lod === 0 ? 7 : 4;
      for (let i = 0; i < nbulb; i++) {
        const ang = r() * Math.PI * 2;
        const c = add(top, [Math.cos(ang) * 0.6, -0.6 - r() * 0.8, Math.sin(ang) * 0.6]);
        const rad = 0.28 + r() * 0.18;
        mb.lathe([[0, -rad], [rad * 0.7, -rad * 0.7], [rad, 0], [rad * 0.7, rad * 0.7], [0.05, rad]], lod === 0 ? 10 : 6, c, PART.BULB, () => 0.9, () => 1);
      }
    }
  } else {
    // "Coral spire": branching coral-like tree with glowing nodes at tips
    const H = 7 + r() * 3;
    const trunk = curvePath([0, -0.5, 0], [0, 1, 0], H * 0.55, Math.max(3, segs / 2), [0.3, 0, 0.2], r, 0.12);
    mb.tube(trunk, trunk.map((_, i) => 0.4 * (1 - i / trunk.length) + 0.14), sides, PART.STEM, (t) => t * 0.4, (t) => 0.55 + 0.45 * t);
    const grow = (base: V3, dir: V3, L: number, rad: number, depth: number) => {
      const p = curvePath(base, dir, L, Math.max(2, segs / 3), [0, 0.6, 0], r, 0.25);
      mb.tube(p, p.map((_, i) => rad * (1 - 0.6 * i / p.length)), Math.max(3, sides - 4), PART.STEM, (t) => 0.4 + 0.3 * depth + t * 0.3, () => 0.75);
      const end = p[p.length - 1];
      if (depth < (lod === 0 ? 3 : 2)) {
        const n = 2 + (r() < 0.4 ? 1 : 0);
        for (let i = 0; i < n; i++) {
          const ang = r() * Math.PI * 2;
          grow(end, norm(add(dir, [Math.cos(ang) * 0.9, 0.35, Math.sin(ang) * 0.9])), L * 0.68, rad * 0.62, depth + 1);
        }
      } else if (lod < 2) {
        const rr = rad * 2.6 + 0.12;
        mb.lathe([[0, -rr], [rr, 0], [rr * 0.6, rr * 0.8], [0, rr]], lod === 0 ? 8 : 5, end, PART.BULB, () => 0.9, () => 1);
      }
    };
    const top = trunk[trunk.length - 1];
    for (let i = 0; i < 3; i++) {
      const ang = i * 2.1 + r();
      grow(top, [Math.cos(ang) * 0.6, 1, Math.sin(ang) * 0.6], H * 0.35, 0.22, 0);
    }
  }
  return mb;
}

export function genMushroom(seed: number, variant: number, lod: number): MeshBuilder {
  const r = rng(seed * 4513 + variant * 7907);
  const mb = new MeshBuilder();
  const sides = lod === 0 ? 20 : lod === 1 ? 10 : 6;
  const make = (base: V3, H: number, capR: number, shape: number, lean: V3) => {
    const segs = lod === 0 ? 10 : lod === 1 ? 5 : 3;
    const stem = curvePath(base, add([0, 1, 0], lean), H, segs, [lean[0] * 2, 0, lean[2] * 2], r, 0.03);
    const rb = capR * 0.16;
    mb.tube(stem, stem.map((_, i) => { const t = i / (stem.length - 1); return rb * (1 + 1.4 * Math.pow(1 - t, 4)) * (1 - 0.25 * t); }), Math.max(5, sides / 2), PART.STEM, (t) => t * 0.35, (t) => 0.5 + 0.5 * t);
    const top = stem[stem.length - 1];
    const tilt = norm(sub(top, stem[stem.length - 2]));
    const n = lod === 0 ? 12 : lod === 1 ? 7 : 4;
    const prof: [number, number][] = [];
    const under: [number, number][] = [];
    for (let i = 0; i <= n; i++) {
      const t = i / n;
      let rr: number, y: number;
      if (shape === 0) { rr = capR * Math.sin(t * Math.PI * 0.5); y = capR * 0.55 * Math.cos(t * Math.PI * 0.5) - capR * 0.12 * t * t; }
      else if (shape === 1) { rr = capR * t; y = capR * 0.22 * (1 - t * t) + capR * 0.1 * Math.pow(t, 6); }
      else { rr = capR * 0.8 * Math.sin(t * Math.PI * 0.55); y = capR * 0.9 * Math.cos(t * Math.PI * 0.5) - capR * 0.35 * t; }
      prof.push([rr, y]);
    }
    for (let i = 0; i <= Math.max(2, n >> 1); i++) {
      const t = i / Math.max(2, n >> 1);
      under.push([capR * (0.12 + 0.86 * t), prof[prof.length - 1][1] * t * 0.9 - capR * 0.04 * (1 - t)]);
    }
    mb.lathe(prof, sides, add(top, mul(tilt, -0.05)), PART.CAP, () => 0.25, (t) => 0.8 + 0.2 * (1 - t), false, tilt);
    mb.lathe(under, sides, add(top, mul(tilt, -0.05)), PART.GILL, () => 0.25, () => 0.6, true, tilt);
  };
  if (variant === 0) {
    make([0, -0.3, 0], 8 + r() * 3, 4.5 + r() * 1.5, 1, [0.08, 0, 0.05]);
  } else if (variant === 1) {
    make([0, -0.3, 0], 10 + r() * 3, 3.2 + r() * 0.8, 0, [0.12, 0, -0.06]);
  } else {
    make([0, -0.3, 0], 6.5, 3.4, 2, [0.05, 0, 0.1]);
    if (lod < 2) {
      for (let i = 0; i < 3; i++) {
        const a = i * 2.1 + r();
        make([Math.cos(a) * 3.2, -0.2, Math.sin(a) * 3.2], 2.4 + r() * 2, 1.2 + r() * 0.7, Math.floor(r() * 3), [Math.cos(a) * 0.25, 0, Math.sin(a) * 0.25]);
      }
    }
  }
  return mb;
}

export function genCrystal(seed: number, variant: number, lod: number): MeshBuilder {
  const r = rng(seed * 3301 + variant * 997);
  const mb = new MeshBuilder();
  const count = (variant === 2 ? 5 : 9) - lod * 3;
  for (let i = 0; i < Math.max(2, count); i++) {
    const big = i === 0;
    const H = (variant === 2 ? 7 : 2.2) * (big ? 1.4 : 0.5 + r() * 0.7);
    const rad = H * (0.11 + r() * 0.05);
    const ang = r() * Math.PI * 2;
    const tiltA = big ? r() * 0.15 : 0.25 + r() * 0.55;
    const axis = norm([Math.cos(ang) * Math.sin(tiltA), Math.cos(tiltA), Math.sin(ang) * Math.sin(tiltA)]);
    const base: V3 = [Math.cos(ang) * rad * 1.5 * (big ? 0 : 1), -0.3, Math.sin(ang) * rad * 1.5 * (big ? 0 : 1)];
    const ref: V3 = Math.abs(axis[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
    const ex = norm(cross(ref, axis)), ez = cross(axis, ex);
    const ring = (h: number, rr: number) => {
      const pts: V3[] = [];
      for (let k = 0; k < 6; k++) {
        const a = (k / 6) * Math.PI * 2 + 0.3;
        pts.push(add(add(base, mul(axis, h)), add(mul(ex, Math.cos(a) * rr), mul(ez, Math.sin(a) * rr))));
      }
      return pts;
    };
    const b0 = ring(0, rad * 0.9), b1 = ring(H * 0.78, rad), tip = add(base, mul(axis, H));
    for (let k = 0; k < 6; k++) {
      const k1 = (k + 1) % 6;
      const fn = norm(cross(sub(b1[k], b0[k]), sub(b0[k1], b0[k])));
      const g0 = 0, g1 = 0.78;
      const a = mb.v(b0[k], fn, 0, 0, PART.CRYSTAL, 0, 0.55, g0);
      const b = mb.v(b0[k1], fn, 1, 0, PART.CRYSTAL, 0, 0.55, g0);
      const c = mb.v(b1[k1], fn, 1, 0.78, PART.CRYSTAL, 0, 1, g1);
      const d = mb.v(b1[k], fn, 0, 0.78, PART.CRYSTAL, 0, 1, g1);
      mb.tri(a, d, b); mb.tri(b, d, c);
      const tn = norm(cross(sub(tip, b1[k]), sub(b1[k1], b1[k])));
      const e = mb.v(b1[k], tn, 0, 0.78, PART.CRYSTAL, 0, 1, g1);
      const f = mb.v(b1[k1], tn, 1, 0.78, PART.CRYSTAL, 0, 1, g1);
      const g = mb.v(tip, tn, 0.5, 1, PART.CRYSTAL, 0, 1, 1);
      mb.tri(e, g, f);
    }
  }
  return mb;
}

export function genFlower(seed: number, variant: number, lod: number): MeshBuilder {
  const r = rng(seed * 2003 + variant * 409);
  const mb = new MeshBuilder();
  const H = variant === 1 ? 4.5 + r() * 1.5 : 2.4 + r();
  const segs = lod === 0 ? 10 : lod === 1 ? 5 : 3;
  const stalk = curvePath([0, -0.2, 0], [0.1, 1, 0], H, segs, [0.6, 0, 0.3], r, 0.05);
  mb.tube(stalk, stalk.map((_, i) => 0.09 * (1 - 0.4 * i / stalk.length) + 0.03), lod === 0 ? 7 : 4, PART.STEM, (t) => t * t, (t) => 0.6 + 0.4 * t);
  const top = stalk[stalk.length - 1];
  const up = norm(sub(top, stalk[stalk.length - 2]));
  const np = variant === 2 ? 11 : 7;
  const petalL = variant === 1 ? 2.0 : 1.3;
  for (let i = 0; i < np; i++) {
    const ang = (i / np) * Math.PI * 2 + r() * 0.2;
    const out: V3 = norm(add(mul(up, 0.55), [Math.cos(ang), 0, Math.sin(ang)]));
    const spine = curvePath(top, out, petalL * (0.85 + r() * 0.3), lod === 0 ? 7 : 4, [0, variant === 2 ? 1.5 : -1.1, 0], r, 0.02);
    const side = norm(cross(out, up));
    mb.strip(spine, side, (t) => Math.sin(Math.PI * Math.pow(t, 0.7)) * petalL * 0.34, PART.PETAL, (t) => 0.6 + 0.4 * t, false, () => mul(up, 0.3));
  }
  if (lod < 2) {
    const ns = lod === 0 ? 9 : 4;
    for (let i = 0; i < ns; i++) {
      const ang = r() * Math.PI * 2;
      const p = curvePath(top, norm(add(up, [Math.cos(ang) * 0.5, 0, Math.sin(ang) * 0.5])), 0.5 + r() * 0.4, 3, [0, 0.4, 0], r, 0.1);
      mb.tube(p, p.map(() => 0.018), 3, PART.STEM, () => 0.8, () => 1);
      const e = p[p.length - 1];
      mb.lathe([[0, -0.06], [0.06, 0], [0, 0.06]], 5, e, PART.BULB, () => 0.8, () => 1);
    }
    // base leaves
    for (let i = 0; i < 4; i++) {
      const ang = i * 1.6 + r();
      const out: V3 = [Math.cos(ang), 0.5, Math.sin(ang)];
      const spine = curvePath([0, 0, 0], out, 1.4 + r() * 0.6, lod === 0 ? 6 : 3, [0, -1.6, 0], r, 0.05);
      mb.strip(spine, norm(cross(out, [0, 1, 0])), (t) => Math.sin(Math.PI * t) * 0.32, PART.FROND, (t) => 0.3 + 0.5 * t, true, () => [0, 0.5, 0]);
    }
  }
  return mb;
}

export function genGlowpod(seed: number, variant: number, lod: number): MeshBuilder {
  const r = rng(seed * 6007 + variant * 31);
  const mb = new MeshBuilder();
  const n = (variant === 0 ? 7 : 5) - lod * 2;
  for (let i = 0; i < Math.max(2, n); i++) {
    const ang = r() * Math.PI * 2;
    const d = r() * 0.35;
    const base: V3 = [Math.cos(ang) * d, -0.1, Math.sin(ang) * d];
    const H = (variant === 0 ? 0.6 : 1.3) * (0.5 + r() * 0.8);
    const p = curvePath(base, [Math.cos(ang) * 0.3, 1, Math.sin(ang) * 0.3], H, lod === 0 ? 5 : 3, [Math.cos(ang) * 0.4, 0, Math.sin(ang) * 0.4], r, 0.1);
    mb.tube(p, p.map((_, k) => 0.03 * (1 - 0.5 * k / p.length) + 0.012), lod === 0 ? 5 : 3, PART.STEM, (t) => t, () => 0.8);
    const e = p[p.length - 1];
    const rr = 0.07 + r() * 0.08;
    mb.lathe([[0, -rr * 1.2], [rr * 0.8, -rr * 0.6], [rr, rr * 0.3], [rr * 0.5, rr * 1.1], [0, rr * 1.3]], lod === 0 ? 8 : 5, e, PART.BULB, () => 1, () => 1);
  }
  return mb;
}

export function genShrub(seed: number, variant: number, lod: number): MeshBuilder {
  const r = rng(seed * 1709 + variant * 113);
  const mb = new MeshBuilder();
  const nf = (variant === 0 ? 12 : 8) - lod * 3;
  for (let i = 0; i < nf; i++) {
    const ang = (i / nf) * Math.PI * 2 + r() * 0.4;
    const el = 0.6 + r() * 0.7;
    const out: V3 = [Math.cos(ang), el, Math.sin(ang)];
    const L = (variant === 0 ? 1.3 : 0.9) * (0.7 + r() * 0.5);
    const spine = curvePath([0, -0.05, 0], out, L, lod === 0 ? 7 : 4, [0, variant === 1 ? 0.8 : -1.8, 0], r, 0.05);
    mb.strip(spine, norm(cross(out, [0, 1, 0])), (t) => Math.sin(Math.PI * Math.pow(t, 0.8)) * L * 0.2, PART.FROND, (t) => 0.3 + t * 0.7, true, () => [0, 0.6, 0], 4);
  }
  return mb;
}

export function genBoulder(seed: number, variant: number, lod: number): MeshBuilder {
  const mb = new MeshBuilder();
  const sd = lod === 0 ? 3 : lod === 1 ? 2 : 1;
  const sc: V3 = variant === 0 ? [1.6, 1.0, 1.3] : variant === 1 ? [1.0, 1.6, 0.9] : [2.4, 0.9, 1.7];
  blob(mb, [0, 0.2, 0], sc, sd, seed * 13 + variant * 7, PART.ROCK, 0.42, true, 0, 1);
  return mb;
}
