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
    // counter-clockwise seen from outside (front faces match the outward normals)
    for (let i = 0; i < n - 1; i++) for (let j = 0; j < sides; j++) {
      const a = start + i * (sides + 1) + j, b = a + 1, c = a + sides + 1, d = c + 1;
      this.tri(a, b, c); this.tri(b, d, c);
    }
    if (cap) {
      const tip = this.v(add(path[n - 1], mul(tan, radius[n - 1] * 0.6)), tan, 0.5, 0, part, flexOf(1), aoOf(1), 1);
      const base = start + (n - 1) * (sides + 1);
      for (let j = 0; j < sides; j++) this.tri(base + j, base + j + 1, tip);
    }
    return start;
  }

  // Surface of revolution around +Y: profile [r, y] pairs, with explicit normals derived from the profile.
  // The normal is the right-hand side of the profile direction ((dy, -dr): outward for bottom-to-top
  // profiles); flip selects the left-hand side (axis-to-rim cap tops). Winding always matches the normal.
  // rmod(angle, r / rRef) optionally scales the radius per angle (lobed rims).
  lathe(profile: [number, number][], sides: number, center: V3, part: number, flexOf: (t: number) => number, ao: (t: number) => number, flip = false, tilt: V3 = [0, 1, 0], rmod?: (a: number, rf: number) => number, rRef = 1) {
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
        const k = rmod ? rmod(a, r / rRef) : 1;
        const pos = add(center, add(mul(radial, r * k), mul(up, y)));
        const nrm = add(mul(radial, pn[0]), mul(up, pn[1]));
        this.v(pos, nrm, j / sides, t, part, flexOf(t), ao(t), t);
      }
    }
    for (let i = 0; i < m - 1; i++) for (let j = 0; j < sides; j++) {
      const a = start + i * (sides + 1) + j, b = a + 1, c = a + sides + 1, d = c + 1;
      if (flip) { this.tri(a, c, b); this.tri(b, c, d); } else { this.tri(a, b, c); this.tri(b, d, c); }
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
  const sides = lod === 0 ? 48 : lod === 1 ? 18 : 8;
  const make = (base: V3, H: number, capR: number, shape: number, lean: V3) => {
    const segs = lod === 0 ? 10 : lod === 1 ? 5 : 3;
    const stem = curvePath(base, add([0, 1, 0], lean), H, segs, [lean[0] * 2, 0, lean[2] * 2], r, 0.03);
    const rb = capR * 0.16;
    mb.tube(stem, stem.map((_, i) => { const t = i / (stem.length - 1); return rb * (1 + 1.4 * Math.pow(1 - t, 4)) * (1 - 0.25 * t); }), Math.max(6, sides / 3), PART.STEM, (t) => t * 0.35, (t) => 0.5 + 0.5 * t);
    const top = stem[stem.length - 1];
    const tilt = norm(sub(top, stem[stem.length - 2]));
    const n = lod === 0 ? 14 : lod === 1 ? 7 : 4;
    const prof: [number, number][] = [];
    for (let i = 0; i <= n; i++) {
      const t = i / n;
      let rr: number, y: number;
      if (shape === 0) { rr = capR * Math.sin(t * Math.PI * 0.5); y = capR * 0.6 * Math.cos(t * Math.PI * 0.5) - capR * 0.12 * t * t; }
      else if (shape === 1) { rr = capR * t; y = capR * (0.34 * (1 - t * t) + 0.07 * Math.exp(-t * t * 16)); }
      else { rr = capR * 0.8 * Math.sin(t * Math.PI * 0.55); y = capR * 0.9 * Math.cos(t * Math.PI * 0.5) - capR * 0.35 * t; }
      prof.push([rr, y]);
    }
    // rolled lip at the rim
    const [er, ey] = prof[prof.length - 1];
    if (lod < 2) { prof.push([er * 1.012, ey - capR * 0.045]); prof.push([er * 0.975, ey - capR * 0.08]); }
    const [lr, ly] = prof[prof.length - 1];
    // gills: from the stem top out to the inside of the lip, facing down
    const under: [number, number][] = [];
    const nu = Math.max(2, n >> 1);
    const y0 = prof[0][1] * 0.42;
    for (let i = 0; i <= nu; i++) {
      const t = i / nu;
      under.push([rb * 0.9 + (lr - rb * 0.9) * t, y0 + (ly - y0) * Math.pow(t, 0.85)]);
    }
    // gently lobed rim (+-6%), identical on cap and gills so the lip stays closed
    const lobes = 5 + Math.floor(r() * 4), ph = r() * 6.283;
    const rmod = (a: number, rf: number) => 1 + 0.06 * Math.sin(a * lobes + ph) * rf * rf;
    const c = add(top, mul(tilt, -0.05));
    mb.lathe(prof, sides, c, PART.CAP, () => 0.25, (t) => 0.8 + 0.2 * (1 - t), true, tilt, rmod, capR);
    mb.lathe(under, sides, c, PART.GILL, () => 0.25, () => 0.6, false, tilt, rmod, capR);
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
    // irregular hexagonal prism: per-edge radius and termination height jitter
    const rj: number[] = [], hj: number[] = [];
    for (let k = 0; k < 6; k++) { rj.push(0.78 + r() * 0.44); hj.push(0.66 + r() * 0.2); }
    const at = (k: number, h: number, rr: number): V3 => {
      const a = (k / 6) * Math.PI * 2 + 0.3;
      return add(add(base, mul(axis, h)), add(mul(ex, Math.cos(a) * rr), mul(ez, Math.sin(a) * rr)));
    };
    const b0: V3[] = [], b1: V3[] = [];
    for (let k = 0; k < 6; k++) { b0.push(at(k, 0, rad * 0.9 * rj[k])); b1.push(at(k, H * hj[k], rad * rj[k])); }
    const tip = add(add(base, mul(axis, H)), add(mul(ex, (r() - 0.5) * rad * 0.5), mul(ez, (r() - 0.5) * rad * 0.5)));
    for (let k = 0; k < 6; k++) {
      const k1 = (k + 1) % 6;
      // outward face normals, counter-clockwise from outside
      const fn = norm(cross(sub(b0[k1], b0[k]), sub(b1[k], b0[k])));
      const a = mb.v(b0[k], fn, 0, 0, PART.CRYSTAL, 0, 0.55, 0);
      const b = mb.v(b0[k1], fn, 1, 0, PART.CRYSTAL, 0, 0.55, 0);
      const c = mb.v(b1[k1], fn, 1, hj[k1], PART.CRYSTAL, 0, 1, hj[k1]);
      const d = mb.v(b1[k], fn, 0, hj[k], PART.CRYSTAL, 0, 1, hj[k]);
      mb.tri(a, b, d); mb.tri(b, c, d);
      const tn = norm(cross(sub(b1[k1], b1[k]), sub(tip, b1[k])));
      const e = mb.v(b1[k], tn, 0, hj[k], PART.CRYSTAL, 0, 1, hj[k]);
      const f = mb.v(b1[k1], tn, 1, hj[k1], PART.CRYSTAL, 0, 1, hj[k1]);
      const g = mb.v(tip, tn, 0.5, 1, PART.CRYSTAL, 0, 1, 1);
      mb.tri(e, f, g);
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

// Chiselled rock: an icosphere projected onto a random convex polytope (soft-min of cutting planes),
// with fine fBm relief. Vertex grad = convexity (worn edges bright, crevices dark).
export function rockMesh(mb: MeshBuilder, center: V3, sc: V3, subdiv: number, seed: number, planes: number, bottom: number) {
  const r = rng(seed * 7919 + 17);
  const pl: [V3, number][] = [];
  for (let i = 0; i < planes; i++) {
    const z = r() * 2 - 1, a = r() * Math.PI * 2, s = Math.sqrt(1 - z * z);
    pl.push([[s * Math.cos(a), z * 0.85, s * Math.sin(a)], 0.7 + r() * 0.28]);
  }
  pl.push([[0, -1, 0], bottom]);
  const pc = polyPiece(center, sc, subdiv, seed, pl);
  rockSurface(mb, pc.pos, pc.f, Math.cbrt(sc[0] * sc[1] * sc[2]), 18,
    (i) => 0.5 + 0.5 * Math.min(1, Math.max(0, (pc.pos[i][1] - center[1]) / sc[1] + 0.6)));
}

// Icosphere projected onto a convex polytope (soft-min over the planes' ray distances gives bevelled
// edges; a sphere bound closes open directions), with fine relief, scale and an optional basis.
interface Piece { pos: V3[]; f: number[] }
function polyPiece(center: V3, sc: V3, subdiv: number, seed: number, pl: [V3, number][], basis?: [V3, V3, V3], relief = 0.05): Piece {
  const { v, f } = icosphere(subdiv);
  const k = 0.035;
  const pn = pl.map(([n, h]) => [norm(n), h] as [V3, number]);
  const pos: V3[] = v.map((d) => {
    let acc = Math.exp(-1.18 / k);
    for (const [n, h] of pn) {
      const c = dot(d, n);
      if (c > 1e-3) acc += Math.exp(-(h / c) / k);
    }
    let rad = -k * Math.log(acc);
    rad *= 1 + relief * fbm3(d[0] * 3.1 + seed, d[1] * 3.1, d[2] * 3.1, seed + 5, 4);
    const l: V3 = [d[0] * rad * sc[0], d[1] * rad * sc[1], d[2] * rad * sc[2]];
    if (!basis) return add(center, l);
    return add(center, add(add(mul(basis[0], l[0]), mul(basis[1], l[1])), mul(basis[2], l[2])));
  });
  return { pos, f: f.slice() };
}
function mergePieces(ps: Piece[]): Piece {
  const pos: V3[] = [], f: number[] = [];
  for (const p of ps) { const o = pos.length; for (const q of p.pos) pos.push(q); for (const i of p.f) f.push(o + i); }
  return { pos, f };
}
// random facet planes around a mean normal direction (y component fixed), jittered distance
function facetRing(r: R, count: number, ny: number, h: number, jit: number): [V3, number][] {
  const out: [V3, number][] = [];
  const a0 = r() * 6.283;
  for (let i = 0; i < count; i++) {
    const a = a0 + (i / count) * 6.283 + (r() - 0.5) * 0.6;
    const y = ny + (r() - 0.5) * 0.25;
    out.push([[Math.cos(a), y, Math.sin(a)], h * (1 - jit * 0.5 + r() * jit)]);
  }
  return out;
}

// Shared finish for displaced rock surfaces: smooth face-weighted normals, convexity in the vertex
// grad channel (worn ridges bright, creases dark) and a caller-supplied occlusion term.
function rockSurface(mb: MeshBuilder, pos: V3[], f: number[], scl: number, cvGain: number, occl: (i: number) => number) {
  const nr: V3[] = pos.map(() => [0, 0, 0]);
  const nsum: V3[] = pos.map(() => [0, 0, 0]);
  const ncnt = new Float32Array(pos.length);
  for (let i = 0; i < f.length; i += 3) {
    const a = pos[f[i]], b = pos[f[i + 1]], c = pos[f[i + 2]];
    const fn = cross(sub(b, a), sub(c, a));
    for (let q = 0; q < 3; q++) {
      const vi = f[i + q], o1 = pos[f[i + (q + 1) % 3]], o2 = pos[f[i + (q + 2) % 3]];
      const m = nr[vi]; m[0] += fn[0]; m[1] += fn[1]; m[2] += fn[2];
      const s = nsum[vi]; s[0] += o1[0] + o2[0]; s[1] += o1[1] + o2[1]; s[2] += o1[2] + o2[2];
      ncnt[vi] += 2;
    }
  }
  const base = mb.count;
  for (let i = 0; i < pos.length; i++) {
    const nn = norm(nr[i]);
    const avg = mul(nsum[i], 1 / Math.max(ncnt[i], 1));
    const convex = dot(nn, sub(pos[i], avg)) / scl; // >0 on ridges, <0 in creases
    const cv = Math.min(1, Math.max(0, 0.5 + convex * cvGain));
    mb.v(pos[i], nn, 0.5, 0.5, PART.ROCK, 0, occl(i), cv);
  }
  for (let i = 0; i < f.length; i += 3) mb.tri(base + f[i], base + f[i + 1], base + f[i + 2]);
}

// Copy a generated mesh into another with a yaw / scale / offset (props on landmarks).
function appendMesh(dst: MeshBuilder, src: MeshBuilder, off: V3, yaw: number, sc: number) {
  const c = Math.cos(yaw), s = Math.sin(yaw);
  const base = dst.count;
  for (let i = 0; i < src.count; i++) {
    const x = src.p[i * 3], y = src.p[i * 3 + 1], z = src.p[i * 3 + 2];
    const nx = src.n[i * 3], ny = src.n[i * 3 + 1], nz = src.n[i * 3 + 2];
    dst.v([off[0] + (x * c + z * s) * sc, off[1] + y * sc, off[2] + (-x * s + z * c) * sc], [nx * c + nz * s, ny, -nx * s + nz * c],
      src.uv[i * 2], src.uv[i * 2 + 1], src.at[i * 4], src.at[i * 4 + 1], src.at[i * 4 + 2], src.at[i * 4 + 3]);
  }
  for (const i of src.opaque) dst.opaque.push(base + i);
  for (const i of src.cutout) dst.cutout.push(base + i);
}

// -------------------------------------------------------------------------------------------
// Landmarks (placed explicitly, see landmarks.ts). Mesh units are metres at instance scale 1.

// Floating island: lobed meadow plateau over a tapering, ridged rock keel; hanging vines and a few
// trees on top. Origin = centre of the rim; the keel hangs to about -D.
const ISLE_R = 60;
export const ISLE_TOP = (s: number) => ISLE_R * (0.1 * (1 - s * s) - 0.07 * Math.pow(s, 6));
export function genIsland(seed: number, variant: number, lod: number): MeshBuilder {
  const r = rng(seed * 911 + variant * 53 + 3);
  const mb = new MeshBuilder();
  const R = ISLE_R;
  const D = R * (variant === 1 ? 1.25 : variant === 2 ? 0.6 : 0.85);
  const elong = variant === 2 ? 1.45 : 1.0 + r() * 0.2;
  const sd = lod === 0 ? 5 : lod === 1 ? 4 : 3;
  const { v, f } = icosphere(sd);
  const tip: V3 = [(r() - 0.5) * R * 0.5, 0, (r() - 0.5) * R * 0.5];
  const ph1 = r() * 50, ph2 = r() * 50;
  const rimAt = (th: number) => {
    const cx = Math.cos(th), sz = Math.sin(th);
    return R * (1 + 0.24 * fbm3(cx * 1.2 + ph1, 0.5, sz * 1.2, seed, 3) + 0.07 * fbm3(cx * 4 + ph2, 1.7, sz * 4, seed + 1, 2));
  };
  // rim cliff band over a shallow faceted bowl; the keel itself is a cluster of hanging polytope
  // crags (an inverted crag field) plus a few loose floating rocks
  const cliff = R * (variant === 2 ? 0.13 : 0.2);
  const bowl = D * 0.4;
  const tc = 0.24, q0 = 0.95;
  const pos: V3[] = v.map((d) => {
    const th = Math.atan2(d[2], d[0]);
    const cx = Math.cos(th), sz = Math.sin(th);
    const rim = rimAt(th);
    const ex = cx * rim * elong, ez = sz * rim;
    if (d[1] >= 0) {
      const s = Math.sqrt(Math.max(0, 1 - d[1] * d[1])); // 0 centre .. 1 rim
      const y = ISLE_TOP(s) + R * 0.02 * fbm3(ex * s * 0.04, 3.3, ez * s * 0.04, seed + 2, 3) * s;
      return [ex * s, y, ez * s];
    }
    const t = -d[1];
    if (t < tc) {
      const k = t / tc;
      const wob = 1 - (1 - q0) * k + 0.02 * fbm3(cx * 3, k * 2, sz * 3, seed + 8, 2);
      return [ex * wob, -R * 0.07 - cliff * k, ez * wob];
    }
    const rq = (Math.sqrt(1 - t * t) / Math.sqrt(1 - tc * tc)) * q0;
    const qx = cx * rq, qz = sz * rq;
    const e = 1 - (rq / q0) * (rq / q0);
    const rn = 1 - Math.abs(2 * fbm3(qx * 3 + ph2, 0.7, qz * 3, seed + 7, 3));
    const dep = bowl * Math.pow(e, 1.1) * (0.75 + 0.5 * (fbm3(qx * 2 + ph1, 0.3, qz * 2, seed + 5, 3) + 0.5)) + D * 0.12 * rn * rn * Math.min(1, e * 3);
    return [qx * rim * elong, -R * 0.07 - cliff - dep, qz * rim];
  });
  const pieces: Piece[] = [{ pos, f }];
  const rc = rng(seed * 59 + variant * 11 + 5);
  const nc = variant === 1 ? 5 : 8;
  const csd = lod === 0 ? 3 : lod === 1 ? 2 : 1;
  for (let i = 0; i < nc; i++) {
    const big = i === 0;
    const a = rc() * 6.283, dr = big ? rc() * 0.12 : 0.18 + rc() * 0.5;
    const px = Math.cos(a) * dr * R * elong + tip[0] * 0.3, pz = Math.sin(a) * dr * R + tip[2] * 0.3;
    const w = R * (big ? (variant === 1 ? 0.36 : 0.44) : 0.13 + rc() * 0.15);
    const hh = D * (big ? (variant === 1 ? 1.6 : 1.05) : 0.32 + rc() * 0.5) * (1 - dr * 0.5);
    const top = -R * 0.07 - cliff - bowl * Math.pow(1 - dr * dr, 1.1) * 0.55;
    const pl = [...facetRing(rc, 6 + Math.floor(rc() * 3), 0.08, 0.82, 0.3), ...facetRing(rc, 4 + Math.floor(rc() * 2), -1.5, 0.66, 0.35)];
    pl.push([[0, 1, 0], 0.95]);
    const yaw = rc() * 6.283, lean = (rc() - 0.5) * 0.25;
    if (lod === 2 && !big && w < R * 0.2) continue;
    const bx: V3 = [Math.cos(yaw), 0, Math.sin(yaw)], bz: V3 = [-Math.sin(yaw), 0, Math.cos(yaw)];
    const by: V3 = norm([lean * bx[0], 1, lean * bx[2]]);
    pieces.push(polyPiece([px, top - hh * 0.42, pz], [w, hh * 0.52, w * (0.8 + rc() * 0.4)], csd, seed + 70 + i, pl, [bx, by, bz], 0.07));
  }
  if (lod < 2) {
    // loose rocks drifting just under and beside the rim
    for (let i = 0; i < (variant === 2 ? 3 : 5); i++) {
      const a = rc() * 6.283, dr = 1.05 + rc() * 0.35;
      const w = R * (0.04 + rc() * 0.06);
      const pl = facetRing(rc, 7, 0.0, 0.8, 0.3).concat(facetRing(rc, 4, -1.2, 0.7, 0.3), facetRing(rc, 3, 1.2, 0.75, 0.3));
      pieces.push(polyPiece([Math.cos(a) * dr * R * elong, -R * 0.07 - cliff * (0.5 + rc() * 2.5), Math.sin(a) * dr * R], [w, w * (0.9 + rc() * 0.8), w], csd, seed + 90 + i, pl, undefined, 0.08));
    }
  }
  const all = mergePieces(pieces);
  rockSurface(mb, all.pos, all.f, R * 0.5, 24, (i) => {
    const y = all.pos[i][1];
    return y > -R * 0.08 ? 1 : 0.38 + 0.47 * Math.max(0, 1 + (y + cliff) / (D * 1.8));
  });  // hanging vines from the rim (own random streams so props stay put across LODs)
  const rv = rng(seed * 37 + variant * 5 + 11);
  const nv = lod === 0 ? 46 : lod === 1 ? 18 : 0;
  for (let i = 0; i < nv; i++) {
    const th = rv() * Math.PI * 2;
    const rim = rimAt(th) * 1.004;
    const p: V3 = [Math.cos(th) * rim * elong, -R * 0.075, Math.sin(th) * rim];
    const L = R * (0.12 + rv() * rv() * 0.5);
    const wv = 0.35 + rv() * 0.25;
    const spine = curvePath(p, [Math.cos(th) * 0.06, -1, Math.sin(th) * 0.06], L, lod === 0 ? 8 : 4, [0, 0, 0], rv, 0.05);
    const side: V3 = [-Math.sin(th), 0, Math.cos(th)];
    mb.strip(spine, side, (t) => (1 - t * 0.55) * wv, PART.FROND, (t) => t, true, () => [Math.cos(th), 0, Math.sin(th)], 10);
  }
  // a few trees and fronds on the plateau
  const rt = rng(seed * 41 + variant * 7 + 13);
  const nt = variant === 1 ? 2 : 5;
  for (let i = 0; i < nt; i++) {
    const th = rt() * Math.PI * 2, s = 0.15 + rt() * 0.5, yaw = rt() * 6.283, ts = 0.55 + rt() * 0.25;
    const rim = rimAt(th);
    const x = Math.cos(th) * rim * elong * s, z = Math.sin(th) * rim * s;
    if (lod === 2 && i >= 3) continue;
    appendMesh(mb, genTree(seed + i * 17, i % 2, lod), [x, ISLE_TOP(s) - 0.6, z], yaw, ts);
  }
  if (lod === 0) {
    const rs = rng(seed * 43 + variant * 3 + 17);
    for (let i = 0; i < 14; i++) {
      const th = rs() * Math.PI * 2, s = rs() * 0.8;
      const rim = rimAt(th);
      appendMesh(mb, genShrub(seed + i, i % 2, 1), [Math.cos(th) * rim * elong * s, ISLE_TOP(s) - 0.2, Math.sin(th) * rim * s], rs() * 6.283, 1.6 + rs());
    }
  }
  return mb;
}

// Natural arch: elliptical rock section swept along a flattened half-superellipse, legs flaring
// into the ground, horizontal erosion grooves. Origin = ground level between the legs.
export function genArch(seed: number, variant: number, lod: number): MeshBuilder {
  const mb = new MeshBuilder();
  const W = variant === 1 ? 34 : variant === 2 ? 120 : 62;
  const H = variant === 1 ? 58 : variant === 2 ? 40 : 44;
  const thick = variant === 1 ? 4.6 : variant === 2 ? 9.5 : 6.5;
  const depth = variant === 1 ? 6 : variant === 2 ? 17 : 9.5;
  const segs = lod === 0 ? 72 : lod === 1 ? 32 : 14;
  const sides = lod === 0 ? 28 : lod === 1 ? 14 : 7;
  const drop = Math.max(14, H * 0.3);
  const path: V3[] = [], legness: number[] = [];
  const nLeg = Math.max(2, Math.round(segs * 0.1));
  for (let i = 0; i < nLeg; i++) { path.push([-W / 2 - thick * 0.25, -drop * (1 - i / nLeg), 0]); legness.push(1); }
  const ex = 2.6;
  for (let i = 0; i <= segs; i++) {
    const phi = Math.PI * (1 - i / segs);
    const c = Math.cos(phi), s = Math.sin(phi);
    const x = (W / 2) * Math.sign(c) * Math.pow(Math.abs(c), 2 / ex);
    const y = H * Math.pow(s, 2 / ex);
    path.push([x, y, 0]); legness.push(1 - Math.pow(s, 1.5));
  }
  for (let i = 1; i <= nLeg; i++) { path.push([W / 2 + thick * 0.25, -drop * (i / nLeg), 0]); legness.push(1); }
  const n = path.length;
  // weathering: asymmetric span, sagging/lumpy thickness along the arc, one heavier leg
  const r = rng(seed * 577 + variant * 13 + 1);
  const ph = r() * 20, heavy = r() < 0.5 ? -1 : 1;
  for (let i = 0; i < n; i++) {
    const u = i / (n - 1), P = path[i];
    path[i] = [P[0] + W * 0.07 * Math.sin(Math.PI * u) * heavy, P[1] * (1 + 0.08 * fbm3(u * 3 + ph, 0, 0, seed + 2, 2)), P[2] + W * 0.05 * fbm3(u * 2.5, ph, 0, seed + 3, 2)];
  }
  const pos: V3[] = [];
  const f: number[] = [];
  const B: V3 = [0, 0, 1];
  for (let i = 0; i < n; i++) {
    const T = norm(sub(path[Math.min(i + 1, n - 1)], path[Math.max(i - 1, 0)]));
    const N = norm(cross(T, B));
    const P = path[i];
    const u = i / (n - 1);
    const lg = legness[i];
    const flare = 1 + 0.9 * Math.max(0, Math.min(1, (6 - P[1]) / (6 + drop)));
    const lump = 0.78 + 0.5 * (fbm3(u * 5 + ph, 1.3, 0, seed + 5, 3) + 0.5);
    const side = 1 + 0.22 * heavy * (u - 0.5) * 2 * lg;
    const a = thick * (0.8 + 0.7 * lg) * flare * lump * side, b = depth * (0.85 + 0.35 * lg) * flare * (0.85 + 0.3 * lump) * side;
    for (let j = 0; j < sides; j++) {
      const al = (j / sides) * Math.PI * 2;
      const dir = add(mul(N, Math.cos(al)), mul(B, Math.sin(al)));
      let q = add(P, add(mul(N, a * Math.cos(al)), mul(B, b * Math.sin(al))));
      const nz = fbm3(q[0] * 0.05, q[1] * 0.05, q[2] * 0.05, seed + 9, 4);
      const groove = Math.sin(q[1] * 0.85 + 3 * fbm3(q[0] * 0.03, q[1] * 0.02, q[2] * 0.03, seed + 4, 2));
      q = add(q, mul(dir, thick * (0.42 * nz + 0.05 * groove)));
      pos.push(q);
    }
  }
  for (let i = 0; i < n - 1; i++) for (let j = 0; j < sides; j++) {
    const j1 = (j + 1) % sides;
    const a = i * sides + j, b = i * sides + j1, c = (i + 1) * sides + j, d = (i + 1) * sides + j1;
    f.push(a, c, b, b, c, d);
  }
  rockSurface(mb, pos, f, thick, 6, (i) => 0.55 + 0.45 * Math.min(1, Math.max(0, pos[i][1] / H + 0.2)));
  return mb;
}

// Rock pillars: v0 stacked hoodoo with a cap rock, v1 spire cluster, v2 balanced rock.
export function genPillar(seed: number, variant: number, lod: number): MeshBuilder {
  const r = rng(seed * 313 + variant * 71);
  const mb = new MeshBuilder();
  const sd = lod === 0 ? 3 : lod === 1 ? 2 : 1;
  if (variant === 0) {
    let y = 0;
    rockMesh(mb, [0, 1, 0], [11, 7, 10], sd, seed + 1, 10, 0.5);
    for (let k = 0; k < 5; k++) {
      const h = 6.5 + r() * 2.5, rad = 5.8 * (1 - 0.07 * k) * (0.85 + r() * 0.3);
      rockMesh(mb, [(r() - 0.5) * 1.2, y + h * 0.5, (r() - 0.5) * 1.2], [rad, h * 0.62, rad * (0.85 + r() * 0.3)], sd, seed + 10 + k, 9, 0.85);
      y += h * 0.92;
    }
    rockMesh(mb, [0.6, y + 2.2, -0.4], [9.5, 3.6, 8.2], sd, seed + 30, 11, 0.7);
  } else if (variant === 1) {
    rockMesh(mb, [0, 0, 0], [12, 6, 11], sd, seed + 2, 10, 0.5);
    rockMesh(mb, [0, 24, 0], [5.5, 28, 5], sd + (lod < 2 ? 1 : 0), seed + 3, 12, 0.95);
    rockMesh(mb, [6.5, 15, 3], [4.2, 18, 4], sd, seed + 4, 11, 0.95);
    rockMesh(mb, [-5, 11, -4.5], [3.6, 13, 3.4], sd, seed + 5, 10, 0.95);
  } else {
    rockMesh(mb, [0, 0, 0], [9, 5, 8], sd, seed + 6, 10, 0.5);
    rockMesh(mb, [0, 9, 0], [3.4, 10, 3.0], sd, seed + 7, 9, 0.95);
    rockMesh(mb, [0.8, 22, 0.3], [12, 7.5, 9.5], sd + (lod < 2 ? 1 : 0), seed + 8, 12, 0.8);
  }
  return mb;
}
export function genBoulder(seed: number, variant: number, lod: number): MeshBuilder {
  const mb = new MeshBuilder();
  const sd = lod === 0 ? 4 : lod === 1 ? 3 : 1;
  const sc: V3 = variant === 0 ? [1.55, 1.05, 1.3] : variant === 1 ? [1.0, 1.75, 0.9] : [2.3, 0.8, 1.7];
  rockMesh(mb, [0, 0.25, 0], sc, sd, seed * 13 + variant * 7, 9 + variant * 2, 0.42);
  return mb;
}
