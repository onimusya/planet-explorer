// Fauna: a grazing herd on the Bioluminescent Valley floor, flocks of flyers circling the three
// destinations, long-necked titans and stone colossi on the Highlands mesa, travellers in
// exosuits, sky jellies over the herd, sky rays over the Vista's plain and starships passing over
// every destination. Meshes are generated once in a rest pose, with one bone and a blend weight per
// vertex. Each frame a compute pass (shaders/fauna.wgsl) writes every pose from the time alone,
// and the vertex shaders in veg.wgsl pose the meshes: legs by two-bone IK with planted feet,
// necks and tails by rotations that grow along their length, wings by shoulder and elbow flaps.
// They share the vegetation's fragment shading, so they sit in the same light and shadow.
// The ships' engine trails are drawn by fx.ts.
import faunaSrc from './shaders/fauna.wgsl';
import vegSrc from './shaders/veg.wgsl';
import type { Ctx } from './ctx';
import { L, bgl, buffer, bufferWith, computePipe, U } from './gpu';
import { MeshBuilder, STRIDE } from './plants';
import {
  FAUNA_NC, FAUNA_NB, FAUNA_NT, FAUNA_NH, FAUNA_NK, FAUNA_NJ, FAUNA_NR, FAUNA_NS,
  FAUNA_OT, FAUNA_OH, FAUNA_OK, FAUNA_OJ, FAUNA_OR, FAUNA_OS, FAUNA_N, HERD_TAB_NS, HERD_TAB_NL,
} from './world';

type V3 = [number, number, number];
const PART = {
  SKIN: 10, FIN: 11, EYE: 12, HORN: 13, FEATHER: 14, WING: 15, HULL: 16, GLASS: 17, ENGINE: 18,
  SUIT: 19, VISOR: 20, JET: 21, STONE: 22, BELL: 23, TENT: 24, RAY: 25,
} as const;
const POSE_VEC4 = 5; // per animal, see fauna.wgsl
const RIG_VEC4 = 96; // grazer 0-16, bird 20-24, titan 32-48, traveller 52-62, colossus 64-74 (veg.wgsl)
const SHIP_FIGHTERS = 6; // ships.wgsl: ships 0-5 are fighters, 6-8 haulers
// Per mesh (grazer, bird, titan, traveller, colossus, jelly, ray, fighter, hauler): instance count,
// first instance (= first pose slot, except the birds whose vertex shader offsets past the herd),
// and the vertex shader entry points for the main pass and the shadow cascades (null: no shadow).
const DRAWS: [number, number, string, string | null][] = [
  [FAUNA_NC, 0, 'vsCreature', 'vsCreatureShadow'],
  [FAUNA_NB, 0, 'vsBird', 'vsBirdShadow'],
  [FAUNA_NT, FAUNA_OT, 'vsTitan', 'vsTitanShadow'],
  [FAUNA_NH, FAUNA_OH, 'vsTraveller', 'vsTravellerShadow'],
  [FAUNA_NK, FAUNA_OK, 'vsColossus', 'vsColossusShadow'],
  [FAUNA_NJ, FAUNA_OJ, 'vsJelly', null],
  [FAUNA_NR, FAUNA_OR, 'vsRay', 'vsRayShadow'],
  [SHIP_FIGHTERS, FAUNA_OS, 'vsShip', null],
  [FAUNA_NS - SHIP_FIGHTERS, FAUNA_OS + SHIP_FIGHTERS, 'vsShip', null],
];
// Pose update (fauna.wgsl main): one 64-thread workgroup per group (herd, birds, titans,
// travellers, colossi, jellies, rays, ships).
const UPDATE_GROUPS = 8;

const add = (a: V3, b: V3): V3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const mul = (a: V3, s: number): V3 => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a: V3): V3 => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
const lerp3 = (a: V3, b: V3, t: number): V3 => [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)];
const smooth = (a: number, b: number, x: number) => { const t = Math.min(Math.max((x - a) / (b - a), 0), 1); return t * t * (3 - 2 * t); };
const bez = (p0: V3, p1: V3, p2: V3, p3: V3, t: number): V3 => {
  const u = 1 - t;
  return add(add(mul(p0, u * u * u), mul(p1, 3 * u * u * t)), add(mul(p2, 3 * u * t * t), mul(p3, t * t * t)));
};
// Catmull-Rom through key rows (every column interpolated), t in [0, 1] over the rows.
function spline(rows: number[][], t: number): number[] {
  const n = rows.length - 1;
  const f = Math.min(Math.max(t, 0), 1) * n;
  const i = Math.min(Math.floor(f), n - 1), u = f - i;
  const r0 = rows[Math.max(i - 1, 0)], r1 = rows[i], r2 = rows[i + 1], r3 = rows[Math.min(i + 2, n)];
  return r1.map((_, k) => 0.5 * (2 * r1[k] + (r2[k] - r0[k]) * u + (2 * r0[k] - 5 * r1[k] + 4 * r2[k] - r3[k]) * u * u
    + (3 * r1[k] - r0[k] - 3 * r2[k] + r3[k]) * u * u * u));
}

// Lofted cross-section: centre, half width (rx) and height (ry), lo scales the lower half, up
// orients it, and n is the superellipse exponent (2 = ellipse; larger is boxier).
interface Section { c: V3; rx: number; ry: number; lo?: number; up?: V3; n?: number }

// Mesh builder for rigged meshes: bone index and blend weight ride in the flex and grad slots.
class Rigged {
  mb = new MeshBuilder();
  vert(p: V3, n: V3, u: number, t: number, part: number, bone: number, w: number, ao: number) {
    return this.mb.v(p, n, u, t, part, bone / 255, ao, w);
  }
  // triangle wound to face the mean of its vertex normals
  tri(a: number, b: number, c: number) {
    const P = this.mb.p, N = this.mb.n;
    const e1: V3 = [P[b * 3] - P[a * 3], P[b * 3 + 1] - P[a * 3 + 1], P[b * 3 + 2] - P[a * 3 + 2]];
    const e2: V3 = [P[c * 3] - P[a * 3], P[c * 3 + 1] - P[a * 3 + 1], P[c * 3 + 2] - P[a * 3 + 2]];
    const nn: V3 = [N[a * 3] + N[b * 3] + N[c * 3], N[a * 3 + 1] + N[b * 3 + 1] + N[c * 3 + 1], N[a * 3 + 2] + N[b * 3 + 2] + N[c * 3 + 2]];
    if (dot(cross(e1, e2), nn) < 0) this.mb.tri(a, c, b); else this.mb.tri(a, b, c);
  }
  // Rings of elliptical section along a spine (rx across, ry up; lo scales the lower half), ends closed.
  loft(rings: number, sides: number, at: (t: number) => Section, part: number,
    bone: (t: number) => [number, number], ao: (t: number, s: number) => number) {
    const start = this.mb.count;
    const frame = (t: number) => {
      const s = at(t);
      const a = at(Math.max(t - 0.01, 0)), b = at(Math.min(t + 0.01, 1));
      const T = norm(sub(b.c, a.c));
      const up = s.up ?? [0, 1, 0];
      const R = norm(cross(up, T));
      return { s, a, b, T, R, U: cross(T, R), ds: Math.hypot(...sub(b.c, a.c)) || 1e-4 };
    };
    for (let i = 0; i <= rings; i++) {
      const t = i / rings;
      const { s, a, b, T, R, U: Uv, ds } = frame(t);
      const drx = (b.rx - a.rx) / ds, dry = (b.ry - a.ry) / ds;
      const [bn, bw] = bone(t);
      const ex = s.n ?? 2;
      for (let j = 0; j <= sides; j++) {
        const ang = (j / sides) * Math.PI * 2;
        const ca = Math.cos(ang), sa = Math.sin(ang);
        // superellipse point and its normal (the gradient of |x/rx|^n + |y/ry|^n)
        const cx = Math.sign(ca) * Math.pow(Math.abs(ca), 2 / ex), sy = Math.sign(sa) * Math.pow(Math.abs(sa), 2 / ex);
        const ry = s.ry * (sa < 0 ? s.lo ?? 1 : 1);
        const pos = add(s.c, add(mul(R, cx * s.rx), mul(Uv, sy * ry)));
        const gx = Math.sign(ca) * Math.pow(Math.abs(cx), ex - 1) / s.rx, gy = Math.sign(sa) * Math.pow(Math.abs(sy), ex - 1) / ry;
        const radial = norm(add(mul(R, gx), mul(Uv, gy)));
        const n = norm(sub(radial, mul(T, ca * ca * drx + sa * sa * dry)));
        this.vert(pos, n, j / sides, t, part, bn, bw, ao(t, sa));
      }
    }
    for (let i = 0; i < rings; i++) for (let j = 0; j < sides; j++) {
      const a = start + i * (sides + 1) + j, b = a + 1, c = a + sides + 1, d = c + 1;
      this.tri(a, b, c); this.tri(b, d, c);
    }
    for (const end of [0, 1]) {
      const { s, T } = frame(end);
      const dir = mul(T, end ? 1 : -1);
      const [bn, bw] = bone(end);
      const tip = this.vert(add(s.c, mul(dir, 0.35 * Math.min(s.rx, s.ry))), dir, 0.5, end, part, bn, bw, ao(end, 0));
      const ring = start + end * rings * (sides + 1);
      for (let j = 0; j < sides; j++) this.tri(ring + j, ring + j + 1, tip);
    }
  }
  ellipsoid(c: V3, r: V3, axis: V3, rings: number, sides: number, part: number, bone: number, w: number, ao: number, up: V3 = [0, 1, 0]) {
    const ax = norm(axis);
    const upRef: V3 = Math.abs(dot(ax, up)) > 0.9 ? [0, 0, 1] : up;
    this.loft(rings, sides, (t) => {
      const k = Math.sqrt(Math.max(1 - (2 * t - 1) * (2 * t - 1), 0.0004));
      return { c: add(c, mul(ax, (2 * t - 1) * r[2])), rx: r[0] * k, ry: r[1] * k, up: upRef };
    }, part, () => [bone, w], () => ao);
  }
  // Thin two-sided blade from a base segment (b0 -> b1) to a tip, normal n, t = 0 at the base.
  blade(b0: V3, b1: V3, tip: V3, n: V3, part: number, bone: number, w: number) {
    const mid = lerp3(lerp3(b0, b1, 0.5), tip, 0.55);
    const a = this.vert(b0, n, 0, 0, part, bone, w, 0.8), b = this.vert(b1, n, 1, 0, part, bone, w, 0.8);
    const c = this.vert(add(mid, mul(sub(b0, b1), 0.18)), n, 0.2, 0.6, part, bone, w, 0.9);
    const d = this.vert(add(mid, mul(sub(b1, b0), 0.18)), n, 0.8, 0.6, part, bone, w, 0.9);
    const e = this.vert(tip, n, 0.5, 1, part, bone, w, 1);
    this.tri(a, b, d); this.tri(a, d, c); this.tri(c, d, e);
  }
}

// ---------------------------------------------------------------------------------------------
// Herd animal: a long-necked grazer about 1.5 m at the shoulder (model space: z forward, y up,
// ground at y = 0) with a luminous dorsal sail, antennae and a tail.
function genGrazer(rig: Float32Array): Rigged {
  const r = new Rigged();
  const set = (i: number, a: number, b: number, c: number, d: number) => rig.set([a, b, c, d], i * 4);
  const body = [ // z, centre y, half width, half height
    [-1.22, 1.30, 0.05, 0.05],
    [-1.12, 1.31, 0.22, 0.22],
    [-0.95, 1.31, 0.34, 0.33],
    [-0.72, 1.29, 0.40, 0.38],
    [-0.40, 1.25, 0.37, 0.35],
    [-0.05, 1.25, 0.39, 0.40],
    [0.30, 1.30, 0.42, 0.44],
    [0.62, 1.38, 0.37, 0.41],
    [0.88, 1.47, 0.27, 0.32],
    [1.04, 1.53, 0.15, 0.18],
    [1.10, 1.55, 0.04, 0.05],
  ];
  const bodyAt = (t: number): Section => {
    const [z, cy, rx, ry] = spline(body, t);
    return { c: [0, cy, z], rx, ry, lo: 1.14 };
  };
  r.loft(30, 28, bodyAt, PART.SKIN, () => [0, 1], (_t, s) => 0.6 + 0.4 * (0.5 + 0.5 * s));
  // neck (angle ramps from its base to the head)
  const n0: V3 = [0, 1.44, 0.84], n1: V3 = [0, 1.78, 1.06], n2: V3 = [0, 2.02, 1.24], n3: V3 = [0, 2.15, 1.42];
  r.loft(14, 20, (t) => ({ c: bez(n0, n1, n2, n3, t), rx: lerp(0.21, 0.115, t), ry: lerp(0.23, 0.125, t) }),
    PART.SKIN, (t) => [1, t], (_t, s) => 0.7 + 0.3 * (0.5 + 0.5 * s));
  set(8, n0[0], n0[1], n0[2], 0);
  // luminous mane: swept-back blades along the top of the neck, continuing the dorsal sail
  for (let k = 0; k < 7; k++) {
    const t = 0.06 + k * 0.125;
    const c = bez(n0, n1, n2, n3, t);
    const T = norm(sub(bez(n0, n1, n2, n3, t + 0.01), bez(n0, n1, n2, n3, t - 0.01)));
    const N = norm([0, T[2], -T[1]]);
    const base = add(c, mul(N, lerp(0.23, 0.125, t) - 0.025));
    const h = 0.1 + 0.1 * Math.sin((Math.PI * (k + 0.8)) / 8);
    r.blade(add(base, mul(T, 0.07)), sub(base, mul(T, 0.07)), add(add(base, mul(N, h)), mul(T, -0.09)), [1, 0, 0], PART.FIN, 1, t);
  }
  // head, pointing forward and down from its pivot at the top of the neck
  const hb = n3;
  const hd = norm([0, -0.45, 1]);
  const head = [ // distance along hd, half width, half height, lift
    [-0.06, 0.10, 0.11, 0.0],
    [0.06, 0.135, 0.145, 0.0],
    [0.22, 0.14, 0.14, 0.01],
    [0.40, 0.11, 0.10, 0.0],
    [0.54, 0.075, 0.07, -0.01],
    [0.62, 0.04, 0.04, -0.01],
  ];
  r.loft(16, 20, (t) => {
    const [s, rx, ry, dy] = spline(head, t);
    return { c: add(add(hb, mul(hd, s)), [0, dy, 0]), rx, ry };
  }, PART.SKIN, () => [2, 1], (_t, s) => 0.75 + 0.25 * (0.5 + 0.5 * s));
  set(9, hb[0], hb[1], hb[2], 0);
  for (const sx of [-1, 1]) {
    // eyes
    r.ellipsoid(add(add(hb, mul(hd, 0.2)), [sx * 0.112, 0.06, 0]), [0.05, 0.05, 0.05], [sx, 0, 0], 6, 10, PART.EYE, 2, 1, 1);
    // antennae with luminous bulbs
    const a0 = add(add(hb, mul(hd, 0.1)), [sx * 0.05, 0.12, 0]);
    const a1 = add(a0, [sx * 0.07, 0.2, -0.05]), a2 = add(a0, [sx * 0.15, 0.34, -0.17]), a3 = add(a0, [sx * 0.19, 0.4, -0.32]);
    r.loft(10, 6, (t) => ({ c: bez(a0, a1, a2, a3, t), rx: lerp(0.024, 0.01, t), ry: lerp(0.024, 0.01, t) }), PART.FIN, () => [2, 1], () => 0.9);
    r.ellipsoid(a3, [0.035, 0.035, 0.035], sub(a3, a2), 5, 8, PART.FIN, 2, 1, 1);
    // gill frills behind the jaw, luminous along their edges
    const f0 = add(add(hb, mul(hd, 0.1)), [sx * 0.11, 0.05, 0]), f1 = add(add(hb, mul(hd, 0.02)), [sx * 0.1, -0.07, 0]);
    const ft = add(hb, [sx * 0.27, 0.06, -0.2]);
    r.blade(f0, f1, ft, norm(cross(sub(f1, f0), sub(ft, f0))), PART.FIN, 2, 1);
  }
  // dorsal sail: a row of luminous blades along the spine
  const topAt = (z: number) => {
    let best = bodyAt(0), bd = 1e9;
    for (let i = 0; i <= 200; i++) { const s = bodyAt(i / 200); const d = Math.abs(s.c[2] - z); if (d < bd) { bd = d; best = s; } }
    return best.c[1] + best.ry;
  };
  for (let k = 0; k < 9; k++) {
    const z = -0.82 + k * 0.2;
    const y = topAt(z) - 0.03;
    const h = 0.1 + 0.2 * Math.sin((Math.PI * (k + 0.5)) / 9);
    r.blade([0, y, z + 0.08], [0, y, z - 0.08], [0, y + h, z - 0.12], [1, 0, 0], PART.FIN, 0, 1);
  }
  // tail
  const t0: V3 = [0, 1.31, -1.1], t1: V3 = [0, 1.28, -1.45], t2: V3 = [0, 1.34, -1.8], t3: V3 = [0, 1.46, -2.05];
  r.loft(14, 12, (t) => ({ c: bez(t0, t1, t2, t3, t), rx: lerp(0.13, 0.025, Math.pow(t, 0.8)), ry: lerp(0.14, 0.03, Math.pow(t, 0.8)) }),
    PART.SKIN, (t) => [3, t], () => 0.8);
  r.blade(add(t3, [0, 0.02, 0.06]), add(t3, [0, -0.02, 0.0]), add(t3, [0, 0.12, -0.2]), [1, 0, 0], PART.FIN, 3, 1);
  set(10, t0[0], t0[1], t0[2], 0);
  // legs: LF, RF, LH, RH. Front knees point forward, hind hocks backward.
  const legs = [
    { hip: [-0.24, 1.18, 0.62] as V3, aU: 0.3, aL: -0.3, bend: 1 },
    { hip: [0.24, 1.18, 0.62] as V3, aU: 0.3, aL: -0.3, bend: 1 },
    { hip: [-0.26, 1.18, -0.78] as V3, aU: -0.32, aL: 0.32, bend: -1 },
    { hip: [0.26, 1.18, -0.78] as V3, aU: -0.32, aL: 0.32, bend: -1 },
  ];
  const l1 = 0.58, l2 = 0.6;
  legs.forEach((g, i) => {
    const hip = g.hip;
    const knee: V3 = [hip[0], hip[1] - l1 * Math.cos(g.aU), hip[2] + l1 * Math.sin(g.aU)];
    const foot: V3 = [hip[0], knee[1] - l2 * Math.cos(g.aL), knee[2] + l2 * Math.sin(g.aL)];
    const front = i < 2;
    const root = add(hip, [0, 0.16, 0]); // thigh starts inside the body
    r.loft(10, 14, (t) => ({ c: lerp3(root, knee, t), rx: lerp(front ? 0.13 : 0.16, 0.08, smooth(0, 1, t)), ry: lerp(front ? 0.15 : 0.19, 0.085, t), up: [0, 0, 1] }),
      PART.SKIN, (t) => [4 + i, smooth(0.05, 0.35, t)], (t) => lerp(0.62, 0.82, t));
    r.ellipsoid(knee, [0.085, 0.085, 0.085], [0, -1, 0], 5, 10, PART.SKIN, 4 + i, 1, 0.8);
    r.loft(10, 12, (t) => ({ c: lerp3(knee, foot, t), rx: lerp(0.075, 0.045, t), ry: lerp(0.075, 0.045, t), up: [0, 0, 1] }),
      PART.SKIN, (t) => [8 + i, smooth(0, 0.3, t)], () => 0.7);
    r.ellipsoid(add(foot, [0, -0.005, 0.03]), [0.065, 0.05, 0.085], [0, 0, 1], 6, 12, PART.HORN, 8 + i, 1, 0.6);
    set(i, hip[0], hip[1], hip[2], l1);
    set(4 + i, knee[0], knee[1], knee[2], l2);
    rig[12 * 4 + i] = g.aU;
    rig[13 * 4 + i] = g.aL;
    rig[14 * 4 + i] = g.bend;
    rig[15 * 4 + i] = foot[2];
    rig[16 * 4 + i] = foot[1];
  });
  // gait: stance excursion = duty x metres per cycle (STRIDE_W in fauna.wgsl), foot lift, duty factor
  set(11, 0.62 * 1.35, 0.12, 0.62, 0);
  return r;
}

// Flyer: long-winged glider, 2.6 m wingspan at scale 1 (z forward, x right, y up).
function genBird(rig: Float32Array): Rigged {
  const r = new Rigged();
  const set = (i: number, v: V3) => rig.set([v[0], v[1], v[2], 0], i * 4);
  const body = [
    [-0.36, 0.0, 0.02, 0.018],
    [-0.26, 0.0, 0.065, 0.055],
    [-0.08, 0.0, 0.105, 0.095],
    [0.1, 0.01, 0.1, 0.095],
    [0.22, 0.03, 0.072, 0.075],
    [0.3, 0.045, 0.062, 0.066],
    [0.36, 0.045, 0.048, 0.05],
    [0.4, 0.04, 0.018, 0.018],
  ];
  r.loft(20, 16, (t) => { const [z, cy, rx, ry] = spline(body, t); return { c: [0, cy, z], rx, ry }; },
    PART.FEATHER, () => [20, 1], (_t, s) => 0.7 + 0.3 * (0.5 + 0.5 * s));
  r.loft(6, 8, (t) => ({ c: lerp3([0, 0.04, 0.37], [0, 0.022, 0.52], t), rx: lerp(0.026, 0.003, t), ry: lerp(0.022, 0.003, t) }),
    PART.HORN, () => [20, 1], () => 0.8);
  for (const sx of [-1, 1]) {
    r.ellipsoid([sx * 0.042, 0.062, 0.31], [0.013, 0.013, 0.013], [sx, 0, 0], 4, 6, PART.EYE, 20, 1, 1);
    // wing: inner panel shoulder -> elbow, swept outer panel elbow -> tip, primaries notched
    const right = sx > 0;
    const S: V3 = [sx * 0.07, 0.03, 0.04], E: V3 = [sx * 0.6, 0.035, 0], Tp: V3 = [sx * 1.32, 0.02, -0.26];
    const NS = 20, NCH = 4, EL = 0.42;
    const start = r.mb.count;
    for (let i = 0; i <= NS; i++) {
      const sg = i / NS;
      const spar = sg <= EL ? lerp3(S, E, sg / EL) : lerp3(E, Tp, (sg - EL) / (1 - EL));
      const chord = sg <= EL ? lerp(0.4, 0.36, sg / EL) : lerp(0.36, 0.1, Math.pow((sg - EL) / (1 - EL), 1.3));
      const notch = 1 - 0.22 * smooth(0.55, 0.9, sg) * (0.5 + 0.5 * Math.cos(sg * Math.PI * 22));
      const bone = sg <= EL ? (right ? 22 : 21) : (right ? 24 : 23);
      const w = sg <= EL ? smooth(0, 0.12, sg) : smooth(EL, EL + 0.1, sg);
      for (let j = 0; j <= NCH; j++) {
        const cf = j / NCH;
        const off = -(cf - 0.25) * chord * (j === NCH ? notch : 1);
        r.vert(add(spar, [0, 0.035 * chord * Math.sin(Math.PI * cf), off]), [0, 1, 0], cf, sg, PART.WING, bone, w, 0.85 + 0.15 * cf);
      }
    }
    for (let i = 0; i < NS; i++) for (let j = 0; j < NCH; j++) {
      const a = start + i * (NCH + 1) + j, b = a + 1, c = a + NCH + 1, d = c + 1;
      r.tri(a, b, c); r.tri(b, d, c);
    }
    set(right ? 21 : 20, S);
    set(right ? 23 : 22, E);
  }
  // forked tail fan
  const start = r.mb.count;
  const NT = 6;
  for (let i = 0; i <= NT; i++) {
    const t = i / NT;
    const z = lerp(-0.3, -0.62, t), y = lerp(0, -0.012, t);
    const half = lerp(0.05, 0.22, t);
    const fork = 0.12 * smooth(0.6, 1, t);
    for (let j = 0; j <= 2; j++) {
      const x = (j - 1) * half;
      r.vert([x, y, z + (j === 1 ? fork : 0)], [0, 1, 0], j / 2, t, PART.WING, 25, t, 0.9);
    }
  }
  for (let i = 0; i < NT; i++) for (let j = 0; j < 2; j++) {
    const a = start + i * 3 + j, b = a + 1, c = a + 3, d = c + 1;
    r.tri(a, b, c); r.tri(b, d, c);
  }
  set(24, [0, 0, -0.3]);
  return r;
}

// ---------------------------------------------------------------------------------------------
// Titan: a long-necked giant grazing the Highlands mesa, at unit scale 1.6 m at the shoulder with
// its head up to 3.5 m and its tail 3.4 m behind (instances are scaled 6-11x). It uses the herd
// animal's skeleton and leg rig, with its rig constants at base rb (poseCreature in veg.wgsl).
function genTitan(rig: Float32Array, rb: number): Rigged {
  const r = new Rigged();
  const set = (i: number, a: number, b: number, c: number, d: number) => rig.set([a, b, c, d], (rb + i) * 4);
  const body = [ // z, centre y, half width, half height
    [-1.12, 1.46, 0.06, 0.06],
    [-1.02, 1.48, 0.27, 0.27],
    [-0.82, 1.5, 0.43, 0.44],
    [-0.48, 1.5, 0.51, 0.52],
    [-0.08, 1.49, 0.53, 0.54],
    [0.3, 1.53, 0.5, 0.53],
    [0.62, 1.62, 0.42, 0.46],
    [0.86, 1.73, 0.3, 0.33],
    [1.0, 1.8, 0.17, 0.19],
    [1.06, 1.82, 0.05, 0.05],
  ];
  const bodyAt = (t: number): Section => { const [z, cy, rx, ry] = spline(body, t); return { c: [0, cy, z], rx, ry, lo: 1.1 }; };
  r.loft(30, 28, bodyAt, PART.SKIN, () => [0, 1], (_t, s) => 0.6 + 0.4 * (0.5 + 0.5 * s));
  // neck rising high in front, its bend growing toward the head
  const n0: V3 = [0, 1.74, 0.82], n1: V3 = [0, 2.45, 1.22], n2: V3 = [0, 3.12, 1.5], n3: V3 = [0, 3.45, 1.88];
  const neckR = (t: number) => lerp(0.3, 0.1, Math.pow(t, 0.8));
  r.loft(18, 20, (t) => ({ c: bez(n0, n1, n2, n3, t), rx: neckR(t), ry: neckR(t) * 1.06 }),
    PART.SKIN, (t) => [1, t], (_t, s) => 0.7 + 0.3 * (0.5 + 0.5 * s));
  set(8, n0[0], n0[1], n0[2], 0);
  // small head pointing forward and down, eyes, swept crest horns
  const hb = n3, hd = norm([0, -0.35, 1]);
  const head = [[-0.05, 0.095, 0.105], [0.05, 0.12, 0.125], [0.18, 0.11, 0.1], [0.3, 0.08, 0.07], [0.37, 0.04, 0.04]];
  r.loft(12, 16, (t) => { const [s, rx, ry] = spline(head, t); return { c: add(hb, mul(hd, s)), rx, ry }; }, PART.SKIN, () => [2, 1], () => 0.85);
  set(9, hb[0], hb[1], hb[2], 0);
  for (const sx of [-1, 1]) {
    r.ellipsoid(add(add(hb, mul(hd, 0.14)), [sx * 0.1, 0.05, 0]), [0.035, 0.035, 0.035], [sx, 0, 0], 5, 8, PART.EYE, 2, 1, 1);
    const c0 = add(add(hb, mul(hd, 0.02)), [sx * 0.06, 0.1, 0]);
    r.blade(add(c0, [0, 0, 0.05]), add(c0, [0, 0, -0.05]), add(c0, [sx * 0.08, 0.16, -0.22]), [1, 0, 0], PART.FIN, 2, 1);
  }
  // luminous plates along the neck and the spine
  for (let k = 0; k < 6; k++) {
    const t = 0.1 + k * 0.15;
    const c = bez(n0, n1, n2, n3, t);
    const T = norm(sub(bez(n0, n1, n2, n3, t + 0.01), bez(n0, n1, n2, n3, t - 0.01)));
    const N = norm([0, T[2], -T[1]]);
    const base = add(c, mul(N, neckR(t) - 0.03));
    r.blade(add(base, mul(T, 0.09)), sub(base, mul(T, 0.09)), add(add(base, mul(N, lerp(0.2, 0.08, t))), mul(T, -0.08)), [1, 0, 0], PART.FIN, 1, t);
  }
  const topAt = (z: number) => {
    let best = bodyAt(0), bd = 1e9;
    for (let i = 0; i <= 200; i++) { const s = bodyAt(i / 200); const d = Math.abs(s.c[2] - z); if (d < bd) { bd = d; best = s; } }
    return best.c[1] + best.ry;
  };
  for (let k = 0; k < 9; k++) {
    const z = -0.9 + k * 0.21;
    const y = topAt(z) - 0.04;
    const h = 0.16 + 0.22 * Math.sin((Math.PI * (k + 0.5)) / 9);
    r.blade([0, y, z + 0.1], [0, y, z - 0.1], [0, y + h, z - 0.14], [1, 0, 0], PART.FIN, 0, 1);
  }
  // long whip tail
  const t0: V3 = [0, 1.5, -1.05], t1: V3 = [0, 1.36, -1.8], t2: V3 = [0, 1.18, -2.6], t3: V3 = [0, 1.02, -3.4];
  r.loft(20, 14, (t) => ({ c: bez(t0, t1, t2, t3, t), rx: lerp(0.3, 0.025, Math.pow(t, 0.7)), ry: lerp(0.32, 0.03, Math.pow(t, 0.7)) }),
    PART.SKIN, (t) => [3, t], () => 0.8);
  set(10, t0[0], t0[1], t0[2], 0);
  // columnar legs: LF, RF, LH, RH
  const legs = [
    { hip: [-0.3, 1.3, 0.55] as V3, aU: 0.08, aL: -0.06, bend: 1 },
    { hip: [0.3, 1.3, 0.55] as V3, aU: 0.08, aL: -0.06, bend: 1 },
    { hip: [-0.32, 1.3, -0.72] as V3, aU: -0.12, aL: 0.1, bend: -1 },
    { hip: [0.32, 1.3, -0.72] as V3, aU: -0.12, aL: 0.1, bend: -1 },
  ];
  const l1 = 0.62, l2 = 0.66;
  legs.forEach((g, i) => {
    const hip = g.hip;
    const knee: V3 = [hip[0], hip[1] - l1 * Math.cos(g.aU), hip[2] + l1 * Math.sin(g.aU)];
    const foot: V3 = [hip[0], knee[1] - l2 * Math.cos(g.aL), knee[2] + l2 * Math.sin(g.aL)];
    const root = add(hip, [0, 0.2, 0]);
    r.loft(10, 16, (t) => ({ c: lerp3(root, knee, t), rx: lerp(0.24, 0.16, smooth(0, 1, t)), ry: lerp(0.26, 0.17, t), up: [0, 0, 1] }),
      PART.SKIN, (t) => [4 + i, smooth(0.05, 0.35, t)], (t) => lerp(0.62, 0.82, t));
    r.ellipsoid(knee, [0.16, 0.16, 0.16], [0, -1, 0], 5, 12, PART.SKIN, 4 + i, 1, 0.8);
    r.loft(10, 14, (t) => ({ c: lerp3(knee, foot, t), rx: lerp(0.15, 0.135, t), ry: lerp(0.15, 0.135, t), up: [0, 0, 1] }),
      PART.SKIN, (t) => [8 + i, smooth(0, 0.3, t)], () => 0.7);
    r.ellipsoid(add(foot, [0, 0.045, 0.02]), [0.17, 0.07, 0.19], [0, 0, 1], 6, 14, PART.HORN, 8 + i, 1, 0.6);
    set(i, hip[0], hip[1], hip[2], l1);
    set(4 + i, knee[0], knee[1], knee[2], l2);
    rig[(rb + 12) * 4 + i] = g.aU;
    rig[(rb + 13) * 4 + i] = g.aL;
    rig[(rb + 14) * 4 + i] = g.bend;
    rig[(rb + 15) * 4 + i] = foot[2];
    rig[(rb + 16) * 4 + i] = foot[1];
  });
  // gait: stance excursion = duty x stride (TITAN_STRIDE in fauna.wgsl), foot lift, duty factor
  set(11, 0.68 * 1.4, 0.07, 0.68, 0);
  return r;
}

// Humanoid standing on y = 0 and facing +z, at rig base rb (bipedV in veg.wgsl): a traveller in an
// exosuit ('suit': 1.9 m with the helmet antenna, glowing visor, backpack with jet nozzles) or a
// colossus of basalt and crystal ('stone': hunched, long arms, burning eyes; instances ~15x).
function genBiped(rig: Float32Array, rb: number, kind: 'suit' | 'stone'): Rigged {
  const r = new Rigged();
  const set = (i: number, v: V3) => rig.set([v[0], v[1], v[2], 0], (rb + i) * 4);
  const suit = kind === 'suit';
  const S = suit ? PART.SUIT : PART.STONE;
  const P = suit
    ? { hipY: 0.94, hipX: 0.095, knee: [0.1, 0.51, 0.012], ankle: [0.1, 0.085, -0.01], sh: [0.205, 1.415, -0.005],
      el: [0.24, 1.13, -0.02], wr: [0.255, 0.87, 0.015], neck: [0, 1.52, 0] as V3,
      thigh: [0.088, 0.066], shin: [0.064, 0.05], upper: [0.06, 0.05], fore: [0.05, 0.042] }
    : { hipY: 0.86, hipX: 0.14, knee: [0.155, 0.47, 0.035], ankle: [0.165, 0.08, 0.0], sh: [0.33, 1.4, 0.08],
      el: [0.39, 1.02, 0.1], wr: [0.41, 0.64, 0.15], neck: [0, 1.47, 0.12] as V3,
      thigh: [0.13, 0.1], shin: [0.11, 0.09], upper: [0.1, 0.085], fore: [0.09, 0.08] };
  // torso lofted up the spine: y, half width, half depth, z offset, superellipse n
  const torso = suit
    ? [[0.86, 0.15, 0.105, 0, 2.4], [0.95, 0.168, 0.112, 0, 2.4], [1.06, 0.15, 0.1, 0.006, 2.2], [1.2, 0.178, 0.118, 0.012, 2.6],
      [1.33, 0.205, 0.13, 0.006, 3], [1.43, 0.195, 0.118, -0.004, 3], [1.49, 0.09, 0.075, 0, 2.2]]
    : [[0.78, 0.21, 0.15, 0, 2.2], [0.9, 0.24, 0.17, 0.01, 2.4], [1.04, 0.23, 0.17, 0.04, 2.3], [1.2, 0.3, 0.21, 0.08, 2.6],
      [1.34, 0.36, 0.23, 0.1, 2.8], [1.44, 0.33, 0.2, 0.1, 2.8], [1.5, 0.14, 0.12, 0.12, 2.2]];
  r.loft(16, 24, (t) => { const [y, rx, ry, z, n] = spline(torso, t); return { c: [0, y, z], rx, ry, n, up: [0, 0, 1] }; },
    S, () => [30, 1], (t, s) => 0.7 + 0.3 * t * (0.5 + 0.5 * s));
  const nk = P.neck;
  r.loft(6, 14, (t) => ({ c: lerp3(add(nk, [0, -0.05, 0]), add(nk, [0, 0.07, 0.01]), t), rx: suit ? 0.056 : 0.1, ry: suit ? 0.052 : 0.09, up: [0, 0, 1] }),
    S, (t) => [31, t], () => 0.7);
  set(0, nk);
  if (suit) {
    r.ellipsoid([0, 1.67, 0.012], [0.128, 0.15, 0.142], [0, 0, 1], 12, 20, PART.SUIT, 31, 1, 1);
    r.ellipsoid([0, 1.675, 0.078], [0.098, 0.062, 0.088], [0, 0, 1], 8, 18, PART.VISOR, 31, 1, 1);
    r.loft(5, 5, (t) => ({ c: lerp3([-0.105, 1.74, -0.05], [-0.13, 1.9, -0.09], t), rx: lerp(0.009, 0.005, t), ry: lerp(0.009, 0.005, t) }), PART.SUIT, () => [31, 1], () => 1);
    r.ellipsoid([-0.13, 1.905, -0.09], [0.013, 0.013, 0.013], [0, 1, 0], 4, 6, PART.JET, 31, 1, 1);
    // backpack, jet nozzles, side tanks
    r.loft(8, 20, (t) => ({ c: lerp3([0, 1.04, -0.205], [0, 1.45, -0.19], t), rx: 0.155, ry: 0.072, n: 4, up: [0, 0, 1] }), PART.SUIT, () => [30, 1], () => 0.8);
    for (const sx of [-1, 1]) {
      r.loft(4, 10, (t) => ({ c: lerp3([sx * 0.075, 1.08, -0.205], [sx * 0.078, 1.0, -0.21], t), rx: lerp(0.032, 0.045, t), ry: lerp(0.032, 0.045, t), up: [0, 0, 1] }),
        PART.JET, () => [30, 1], () => 0.8);
      r.loft(6, 10, (t) => ({ c: lerp3([sx * 0.165, 1.1, -0.19], [sx * 0.165, 1.4, -0.19], t), rx: 0.035, ry: 0.035, up: [0, 0, 1] }), PART.SUIT, () => [30, 1], () => 0.8);
    }
  } else {
    // a blunt head jutting forward under a brow ridge, burning eyes, crystal shards on the back
    r.ellipsoid([0, 1.57, 0.2], [0.11, 0.115, 0.13], [0, 0, 1], 10, 16, PART.STONE, 31, 1, 0.9);
    r.ellipsoid([0, 1.61, 0.29], [0.035, 0.03, 0.1], [1, 0, 0], 6, 10, PART.STONE, 31, 1, 0.9);
    for (const sx of [-1, 1]) r.ellipsoid([sx * 0.045, 1.575, 0.315], [0.022, 0.018, 0.018], [0, 0, 1], 5, 8, PART.EYE, 31, 1, 1);
    const shards = [[0.22, 1.46, -0.05, 0.25, 0.3], [-0.24, 1.48, -0.02, -0.22, 0.34], [0.1, 1.4, -0.16, 0.05, 0.42], [-0.12, 1.32, -0.18, -0.1, 0.3],
      [0.32, 1.42, 0.06, 0.4, 0.22], [-0.33, 1.43, 0.05, -0.38, 0.26], [0, 1.22, -0.2, 0, 0.28]];
    for (const [x, y, z, dx, h] of shards) {
      const b: V3 = [x, y, z], tip = add(b, [dx * h, h, -0.25 * h]);
      const side = norm(cross(sub(tip, b), [0, 0, 1])), side2 = norm(cross(sub(tip, b), side));
      r.blade(add(b, mul(side, 0.04)), sub(b, mul(side, 0.04)), tip, side2, PART.FIN, 30, 1);
      r.blade(add(b, mul(side2, 0.035)), sub(b, mul(side2, 0.035)), tip, side, PART.FIN, 30, 1);
    }
  }
  // arms (s = 0 at -x, 1 at +x): shoulder pad, upper arm, elbow, forearm, hand
  for (const s of [0, 1]) {
    const sx = s ? 1 : -1;
    const sh: V3 = [sx * P.sh[0], P.sh[1], P.sh[2]], el: V3 = [sx * P.el[0], P.el[1], P.el[2]], wr: V3 = [sx * P.wr[0], P.wr[1], P.wr[2]];
    set(1 + s, sh);
    set(3 + s, el);
    r.ellipsoid(add(sh, [sx * 0.01, 0.015, 0]), suit ? [0.082, 0.07, 0.085] : [0.16, 0.14, 0.16], [0, 0, 1], 8, 14, S, 32 + s, 1, 0.9);
    r.loft(10, 14, (t) => ({ c: lerp3(add(sh, [0, 0.02, 0]), el, t), rx: lerp(P.upper[0], P.upper[1], t), ry: lerp(P.upper[0], P.upper[1], t) * 1.05, up: [0, 0, 1] }),
      S, (t) => [32 + s, smooth(0, 0.15, t)], () => 0.8);
    r.ellipsoid(el, [P.fore[0] * 1.05, P.fore[0] * 1.05, P.fore[0] * 1.05], [0, -1, 0], 5, 10, S, 34 + s, 0.5, 0.8);
    r.loft(10, 14, (t) => ({ c: lerp3(el, wr, t), rx: lerp(P.fore[0], P.fore[1], t), ry: lerp(P.fore[0], P.fore[1], t), up: [0, 0, 1] }),
      S, (t) => [34 + s, smooth(0, 0.3, t)], () => 0.8);
    r.ellipsoid(add(wr, [0, suit ? -0.055 : -0.07, 0.008]), suit ? [0.038, 0.058, 0.045] : [0.085, 0.1, 0.09], [0, -1, 0], 6, 10, S, 34 + s, 1, 0.75);
  }
  // legs: thigh, knee pad, shin, boot or foot (heel to toe)
  for (const s of [0, 1]) {
    const sx = s ? 1 : -1;
    const hip: V3 = [sx * P.hipX, P.hipY, 0], knee: V3 = [sx * P.knee[0], P.knee[1], P.knee[2]], ankle: V3 = [sx * P.ankle[0], P.ankle[1], P.ankle[2]];
    set(5 + s, hip);
    set(7 + s, knee);
    r.loft(12, 16, (t) => ({ c: lerp3(add(hip, [0, 0.06, 0]), knee, t), rx: lerp(P.thigh[0], P.thigh[1], t), ry: lerp(P.thigh[0], P.thigh[1], t) * 1.05, up: [0, 0, 1] }),
      S, (t) => [36 + s, smooth(0, 0.25, t)], () => 0.8);
    r.ellipsoid(add(knee, [0, 0, suit ? 0.03 : 0.04]), suit ? [0.055, 0.06, 0.035] : [0.11, 0.11, 0.09], [0, 0, 1], 6, 12, S, 38 + s, 0.4, 0.85);
    r.loft(12, 16, (t) => ({ c: lerp3(knee, ankle, t), rx: lerp(P.shin[0], P.shin[1], t), ry: lerp(P.shin[0], P.shin[1], t) * 1.05, up: [0, 0, 1] }),
      S, (t) => [38 + s, smooth(0, 0.25, t)], () => 0.75);
    const fy = suit ? 0.06 : 0.075, back = suit ? 0.075 : 0.1, toe = suit ? 0.17 : 0.24;
    r.loft(10, 16, (t) => ({ c: lerp3([ankle[0], fy, ankle[2] - back], [ankle[0], fy - 0.005, ankle[2] + toe], t), rx: suit ? 0.058 : 0.11, ry: fy, n: 3 }),
      S, () => [38 + s, 1], () => 0.6);
  }
  // gait: hip swing, knee lift, arm swing, elbow rest bend; pelvis height
  rig.set(suit ? [0.42, 0.75, 0.35, 0.18] : [0.36, 0.6, 0.28, 0.35], (rb + 9) * 4);
  rig.set([P.hipY, 0, 0, 0], (rb + 10) * 4);
  return r;
}

// Sky jelly, bell radius 1: a domed bell with a flared rim, four frilled oral arms and a ring of
// long fine tentacles. Bone 0 is the bell (w from the apex to the rim), 1 the arms and tentacles
// (w from root to tip); uv.y runs the same way, for the glow.
function genJelly(): Rigged {
  const r = new Rigged();
  const R = 18, S = 32;
  const prof = (t: number): [number, number] => {
    const a = t * 1.5, flare = smooth(0.75, 1, t) * 0.12;
    return [Math.sin(a) * (1 + 0.06 * t) + flare, Math.cos(a) * 0.9 - flare * 0.8];
  };
  const start = r.mb.count;
  for (let i = 0; i <= R; i++) {
    const t = i / R;
    const [rr, y] = prof(t);
    const [r2, y2] = prof(Math.min(t + 0.01, 1)), [r1, y1] = prof(Math.max(t - 0.01, 0));
    const dr = r2 - r1, dy = y2 - y1;
    for (let j = 0; j <= S; j++) {
      const a = (j / S) * Math.PI * 2, ca = Math.cos(a), sa = Math.sin(a);
      r.vert([ca * rr, y, sa * rr], norm([-ca * dy, dr, -sa * dy]), j / S, t, PART.BELL, 0, t, 0.9);
    }
  }
  for (let i = 0; i < R; i++) for (let j = 0; j < S; j++) {
    const a = start + i * (S + 1) + j, b = a + 1, c = a + S + 1, d = c + 1;
    r.tri(a, b, c); r.tri(b, d, c);
  }
  for (let k = 0; k < 4; k++) {
    const a = (k / 4) * Math.PI * 2 + 0.4;
    const dir: V3 = [Math.cos(a), 0, Math.sin(a)];
    r.loft(16, 6, (t) => ({ c: add(mul(dir, 0.12 + 0.15 * Math.sin(t * 3)), [0.06 * Math.sin(t * 9 + k), 0.25 - 1.7 * t, 0.06 * Math.cos(t * 7 + k)]),
      rx: lerp(0.13, 0.04, t), ry: 0.025, up: dir }), PART.TENT, (t) => [1, t], () => 0.9);
  }
  for (let k = 0; k < 12; k++) {
    const a = (k / 12) * Math.PI * 2, ca = Math.cos(a), sa = Math.sin(a);
    const len = 2.4 + 1.2 * ((k * 7) % 5) / 4;
    const p0: V3 = [ca * 1.02, 0.05, sa * 1.02];
    r.loft(14, 5, (t) => ({ c: add(p0, [ca * 0.15 * Math.sin(t * 2.5), -len * t, sa * 0.15 * Math.sin(t * 2.5)]), rx: lerp(0.028, 0.006, t), ry: lerp(0.028, 0.006, t), up: [ca, 0, sa] }),
      PART.TENT, (t) => [1, t], () => 0.9);
  }
  return r;
}

// Sky ray, span 2 (x = -1..1, nose at +z): one closed surface with a body bulge between wings that
// thin to their edges, cephalic fins, eyes and a whip tail. Bone 0 body and wings, 1 tail.
// uv: x = |x| (span), y = chord position (0 trailing edge, 1 leading edge).
function genRay(): Rigged {
  const r = new Rigged();
  const NX = 18, NZ = 10;
  const le = (ax: number) => 0.52 - 0.68 * Math.pow(ax, 1.15);
  const te = (ax: number) => -0.42 + 0.24 * Math.pow(ax, 0.9);
  const th = (ax: number, c: number) => 0.13 * Math.pow(1 - ax, 1.6) * (1 - c * c) + 0.004;
  for (const top of [true, false]) {
    const start = r.mb.count;
    const k = top ? 1 : -0.55;
    const pos = (i: number, j: number): V3 => {
      const x = -1 + i / NX, ax = Math.abs(x), c = -1 + (2 * j) / NZ;
      return [x, k * th(ax, c), lerp(te(ax), le(ax), (c + 1) / 2)];
    };
    for (let i = 0; i <= 2 * NX; i++) for (let j = 0; j <= NZ; j++) {
      const p = pos(i, j);
      let n = norm(cross(sub(pos(i, Math.min(j + 1, NZ)), pos(i, Math.max(j - 1, 0))), sub(pos(Math.min(i + 1, 2 * NX), j), pos(Math.max(i - 1, 0), j))));
      if (n[1] > 0 !== top) n = mul(n, -1);
      r.vert(p, n, Math.abs(p[0]), j / NZ, PART.RAY, 0, Math.abs(p[0]), 0.9);
    }
    for (let i = 0; i < 2 * NX; i++) for (let j = 0; j < NZ; j++) {
      const a = start + i * (NZ + 1) + j, b = a + 1, c = a + NZ + 1, d = c + 1;
      r.tri(a, b, c); r.tri(b, d, c);
    }
  }
  for (const sx of [-1, 1]) {
    const b: V3 = [sx * 0.1, 0, 0.5];
    r.blade(add(b, [sx * 0.03, 0, -0.04]), add(b, [-sx * 0.03, 0, 0]), add(b, [sx * 0.05, -0.04, 0.2]), [0, 1, 0], PART.RAY, 0, 0.1);
    r.ellipsoid([sx * 0.13, 0.035, 0.4], [0.022, 0.022, 0.022], [sx, 0, 0], 4, 6, PART.EYE, 0, 0.1, 1);
  }
  r.loft(14, 6, (t) => ({ c: lerp3([0, 0.01, -0.42], [0, 0.03, -1.55], t), rx: lerp(0.035, 0.004, t), ry: lerp(0.03, 0.004, t) }), PART.RAY, (t) => [1, t], () => 0.85);
  return r;
}

// Wing or fin slab lofted from root to tip: chord along the side vector, thickness along up.
function slab(r: Rigged, root: V3, tip: V3, chord: [number, number], thick: [number, number], part: number, up: V3 = [0, 1, 0]) {
  r.loft(10, 16, (t) => ({ c: lerp3(root, tip, t), rx: lerp(chord[0], chord[1], t) / 2, ry: lerp(thick[0], thick[1], t) / 2, n: 4, up }),
    part, () => [0, 1], () => 0.85);
}

// Fighter, 16 m from the nozzles to the nose (+z): fuselage, canopy, swept wings with tip pods,
// twin engines with glowing nozzles (ships.wgsl shipEngine), canted tail fins.
function genFighter(): Rigged {
  const r = new Rigged();
  const fus = [ // z, y, half width, half height, superellipse n
    [-7.6, 0.15, 0.95, 0.55, 3.2], [-7.0, 0.15, 1.25, 0.72, 3.2], [-4.5, 0.2, 1.45, 0.84, 3.2], [-1.5, 0.25, 1.3, 0.82, 3],
    [1.5, 0.25, 1.0, 0.72, 2.7], [4.2, 0.12, 0.66, 0.52, 2.4], [6.6, -0.02, 0.3, 0.26, 2.2], [8.0, -0.08, 0.05, 0.05, 2],
  ];
  r.loft(28, 24, (t) => { const [z, y, rx, ry, n] = spline(fus, t); return { c: [0, y, z], rx, ry, n, lo: 0.85 }; },
    PART.HULL, () => [0, 1], (_t, s) => 0.75 + 0.25 * (0.5 + 0.5 * s));
  r.ellipsoid([0, 0.95, 2.2], [0.52, 0.42, 1.75], [0, 0, 1], 10, 18, PART.GLASS, 0, 1, 1);
  for (const sx of [-1, 1]) {
    slab(r, [sx * 1.1, 0, -2.0], [sx * 7.0, 0.42, -5.05], [5.6, 1.5], [0.34, 0.08], PART.HULL);
    r.ellipsoid([sx * 7.05, 0.44, -4.9], [0.13, 0.13, 0.75], [0, 0, 1], 6, 10, PART.HULL, 0, 1, 1);
    r.loft(12, 18, (t) => {
      const rr = t < 0.85 ? lerp(0.6, 0.66, t / 0.85) : lerp(0.66, 0.48, (t - 0.85) / 0.15);
      return { c: [sx * 1.75, -0.05, lerp(-7.9, -2.9, t)], rx: rr, ry: rr };
    }, PART.HULL, () => [0, 1], () => 0.8);
    r.ellipsoid([sx * 1.75, -0.05, -7.93], [0.5, 0.5, 0.09], [0, 0, 1], 4, 16, PART.ENGINE, 0, 1, 1);
    slab(r, [sx * 0.75, 0.7, -5.4], [sx * 1.55, 2.3, -6.9], [2.0, 0.7], [0.12, 0.05], PART.HULL, [sx, 0, 0]);
  }
  return r;
}

// Hauler, 22 m: boxy hull, cockpit band, dorsal fin, two cargo pods on pylons, four engines.
function genHauler(): Rigged {
  const r = new Rigged();
  const fus = [
    [-10.9, 0.4, 1.9, 1.5, 4.5], [-10.2, 0.4, 2.3, 1.85, 4.5], [-4, 0.45, 2.45, 1.95, 4.5], [3, 0.45, 2.35, 1.9, 4.2],
    [7.5, 0.35, 2.0, 1.6, 3.6], [10, 0.2, 1.3, 1.1, 3], [11.2, 0.1, 0.4, 0.4, 2.4],
  ];
  r.loft(26, 28, (t) => { const [z, y, rx, ry, n] = spline(fus, t); return { c: [0, y, z], rx, ry, n }; },
    PART.HULL, () => [0, 1], (_t, s) => 0.75 + 0.25 * (0.5 + 0.5 * s));
  r.ellipsoid([0, 1.2, 8.6], [1.25, 0.5, 1.5], [0, 0, 1], 8, 18, PART.GLASS, 0, 1, 1);
  slab(r, [0, 2.1, -6.5], [0, 3.7, -9.6], [3.2, 1.2], [0.22, 0.08], PART.HULL, [1, 0, 0]);
  const pod = [[-8.7, 0.2, 0.9, 1.05, 3], [-8.0, 0.2, 1.15, 1.35, 6], [3.5, 0.2, 1.15, 1.35, 6], [4.6, 0.2, 0.7, 0.8, 3]];
  for (const sx of [-1, 1]) {
    r.loft(14, 22, (t) => { const [z, y, rx, ry, n] = spline(pod, t); return { c: [sx * 3.9, y, z], rx, ry, n }; }, PART.HULL, () => [0, 1], () => 0.8);
    slab(r, [sx * 2.2, 0.35, -2.5], [sx * 3.2, 0.3, -2.5], [6.0, 5.0], [0.5, 0.4], PART.HULL);
    r.ellipsoid([sx * 3.9, 0.2, -8.82], [0.78, 0.9, 0.1], [0, 0, 1], 4, 16, PART.ENGINE, 0, 1, 1);
    r.ellipsoid([sx * 1.15, 0.45, -10.95], [0.6, 0.6, 0.1], [0, 0, 1], 4, 14, PART.ENGINE, 0, 1, 1);
  }
  return r;
}

// Pack meshes into one vertex buffer (vegetation layout, 24 bytes) and one index buffer.
function pack(meshes: MeshBuilder[]) {
  let vcount = 0, icount = 0;
  for (const m of meshes) { vcount += m.count; icount += m.opaque.length; }
  const vb = new ArrayBuffer(vcount * STRIDE);
  const f32 = new Float32Array(vb), i8 = new Int8Array(vb), u16 = new Uint16Array(vb), u8 = new Uint8Array(vb);
  const ib = new Uint32Array(icount);
  const draws: { first: number; count: number; base: number }[] = [];
  let vo = 0, io = 0;
  const c01 = (x: number) => Math.min(Math.max(x, 0), 1);
  for (const m of meshes) {
    for (let i = 0; i < m.count; i++) {
      const b = (vo + i) * STRIDE;
      f32[b / 4] = m.p[i * 3]; f32[b / 4 + 1] = m.p[i * 3 + 1]; f32[b / 4 + 2] = m.p[i * 3 + 2];
      i8[b + 12] = Math.round(m.n[i * 3] * 127); i8[b + 13] = Math.round(m.n[i * 3 + 1] * 127); i8[b + 14] = Math.round(m.n[i * 3 + 2] * 127);
      u16[b / 2 + 8] = Math.round(c01(m.uv[i * 2]) * 65535);
      u16[b / 2 + 9] = Math.round(c01(m.uv[i * 2 + 1] / 64) * 65535);
      u8[b + 20] = m.at[i * 4];
      u8[b + 21] = Math.round(c01(m.at[i * 4 + 1]) * 255);
      u8[b + 22] = Math.round(c01(m.at[i * 4 + 2]) * 255);
      u8[b + 23] = Math.round(c01(m.at[i * 4 + 3]) * 255);
    }
    ib.set(m.opaque, io);
    draws.push({ first: io, count: m.opaque.length, base: vo });
    io += m.opaque.length;
    vo += m.count;
  }
  return { vb: u8, ib, draws, vcount, tris: icount / 3 };
}

export class Fauna {
  private poseBuf!: GPUBuffer;
  private vbuf!: GPUBuffer;
  private ibuf!: GPUBuffer;
  private draws: { first: number; count: number; base: number }[] = [];
  private pUpdate!: GPUComputePipeline;
  private updateBG!: GPUBindGroup;
  private drawBG: GPUBindGroup[] = [];
  private pMain: GPURenderPipeline[] = [];
  private pShadow: (GPURenderPipeline | null)[] = [];
  stats = { vertices: 0, triangles: 0 };

  constructor(private ctx: Ctx) {}

  async init() {
    const { device } = this.ctx;
    const fb = this.ctx.frameBGL, wb = this.ctx.worldBGL;
    // layouts: poses and the herd trail table (compute); cascade index (shadows), poses, rig (draw)
    const lu = bgl(device, [L.storage(0), L.storage(1)]);
    const V = GPUShaderStage.VERTEX;
    const ld = bgl(device, [L.uniform(0, V | GPUShaderStage.FRAGMENT), L.storageRO(1, V), L.uniform(2, V)]);
    const [cmod, vmod] = await Promise.all([this.ctx.module(faunaSrc, 'fauna'), this.ctx.module(vegSrc, 'veg')]);
    const vbl: GPUVertexBufferLayout[] = [{ arrayStride: STRIDE, attributes: [
      { shaderLocation: 0, offset: 0, format: 'float32x3' },
      { shaderLocation: 1, offset: 12, format: 'snorm8x4' },
      { shaderLocation: 2, offset: 16, format: 'unorm16x2' },
      { shaderLocation: 3, offset: 20, format: 'unorm8x4' },
    ] }];
    const mainLayout = device.createPipelineLayout({ bindGroupLayouts: [fb, wb, this.ctx.lightBGL, ld] });
    const shLayout = device.createPipelineLayout({ bindGroupLayouts: [fb, this.ctx.emptyBGL, this.ctx.emptyBGL, ld] });
    const depth: GPUDepthStencilState = { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'greater' };
    const shDepth: GPUDepthStencilState = { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less', depthBias: 2, depthBiasSlopeScale: 2.5, depthBiasClamp: 0.01 };
    const prim: GPUPrimitiveState = { topology: 'triangle-list', cullMode: 'none' };
    const main = (label: string, entryPoint: string) => device.createRenderPipelineAsync({ label, layout: mainLayout,
      vertex: { module: vmod, entryPoint, buffers: vbl },
      fragment: { module: vmod, entryPoint: 'fsOpaque', targets: [{ format: 'rgba16float' }] }, primitive: prim, depthStencil: depth });
    const shadow = (label: string, entryPoint: string) => device.createRenderPipelineAsync({ label, layout: shLayout,
      vertex: { module: vmod, entryPoint, buffers: vbl }, primitive: prim, depthStencil: shDepth });
    // one pipeline per entry point (the two ship meshes share theirs)
    const made = new Map<string, Promise<GPURenderPipeline>>();
    const get = (ep: string, sh: boolean) => {
      if (!made.has(ep)) made.set(ep, sh ? shadow(ep, ep) : main(ep, ep));
      return made.get(ep)!;
    };
    // Pipelines compile while the meshes are generated below. All main-pass pipelines share one
    // large fragment shader (fsOpaque): the first compiles alone so the GPU process caches it, and
    // the others then only add their vertex shaders. Started together, each would compile its own
    // copy (measured on a cold shader cache: 14 s for these pipelines together, 2.6 s primed).
    const update = computePipe(device, cmod, 'main', [fb, wb, lu]);
    const table = computePipe(device, cmod, 'trailTable', [fb, wb, lu]);
    const shadows = Promise.all(DRAWS.map((d) => (d[3] ? get(d[3], true) : Promise.resolve(null))));
    const primed = get(DRAWS[0][2], false);

    // ---- meshes, poses, rig ----
    const rig = new Float32Array(RIG_VEC4 * 4);
    const packed = pack([
      genGrazer(rig).mb, genBird(rig).mb, genTitan(rig, 32).mb, genBiped(rig, 52, 'suit').mb, genBiped(rig, 64, 'stone').mb,
      genJelly().mb, genRay().mb, genFighter().mb, genHauler().mb,
    ]);
    this.draws = packed.draws;
    this.stats.vertices = packed.vcount;
    this.stats.triangles = packed.tris;
    this.vbuf = bufferWith(device, packed.vb, U.VERTEX, 'faunaVerts');
    this.ibuf = bufferWith(device, packed.ib, U.INDEX, 'faunaIdx');
    this.poseBuf = buffer(device, FAUNA_N * POSE_VEC4 * 16, U.STORAGE | U.COPY_SRC, 'faunaPose');
    const trailBuf = buffer(device, HERD_TAB_NS * HERD_TAB_NL * 8, U.STORAGE, 'herdTrail');
    const rigBuf = bufferWith(device, rig, U.UNIFORM, 'faunaRig');
    this.updateBG = device.createBindGroup({ layout: lu, entries: [
      { binding: 0, resource: { buffer: this.poseBuf } },
      { binding: 1, resource: { buffer: trailBuf } },
    ] });
    for (let c = 0; c < 4; c++) {
      const u = bufferWith(device, new Uint32Array([c, 0, 0, 0]), U.UNIFORM);
      this.drawBG.push(device.createBindGroup({ layout: ld, entries: [
        { binding: 0, resource: { buffer: u } },
        { binding: 1, resource: { buffer: this.poseBuf } },
        { binding: 2, resource: { buffer: rigBuf } },
      ] }));
    }
    this.pTable = await table;
    await primed;
    [this.pMain, this.pShadow, this.pUpdate] = await Promise.all([Promise.all(DRAWS.map((d) => get(d[2], false))), shadows, update]);
  }

  // Fill the herd trail table, once. Its pass shares the update's layout, which includes the world
  // bind group, so the renderer calls this at the end of its init, once that group exists.
  encodeTable(enc: GPUCommandEncoder, worldBG: GPUBindGroup) {
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pTable!);
    pass.setBindGroup(0, this.ctx.frameBG);
    pass.setBindGroup(1, worldBG);
    pass.setBindGroup(2, this.updateBG);
    pass.dispatchWorkgroups(Math.ceil((HERD_TAB_NS * HERD_TAB_NL) / 64));
    pass.end();
    this.pTable = null;
  }
  private pTable: GPUComputePipeline | null = null;

  encodeUpdate(pass: GPUComputePassEncoder, worldBG: GPUBindGroup) {
    pass.setPipeline(this.pUpdate);
    pass.setBindGroup(0, this.ctx.frameBG);
    pass.setBindGroup(1, worldBG);
    pass.setBindGroup(2, this.updateBG);
    pass.dispatchWorkgroups(UPDATE_GROUPS);
  }

  private record(enc: GPURenderBundleEncoder, cascade: number, pipes: (GPURenderPipeline | null)[]) {
    enc.setBindGroup(3, this.drawBG[cascade]);
    enc.setVertexBuffer(0, this.vbuf);
    enc.setIndexBuffer(this.ibuf, 'uint32');
    for (let m = 0; m < DRAWS.length; m++) {
      const p = pipes[m];
      if (!p) continue;
      const d = this.draws[m];
      enc.setPipeline(p);
      enc.drawIndexed(d.count, DRAWS[m][0], d.first, d.base, DRAWS[m][1]);
    }
  }

  recordMain(enc: GPURenderBundleEncoder) {
    this.record(enc, 0, this.pMain);
  }

  recordShadow(enc: GPURenderBundleEncoder, cascade: number) {
    enc.setBindGroup(0, this.ctx.frameBG);
    enc.setBindGroup(1, this.ctx.emptyBG);
    enc.setBindGroup(2, this.ctx.emptyBG);
    this.record(enc, cascade, this.pShadow);
  }

  // Debug: this frame's full pose buffer (POSE_VEC4 vec4 = 20 floats per slot, in the world.ts
  // order: herd, birds, titans, travellers, colossi, jellies, rays, ships). Allocates; not per frame.
  async debugPoses(): Promise<number[]> {
    const { device } = this.ctx;
    const n = FAUNA_N * POSE_VEC4 * 16;
    const rb = buffer(device, n, U.MAP_READ | U.COPY_DST);
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(this.poseBuf, 0, rb, 0, n);
    device.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const a = Array.from(new Float32Array(rb.getMappedRange().slice(0)));
    rb.unmap(); rb.destroy();
    return a;
  }
}
