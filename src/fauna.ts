// Fauna: a grazing herd on the Bioluminescent Valley floor and flocks of flyers circling the three
// destinations. Meshes are generated once in a rest pose, with one bone and a blend weight per
// vertex. Each frame a compute pass (shaders/fauna.wgsl) writes every animal's pose from the time
// alone, and the vertex shaders in veg.wgsl pose the meshes: legs by two-bone IK with planted feet,
// necks and tails by rotations that grow along their length, wings by shoulder and elbow flaps.
// The animals share the vegetation's fragment shading, so they sit in the same light and shadow.
import faunaSrc from './shaders/fauna.wgsl';
import vegSrc from './shaders/veg.wgsl';
import type { Ctx } from './ctx';
import { L, bgl, buffer, bufferWith, computePipe, U } from './gpu';
import { MeshBuilder, STRIDE } from './plants';
import { FAUNA_NC, FAUNA_NB } from './world';

type V3 = [number, number, number];
const PART = { SKIN: 10, FIN: 11, EYE: 12, HORN: 13, FEATHER: 14, WING: 15 } as const;
const POSE_VEC4 = 5; // per animal, see fauna.wgsl
const RIG_VEC4 = 28; // see poseCreature / birdV in veg.wgsl

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

interface Section { c: V3; rx: number; ry: number; lo?: number; up?: V3 }

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
      for (let j = 0; j <= sides; j++) {
        const ang = (j / sides) * Math.PI * 2;
        const ca = Math.cos(ang), sa = Math.sin(ang);
        const ry = s.ry * (sa < 0 ? s.lo ?? 1 : 1);
        const pos = add(s.c, add(mul(R, ca * s.rx), mul(Uv, sa * ry)));
        const radial = norm(add(mul(R, ca * ry), mul(Uv, sa * s.rx)));
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
  private pCreature!: GPURenderPipeline;
  private pBird!: GPURenderPipeline;
  private pCreatureSh!: GPURenderPipeline;
  private pBirdSh!: GPURenderPipeline;
  stats = { vertices: 0, triangles: 0 };

  constructor(private ctx: Ctx) {}

  async init() {
    const { device } = this.ctx;
    const rig = new Float32Array(RIG_VEC4 * 4);
    const packed = pack([genGrazer(rig).mb, genBird(rig).mb]);
    this.draws = packed.draws;
    this.stats.vertices = packed.vcount;
    this.stats.triangles = packed.tris;
    this.vbuf = bufferWith(device, packed.vb, U.VERTEX, 'faunaVerts');
    this.ibuf = bufferWith(device, packed.ib, U.INDEX, 'faunaIdx');
    this.poseBuf = buffer(device, (FAUNA_NC + FAUNA_NB) * POSE_VEC4 * 16, U.STORAGE | U.COPY_SRC, 'faunaPose');
    const rigBuf = bufferWith(device, rig, U.UNIFORM, 'faunaRig');
    const fb = this.ctx.frameBGL, wb = this.ctx.worldBGL;
    // compute: poses
    const lu = bgl(device, [L.storage(0)]);
    const cmod = await this.ctx.module(faunaSrc, 'fauna');
    this.pUpdate = await computePipe(device, cmod, 'main', [fb, wb, lu]);
    this.updateBG = device.createBindGroup({ layout: lu, entries: [{ binding: 0, resource: { buffer: this.poseBuf } }] });
    // draw: cascade index (shadows), poses, rig
    const V = GPUShaderStage.VERTEX;
    const ld = bgl(device, [L.uniform(0, V | GPUShaderStage.FRAGMENT), L.storageRO(1, V), L.uniform(2, V)]);
    for (let c = 0; c < 4; c++) {
      const u = bufferWith(device, new Uint32Array([c, 0, 0, 0]), U.UNIFORM);
      this.drawBG.push(device.createBindGroup({ layout: ld, entries: [
        { binding: 0, resource: { buffer: u } },
        { binding: 1, resource: { buffer: this.poseBuf } },
        { binding: 2, resource: { buffer: rigBuf } },
      ] }));
    }
    const vmod = await this.ctx.module(vegSrc, 'veg');
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
    [this.pCreature, this.pBird, this.pCreatureSh, this.pBirdSh] = await Promise.all([
      main('creature', 'vsCreature'), main('bird', 'vsBird'),
      shadow('creatureShadow', 'vsCreatureShadow'), shadow('birdShadow', 'vsBirdShadow'),
    ]);
  }

  encodeUpdate(pass: GPUComputePassEncoder, worldBG: GPUBindGroup) {
    pass.setPipeline(this.pUpdate);
    pass.setBindGroup(0, this.ctx.frameBG);
    pass.setBindGroup(1, worldBG);
    pass.setBindGroup(2, this.updateBG);
    pass.dispatchWorkgroups(Math.ceil((FAUNA_NC + FAUNA_NB) / 64));
  }

  private record(enc: GPURenderBundleEncoder, cascade: number, creature: GPURenderPipeline, bird: GPURenderPipeline) {
    enc.setBindGroup(3, this.drawBG[cascade]);
    enc.setVertexBuffer(0, this.vbuf);
    enc.setIndexBuffer(this.ibuf, 'uint32');
    const [g, b] = this.draws;
    enc.setPipeline(creature);
    enc.drawIndexed(g.count, FAUNA_NC, g.first, g.base, 0);
    enc.setPipeline(bird);
    enc.drawIndexed(b.count, FAUNA_NB, b.first, b.base, 0);
  }

  recordMain(enc: GPURenderBundleEncoder) {
    this.record(enc, 0, this.pCreature, this.pBird);
  }

  recordShadow(enc: GPURenderBundleEncoder, cascade: number) {
    enc.setBindGroup(0, this.ctx.frameBG);
    enc.setBindGroup(1, this.ctx.emptyBG);
    enc.setBindGroup(2, this.ctx.emptyBG);
    this.record(enc, cascade, this.pCreatureSh, this.pBirdSh);
  }

  // Debug: this frame's full pose buffer (POSE_VEC4 vec4 = 20 floats per animal, herd first, then
  // birds). Allocates; not used per frame.
  async debugPoses(): Promise<number[]> {
    const { device } = this.ctx;
    const n = (FAUNA_NC + FAUNA_NB) * POSE_VEC4 * 16;
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
