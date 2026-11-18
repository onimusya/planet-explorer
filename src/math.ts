// Allocation-free math helpers. All functions write into caller-provided typed arrays.
// Matrices are column-major Float32Array(16) (WGSL mat4x4f layout).
// Helpers used every frame take their fractional inputs in arrays too (see clock.ts).

export type V3 = Float64Array; // length 3
export type Q4 = Float64Array; // quaternion x,y,z,w
export type M4 = Float32Array; // length 16

export const v3 = (x = 0, y = 0, z = 0): V3 => {
  const a = new Float64Array(3);
  a[0] = x; a[1] = y; a[2] = z;
  return a;
};
export const q4 = (): Q4 => {
  const a = new Float64Array(4);
  a[3] = 1;
  return a;
};
export const m4 = (): M4 => new Float32Array(16);

export function v3set(o: V3, x: number, y: number, z: number): V3 {
  o[0] = x; o[1] = y; o[2] = z;
  return o;
}
export function v3copy(o: V3, a: V3): V3 {
  o[0] = a[0]; o[1] = a[1]; o[2] = a[2];
  return o;
}
export function v3len(a: V3): number {
  return Math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
}
export function v3norm(o: V3, a: V3): V3 {
  const l = Math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]) || 1;
  o[0] = a[0] / l; o[1] = a[1] / l; o[2] = a[2] / l;
  return o;
}
export function v3cross(o: V3, a: V3, b: V3): V3 {
  const x = a[1] * b[2] - a[2] * b[1];
  const y = a[2] * b[0] - a[0] * b[2];
  const z = a[0] * b[1] - a[1] * b[0];
  o[0] = x; o[1] = y; o[2] = z;
  return o;
}
export function v3dot(a: V3, b: V3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

// Quaternions -------------------------------------------------------------------------------
export function qnorm(q: Q4): Q4 {
  // not Math.hypot: V8's hypot builtin allocates a temporary array
  const l = Math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]) || 1;
  q[0] /= l; q[1] /= l; q[2] /= l; q[3] /= l;
  return q;
}
export function qmul(o: Q4, a: Q4, b: Q4): Q4 {
  const ax = a[0], ay = a[1], az = a[2], aw = a[3];
  const bx = b[0], by = b[1], bz = b[2], bw = b[3];
  o[0] = aw * bx + ax * bw + ay * bz - az * by;
  o[1] = aw * by - ax * bz + ay * bw + az * bx;
  o[2] = aw * bz + ax * by - ay * bx + az * bw;
  o[3] = aw * bw - ax * bx - ay * by - az * bz;
  return o;
}
export function qaxis(o: Q4, x: number, y: number, z: number, angle: number): Q4 {
  const s = Math.sin(angle * 0.5);
  o[0] = x * s; o[1] = y * s; o[2] = z * s; o[3] = Math.cos(angle * 0.5);
  return o;
}
// rotate vector v by quaternion q
export function qrot(o: V3, q: Q4, x: number, y: number, z: number): V3 {
  const qx = q[0], qy = q[1], qz = q[2], qw = q[3];
  const tx = 2 * (qy * z - qz * y);
  const ty = 2 * (qz * x - qx * z);
  const tz = 2 * (qx * y - qy * x);
  o[0] = x + qw * tx + (qy * tz - qz * ty);
  o[1] = y + qw * ty + (qz * tx - qx * tz);
  o[2] = z + qw * tz + (qx * ty - qy * tx);
  return o;
}
// The parameter is tt[0], passed in an array because the camera calls this every frame of a flight.
export function qslerp(o: Q4, a: Q4, b: Q4, tt: Float64Array): Q4 {
  const t = tt[0];
  let bx = b[0], by = b[1], bz = b[2], bw = b[3];
  let c = a[0] * bx + a[1] * by + a[2] * bz + a[3] * bw;
  if (c < 0) { c = -c; bx = -bx; by = -by; bz = -bz; bw = -bw; }
  let k0: number, k1: number;
  if (c > 0.9995) { k0 = 1 - t; k1 = t; }
  else {
    const s = Math.sqrt(1 - c * c);
    const ang = Math.atan2(s, c);
    k0 = Math.sin((1 - t) * ang) / s;
    k1 = Math.sin(t * ang) / s;
  }
  o[0] = a[0] * k0 + bx * k1; o[1] = a[1] * k0 + by * k1;
  o[2] = a[2] * k0 + bz * k1; o[3] = a[3] * k0 + bw * k1;
  return qnorm(o);
}
// Quaternion from an orthonormal basis (right, up, back) where the camera looks down -back.
export function qfromBasis(o: Q4, r: V3, u: V3, b: V3): Q4 {
  const m00 = r[0], m01 = u[0], m02 = b[0];
  const m10 = r[1], m11 = u[1], m12 = b[1];
  const m20 = r[2], m21 = u[2], m22 = b[2];
  const tr = m00 + m11 + m22;
  if (tr > 0) {
    const s = Math.sqrt(tr + 1) * 2;
    o[3] = 0.25 * s; o[0] = (m21 - m12) / s; o[1] = (m02 - m20) / s; o[2] = (m10 - m01) / s;
  } else if (m00 > m11 && m00 > m22) {
    const s = Math.sqrt(1 + m00 - m11 - m22) * 2;
    o[3] = (m21 - m12) / s; o[0] = 0.25 * s; o[1] = (m01 + m10) / s; o[2] = (m02 + m20) / s;
  } else if (m11 > m22) {
    const s = Math.sqrt(1 + m11 - m00 - m22) * 2;
    o[3] = (m02 - m20) / s; o[0] = (m01 + m10) / s; o[1] = 0.25 * s; o[2] = (m12 + m21) / s;
  } else {
    const s = Math.sqrt(1 + m22 - m00 - m11) * 2;
    o[3] = (m10 - m01) / s; o[0] = (m02 + m20) / s; o[1] = (m12 + m21) / s; o[2] = 0.25 * s;
  }
  return qnorm(o);
}
const _r = v3(), _u = v3(), _b = v3(), _d = v3();
// Look-at orientation (world up = +Y): the camera looks along d (any length).
export function qlookDirV(o: Q4, d: V3): Q4 {
  _b[0] = -d[0]; _b[1] = -d[1]; _b[2] = -d[2]; v3norm(_b, _b);
  v3set(_u, 0, 1, 0);
  v3cross(_r, _u, _b); v3norm(_r, _r);
  v3cross(_u, _b, _r);
  return qfromBasis(o, _r, _u, _b);
}
export function qlookDir(o: Q4, dx: number, dy: number, dz: number): Q4 {
  _d[0] = dx; _d[1] = dy; _d[2] = dz;
  return qlookDirV(o, _d);
}

// Matrices -----------------------------------------------------------------------------------
export function m4mul(o: M4, a: M4, b: M4): M4 {
  // o = a * b (o must not alias a or b)
  for (let c = 0; c < 4; c++) {
    const b0 = b[c * 4], b1 = b[c * 4 + 1], b2 = b[c * 4 + 2], b3 = b[c * 4 + 3];
    o[c * 4] = a[0] * b0 + a[4] * b1 + a[8] * b2 + a[12] * b3;
    o[c * 4 + 1] = a[1] * b0 + a[5] * b1 + a[9] * b2 + a[13] * b3;
    o[c * 4 + 2] = a[2] * b0 + a[6] * b1 + a[10] * b2 + a[14] * b3;
    o[c * 4 + 3] = a[3] * b0 + a[7] * b1 + a[11] * b2 + a[15] * b3;
  }
  return o;
}
export function m4invert(o: M4, m: M4): boolean {
  const a00 = m[0], a01 = m[1], a02 = m[2], a03 = m[3];
  const a10 = m[4], a11 = m[5], a12 = m[6], a13 = m[7];
  const a20 = m[8], a21 = m[9], a22 = m[10], a23 = m[11];
  const a30 = m[12], a31 = m[13], a32 = m[14], a33 = m[15];
  const b00 = a00 * a11 - a01 * a10, b01 = a00 * a12 - a02 * a10;
  const b02 = a00 * a13 - a03 * a10, b03 = a01 * a12 - a02 * a11;
  const b04 = a01 * a13 - a03 * a11, b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30, b07 = a20 * a32 - a22 * a30;
  const b08 = a20 * a33 - a23 * a30, b09 = a21 * a32 - a22 * a31;
  const b10 = a21 * a33 - a23 * a31, b11 = a22 * a33 - a23 * a32;
  let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (!det) return false;
  det = 1 / det;
  o[0] = (a11 * b11 - a12 * b10 + a13 * b09) * det;
  o[1] = (a02 * b10 - a01 * b11 - a03 * b09) * det;
  o[2] = (a31 * b05 - a32 * b04 + a33 * b03) * det;
  o[3] = (a22 * b04 - a21 * b05 - a23 * b03) * det;
  o[4] = (a12 * b08 - a10 * b11 - a13 * b07) * det;
  o[5] = (a00 * b11 - a02 * b08 + a03 * b07) * det;
  o[6] = (a32 * b02 - a30 * b05 - a33 * b01) * det;
  o[7] = (a20 * b05 - a22 * b02 + a23 * b01) * det;
  o[8] = (a10 * b10 - a11 * b08 + a13 * b06) * det;
  o[9] = (a01 * b08 - a00 * b10 - a03 * b06) * det;
  o[10] = (a30 * b04 - a31 * b02 + a33 * b00) * det;
  o[11] = (a21 * b02 - a20 * b04 - a23 * b00) * det;
  o[12] = (a11 * b07 - a10 * b09 - a12 * b06) * det;
  o[13] = (a00 * b09 - a01 * b07 + a02 * b06) * det;
  o[14] = (a31 * b01 - a30 * b03 - a32 * b00) * det;
  o[15] = (a20 * b03 - a21 * b01 + a22 * b00) * det;
  return true;
}
// Reversed-Z infinite perspective (depth 1 at near plane, 0 at infinity), right-handed view.
export function m4perspRevInf(o: M4, fovY: number, aspect: number, near: number, jx: number, jy: number): M4 {
  const f = 1 / Math.tan(fovY * 0.5);
  o.fill(0);
  o[0] = f / aspect;
  o[5] = f;
  o[8] = -jx; // NDC jitter: clip.x += -jx * z_view = jx * w (w = -z_view)
  o[9] = -jy;
  o[11] = -1;
  o[14] = near;
  return o;
}
// Rotation-only view matrix from camera basis (right, up, forward).
export function m4viewBasis(o: M4, r: V3, u: V3, f: V3): M4 {
  o[0] = r[0]; o[4] = r[1]; o[8] = r[2]; o[12] = 0;
  o[1] = u[0]; o[5] = u[1]; o[9] = u[2]; o[13] = 0;
  o[2] = -f[0]; o[6] = -f[1]; o[10] = -f[2]; o[14] = 0;
  o[3] = 0; o[7] = 0; o[11] = 0; o[15] = 1;
  return o;
}
// Orthographic projection for shadow maps (WebGPU depth 0..1), right-handed light view.
export function m4ortho(o: M4, l: number, r: number, b: number, t: number, n: number, f: number): M4 {
  o.fill(0);
  o[0] = 2 / (r - l);
  o[5] = 2 / (t - b);
  o[10] = -1 / (f - n);
  o[12] = -(r + l) / (r - l);
  o[13] = -(t + b) / (t - b);
  o[14] = -n / (f - n);
  o[15] = 1;
  return o;
}

export const clamp = (x: number, a: number, b: number) => (x < a ? a : x > b ? b : x);
export const smoothstep = (a: number, b: number, x: number) => {
  const t = clamp((x - a) / (b - a), 0, 1);
  return t * t * (3 - 2 * t);
};
export const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
export function halton(i: number, b: number): number {
  let f = 1, r = 0;
  while (i > 0) { f /= b; r += f * (i % b); i = Math.floor(i / b); }
  return r;
}
// Deterministic PRNG (mulberry32) for startup-time generation.
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
