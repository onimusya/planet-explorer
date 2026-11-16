// Six-degree-of-freedom free-flight camera with smoothing, cinematic fly-to and scripted paths.
import { Input, K } from './input';
import { v3, q4, qaxis, qmul, qnorm, qrot, qslerp, qlookDir, clamp, type V3, type Q4 } from './math';

export interface Pose { pos: V3; q: Q4 }

const TMPQ = q4();
const TMPV = v3();

function smoother(t: number): number {
  return t * t * t * (t * (t * 6 - 15) + 10);
}

export class Camera {
  readonly pos = v3();
  readonly prevPos = v3();
  readonly q = q4();
  readonly right = v3(1, 0, 0);
  readonly up = v3(0, 1, 0);
  readonly fwd = v3(0, 0, -1);
  readonly vel = v3();
  readonly fovY = (62 * Math.PI) / 180;
  speed = 35;
  private pending = new Float64Array(3); // pending pitch/yaw rotation (rad)
  private rollVel = 0;
  // cinematic flight state
  private flying = false;
  private flyT = 0;
  private flyDur = 1;
  private readonly fromPos = v3();
  private readonly toPos = v3();
  private readonly fromQ = q4();
  private readonly toQ = q4();
  private arc = 0;
  // scripted path (benchmark): flat arrays of keyframes
  private path: Float64Array | null = null;
  private pathT = 0;
  private pathDur = 0;
  private pathKeys = 0;
  onPathDone: (() => void) | null = null;
  groundAt: ((x: number, z: number) => number) | null = null;

  setPose(x: number, y: number, z: number, dx: number, dy: number, dz: number) {
    this.pos[0] = x; this.pos[1] = y; this.pos[2] = z;
    this.prevPos[0] = x; this.prevPos[1] = y; this.prevPos[2] = z;
    qlookDir(this.q, dx, dy, dz);
    this.vel.fill(0);
    this.pending.fill(0);
    this.rollVel = 0;
    this.flying = false;
    this.path = null;
    this.updateBasis();
  }

  flyTo(x: number, y: number, z: number, dx: number, dy: number, dz: number) {
    this.path = null;
    this.fromPos.set(this.pos);
    this.toPos[0] = x; this.toPos[1] = y; this.toPos[2] = z;
    this.fromQ.set(this.q);
    qlookDir(this.toQ, dx, dy, dz);
    const dist = Math.hypot(x - this.pos[0], y - this.pos[1], z - this.pos[2]);
    this.flyDur = clamp(dist / 450, 3.5, 11);
    this.arc = Math.min(dist * 0.12, 900);
    this.flyT = 0;
    this.flying = true;
    this.vel.fill(0);
  }

  // keys: [x, y, z, dx, dy, dz] * n ; segment duration in seconds
  playPath(keys: Float64Array, segDur: number) {
    this.path = keys;
    this.pathKeys = keys.length / 6;
    this.pathDur = segDur;
    this.pathT = 0;
    this.flying = false;
    this.vel.fill(0);
  }
  get busy() { return this.flying || this.path !== null; }

  private updateBasis() {
    qrot(this.right, this.q, 1, 0, 0);
    qrot(this.up, this.q, 0, 1, 0);
    qrot(this.fwd, this.q, 0, 0, -1);
  }

  update(dt: number, input: Input) {
    this.prevPos.set(this.pos);
    const m = input.mouse;
    if (m[2] !== 0) {
      this.speed = clamp(this.speed * Math.pow(1.3, m[2]), 1, 4000);
      m[2] = 0;
    }
    const sens = 0.0021;
    this.pending[0] += -m[1] * sens; // pitch
    this.pending[1] += -m[0] * sens; // yaw
    m[0] = 0; m[1] = 0;
    const userActive = this.pending[0] !== 0 || this.pending[1] !== 0 || input.keys[K.W] || input.keys[K.S] || input.keys[K.A] || input.keys[K.D] || input.keys[K.UP] || input.keys[K.DOWN] || input.keys[K.Q] || input.keys[K.E];
    if (userActive && (this.flying || this.path)) { this.flying = false; this.path = null; }

    if (this.path) { this.stepPath(dt); this.updateBasis(); return; }
    if (this.flying) { this.stepFly(dt); this.updateBasis(); return; }

    // smoothed look
    const f = 1 - Math.exp(-dt / 0.03);
    const pitch = this.pending[0] * f;
    const yaw = this.pending[1] * f;
    this.pending[0] -= pitch;
    this.pending[1] -= yaw;
    const rollIn = (input.keys[K.Q] ? 1 : 0) - (input.keys[K.E] ? 1 : 0);
    this.rollVel += (rollIn * 1.4 - this.rollVel) * (1 - Math.exp(-dt * 6));
    if (yaw !== 0) qmul(this.q, this.q, qaxis(TMPQ, 0, 1, 0, yaw));
    if (pitch !== 0) qmul(this.q, this.q, qaxis(TMPQ, 1, 0, 0, pitch));
    if (this.rollVel !== 0) qmul(this.q, this.q, qaxis(TMPQ, 0, 0, 1, this.rollVel * dt));
    qnorm(this.q);
    this.updateBasis();

    // movement in camera space
    const mz = (input.keys[K.W] ? 1 : 0) - (input.keys[K.S] ? 1 : 0);
    const mx = (input.keys[K.D] ? 1 : 0) - (input.keys[K.A] ? 1 : 0);
    const my = (input.keys[K.UP] ? 1 : 0) - (input.keys[K.DOWN] ? 1 : 0);
    let sp = this.speed * (input.keys[K.SHIFT] ? 6 : 1);
    const g = this.groundAt ? this.groundAt(this.pos[0], this.pos[2]) : 0;
    const agl = this.pos[1] - g;
    sp *= clamp(0.55 + agl / 220, 0.55, 5);
    const tx = (this.fwd[0] * mz + this.right[0] * mx + this.up[0] * my) * sp;
    const ty = (this.fwd[1] * mz + this.right[1] * mx + this.up[1] * my) * sp;
    const tz = (this.fwd[2] * mz + this.right[2] * mx + this.up[2] * my) * sp;
    const k = 1 - Math.exp(-dt * 5);
    this.vel[0] += (tx - this.vel[0]) * k;
    this.vel[1] += (ty - this.vel[1]) * k;
    this.vel[2] += (tz - this.vel[2]) * k;
    this.pos[0] += this.vel[0] * dt;
    this.pos[1] += this.vel[1] * dt;
    this.pos[2] += this.vel[2] * dt;
    // soft ground clearance (keeps the camera from slipping under the surface)
    if (this.groundAt) {
      const gh = this.groundAt(this.pos[0], this.pos[2]) + 1.2;
      if (this.pos[1] < gh) { this.pos[1] = gh; if (this.vel[1] < 0) this.vel[1] = 0; }
    }
  }

  private stepFly(dt: number) {
    this.flyT += dt / this.flyDur;
    const t = clamp(this.flyT, 0, 1);
    const s = smoother(t);
    const lift = Math.sin(Math.PI * s) * this.arc;
    this.pos[0] = this.fromPos[0] + (this.toPos[0] - this.fromPos[0]) * s;
    this.pos[1] = this.fromPos[1] + (this.toPos[1] - this.fromPos[1]) * s + lift;
    this.pos[2] = this.fromPos[2] + (this.toPos[2] - this.fromPos[2]) * s;
    if (this.groundAt) {
      const gh = this.groundAt(this.pos[0], this.pos[2]) + 25 * Math.sin(Math.PI * s);
      if (this.pos[1] < gh) this.pos[1] = gh;
    }
    qslerp(this.q, this.fromQ, this.toQ, smoother(clamp(t * 1.15, 0, 1)));
    if (this.flyT >= 1) this.flying = false;
  }

  private stepPath(dt: number) {
    const keys = this.path!;
    this.pathT += dt / this.pathDur;
    const n = this.pathKeys;
    if (this.pathT >= n - 1) {
      this.pathT = n - 1;
      this.path = null;
      const cb = this.onPathDone;
      if (cb) cb();
    }
    const seg = Math.min(Math.floor(this.pathT), n - 2);
    const t = this.pathT - seg;
    // Catmull-Rom through key positions and look directions
    const i0 = Math.max(seg - 1, 0), i1 = seg, i2 = seg + 1, i3 = Math.min(seg + 2, n - 1);
    for (let c = 0; c < 6; c++) {
      const p0 = keys[i0 * 6 + c], p1 = keys[i1 * 6 + c], p2 = keys[i2 * 6 + c], p3 = keys[i3 * 6 + c];
      const t2 = t * t, t3 = t2 * t;
      const v = 0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3);
      if (c < 3) this.pos[c] = v; else TMPV[c - 3] = v;
    }
    qlookDir(this.q, TMPV[0], TMPV[1], TMPV[2]);
  }
}
