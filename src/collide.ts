// Camera collision with the world's solid things.
// Every few frames the GPU gathers what is solid near a probe point just ahead of the camera
// (the col* entry points in shaders/veg_cull.wgsl): the flora instances (from the same scatter as the drawn ones), the
// landmarks, the animals, and a window of exact terrain heights. The CPU reads that back without
// waiting, through a ring of three buffers whose map state it polls. Each frame the camera's move
// is checked against it in short steps: flora, rocks and landmarks by signed distance fields built
// at startup from their meshes (so the camera fits under a mushroom cap or through an arch), the
// animals by capsules, the terrain by its exact heights (the 16 m CPU grid outside the window).
// Whatever the camera would enter pushes it back out along the surface normal, and the velocity
// into that surface is removed, so the camera slides along trunks, cliffs and arches.
// resolve() runs every frame and allocates nothing (scalars live in typed arrays, see clock.ts).
import type { Ctx } from './ctx';
import type { MeshBuilder } from './plants';
import { PART } from './plants';
import { L, bgl, buffer, computePipe, U } from './gpu';
import { HMAP_N, WORLD_L, WATER_Y } from './world';
import type { GroundQuery } from './camera';
import type { SdfData } from './sdf';
import SdfWorker from './sdf.worker?worker&inline';

export const COL_CAP = 256; // veg_cull.wgsl COL_CAP
const COL_WIN = 32; // veg_cull.wgsl COL_WIN
const F_HEAD = 4, F_HEIGHTS = 12, F_INST = F_HEIGHTS + COL_WIN * COL_WIN; // float offsets in the gather output
const OUT_BYTES = (F_INST + COL_CAP * 8) * 4;
const TEXEL = WORLD_L / HMAP_N;
// Flora species gathered (scatter species with solid parts): tree, mushroom, crystal, boulder.
const FLORA = [0, 1, 2, 6];
const SOLID_SPECIES = [0, 1, 2, 6, 7, 8, 9]; // + arch, floating island, pillar (landmarks)
// Mesh parts that are solid: bark, cap, gill, crystal, rock, stem (leaves, petals, bulbs and
// fronds stay passable).
const SOLID_PART = new Uint8Array(16);
for (const p of [PART.BARK, PART.CAP, PART.GILL, PART.CRYSTAL, PART.ROCK, PART.STEM]) SOLID_PART[p] = 1;
export const isSolidSpecies = (s: number) => SOLID_SPECIES.includes(s);

const R_OBJ = 0.6; // camera radius against objects (m)
const CLEAR = 1.2; // camera height kept above the ground (m)
const STEP = 0.35; // longest step of the move between collision checks (m)
const GATHER_EVERY = 4; // frames between gathers

// Animal capsules in model space at scale 1 (z forward, y up, ground at 0; fauna.ts): segment
// endpoints and radius, per kind (veg_cull.wgsl colAnimals: 100 herd, 102 titan, 103 traveller, 104 colossus,
// 105 jelly).
const CAPS: Float64Array[] = [
  new Float64Array([0, 1.2, -0.8, 0, 1.25, 0.75, 0.42, 0, 1.4, 0.8, 0, 2.1, 1.25, 0.2]),
  new Float64Array(0),
  new Float64Array([0, 1.5, -0.95, 0, 1.55, 0.75, 0.5, 0, 1.75, 0.82, 0, 3.2, 1.6, 0.22, 0, 1.45, -1.1, 0, 1.1, -2.9, 0.18,
    -0.3, 0.1, 0.55, -0.3, 1.3, 0.55, 0.17, 0.3, 0.1, 0.55, 0.3, 1.3, 0.55, 0.17,
    -0.32, 0.1, -0.72, -0.32, 1.3, -0.72, 0.17, 0.32, 0.1, -0.72, 0.32, 1.3, -0.72, 0.17]),
  new Float64Array([0, 0.35, 0, 0, 1.55, 0, 0.32]),
  new Float64Array([0, 0.4, 0, 0, 1.45, 0.1, 0.38]),
  new Float64Array([0, -0.2, 0, 0, 0.5, 0, 0.9]),
];

export class Collider {
  private shapes: (SdfData | null)[] = new Array(40).fill(null); // species * 4 + variant
  private pipes: GPUComputePipeline[] = [];
  private layout!: GPUBindGroupLayout;
  private bg!: GPUBindGroup;
  private outBuf!: GPUBuffer;
  private probeBuf!: GPUBuffer;
  private readonly probeData = new Float32Array(8);
  private rb: GPUBuffer[] = [];
  private readonly rbState = new Uint8Array(3); // 0 free, 1 copied, 2 mapping
  private readonly rbProbe = new Float64Array(3 * 3); // unwrapped probe x, z of each copy, and its seq
  private readonly rbLast = new Float64Array(3); // last probe x, z and its frame
  private nMarks = 0;
  private frame = 0;
  private seq = 0;
  // latest gather: unwrapped probe centre (x, z), wrapped centre (x, z), window texel origin (x, z)
  private readonly gp = new Float64Array(6);
  private readonly inst = new Float32Array(COL_CAP * 8);
  private readonly heights = new Float32Array(COL_WIN * COL_WIN);
  private n = 0;
  private hasData = false;
  enabled = true;
  ground: GroundQuery | null = null; // 16 m CPU grid, outside the window
  // registers (no fractional number passes through a call): query point, result, scratch
  private readonly q = new Float64Array(3); // point relative to the probe centre
  private readonly res = new Float64Array(4); // distance (m, negative inside), outward normal
  private readonly l = new Float64Array(4); // local point; [3] = sampled distance
  private readonly th = new Float64Array(3); // terrain height, dh/dx, dh/dz
  private readonly wx = new Float64Array(4);
  private readonly wz = new Float64Array(4);
  private readonly dwx = new Float64Array(4);
  private readonly dwz = new Float64Array(4);
  private readonly cn = new Float64Array(9); // up to three contact normals this frame
  private ncn = 0;
  readonly stats = { voxels: 0, shapes: 0, buildMs: 0 };

  constructor(private ctx: Ctx) {}

  // Pipelines (compiled with the others at startup). The gather's entry points live in the
  // vegetation culling module (veg_cull.wgsl), which shares the scatter and ecology code.
  async init(cullModule: Promise<GPUShaderModule>) {
    const { device } = this.ctx;
    const C = GPUShaderStage.COMPUTE;
    this.layout = bgl(device, [L.uniform(0, C), L.storageRO(6, C), L.storageRO(7, C), L.storage(8), L.uniform(9, C)]);
    const mod = await cullModule;
    const lays = [this.ctx.frameBGL, this.ctx.worldBGL, this.layout];
    this.pipes = await Promise.all(['colBegin', 'colFlora', 'colMarks', 'colAnimals', 'colTerrain'].map((e) => computePipe(device, mod, e, lays)));
    this.outBuf = buffer(device, OUT_BYTES, U.STORAGE | U.COPY_SRC, 'collideOut');
    this.probeBuf = buffer(device, 32, U.UNIFORM | U.COPY_DST, 'collideProbe');
    for (let i = 0; i < 3; i++) this.rb.push(buffer(device, OUT_BYTES, U.MAP_READ | U.COPY_DST, 'collideRead'));
  }

  // Distance fields of the solid flora and landmark meshes (LOD 0), built by a worker while the
  // main thread goes on generating the world. Only the solid parts' triangles are sent.
  buildShapes(meshes: { s: number; v: number; mb: MeshBuilder }[]): Promise<void> {
    const t0 = performance.now();
    const jobs: { key: number; P: Float32Array; tris: Uint32Array; res: number; closeR: number }[] = [];
    const transfer: ArrayBuffer[] = [];
    for (const { s, v, mb } of meshes) {
      const tris: number[] = [];
      for (const list of [mb.opaque, mb.cutout]) {
        for (let i = 0; i < list.length; i += 3) if (SOLID_PART[mb.at[list[i] * 4]]) tris.push(list[i], list[i + 1], list[i + 2]);
      }
      const P = new Float32Array(mb.p), T = new Uint32Array(tris);
      // landmarks get finer grids; tree crowns close by one voxel only, so the gaps between
      // their branches stay open
      jobs.push({ key: s * 4 + v, P, tris: T, res: s >= 7 ? 48 : 32, closeR: s === 0 ? 1 : 2 });
      transfer.push(P.buffer, T.buffer);
    }
    if (jobs.length === 0) return Promise.resolve();
    const worker = new SdfWorker();
    return new Promise((resolve, reject) => {
      let left = jobs.length;
      worker.onerror = (e) => { worker.terminate(); reject(new Error('collision shapes: ' + e.message)); };
      worker.onmessage = (e: MessageEvent<{ key: number; sdf: SdfData | null }>) => {
        const { key, sdf } = e.data;
        this.shapes[key] = sdf;
        if (sdf) { this.stats.voxels += sdf.d.length; this.stats.shapes++; }
        if (--left === 0) {
          worker.terminate();
          this.stats.buildMs = performance.now() - t0;
          resolve();
        }
      };
      worker.postMessage(jobs, transfer);
    });
  }

  bind(cullU: GPUBuffer, marks: GPUBuffer, nMarks: number, pose: GPUBuffer) {
    this.nMarks = nMarks;
    this.bg = this.ctx.device.createBindGroup({ layout: this.layout, entries: [
      { binding: 0, resource: { buffer: cullU } },
      { binding: 6, resource: { buffer: marks } },
      { binding: 7, resource: { buffer: pose } },
      { binding: 8, resource: { buffer: this.outBuf } },
      { binding: 9, resource: { buffer: this.probeBuf } },
    ] });
  }

  // Every GATHER_EVERY frames, while a readback buffer is free: gather around a point a little
  // ahead of the camera (pos, vel), into the compute pass that has just updated the animal poses.
  // Returns the ring slot to copy into after the pass, or -1.
  encode(pass: GPUComputePassEncoder, worldBG: GPUBindGroup, pos: Float64Array, vel: Float64Array): number {
    if (!this.enabled || ++this.frame % GATHER_EVERY !== 0) return -1;
    let slot = -1;
    for (let i = 0; i < 3; i++) if (this.rbState[i] === 0) { slot = i; break; }
    if (slot < 0) return -1;
    const px = pos[0] + vel[0] * 0.12, pz = pos[2] + vel[2] * 0.12;
    const sp = Math.sqrt(vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2]);
    // a camera at rest near the last probe needs only an occasional refresh (for the animals):
    // each readback allocates its mapping objects
    const lx = px - this.rbLast[0], lz = pz - this.rbLast[1];
    if (this.hasData && sp < 0.5 && lx * lx + lz * lz < 16 && this.frame - this.rbLast[2] < 60) return -1;
    this.rbLast[0] = px; this.rbLast[1] = pz; this.rbLast[2] = this.frame;
    this.seq++;
    const d = this.probeData;
    d[0] = px - Math.floor(px / WORLD_L) * WORLD_L;
    d[1] = pz - Math.floor(pz / WORLD_L) * WORLD_L;
    d[2] = Math.min(24 + sp * 0.25, 60);
    d[3] = this.seq;
    d[4] = FLORA[0]; d[5] = FLORA[1]; d[6] = FLORA[2]; d[7] = FLORA[3];
    this.ctx.device.queue.writeBuffer(this.probeBuf, 0, d);
    this.rbProbe[slot * 3] = px; this.rbProbe[slot * 3 + 1] = pz; this.rbProbe[slot * 3 + 2] = this.seq;
    pass.setBindGroup(0, this.ctx.frameBG);
    pass.setBindGroup(1, worldBG);
    pass.setBindGroup(2, this.bg);
    pass.setPipeline(this.pipes[0]);
    pass.dispatchWorkgroups(1);
    pass.setPipeline(this.pipes[1]);
    pass.dispatchWorkgroups(3, 3, 4);
    if (this.nMarks > 0) {
      pass.setPipeline(this.pipes[2]);
      pass.dispatchWorkgroups(Math.ceil(this.nMarks / 64));
    }
    pass.setPipeline(this.pipes[3]);
    pass.dispatchWorkgroups(2);
    pass.setPipeline(this.pipes[4]);
    pass.dispatchWorkgroups(COL_WIN / 8, COL_WIN / 8);
    return slot;
  }

  // After the compute pass: copy the gather into its readback buffer.
  copy(enc: GPUCommandEncoder, slot: number) {
    if (slot < 0) return;
    enc.copyBufferToBuffer(this.outBuf, 0, this.rb[slot], 0, OUT_BYTES);
    this.rbState[slot] = 1;
  }

  // After submit: start mapping fresh copies and take in the newest finished one (no promise
  // callbacks; mapState is polled).
  poll() {
    let best = -1;
    for (let s = 0; s < 3; s++) {
      const b = this.rb[s];
      if (this.rbState[s] === 1) {
        b.mapAsync(GPUMapMode.READ);
        this.rbState[s] = 2;
      } else if (this.rbState[s] === 2 && b.mapState === 'mapped') {
        if (best < 0 || this.rbProbe[s * 3 + 2] > this.rbProbe[best * 3 + 2]) {
          if (best >= 0) { this.rb[best].unmap(); this.rbState[best] = 0; }
          best = s;
        } else { b.unmap(); this.rbState[s] = 0; }
      }
    }
    if (best < 0) return;
    // every mapping returns a new ArrayBuffer, so the view cannot be kept
    const f = new Float32Array(this.rb[best].getMappedRange());
    if (f[F_HEAD + 1] === this.rbProbe[best * 3 + 2]) {
      this.n = Math.min(f[F_HEAD], COL_CAP);
      this.gp[0] = this.rbProbe[best * 3]; this.gp[1] = this.rbProbe[best * 3 + 1];
      this.gp[2] = f[F_HEAD + 2]; this.gp[3] = f[F_HEAD + 3];
      this.gp[4] = f[F_HEAD + 5]; this.gp[5] = f[F_HEAD + 6];
      for (let i = 0; i < COL_WIN * COL_WIN; i++) this.heights[i] = f[F_HEIGHTS + i];
      for (let i = 0; i < this.n * 8; i++) this.inst[i] = f[F_INST + i];
      this.hasData = true;
    }
    this.rb[best].unmap();
    this.rbState[best] = 0;
  }

  // Debug only: the latest gather (world positions of its instances) and the distance from a
  // point to the nearest solid thing.
  debug(x?: number, y?: number, z?: number) {
    const list: number[][] = [];
    for (let e = 0; e < this.n; e++) {
      const o = e * 8, it = this.inst;
      list.push([it[o + 6], it[o + 7], +(it[o] + this.gp[0]).toFixed(1), +it[o + 1].toFixed(1), +(it[o + 2] + this.gp[1]).toFixed(1), +it[o + 3].toFixed(2)]);
    }
    let near: number[] | null = null;
    if (x !== undefined && y !== undefined && z !== undefined) {
      this.q[0] = x - this.gp[0]; this.q[1] = y; this.q[2] = z - this.gp[1];
      this.nearest();
      near = Array.from(this.res);
    }
    return { hasData: this.hasData, n: this.n, probe: Array.from(this.gp), near, inst: list };
  }
  // Debug only: where a move from a to b ends.
  debugMove(a: number[], b: number[]) {
    const from = new Float64Array(a), to = new Float64Array(b), vel = new Float64Array(3);
    this.resolve(from, to, vel);
    return Array.from(to);
  }

  // Terrain height and gradient at the probe-relative point q into th: Catmull-Rom over the exact
  // window (as heightCubic in world.wgsl), or the CPU grid outside it.
  private terrainAt() {
    const q = this.q, th = this.th, gp = this.gp;
    if (this.hasData) {
      // texel coordinates inside the window (texel i sits at i * TEXEL, wrapped coordinates)
      const tx = (gp[2] + q[0]) / TEXEL - gp[4], tz = (gp[3] + q[2]) / TEXEL - gp[5];
      if (tx >= 1 && tz >= 1 && tx < COL_WIN - 2 && tz < COL_WIN - 2) {
        const ix = Math.floor(tx), iz = Math.floor(tz);
        const fx = tx - ix, fz = tz - iz;
        const wx = this.wx, wz = this.wz, dwx = this.dwx, dwz = this.dwz;
        let t = fx, t2 = t * t, t3 = t2 * t;
        wx[0] = -0.5 * t3 + t2 - 0.5 * t; wx[1] = 1.5 * t3 - 2.5 * t2 + 1; wx[2] = -1.5 * t3 + 2 * t2 + 0.5 * t; wx[3] = 0.5 * t3 - 0.5 * t2;
        dwx[0] = -1.5 * t2 + 2 * t - 0.5; dwx[1] = 4.5 * t2 - 5 * t; dwx[2] = -4.5 * t2 + 4 * t + 0.5; dwx[3] = 1.5 * t2 - t;
        t = fz; t2 = t * t; t3 = t2 * t;
        wz[0] = -0.5 * t3 + t2 - 0.5 * t; wz[1] = 1.5 * t3 - 2.5 * t2 + 1; wz[2] = -1.5 * t3 + 2 * t2 + 0.5 * t; wz[3] = 0.5 * t3 - 0.5 * t2;
        dwz[0] = -1.5 * t2 + 2 * t - 0.5; dwz[1] = 4.5 * t2 - 5 * t; dwz[2] = -4.5 * t2 + 4 * t + 0.5; dwz[3] = 1.5 * t2 - t;
        let h = 0, gx = 0, gz = 0;
        const hs = this.heights;
        for (let j = 0; j < 4; j++) {
          const row = (iz - 1 + j) * COL_WIN + ix - 1;
          for (let i = 0; i < 4; i++) {
            const v = hs[row + i];
            h += v * wx[i] * wz[j];
            gx += v * dwx[i] * wz[j];
            gz += v * wx[i] * dwz[j];
          }
        }
        th[0] = h; th[1] = gx / TEXEL; th[2] = gz / TEXEL;
        return;
      }
    }
    // outside the window: the 16 m grid, gradient by central differences
    const g = this.ground;
    if (!g) { th[0] = 0; th[1] = 0; th[2] = 0; return; }
    const x = this.gp[0] + q[0], z = this.gp[1] + q[2];
    g.q[0] = x; g.q[1] = z; g.heightQ(); th[0] = g.q[2];
    g.q[0] = x + 2; g.heightQ(); th[1] = g.q[2];
    g.q[0] = x - 2; g.heightQ(); th[1] = (th[1] - g.q[2]) / 4;
    g.q[0] = x; g.q[1] = z + 2; g.heightQ(); th[2] = g.q[2];
    g.q[1] = z - 2; g.heightQ(); th[2] = (th[2] - g.q[2]) / 4;
  }

  // Trilinear sample of an SDF at the local point l[0..2] into l[3] (outside the grid: the
  // nearest border value plus the distance to the grid).
  private sample(s: SdfData) {
    const l = this.l;
    let gx = (l[0] - s.ox) / s.vs - 0.5, gy = (l[1] - s.oy) / s.vs - 0.5, gz = (l[2] - s.oz) / s.vs - 0.5;
    const cx = Math.min(Math.max(gx, 0), s.nx - 1.001), cy = Math.min(Math.max(gy, 0), s.ny - 1.001), cz = Math.min(Math.max(gz, 0), s.nz - 1.001);
    const ex = (gx - cx) * s.vs, ey = (gy - cy) * s.vs, ez = (gz - cz) * s.vs;
    gx = cx; gy = cy; gz = cz;
    const i = Math.floor(gx), j = Math.floor(gy), k = Math.floor(gz);
    const fx = gx - i, fy = gy - j, fz = gz - k;
    const nx = s.nx, nxy = s.nx * s.ny, d = s.d;
    const o = (k * s.ny + j) * nx + i;
    const c00 = d[o] + (d[o + 1] - d[o]) * fx;
    const c10 = d[o + nx] + (d[o + nx + 1] - d[o + nx]) * fx;
    const c01 = d[o + nxy] + (d[o + nxy + 1] - d[o + nxy]) * fx;
    const c11 = d[o + nxy + nx] + (d[o + nxy + nx + 1] - d[o + nxy + nx]) * fx;
    const c0 = c00 + (c10 - c00) * fy, c1 = c01 + (c11 - c01) * fy;
    l[3] = c0 + (c1 - c0) * fz + Math.sqrt(ex * ex + ey * ey + ez * ez);
  }

  // Closest solid thing to the probe-relative point q: distance (m, camera radius not included)
  // and outward normal into res. Terrain distance is measured from CLEAR above the ground.
  private nearest() {
    const q = this.q, res = this.res, th = this.th, l = this.l;
    // terrain (and the water surface): distance to the tangent plane above the ground
    this.terrainAt();
    let nx = -th[1], ny = 1, nz = -th[2];
    let inv = 1 / Math.sqrt(nx * nx + ny * ny + nz * nz);
    nx *= inv; ny *= inv; nz *= inv;
    let h = th[0];
    if (h < WATER_Y) { h = WATER_Y; nx = 0; ny = 1; nz = 0; }
    res[0] = (q[1] - h) * ny - CLEAR + R_OBJ; res[1] = nx; res[2] = ny; res[3] = nz;
    // objects
    const it = this.inst;
    for (let e = 0; e < this.n; e++) {
      const o = e * 8;
      const kind = it[o + 6] | 0; // (integer keys: a fractional one would be boxed)
      const dx = q[0] - it[o], dy = q[1] - it[o + 1], dz = q[2] - it[o + 2];
      const sc = it[o + 3];
      if (kind < 100) {
        const s = this.shapes[(kind * 4 + it[o + 7]) | 0];
        if (!s) continue;
        const rr = s.br * sc + R_OBJ + 1;
        if (dx * dx + dy * dy + dz * dz > rr * rr) continue;
        const c = Math.cos(it[o + 4]), sn = Math.sin(it[o + 4]), ys = it[o + 5];
        const k = 1 / sc;
        const lx = (dx * c - dz * sn) * k, ly = dy * k / ys, lz = (dx * sn + dz * c) * k;
        l[0] = lx; l[1] = ly; l[2] = lz;
        this.sample(s);
        const dist = l[3] * sc * Math.min(ys, 1);
        if (dist >= res[0] + 0.5) continue;
        // gradient by central differences in local space, back to world (as normals transform)
        const ep = s.vs * 0.5;
        l[0] = lx + ep; this.sample(s); let gx = l[3];
        l[0] = lx - ep; this.sample(s); gx -= l[3];
        l[0] = lx; l[1] = ly + ep; this.sample(s); let gy = l[3];
        l[1] = ly - ep; this.sample(s); gy -= l[3];
        l[1] = ly; l[2] = lz + ep; this.sample(s); let gz = l[3];
        l[2] = lz - ep; this.sample(s); gz -= l[3];
        let wx = gx * c + gz * sn, wy = gy / ys, wz = -gx * sn + gz * c;
        const wl = Math.sqrt(wx * wx + wy * wy + wz * wz);
        if (wl < 1e-9) continue;
        if (dist < res[0]) { res[0] = dist; res[1] = wx / wl; res[2] = wy / wl; res[3] = wz / wl; }
      } else {
        if (kind - 100 >= CAPS.length) continue;
        const cap = CAPS[kind - 100];
        const rr = sc * 4 + R_OBJ;
        if (dx * dx + dy * dy + dz * dz > rr * rr) continue;
        const c = Math.cos(it[o + 4]), sn = Math.sin(it[o + 4]);
        for (let m = 0; m < cap.length; m += 7) {
          // segment a-b in the world (rotY by the heading, scaled), relative to the instance
          const ax = (cap[m] * c + cap[m + 2] * sn) * sc, ay = cap[m + 1] * sc, az = (-cap[m] * sn + cap[m + 2] * c) * sc;
          const bx = (cap[m + 3] * c + cap[m + 5] * sn) * sc, by = cap[m + 4] * sc, bz = (-cap[m + 3] * sn + cap[m + 5] * c) * sc;
          const ux = bx - ax, uy = by - ay, uz = bz - az;
          const px = dx - ax, py = dy - ay, pz = dz - az;
          const t = Math.min(Math.max((px * ux + py * uy + pz * uz) / Math.max(ux * ux + uy * uy + uz * uz, 1e-9), 0), 1);
          const vx = px - ux * t, vy = py - uy * t, vz = pz - uz * t;
          const vl = Math.sqrt(vx * vx + vy * vy + vz * vz);
          const dist = vl - cap[m + 6] * sc;
          if (dist < res[0] && vl > 1e-6) { res[0] = dist; res[1] = vx / vl; res[2] = vy / vl; res[3] = vz / vl; }
        }
      }
    }
  }

  // Moves the camera from `from` toward `to` (both unwrapped world positions), stopping it at
  // solid things; `to` receives the result and vel loses its component into what it touched.
  resolve(from: Float64Array, to: Float64Array, vel: Float64Array) {
    const q = this.q, res = this.res, gp = this.gp;
    // the gather must cover the move: otherwise only the terrain (CPU grid) is checked
    const ox = to[0] - gp[0], oz = to[2] - gp[1];
    if (this.hasData && ox * ox + oz * oz > 48 * 48) this.hasData = false;
    if (!this.hasData) { gp[0] = to[0]; gp[1] = to[2]; this.n = 0; }
    const mx = to[0] - from[0], my = to[1] - from[1], mz = to[2] - from[2];
    const len = Math.sqrt(mx * mx + my * my + mz * mz);
    const steps = Math.min(Math.max(Math.ceil(len / STEP), 1), 32);
    let px = from[0] - gp[0], py = from[1], pz = from[2] - gp[1];
    this.ncn = 0;
    for (let s = 1; s <= steps; s++) {
      px += mx / steps; py += my / steps; pz += mz / steps;
      for (let iter = 0; iter < 4; iter++) {
        q[0] = px; q[1] = py; q[2] = pz;
        this.nearest();
        const pen = R_OBJ - res[0];
        if (pen <= 0) break;
        px += res[1] * pen; py += res[2] * pen; pz += res[3] * pen;
        if (this.ncn < 3) {
          const c = this.ncn * 3;
          this.cn[c] = res[1]; this.cn[c + 1] = res[2]; this.cn[c + 2] = res[3];
          this.ncn++;
        }
      }
    }
    // never below the ground itself, whatever the normals did
    q[0] = px; q[1] = py; q[2] = pz;
    this.terrainAt();
    const floor = Math.max(this.th[0], WATER_Y) + 0.3;
    if (py < floor) py = floor;
    to[0] = px + gp[0]; to[1] = py; to[2] = pz + gp[1];
    // velocity: drop the part into each surface touched; a head-on hit also brakes the slide, so the
    // camera stops at a trunk instead of skating around it (a glancing one keeps its speed)
    for (let i = 0; i < this.ncn; i++) {
      const nx = this.cn[i * 3], ny = this.cn[i * 3 + 1], nz = this.cn[i * 3 + 2];
      const vn = vel[0] * nx + vel[1] * ny + vel[2] * nz;
      if (vn >= 0) continue;
      const sp = Math.sqrt(vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2]);
      vel[0] -= nx * vn; vel[1] -= ny * vn; vel[2] -= nz * vn;
      const k = 1 - 0.75 * Math.min(-vn / Math.max(sp, 1e-6), 1);
      vel[0] *= k; vel[1] *= k; vel[2] *= k;
    }
  }
}
