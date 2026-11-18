// Terrain: one-time GPU field generation + CPU CDLOD quadtree selection + geomorphed rendering.
import genSrc from './shaders/terrain_gen.wgsl';
import terrainSrc from './shaders/terrain.wgsl';
import { L, bgl, buffer, bufferWith, computePipe, tex2d, U, T } from './gpu';
import type { Ctx } from './ctx';
import { HMAP_N, WORLD_L, PLANET_R, TRAIL_N } from './world';

const LODS = 11;
const LEAF = 16;
const GRID = 64;
const MAXN = 768; // per list
const QIDX = (GRID / 2) * (GRID / 2) * 6;
const BLOCKS = HMAP_N / 64;
const BLOCK_M = WORLD_L / BLOCKS; // 256 m
const CPU_N = HMAP_N / 4;
const STACK = 4 * LODS; // the walk keeps at most 3 waiting nodes per level, plus the current one

export class Terrain {
  hTex!: GPUTexture;
  gTex!: GPUTexture;
  aTex!: GPUTexture;
  fTex!: GPUTexture; // coarse (1/4 res, mipmapped) height / shadow height / occluder distance
  tTex!: GPUTexture; // herd trail mask (see genTrail in terrain_gen.wgsl)
  cpuGrid = new Float32Array(CPU_N * CPU_N);
  private pyr: Float32Array[] = []; // min/max pyramid, level 0 = 256 m blocks
  pipeline!: GPURenderPipeline;
  private indexBuf!: GPUBuffer;
  private instBuf!: GPUBuffer;
  indirectBuf!: GPUBuffer;
  private tuBuf!: GPUBuffer;
  tuBG!: GPUBindGroup;
  tuBGL!: GPUBindGroupLayout;
  private inst = new Float32Array(5 * MAXN * 4);
  private counts = new Uint32Array(5);
  private indirect = new Uint32Array(25);
  private ranges = new Float64Array(LODS);
  private morph = new Float32Array(64);
  readonly planes = new Float64Array(20);
  private cam = new Float64Array(3);
  readonly q = new Float64Array(3); // height query registers: x, z in; height out
  // quadtree walk stack: node x, z, LOD, height bounds and squared camera distance
  private stX = new Float64Array(STACK);
  private stZ = new Float64Array(STACK);
  private stL = new Int32Array(STACK);
  private stMn = new Float64Array(STACK);
  private stMx = new Float64Array(STACK);
  private stD = new Float64Array(STACK);
  nodeCount = 0;
  rangeScale = 1;

  constructor(private ctx: Ctx) {}

  // ------------------------------------------------------------------------------------------
  async generate(onProgress: (f: number, msg: string) => void) {
    const { device } = this.ctx;
    const N = HMAP_N;
    const mips = Math.log2(N) + 1;
    const base = tex2d(device, N, N, 'r32float', T.STORAGE_BINDING | T.TEXTURE_BINDING, 'baseH');
    const attrTmp = tex2d(device, N, N, 'rgba8unorm', T.STORAGE_BINDING | T.TEXTURE_BINDING, 'attrTmp');
    this.hTex = tex2d(device, N, N, 'r32float', T.STORAGE_BINDING | T.TEXTURE_BINDING | T.COPY_SRC, 'height', mips);
    this.gTex = tex2d(device, N, N, 'rgba16float', T.STORAGE_BINDING | T.TEXTURE_BINDING, 'gradShadow');
    this.aTex = tex2d(device, N, N, 'rgba8unorm', T.STORAGE_BINDING | T.TEXTURE_BINDING, 'attr');
    const gBuf = buffer(device, 16, U.UNIFORM | U.COPY_DST, 'genU');
    const mmBuf = buffer(device, BLOCKS * BLOCKS * 8, U.STORAGE | U.COPY_SRC, 'minmax');
    const cgBuf = buffer(device, CPU_N * CPU_N * 4, U.STORAGE | U.COPY_SRC, 'cpuGrid');
    const mod = await this.ctx.module(genSrc, 'terrain_gen');
    const C = GPUShaderStage.COMPUTE;
    const E = {
      inA: L.tex(0, 'unfilterable-float', '2d', C),
      outR32: L.stTex(1, 'r32float'),
      outRGBA8: L.stTex(2, 'rgba8unorm'),
      inB: L.tex(3, 'unfilterable-float', '2d', C),
      outRGBA16: L.stTex(4, 'rgba16float'),
      inMax: L.tex(5, 'unfilterable-float', '2d', C),
      minmax: L.storage(6),
      G: L.uniform(7, C),
      cpu: L.storage(8),
    };
    const fb = this.ctx.frameBGL;
    const lay = {
      base: bgl(device, [E.outR32, E.G]),
      fin: bgl(device, [E.inA, E.outR32, E.outRGBA8, E.G]),
      mip: bgl(device, [E.outR32, E.inMax]),
      attr: bgl(device, [E.inA, E.outRGBA8, E.inB, E.outRGBA16, E.inMax, E.G]),
      cpu: bgl(device, [E.inA, E.cpu]),
      mm: bgl(device, [E.inA, E.minmax]),
      coarse: bgl(device, [E.inA, E.inB, E.outRGBA16]),
      cmip: bgl(device, [E.inB, E.outRGBA16]),
      trail: bgl(device, [E.outRGBA8]),
    };
    const [pBase, pFin, pMip, pAttr, pCpu, pMm, pCoarse, pCmip, pTrail] = await Promise.all([
      computePipe(device, mod, 'genBase', [fb, lay.base]),
      computePipe(device, mod, 'genFinal', [fb, lay.fin]),
      computePipe(device, mod, 'genMaxMip', [fb, lay.mip]),
      computePipe(device, mod, 'genAttr', [fb, lay.attr]),
      computePipe(device, mod, 'genCpuGrid', [fb, lay.cpu]),
      computePipe(device, mod, 'genMinMax', [fb, lay.mm]),
      computePipe(device, mod, 'genCoarse', [fb, lay.coarse]),
      computePipe(device, mod, 'genCoarseMip', [fb, lay.cmip]),
      computePipe(device, mod, 'genTrail', [fb, lay.trail]),
    ]);
    const v = (t: GPUTexture, mip = 0, count = 1) => t.createView({ baseMipLevel: mip, mipLevelCount: count });
    const mk = (layout: GPUBindGroupLayout, entries: [number, GPUBindingResource][]) =>
      device.createBindGroup({ layout, entries: entries.map(([binding, resource]) => ({ binding, resource })) });
    const bgBase = mk(lay.base, [[1, v(base)], [7, { buffer: gBuf }]]);
    const bgFin = mk(lay.fin, [[0, v(base)], [1, v(this.hTex, 0)], [2, v(attrTmp)], [7, { buffer: gBuf }]]);
    const bgAttr = mk(lay.attr, [[0, v(this.hTex, 0)], [2, v(this.aTex)], [3, v(attrTmp)], [4, v(this.gTex)], [5, v(this.hTex, 0, mips)], [7, { buffer: gBuf }]]);
    const bgCpu = mk(lay.cpu, [[0, v(this.hTex, 0)], [8, { buffer: cgBuf }]]);
    const bgMm = mk(lay.mm, [[0, v(this.hTex, 0)], [6, { buffer: mmBuf }]]);
    const gData = new Uint32Array(4);
    const SLICE = 256;
    const slices = N / SLICE;
    const runSliced = async (pipe: GPUComputePipeline, group: GPUBindGroup, label: string, p0: number, p1: number) => {
      for (let s = 0; s < slices; s++) {
        gData[0] = s * SLICE;
        device.queue.writeBuffer(gBuf, 0, gData);
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(pipe);
        pass.setBindGroup(0, this.ctx.frameBG);
        pass.setBindGroup(1, group);
        pass.dispatchWorkgroups(N / 8, SLICE / 8);
        pass.end();
        device.queue.submit([enc.finish()]);
        if (s % 4 === 3) {
          await device.queue.onSubmittedWorkDone();
          onProgress(p0 + ((p1 - p0) * (s + 1)) / slices, label);
        }
      }
    };
    await runSliced(pBase, bgBase, 'Shaping continents', 0.0, 0.25);
    await runSliced(pFin, bgFin, 'Eroding mountains', 0.25, 0.45);
    {
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(pMip);
      pass.setBindGroup(0, this.ctx.frameBG);
      for (let m = 1; m < mips; m++) {
        const g = mk(lay.mip, [[1, v(this.hTex, m)], [5, v(this.hTex, m - 1)]]);
        pass.setBindGroup(1, g);
        const s = Math.max(1, (N >> m));
        pass.dispatchWorkgroups(Math.ceil(s / 8), Math.ceil(s / 8));
      }
      pass.end();
      device.queue.submit([enc.finish()]);
    }
    await runSliced(pAttr, bgAttr, 'Tracing sunlight', 0.45, 0.9);
    {
      const CN = N / 4, cm = Math.log2(CN) + 1;
      this.fTex = tex2d(device, CN, CN, 'rgba16float', T.STORAGE_BINDING | T.TEXTURE_BINDING, 'coarseField', cm);
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setBindGroup(0, this.ctx.frameBG);
      pass.setPipeline(pCoarse);
      pass.setBindGroup(1, mk(lay.coarse, [[0, v(this.hTex, 0)], [3, v(this.gTex)], [4, v(this.fTex, 0)]]));
      pass.dispatchWorkgroups(CN / 8, CN / 8);
      pass.setPipeline(pCmip);
      for (let m = 1; m < cm; m++) {
        pass.setBindGroup(1, mk(lay.cmip, [[3, v(this.fTex, m - 1)], [4, v(this.fTex, m)]]));
        const s = Math.max(1, CN >> m);
        pass.dispatchWorkgroups(Math.ceil(s / 8), Math.ceil(s / 8));
      }
      this.tTex = tex2d(device, TRAIL_N, TRAIL_N, 'rgba8unorm', T.STORAGE_BINDING | T.TEXTURE_BINDING, 'trail');
      pass.setPipeline(pTrail);
      pass.setBindGroup(1, mk(lay.trail, [[2, v(this.tTex)]]));
      pass.dispatchWorkgroups(Math.ceil(TRAIL_N / 8), Math.ceil(TRAIL_N / 8));
      pass.end();
      device.queue.submit([enc.finish()]);
    }
    {
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setBindGroup(0, this.ctx.frameBG);
      pass.setPipeline(pCpu);
      pass.setBindGroup(1, bgCpu);
      pass.dispatchWorkgroups(CPU_N / 8, CPU_N / 8);
      pass.setPipeline(pMm);
      pass.setBindGroup(1, bgMm);
      pass.dispatchWorkgroups(BLOCKS / 8, BLOCKS / 8);
      pass.end();
      const rb1 = buffer(device, mmBuf.size, U.MAP_READ | U.COPY_DST);
      const rb2 = buffer(device, cgBuf.size, U.MAP_READ | U.COPY_DST);
      enc.copyBufferToBuffer(mmBuf, 0, rb1, 0, mmBuf.size);
      enc.copyBufferToBuffer(cgBuf, 0, rb2, 0, cgBuf.size);
      device.queue.submit([enc.finish()]);
      await Promise.all([rb1.mapAsync(GPUMapMode.READ), rb2.mapAsync(GPUMapMode.READ)]);
      const mm = new Float32Array(rb1.getMappedRange().slice(0));
      this.cpuGrid.set(new Float32Array(rb2.getMappedRange()));
      rb1.unmap(); rb2.unmap(); rb1.destroy(); rb2.destroy();
      this.buildPyramid(mm);
    }
    onProgress(0.92, 'Terrain ready');
    base.destroy();
    attrTmp.destroy();
    mmBuf.destroy();
    cgBuf.destroy();
  }

  private buildPyramid(mm: Float32Array) {
    let n = BLOCKS;
    let cur = new Float32Array(n * n * 2);
    for (let i = 0; i < n * n; i++) {
      cur[i * 2] = mm[i * 2] - 4;
      cur[i * 2 + 1] = mm[i * 2 + 1] + 4;
    }
    this.pyr.push(cur);
    while (n > 1) {
      const m = n >> 1;
      const nx = new Float32Array(m * m * 2);
      for (let y = 0; y < m; y++) for (let x = 0; x < m; x++) {
        let mn = 1e9, mx = -1e9;
        for (let k = 0; k < 4; k++) {
          const i = ((y * 2 + (k >> 1)) * n + x * 2 + (k & 1)) * 2;
          mn = Math.min(mn, cur[i]); mx = Math.max(mx, cur[i + 1]);
        }
        nx[(y * m + x) * 2] = mn; nx[(y * m + x) * 2 + 1] = mx;
      }
      this.pyr.push(nx);
      cur = nx;
      n = m;
    }
  }

  // Debug only (called from the automation API, never per frame): read back a 64x64-texel block of
  // the full-resolution heightmap whose first texel contains (x, z).
  async probe(x: number, z: number): Promise<Float32Array> {
    const { device } = this.ctx;
    const tx = WORLD_L / HMAP_N;
    const ox = Math.floor(x / tx), oz = Math.floor(z / tx);
    const rb = buffer(device, 64 * 64 * 4, U.MAP_READ | U.COPY_DST, 'probe');
    const enc = device.createCommandEncoder();
    enc.copyTextureToBuffer({ texture: this.hTex, origin: { x: ox, y: oz } }, { buffer: rb, bytesPerRow: 256 }, { width: 64, height: 64 });
    device.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(rb.getMappedRange().slice(0));
    rb.unmap(); rb.destroy();
    return out;
  }

  // CPU height query (bilinear on the 16 m grid). Per-frame callers use the query registers:
  // write x, z into q[0], q[1], call heightQ(), read the height from q[2]. No fractional number
  // crosses the call, so nothing is boxed (see clock.ts).
  heightQ(): void {
    const q = this.q, s = CPU_N / WORLD_L, M = CPU_N - 1; // CPU_N is a power of two
    let fx = q[0] * s, fz = q[1] * s;
    const ix = Math.floor(fx), iz = Math.floor(fz);
    fx -= ix; fz -= iz;
    const g = this.cpuGrid;
    const x0 = ix & M, x1 = (ix + 1) & M, z0 = (iz & M) * CPU_N, z1 = ((iz + 1) & M) * CPU_N;
    const a = g[z0 + x0], b = g[z0 + x1], c = g[z1 + x0], d = g[z1 + x1];
    q[2] = a + (b - a) * fx + (c - a) * fz + (a - b - c + d) * fx * fz;
  }
  heightAt(x: number, z: number): number {
    this.q[0] = x; this.q[1] = z;
    this.heightQ();
    return this.q[2];
  }

  // ------------------------------------------------------------------------------------------
  async initRender(colorFormat: GPUTextureFormat) {
    const { device } = this.ctx;
    // index buffer: 4 quadrants, each 32x32 quads over the 65x65 vertex grid
    const idx = new Uint16Array(QIDX * 4);
    let o = 0;
    for (let q = 0; q < 4; q++) {
      const qx = (q & 1) * (GRID / 2), qy = (q >> 1) * (GRID / 2);
      for (let j = qy; j < qy + GRID / 2; j++) for (let i = qx; i < qx + GRID / 2; i++) {
        const a = j * (GRID + 1) + i, b = a + 1, c = a + GRID + 1, d = c + 1;
        if (((i + j) & 1) === 0) { idx[o++] = a; idx[o++] = c; idx[o++] = b; idx[o++] = b; idx[o++] = c; idx[o++] = d; }
        else { idx[o++] = a; idx[o++] = c; idx[o++] = d; idx[o++] = a; idx[o++] = d; idx[o++] = b; }
      }
    }
    this.indexBuf = bufferWith(device, idx, U.INDEX, 'terrainIdx');
    this.instBuf = buffer(device, this.inst.byteLength, U.VERTEX | U.COPY_DST, 'terrainInst');
    this.indirectBuf = buffer(device, this.indirect.byteLength, U.INDIRECT | U.COPY_DST, 'terrainIndirect');
    this.tuBuf = buffer(device, 16 * 16, U.UNIFORM | U.COPY_DST, 'terrainU');
    this.tuBGL = bgl(device, [L.uniform(0)]);
    this.tuBG = device.createBindGroup({ layout: this.tuBGL, entries: [{ binding: 0, resource: { buffer: this.tuBuf } }] });
    const mod = await this.ctx.module(terrainSrc, 'terrain');
    this.pipeline = await device.createRenderPipelineAsync({
      label: 'terrain',
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.ctx.frameBGL, this.ctx.worldBGL, this.ctx.lightBGL, this.tuBGL] }),
      vertex: {
        module: mod, entryPoint: 'vs',
        buffers: [{ arrayStride: 16, stepMode: 'instance', attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x4' }] }],
      },
      fragment: { module: mod, entryPoint: 'fs', targets: [{ format: colorFormat }] },
      primitive: { topology: 'triangle-list', cullMode: 'back', frontFace: 'ccw' },
      depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'greater' },
    });
    this.setRangeScale(1);
  }

  setRangeScale(s: number) {
    this.rangeScale = s;
    for (let l = 0; l < LODS; l++) this.ranges[l] = 44 * s * Math.pow(2, l);
    for (let l = 0; l < LODS; l++) {
      const prev = l === 0 ? 0 : this.ranges[l - 1];
      const end = this.ranges[l];
      this.morph[l * 4] = prev + (end - prev) * 0.68;
      this.morph[l * 4 + 1] = end;
    }
    this.ctx.device.queue.writeBuffer(this.tuBuf, 0, this.morph);
  }

  recordDraw(enc: GPURenderPassEncoder | GPURenderBundleEncoder) {
    enc.setPipeline(this.pipeline);
    enc.setBindGroup(3, this.tuBG);
    enc.setIndexBuffer(this.indexBuf, 'uint16');
    for (let l = 0; l < 5; l++) {
      enc.setVertexBuffer(0, this.instBuf, l * MAXN * 16, MAXN * 16);
      enc.drawIndexedIndirect(this.indirectBuf, l * 20);
    }
  }

  // ------------------------------------------------------------------------------------------
  // CDLOD selection. camPos: wrapped camera position. vp: camera-relative viewProj (unjittered).
  // The quadtree walk is iterative and its helpers take stack indices: a recursive walk passes
  // fractional numbers through calls, which boxes hundreds of them per frame (see clock.ts).
  select(camPos: Float64Array, vp: Float32Array) {
    const pl = this.planes;
    // left, right, bottom, top, near (reversed-Z: clip.z <= clip.w)
    for (let i = 0; i < 4; i++) {
      const row = i >> 1, sgn = (i & 1) ? -1 : 1;
      pl[i * 4] = vp[3] + sgn * vp[row];
      pl[i * 4 + 1] = vp[7] + sgn * vp[4 + row];
      pl[i * 4 + 2] = vp[11] + sgn * vp[8 + row];
      pl[i * 4 + 3] = vp[15] + sgn * vp[12 + row];
    }
    pl[16] = vp[3] - vp[2]; pl[17] = vp[7] - vp[6]; pl[18] = vp[11] - vp[10]; pl[19] = vp[15] - vp[14];
    const cam = this.cam;
    cam[0] = camPos[0]; cam[1] = camPos[1]; cam[2] = camPos[2];
    this.counts.fill(0);
    const rs = LEAF << (LODS - 1);
    const R = this.ranges[LODS - 1];
    const x0 = Math.floor((cam[0] - R) / rs), x1 = Math.floor((cam[0] + R) / rs);
    const z0 = Math.floor((cam[2] - R) / rs), z1 = Math.floor((cam[2] + R) / rs);
    const sx = this.stX, sz = this.stZ, sl = this.stL, ranges = this.ranges;
    for (let rz = z0; rz <= z1; rz++) for (let rx = x0; rx <= x1; rx++) {
      sx[0] = rx * rs; sz[0] = rz * rs; sl[0] = LODS - 1;
      if (!this.inRange(0)) continue;
      let sp = 1;
      while (sp > 0) {
        const i = --sp;
        if (!this.visible(i)) continue;
        const lod = sl[i], x = sx[i], z = sz[i], size = LEAF << lod;
        if (lod === 0) { this.add(0, x, z, size, 0); continue; }
        const rc = ranges[lod - 1];
        if (this.stD[i] > rc * rc) { this.add(0, x, z, size, lod); continue; }
        // Children replace this entry on the stack, pushed in reverse so quadrant 0 is walked
        // first. A child outside its own LOD range is drawn as a quarter of this patch instead.
        const h = size / 2;
        for (let q = 3; q >= 0; q--) {
          sx[sp] = x + (q & 1) * h; sz[sp] = z + (q >> 1) * h; sl[sp] = lod - 1;
          if (this.inRange(sp)) sp++;
          else this.add(1 + q, x, z, size, lod);
        }
      }
    }
    let total = 0;
    for (let l = 0; l < 5; l++) {
      const b = l * 5;
      this.indirect[b] = l === 0 ? QIDX * 4 : QIDX;
      this.indirect[b + 1] = this.counts[l];
      this.indirect[b + 2] = l === 0 ? 0 : (l - 1) * QIDX;
      this.indirect[b + 3] = 0;
      this.indirect[b + 4] = 0;
      total += this.counts[l];
    }
    this.nodeCount = total;
    const q = this.ctx.device.queue;
    q.writeBuffer(this.instBuf, 0, this.inst);
    q.writeBuffer(this.indirectBuf, 0, this.indirect);
  }

  // Loads the height bounds of stack entry i (x, z and LOD set) from the min/max pyramid, stores
  // its squared distance to the camera, and returns whether it lies within its LOD range.
  private inRange(i: number): boolean {
    const lod = this.stL[i], size = LEAF << lod, x = this.stX[i], z = this.stZ[i];
    let lv = 0, s = BLOCK_M;
    while (s < size && lv < this.pyr.length - 1) { s *= 2; lv++; }
    const n = BLOCKS >> lv;
    const bx = ((Math.floor(x / s) % n) + n) % n, bz = ((Math.floor(z / s) % n) + n) % n;
    const a = this.pyr[lv], o = (bz * n + bx) * 2;
    const mn = a[o], mx = a[o + 1];
    this.stMn[i] = mn; this.stMx[i] = mx;
    const c = this.cam;
    const dx = Math.max(x - c[0], 0, c[0] - (x + size));
    const dy = Math.max(mn - c[1], 0, c[1] - mx);
    const dz = Math.max(z - c[2], 0, c[2] - (z + size));
    const d2 = dx * dx + dy * dy + dz * dz;
    this.stD[i] = d2;
    const r = this.ranges[lod];
    return d2 <= r * r;
  }

  // Frustum test of stack entry i (bounds loaded by inRange).
  private visible(i: number): boolean {
    const c = this.cam, pl = this.planes, size = LEAF << this.stL[i];
    const ax = this.stX[i] - c[0], az = this.stZ[i] - c[2], bx = ax + size, bz = az + size;
    // curvature lowers distant terrain: extend the box downward conservatively
    const far = Math.max(Math.abs(ax), Math.abs(bx)) ** 2 + Math.max(Math.abs(az), Math.abs(bz)) ** 2;
    const ay = this.stMn[i] - c[1] - far / (2 * PLANET_R), by = this.stMx[i] - c[1];
    for (let k = 0; k < 5; k++) {
      const p0 = pl[k * 4], p1 = pl[k * 4 + 1], p2 = pl[k * 4 + 2], p3 = pl[k * 4 + 3];
      const px = p0 > 0 ? bx : ax, py = p1 > 0 ? by : ay, pz = p2 > 0 ? bz : az;
      if (p0 * px + p1 * py + p2 * pz + p3 < 0) return false;
    }
    return true;
  }

  // x, z and size are whole metres, so they pass as small integers without boxing.
  private add(list: number, x: number, z: number, size: number, lod: number) {
    const n = this.counts[list];
    if (n >= MAXN) return;
    const o = (list * MAXN + n) * 4;
    this.inst[o] = x; this.inst[o + 1] = z; this.inst[o + 2] = size; this.inst[o + 3] = lod;
    this.counts[list] = n + 1;
  }

}
