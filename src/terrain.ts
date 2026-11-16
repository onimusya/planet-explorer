// Terrain: one-time GPU field generation + CPU CDLOD quadtree selection + geomorphed rendering.
import genSrc from './shaders/terrain_gen.wgsl';
import terrainSrc from './shaders/terrain.wgsl';
import { L, bgl, buffer, bufferWith, computePipe, tex2d, U, T } from './gpu';
import type { Ctx } from './ctx';
import { HMAP_N, WORLD_L, PLANET_R } from './world';

const LODS = 11;
const LEAF = 16;
const GRID = 64;
const MAXN = 768; // per list
const QIDX = (GRID / 2) * (GRID / 2) * 6;
const BLOCKS = HMAP_N / 64;
const CPU_N = HMAP_N / 4;

export class Terrain {
  hTex!: GPUTexture;
  gTex!: GPUTexture;
  aTex!: GPUTexture;
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
    this.hTex = tex2d(device, N, N, 'r32float', T.STORAGE_BINDING | T.TEXTURE_BINDING, 'height', mips);
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
    };
    const [pBase, pFin, pMip, pAttr, pCpu, pMm] = await Promise.all([
      computePipe(device, mod, 'genBase', [fb, lay.base]),
      computePipe(device, mod, 'genFinal', [fb, lay.fin]),
      computePipe(device, mod, 'genMaxMip', [fb, lay.mip]),
      computePipe(device, mod, 'genAttr', [fb, lay.attr]),
      computePipe(device, mod, 'genCpuGrid', [fb, lay.cpu]),
      computePipe(device, mod, 'genMinMax', [fb, lay.mm]),
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

  // CPU height query (bilinear on the 16 m grid).
  heightAt(x: number, z: number): number {
    const s = (CPU_N / WORLD_L);
    let fx = x * s, fz = z * s;
    const ix = Math.floor(fx), iz = Math.floor(fz);
    fx -= ix; fz -= iz;
    const g = this.cpuGrid;
    const w = (i: number) => ((i % CPU_N) + CPU_N) % CPU_N;
    const x0 = w(ix), x1 = w(ix + 1), z0 = w(iz) * CPU_N, z1 = w(iz + 1) * CPU_N;
    const a = g[z0 + x0], b = g[z0 + x1], c = g[z1 + x0], d = g[z1 + x1];
    return a + (b - a) * fx + (c - a) * fz + (a - b - c + d) * fx * fz;
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
  // CDLOD selection. camX/Y/Z: wrapped camera position. vp: camera-relative viewProj (unjittered).
  select(camX: number, camY: number, camZ: number, vp: Float32Array) {
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
    this.cam[0] = camX; this.cam[1] = camY; this.cam[2] = camZ;
    this.counts.fill(0);
    const rs = LEAF << (LODS - 1);
    const R = this.ranges[LODS - 1];
    const x0 = Math.floor((camX - R) / rs), x1 = Math.floor((camX + R) / rs);
    const z0 = Math.floor((camZ - R) / rs), z1 = Math.floor((camZ + R) / rs);
    for (let z = z0; z <= z1; z++) for (let x = x0; x <= x1; x++) this.node(x * rs, z * rs, rs, LODS - 1);
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

  private bounds(x: number, z: number, size: number, out: Float64Array) {
    const bs = WORLD_L / BLOCKS; // 256 m
    let lv = 0;
    let s = bs;
    while (s < size && lv < this.pyr.length - 1) { s *= 2; lv++; }
    const n = BLOCKS >> lv;
    const bx = (((Math.floor(x / s) % n) + n) % n), bz = (((Math.floor(z / s) % n) + n) % n);
    const a = this.pyr[lv];
    out[0] = a[(bz * n + bx) * 2];
    out[1] = a[(bz * n + bx) * 2 + 1];
  }
  private bb = new Float64Array(2);

  private boxDist2(x: number, z: number, size: number, mn: number, mx: number): number {
    const c = this.cam;
    const dx = Math.max(x - c[0], 0, c[0] - (x + size));
    const dy = Math.max(mn - c[1], 0, c[1] - mx);
    const dz = Math.max(z - c[2], 0, c[2] - (z + size));
    return dx * dx + dy * dy + dz * dz;
  }

  private inFrustum(x: number, z: number, size: number, mn: number, mx: number): boolean {
    const c = this.cam, pl = this.planes;
    const ax = x - c[0], az = z - c[2], bx = ax + size, bz = az + size;
    // curvature lowers distant terrain: extend the box downward conservatively
    const far = Math.max(Math.abs(ax), Math.abs(bx)) ** 2 + Math.max(Math.abs(az), Math.abs(bz)) ** 2;
    const ay = mn - c[1] - far / (2 * PLANET_R), by = mx - c[1];
    for (let i = 0; i < 5; i++) {
      const p0 = pl[i * 4], p1 = pl[i * 4 + 1], p2 = pl[i * 4 + 2], p3 = pl[i * 4 + 3];
      const px = p0 > 0 ? bx : ax, py = p1 > 0 ? by : ay, pz = p2 > 0 ? bz : az;
      if (p0 * px + p1 * py + p2 * pz + p3 < 0) return false;
    }
    return true;
  }

  private add(list: number, x: number, z: number, size: number, lod: number) {
    const n = this.counts[list];
    if (n >= MAXN) return;
    const o = (list * MAXN + n) * 4;
    this.inst[o] = x; this.inst[o + 1] = z; this.inst[o + 2] = size; this.inst[o + 3] = lod;
    this.counts[list] = n + 1;
  }

  private node(x: number, z: number, size: number, lod: number): boolean {
    this.bounds(x, z, size, this.bb);
    const mn = this.bb[0], mx = this.bb[1];
    const r = this.ranges[lod];
    if (this.boxDist2(x, z, size, mn, mx) > r * r) return false;
    if (!this.inFrustum(x, z, size, mn, mx)) return true;
    if (lod === 0) { this.add(0, x, z, size, 0); return true; }
    const rc = this.ranges[lod - 1];
    if (this.boxDist2(x, z, size, mn, mx) > rc * rc) { this.add(0, x, z, size, lod); return true; }
    const h = size / 2;
    for (let q = 0; q < 4; q++) {
      const cx = x + (q & 1) * h, cz = z + (q >> 1) * h;
      if (!this.node(cx, cz, h, lod - 1)) this.add(1 + q, x, z, size, lod);
    }
    return true;
  }
}
