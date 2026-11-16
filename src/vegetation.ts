// Vegetation: startup mesh generation, GPU scatter/cull/LOD into indirect draw lists, rendering.
import cullSrc from './shaders/veg_cull.wgsl';
import vegSrc from './shaders/veg.wgsl';
import { L, bgl, buffer, bufferWith, computePipe, U } from './gpu';
import type { Ctx } from './ctx';
import { MeshBuilder, STRIDE, genTree, genMushroom, genCrystal, genFlower, genGlowpod, genShrub, genBoulder, genArch, genIsland, genPillar } from './plants';
import { SEED } from './world';

interface SpeciesDef {
  gen: (seed: number, variant: number, lod: number) => MeshBuilder;
  variants: number; cell: number; grid: number; maxDist: number; lod0: number; lod1: number;
  shNear: number; shFar: number; smin: number; smax: number; radius: number; sink: number; ystretch: number;
  caps: [number, number, number]; shCaps: [number, number];
  cy?: number; // bounding-sphere centre height as a fraction of the radius (default 0.6)
}
// Species with grid 0 are never scattered; they are only placed explicitly (landmarks.ts).
export const SP = { TREE: 0, MUSHROOM: 1, CRYSTAL: 2, FLOWER: 3, GLOWPOD: 4, SHRUB: 5, BOULDER: 6, ARCH: 7, ISLAND: 8, PILLAR: 9 } as const;
const MAXS = 12;
const INFO = 20 + MAXS * 16;

export const SPECIES: SpeciesDef[] = [
  { gen: genTree, variants: 3, cell: 8, grid: 320, maxDist: 1280, lod0: 95, lod1: 380, shNear: 130, shFar: 1100, smin: 1.25, smax: 1.95, radius: 9, sink: 0.5, ystretch: 1.18, caps: [2048, 6144, 16384], shCaps: [2048, 8192] },
  { gen: genMushroom, variants: 3, cell: 16, grid: 160, maxDist: 1280, lod0: 110, lod1: 420, shNear: 130, shFar: 1100, smin: 1.1, smax: 2.0, radius: 8, sink: 0.3, ystretch: 1.2, caps: [1024, 4096, 8192], shCaps: [1024, 4096] },
  { gen: genCrystal, variants: 3, cell: 8, grid: 192, maxDist: 760, lod0: 65, lod1: 250, shNear: 100, shFar: 600, smin: 1.0, smax: 2.4, radius: 3, sink: 0.3, ystretch: 1.3, caps: [2048, 4096, 8192], shCaps: [2048, 4096] },
  { gen: genFlower, variants: 3, cell: 8, grid: 144, maxDist: 570, lod0: 50, lod1: 190, shNear: 70, shFar: 320, smin: 1.2, smax: 2.0, radius: 3, sink: 0.1, ystretch: 1.2, caps: [2048, 4096, 8192], shCaps: [1024, 2048] },
  { gen: genGlowpod, variants: 2, cell: 4, grid: 96, maxDist: 190, lod0: 35, lod1: 100, shNear: 0, shFar: 0, smin: 1.0, smax: 1.9, radius: 1, sink: 0.05, ystretch: 1.3, caps: [4096, 4096, 4096], shCaps: [16, 16] },
  { gen: genShrub, variants: 2, cell: 4, grid: 128, maxDist: 250, lod0: 35, lod1: 110, shNear: 45, shFar: 0, smin: 1.0, smax: 1.9, radius: 1.5, sink: 0.05, ystretch: 1.2, caps: [4096, 8192, 8192], shCaps: [2048, 16] },
  { gen: genBoulder, variants: 3, cell: 8, grid: 192, maxDist: 760, lod0: 55, lod1: 220, shNear: 100, shFar: 600, smin: 0.4, smax: 2.4, radius: 2.6, sink: 0.6, ystretch: 1.0, caps: [2048, 4096, 8192], shCaps: [2048, 4096] },
  { gen: genArch, variants: 3, cell: 1, grid: 0, maxDist: 9000, lod0: 320, lod1: 1500, shNear: 2500, shFar: 4500, smin: 1, smax: 1, radius: 85, sink: 0, ystretch: 1, caps: [16, 32, 64], shCaps: [32, 64], cy: 0.2 },
  { gen: genIsland, variants: 3, cell: 1, grid: 0, maxDist: 14000, lod0: 520, lod1: 2200, shNear: 2500, shFar: 4500, smin: 1, smax: 1, radius: 125, sink: 0, ystretch: 1, caps: [16, 32, 96], shCaps: [32, 64], cy: -0.3 },
  { gen: genPillar, variants: 3, cell: 1, grid: 0, maxDist: 8000, lod0: 260, lod1: 1100, shNear: 2500, shFar: 4500, smin: 1, smax: 1, radius: 42, sink: 0, ystretch: 1, caps: [32, 64, 128], shCaps: [64, 128], cy: 0.5 },
];

interface MeshRec { baseVertex: number; opFirst: number; opCount: number; cutFirst: number; cutCount: number }
interface Draw { list: number; mesh: MeshRec; cut: boolean }

export class Vegetation {
  private vbuf!: GPUBuffer;
  private ibuf!: GPUBuffer;
  private inst!: GPUBuffer;
  private counts!: GPUBuffer;
  private listsBuf!: GPUBuffer;
  private drawMapBuf!: GPUBuffer;
  indirect!: GPUBuffer;
  private cullU!: GPUBuffer;
  readonly cullData = new Float32Array(INFO + 4);
  private marksBuf!: GPUBuffer;
  private nMarks = 0;
  private cullBG!: GPUBindGroup;
  private pReset!: GPUComputePipeline;
  private pCull!: GPUComputePipeline;
  private pMarks!: GPUComputePipeline;
  private pFinal!: GPUComputePipeline;
  private pOpaque!: GPURenderPipeline;
  private pCut!: GPURenderPipeline;
  private pShadow!: GPURenderPipeline;
  private pShadowCut!: GPURenderPipeline;
  cascadeBGL!: GPUBindGroupLayout;
  cascadeBG: GPUBindGroup[] = [];
  private listOffsets: number[] = [];
  private mainDraws: number[] = [];   // draw indices
  private shadowNear: number[] = [];
  private shadowFar: number[] = [];
  private draws: Draw[] = [];
  private nLists = 0;
  maxGrid = 0;
  distScale = 1;
  stats = { vertices: 0, triangles: 0, meshes: 0, instanceCap: 0 };

  constructor(private ctx: Ctx) {}

  async init(marks: Float32Array) {
    const { device } = this.ctx;
    // ---- meshes ----
    const meshes: MeshRec[][][] = [];
    const parts: MeshBuilder[] = [];
    let vcount = 0;
    const recs: { mb: MeshBuilder; rec: MeshRec }[] = [];
    for (let s = 0; s < SPECIES.length; s++) {
      const sd = SPECIES[s];
      meshes.push([]);
      for (let v = 0; v < sd.variants; v++) {
        meshes[s].push([]);
        for (let l = 0; l < 3; l++) {
          const mb = sd.gen(SEED + s * 101, v, l);
          const rec: MeshRec = { baseVertex: vcount, opFirst: 0, opCount: mb.opaque.length, cutFirst: 0, cutCount: mb.cutout.length };
          vcount += mb.count;
          parts.push(mb);
          recs.push({ mb, rec });
          meshes[s][v].push(rec);
        }
      }
    }
    const vb = new ArrayBuffer(vcount * STRIDE);
    const f32 = new Float32Array(vb), i8 = new Int8Array(vb), u16 = new Uint16Array(vb), u8 = new Uint8Array(vb);
    let icount = 0;
    for (const r of recs) icount += r.rec.opCount + r.rec.cutCount;
    const ib = new Uint32Array(icount);
    let vo = 0, io = 0;
    for (const { mb, rec } of recs) {
      for (let i = 0; i < mb.count; i++) {
        const b = (vo + i) * STRIDE;
        f32[b / 4] = mb.p[i * 3]; f32[b / 4 + 1] = mb.p[i * 3 + 1]; f32[b / 4 + 2] = mb.p[i * 3 + 2];
        i8[b + 12] = Math.round(mb.n[i * 3] * 127); i8[b + 13] = Math.round(mb.n[i * 3 + 1] * 127); i8[b + 14] = Math.round(mb.n[i * 3 + 2] * 127); i8[b + 15] = 0;
        u16[b / 2 + 8] = Math.round(Math.min(Math.max(mb.uv[i * 2], 0), 1) * 65535);
        u16[b / 2 + 9] = Math.round(Math.min(Math.max(mb.uv[i * 2 + 1] / 64, 0), 1) * 65535);
        u8[b + 20] = mb.at[i * 4];
        u8[b + 21] = Math.round(Math.min(Math.max(mb.at[i * 4 + 1], 0), 1) * 255);
        u8[b + 22] = Math.round(Math.min(Math.max(mb.at[i * 4 + 2], 0), 1) * 255);
        u8[b + 23] = Math.round(Math.min(Math.max(mb.at[i * 4 + 3], 0), 1) * 255);
      }
      rec.opFirst = io;
      ib.set(mb.opaque, io); io += mb.opaque.length;
      rec.cutFirst = io;
      ib.set(mb.cutout, io); io += mb.cutout.length;
      vo += mb.count;
    }
    this.stats.vertices = vcount; this.stats.triangles = icount / 3; this.stats.meshes = recs.length;
    this.vbuf = bufferWith(device, u8, U.VERTEX, 'vegVerts');
    this.ibuf = bufferWith(device, ib, U.INDEX, 'vegIdx');

    // ---- lists & draws ----
    const listCaps: number[] = [];
    const base: number[] = [], shBase: number[] = [];
    for (let s = 0; s < SPECIES.length; s++) {
      base.push(listCaps.length);
      for (let v = 0; v < SPECIES[s].variants; v++) for (let l = 0; l < 3; l++) listCaps.push(SPECIES[s].caps[l]);
    }
    for (let s = 0; s < SPECIES.length; s++) {
      shBase.push(listCaps.length);
      for (let v = 0; v < SPECIES[s].variants; v++) for (let k = 0; k < 2; k++) listCaps.push(SPECIES[s].shCaps[k]);
    }
    this.nLists = listCaps.length;
    const listsData = new Uint32Array(this.nLists * 2);
    let off = 0;
    for (let i = 0; i < this.nLists; i++) {
      listsData[i * 2] = off; listsData[i * 2 + 1] = listCaps[i];
      this.listOffsets.push(off);
      off += listCaps[i];
    }
    this.stats.instanceCap = off;
    const addDraw = (list: number, mesh: MeshRec, out: number[]) => {
      if (mesh.opCount > 0) { out.push(this.draws.length); this.draws.push({ list, mesh, cut: false }); }
      if (mesh.cutCount > 0) { out.push(this.draws.length); this.draws.push({ list, mesh, cut: true }); }
    };
    for (let s = 0; s < SPECIES.length; s++) for (let v = 0; v < SPECIES[s].variants; v++) {
      for (let l = 0; l < 3; l++) addDraw(base[s] + v * 3 + l, meshes[s][v][l], this.mainDraws);
      if (SPECIES[s].shNear > 0) addDraw(shBase[s] + v * 2, meshes[s][v][1], this.shadowNear);
      if (SPECIES[s].shFar > 0) addDraw(shBase[s] + v * 2 + 1, meshes[s][v][2], this.shadowFar);
    }
    const nd = this.draws.length;
    const ind = new Uint32Array(nd * 5);
    const dmap = new Uint32Array(nd);
    for (let d = 0; d < nd; d++) {
      const dr = this.draws[d];
      ind[d * 5] = dr.cut ? dr.mesh.cutCount : dr.mesh.opCount;
      ind[d * 5 + 1] = 0;
      ind[d * 5 + 2] = dr.cut ? dr.mesh.cutFirst : dr.mesh.opFirst;
      ind[d * 5 + 3] = dr.mesh.baseVertex;
      ind[d * 5 + 4] = 0;
      dmap[d] = dr.list;
    }
    this.indirect = bufferWith(device, ind, U.INDIRECT | U.STORAGE | U.COPY_SRC, 'vegIndirect');
    this.drawMapBuf = bufferWith(device, dmap, U.STORAGE, 'vegDrawMap');
    this.listsBuf = bufferWith(device, listsData, U.STORAGE, 'vegLists');
    this.counts = buffer(device, this.nLists * 4, U.STORAGE, 'vegCounts');
    this.inst = buffer(device, off * 32, U.STORAGE | U.VERTEX, 'vegInst');
    this.cullU = buffer(device, this.cullData.byteLength, U.UNIFORM | U.COPY_DST, 'cullU');
    // species uniform block (static part)
    const cd = this.cullData;
    for (let s = 0; s < SPECIES.length; s++) {
      const sd = SPECIES[s];
      const o = 20 + s * 16;
      this.maxGrid = Math.max(this.maxGrid, sd.grid);
      cd[o] = sd.cell; cd[o + 1] = sd.grid; cd[o + 2] = sd.maxDist; cd[o + 3] = sd.variants;
      cd[o + 4] = sd.lod0; cd[o + 5] = sd.lod1; cd[o + 6] = sd.shNear; cd[o + 7] = sd.shFar;
      cd[o + 8] = sd.smin; cd[o + 9] = sd.smax; cd[o + 10] = sd.radius; cd[o + 11] = base[s];
      cd[o + 12] = shBase[s]; cd[o + 13] = sd.sink; cd[o + 14] = sd.ystretch; cd[o + 15] = sd.cy ?? 0.6;
    }
    cd[INFO] = SPECIES.length; cd[INFO + 1] = nd; cd[INFO + 2] = 1;
    this.nMarks = marks.length / 8;
    this.marksBuf = bufferWith(device, marks.length ? marks : new Float32Array(8), U.STORAGE, 'vegMarks');

    // ---- pipelines ----
    const C = GPUShaderStage.COMPUTE;
    const cl = bgl(device, [L.uniform(0, C), L.storageRO(1, C), L.storage(2), L.storage(3), L.storageRO(4, C), L.storage(5), L.storageRO(6, C)]);
    const cmod = await this.ctx.module(cullSrc, 'veg_cull');
    const fb = this.ctx.frameBGL, wb = this.ctx.worldBGL;
    [this.pReset, this.pCull, this.pMarks, this.pFinal] = await Promise.all([
      computePipe(device, cmod, 'reset', [fb, wb, cl]),
      computePipe(device, cmod, 'cull', [fb, wb, cl]),
      computePipe(device, cmod, 'cullMarks', [fb, wb, cl]),
      computePipe(device, cmod, 'finalize', [fb, wb, cl]),
    ]);
    this.cullBG = device.createBindGroup({ layout: cl, entries: [
      { binding: 0, resource: { buffer: this.cullU } },
      { binding: 1, resource: { buffer: this.listsBuf } },
      { binding: 2, resource: { buffer: this.counts } },
      { binding: 3, resource: { buffer: this.inst } },
      { binding: 4, resource: { buffer: this.drawMapBuf } },
      { binding: 5, resource: { buffer: this.indirect } },
      { binding: 6, resource: { buffer: this.marksBuf } },
    ] });
    this.cascadeBGL = bgl(device, [L.uniform(0, GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT)]);
    for (let c = 0; c < 4; c++) {
      const u = bufferWith(device, new Uint32Array([c, 0, 0, 0]), U.UNIFORM);
      this.cascadeBG.push(device.createBindGroup({ layout: this.cascadeBGL, entries: [{ binding: 0, resource: { buffer: u } }] }));
    }
    const vmod = await this.ctx.module(vegSrc, 'veg');
    const vbl: GPUVertexBufferLayout[] = [
      { arrayStride: STRIDE, attributes: [
        { shaderLocation: 0, offset: 0, format: 'float32x3' },
        { shaderLocation: 1, offset: 12, format: 'snorm8x4' },
        { shaderLocation: 2, offset: 16, format: 'unorm16x2' },
        { shaderLocation: 3, offset: 20, format: 'unorm8x4' },
      ] },
      { arrayStride: 32, stepMode: 'instance', attributes: [
        { shaderLocation: 4, offset: 0, format: 'float32x4' },
        { shaderLocation: 5, offset: 16, format: 'float32x4' },
      ] },
    ];
    const mainLayout = device.createPipelineLayout({ bindGroupLayouts: [fb, wb, this.ctx.lightBGL, this.cascadeBGL] });
    const shLayout = device.createPipelineLayout({ bindGroupLayouts: [fb, this.ctx.emptyBGL, this.ctx.emptyBGL, this.cascadeBGL] });
    const depth: GPUDepthStencilState = { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'greater' };
    const shDepth: GPUDepthStencilState = { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less', depthBias: 2, depthBiasSlopeScale: 2.5, depthBiasClamp: 0.01 };
    [this.pOpaque, this.pCut, this.pShadow, this.pShadowCut] = await Promise.all([
      device.createRenderPipelineAsync({ label: 'vegOpaque', layout: mainLayout, vertex: { module: vmod, entryPoint: 'vs', buffers: vbl },
        fragment: { module: vmod, entryPoint: 'fsOpaque', targets: [{ format: 'rgba16float' }] }, primitive: { topology: 'triangle-list', cullMode: 'none' }, depthStencil: depth }),
      device.createRenderPipelineAsync({ label: 'vegCut', layout: mainLayout, vertex: { module: vmod, entryPoint: 'vs', buffers: vbl },
        fragment: { module: vmod, entryPoint: 'fsCutout', targets: [{ format: 'rgba16float' }] }, primitive: { topology: 'triangle-list', cullMode: 'none' }, depthStencil: depth }),
      device.createRenderPipelineAsync({ label: 'vegShadow', layout: shLayout, vertex: { module: vmod, entryPoint: 'vsShadow', buffers: vbl },
        primitive: { topology: 'triangle-list', cullMode: 'none' }, depthStencil: shDepth }),
      device.createRenderPipelineAsync({ label: 'vegShadowCut', layout: shLayout, vertex: { module: vmod, entryPoint: 'vsShadow', buffers: vbl },
        fragment: { module: vmod, entryPoint: 'fsShadowCut', targets: [] }, primitive: { topology: 'triangle-list', cullMode: 'none' }, depthStencil: shDepth }),
    ]);
  }

  // Debug: read back visible instance counts per (species, variant, lod) list.
  async debugCounts(): Promise<number[]> {
    const { device } = this.ctx;
    const rb = buffer(device, this.draws.length * 20, U.MAP_READ | U.COPY_DST);
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(this.indirect, 0, rb, 0, this.draws.length * 20);
    device.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const a = new Uint32Array(rb.getMappedRange().slice(0));
    rb.unmap(); rb.destroy();
    const out: number[] = [];
    for (let s = 0; s < SPECIES.length; s++) out.push(0);
    for (let d = 0; d < this.draws.length; d++) {
      if (!this.mainDraws.includes(d) || this.draws[d].cut) continue;
      let s = 0, acc = 0;
      for (; s < SPECIES.length; s++) { const n = SPECIES[s].variants * 3; if (this.draws[d].list < acc + n) break; acc += n; }
      if (s < SPECIES.length) out[s] += a[d * 5 + 1];
    }
    return out;
  }

  setPlanes(planes: Float64Array, distScale: number) {
    for (let i = 0; i < 20; i++) this.cullData[i] = planes[i];
    this.cullData[INFO + 2] = distScale;
    this.ctx.device.queue.writeBuffer(this.cullU, 0, this.cullData);
  }

  encodeCull(pass: GPUComputePassEncoder, worldBG: GPUBindGroup) {
    pass.setBindGroup(0, this.ctx.frameBG);
    pass.setBindGroup(1, worldBG);
    pass.setBindGroup(2, this.cullBG);
    pass.setPipeline(this.pReset);
    pass.dispatchWorkgroups(Math.ceil(this.nLists / 64));
    pass.setPipeline(this.pCull);
    pass.dispatchWorkgroups(Math.ceil(this.maxGrid / 8), Math.ceil(this.maxGrid / 8), SPECIES.length);
    if (this.nMarks > 0) {
      pass.setPipeline(this.pMarks);
      pass.dispatchWorkgroups(Math.ceil(this.nMarks / 64));
    }
    pass.setPipeline(this.pFinal);
    pass.dispatchWorkgroups(Math.ceil(this.draws.length / 64));
  }

  private recordList(enc: GPURenderBundleEncoder, idx: number[], opaque: GPURenderPipeline, cut: GPURenderPipeline) {
    enc.setIndexBuffer(this.ibuf, 'uint32');
    enc.setVertexBuffer(0, this.vbuf);
    for (const pass of [false, true]) {
      enc.setPipeline(pass ? cut : opaque);
      for (const d of idx) {
        const dr = this.draws[d];
        if (dr.cut !== pass) continue;
        enc.setVertexBuffer(1, this.inst, this.listOffsets[dr.list] * 32, SPECIES_CAP(dr, this) * 32);
        enc.drawIndexedIndirect(this.indirect, d * 20);
      }
    }
  }

  listCap(list: number) {
    const next = list + 1 < this.listOffsets.length ? this.listOffsets[list + 1] : this.stats.instanceCap;
    return next - this.listOffsets[list];
  }

  recordMain(enc: GPURenderBundleEncoder) {
    enc.setBindGroup(3, this.cascadeBG[0]);
    this.recordList(enc, this.mainDraws, this.pOpaque, this.pCut);
  }

  recordShadow(enc: GPURenderBundleEncoder, cascade: number) {
    enc.setBindGroup(0, this.ctx.frameBG);
    enc.setBindGroup(1, this.ctx.emptyBG);
    enc.setBindGroup(2, this.ctx.emptyBG);
    enc.setBindGroup(3, this.cascadeBG[cascade]);
    this.recordList(enc, cascade < 2 ? this.shadowNear : this.shadowFar, this.pShadow, this.pShadowCut);
  }
}

function SPECIES_CAP(dr: Draw, v: Vegetation) { return v.listCap(dr.list); }
