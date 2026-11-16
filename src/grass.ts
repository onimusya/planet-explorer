// Grass: GPU tile culling + indexed procedural blades (3 LOD draws via indirect args).
import src from './shaders/grass.wgsl';
import { L, bgl, buffer, bufferWith, computePipe, U } from './gpu';
import type { Ctx } from './ctx';

const CAP = 4096;
const BLADES = [160, 56, 18];
const GRID = 96;

export class Grass {
  private pCull!: GPUComputePipeline;
  private pFinal!: GPUComputePipeline;
  private pipe!: GPURenderPipeline;
  private bg!: GPUBindGroup;
  private tiles!: GPUBuffer;
  private ind!: GPUBuffer;
  private ibuf!: GPUBuffer;
  private ubuf!: GPUBuffer;
  readonly udata = new Float32Array(24);
  constructor(private ctx: Ctx) {}

  async init() {
    const { device } = this.ctx;
    const maxB = BLADES[0];
    const idx = new Uint32Array(maxB * 15);
    for (let b = 0; b < maxB; b++) {
      const v = b * 7, o = b * 15;
      const t = [0, 1, 2, 1, 3, 2, 2, 3, 4, 3, 5, 4, 4, 5, 6];
      for (let i = 0; i < 15; i++) idx[o + i] = v + t[i];
    }
    this.ibuf = bufferWith(device, idx, U.INDEX, 'grassIdx');
    const ind = new Uint32Array(15);
    for (let l = 0; l < 3; l++) { ind[l * 5] = BLADES[l] * 15; }
    this.ind = bufferWith(device, ind, U.INDIRECT | U.STORAGE, 'grassIndirect');
    this.tiles = buffer(device, CAP * 3 * 16, U.STORAGE | U.VERTEX, 'grassTiles');
    const counts = buffer(device, 16, U.STORAGE, 'grassCounts');
    this.ubuf = buffer(device, this.udata.byteLength, U.UNIFORM | U.COPY_DST, 'grassU');
    const C = GPUShaderStage.COMPUTE;
    const lay = bgl(device, [L.uniform(0), L.storage(1), L.storage(2), L.storage(3)]);
    const layR = bgl(device, [L.uniform(0)]);
    const mod = await this.ctx.module(src, 'grass');
    const fb = this.ctx.frameBGL, wb = this.ctx.worldBGL;
    [this.pCull, this.pFinal] = await Promise.all([
      computePipe(device, mod, 'cull', [fb, wb, this.ctx.emptyBGL, lay]),
      computePipe(device, mod, 'finalize', [fb, wb, this.ctx.emptyBGL, lay]),
    ]);
    void C;
    this.bg = device.createBindGroup({ layout: lay, entries: [
      { binding: 0, resource: { buffer: this.ubuf } },
      { binding: 1, resource: { buffer: counts } },
      { binding: 2, resource: { buffer: this.tiles } },
      { binding: 3, resource: { buffer: this.ind } },
    ] });
    // render pipeline uses group 2 = lighting; grass uniforms are read in the vertex stage via group 3
    this.renderBG = device.createBindGroup({ layout: layR, entries: [{ binding: 0, resource: { buffer: this.ubuf } }] });
    this.pipe = await device.createRenderPipelineAsync({
      label: 'grass',
      layout: device.createPipelineLayout({ bindGroupLayouts: [fb, wb, this.ctx.lightBGL, layR] }),
      vertex: { module: mod, entryPoint: 'vs', buffers: [{ arrayStride: 16, stepMode: 'instance', attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x4' }] }] },
      fragment: { module: mod, entryPoint: 'fs', targets: [{ format: 'rgba16float' }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'greater' },
    });
  }
  private renderBG!: GPUBindGroup;

  update(planes: Float64Array, radiusScale: number, densityScale: number) {
    for (let i = 0; i < 20; i++) this.udata[i] = planes[i];
    this.udata[20] = radiusScale;
    this.udata[21] = densityScale;
    this.ctx.device.queue.writeBuffer(this.ubuf, 0, this.udata);
  }

  encodeCull(pass: GPUComputePassEncoder, worldBG: GPUBindGroup) {
    pass.setBindGroup(0, this.ctx.frameBG);
    pass.setBindGroup(1, worldBG);
    pass.setBindGroup(2, this.ctx.emptyBG);
    pass.setBindGroup(3, this.bg);
    pass.setPipeline(this.pCull);
    pass.dispatchWorkgroups(GRID / 8, GRID / 8);
    pass.setPipeline(this.pFinal);
    pass.dispatchWorkgroups(1);
  }

  record(enc: GPURenderBundleEncoder) {
    enc.setPipeline(this.pipe);
    enc.setBindGroup(3, this.renderBG);
    enc.setIndexBuffer(this.ibuf, 'uint32');
    for (let l = 0; l < 3; l++) {
      enc.setVertexBuffer(0, this.tiles, l * CAP * 16, CAP * 16);
      enc.drawIndexedIndirect(this.ind, l * 20);
    }
  }
}
