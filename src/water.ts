// Water surface pass (lakes, canyon rivers, valley stream).
import src from './shaders/water.wgsl';
import { L, bgl } from './gpu';
import type { Ctx } from './ctx';

const VERTS = 72 * 128 * 6;

export class Water {
  pipeline!: GPURenderPipeline;
  bgl!: GPUBindGroupLayout;
  bg!: GPUBindGroup;
  constructor(private ctx: Ctx) {}

  async init() {
    const { device } = this.ctx;
    const F = GPUShaderStage.FRAGMENT;
    this.bgl = bgl(device, [L.tex(0, 'float', '2d', F), L.depth(1, '2d', F)]);
    const mod = await this.ctx.module(src, 'water');
    this.pipeline = await device.createRenderPipelineAsync({
      label: 'water',
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.ctx.frameBGL, this.ctx.worldBGL, this.ctx.lightBGL, this.bgl] }),
      vertex: { module: mod, entryPoint: 'vs' },
      fragment: { module: mod, entryPoint: 'fs', targets: [{ format: 'rgba16float' }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'greater' },
    });
  }

  bind(colorCopy: GPUTexture, depthCopy: GPUTexture) {
    this.bg = this.ctx.device.createBindGroup({ layout: this.bgl, entries: [
      { binding: 0, resource: colorCopy.createView() },
      { binding: 1, resource: depthCopy.createView() },
    ] });
  }

  record(enc: GPURenderPassEncoder | GPURenderBundleEncoder) {
    enc.setPipeline(this.pipeline);
    enc.setBindGroup(3, this.bg);
    enc.draw(VERTS);
  }
}
