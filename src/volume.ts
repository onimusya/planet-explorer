// Volumetric clouds + light-shafted ground fog at half resolution (see shaders/volume.wgsl).
import src from './shaders/volume.wgsl';
import { L, bgl, buffer, computePipe, U } from './gpu';
import type { Ctx } from './ctx';
import type { Targets } from './post';

export class Volume {
  private pClouds!: GPUComputePipeline;
  private pFog!: GPUComputePipeline;
  private layout!: GPUBindGroupLayout;
  private bgs: GPUBindGroup[] = [];
  private readonly u: GPUBuffer;
  readonly uData = new Float32Array(4); // reset, prev internal w, prev internal h, debug mask
  debug = 0; // bit 0: no clouds, bit 1: no fog

  constructor(private ctx: Ctx) {
    this.u = buffer(ctx.device, 16, U.UNIFORM | U.COPY_DST, 'volU');
  }

  async init() {
    const { device } = this.ctx;
    const C = GPUShaderStage.COMPUTE;
    this.layout = bgl(device, [
      L.depth(0, '2d', C), L.tex(1, 'float', '2d', C), L.stTex(2, 'rgba16float'), L.stTex(3, 'rgba16float'),
      L.stTex(4, 'r32float'), L.uniform(5, C),
    ], 'volume');
    const mod = await this.ctx.module(src, 'volume');
    const lays = [this.ctx.frameBGL, this.ctx.worldBGL, this.layout];
    [this.pClouds, this.pFog] = await Promise.all([computePipe(device, mod, 'clouds', lays), computePipe(device, mod, 'fog', lays)]);
  }

  // bind group i writes cloud[i] and reads cloud[1 - i] as history
  bind(t: Targets) {
    const { device } = this.ctx;
    this.bgs = [0, 1].map((i) => device.createBindGroup({ layout: this.layout, entries: [
      { binding: 0, resource: t.depth.createView() },
      { binding: 1, resource: t.cloud[1 - i].createView() },
      { binding: 2, resource: t.cloud[i].createView() },
      { binding: 3, resource: t.fog.createView() },
      { binding: 4, resource: t.volDepth.createView() },
      { binding: 5, resource: { buffer: this.u } },
    ] }));
  }

  update(reset: boolean, prevW: number, prevH: number) {
    this.uData[0] = reset ? 1 : 0;
    this.uData[1] = prevW;
    this.uData[2] = prevH;
    this.uData[3] = this.debug;
    this.ctx.device.queue.writeBuffer(this.u, 0, this.uData);
  }

  encode(pass: GPUComputePassEncoder, worldBG: GPUBindGroup, ci: number, iw: number, ih: number) {
    const gx = Math.ceil(Math.ceil(iw / 2) / 8), gy = Math.ceil(Math.ceil(ih / 2) / 8);
    pass.setBindGroup(0, this.ctx.frameBG);
    pass.setBindGroup(1, worldBG);
    pass.setBindGroup(2, this.bgs[ci]);
    pass.setPipeline(this.pClouds);
    pass.dispatchWorkgroups(gx, gy);
    pass.setPipeline(this.pFog);
    pass.dispatchWorkgroups(gx, gy);
  }
}
