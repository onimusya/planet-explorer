// Airborne particles (shaders/fx.wgsl): luminous spores in the valley and sunlit pollen elsewhere,
// drawn after the opaque scene into their own HDR layer at output resolution (Targets.fx), which
// bloom and the final pass add. One instanced draw replayed from a render bundle.
import fxSrc from './shaders/fx.wgsl';
import type { Ctx } from './ctx';
import type { Targets } from './post';
import { L, bgl } from './gpu';
import { FX_SPORES, FX_MOTES, FX_TRAIL_SEG, FAUNA_NS } from './world';

export class Fx {
  private pipe!: GPURenderPipeline;
  private pTrail!: GPURenderPipeline;
  private layout!: GPUBindGroupLayout;
  private readonly att: GPURenderPassColorAttachment = {
    view: undefined as unknown as GPUTextureView, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 0 },
  };
  private readonly desc: GPURenderPassDescriptor = { colorAttachments: [this.att] };
  private bundles: GPURenderBundle[] = [];

  constructor(private ctx: Ctx) {}

  async init() {
    const { device } = this.ctx;
    this.layout = bgl(device, [L.depth(0, '2d', GPUShaderStage.FRAGMENT)]);
    const mod = await this.ctx.module(fxSrc, 'fx');
    const add: GPUBlendComponent = { srcFactor: 'one', dstFactor: 'one', operation: 'add' };
    const layout = device.createPipelineLayout({ bindGroupLayouts: [this.ctx.frameBGL, this.ctx.worldBGL, this.layout] });
    const targets: GPUColorTargetState[] = [{ format: 'rgba16float', blend: { color: add, alpha: add } }];
    [this.pipe, this.pTrail] = await Promise.all([
      device.createRenderPipelineAsync({ label: 'fx', layout, vertex: { module: mod, entryPoint: 'vs' },
        fragment: { module: mod, entryPoint: 'fs', targets }, primitive: { topology: 'triangle-strip' } }),
      // starship engine trails: a strip of FX_TRAIL_SEG segments per engine, two engines per ship
      device.createRenderPipelineAsync({ label: 'trail', layout, vertex: { module: mod, entryPoint: 'vsTrail' },
        fragment: { module: mod, entryPoint: 'fsTrail', targets }, primitive: { topology: 'triangle-strip' } }),
    ]);
  }

  // On resize: new target and depth views, so the bundle is recorded again.
  bind(t: Targets, worldBG: GPUBindGroup) {
    const { device } = this.ctx;
    this.att.view = t.fx.createView();
    const dbg = device.createBindGroup({ layout: this.layout, entries: [{ binding: 0, resource: t.depth.createView() }] });
    const be = device.createRenderBundleEncoder({ colorFormats: ['rgba16float'] });
    be.setPipeline(this.pipe);
    be.setBindGroup(0, this.ctx.frameBG);
    be.setBindGroup(1, worldBG);
    be.setBindGroup(2, dbg);
    be.draw(4, FX_SPORES + FX_MOTES);
    be.setPipeline(this.pTrail);
    be.draw(2 * (FX_TRAIL_SEG + 1), 2 * FAUNA_NS);
    this.bundles = [be.finish()];
  }

  encode(enc: GPUCommandEncoder, tsw: GPURenderPassTimestampWrites | undefined) {
    this.desc.timestampWrites = tsw;
    const p = enc.beginRenderPass(this.desc);
    p.executeBundles(this.bundles);
    p.end();
  }
}
