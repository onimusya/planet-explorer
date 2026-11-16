// Atmosphere LUTs (transmittance, multi-scattering, sky-view, aerial perspective, ambient SH).
import src from './shaders/atmo_luts.wgsl';
import noiseSrc from './shaders/noise3d.wgsl';
import { L, bgl, buffer, computePipe, tex2d, tex3d, U, T } from './gpu';
import type { Ctx } from './ctx';

export class Atmosphere {
  trans!: GPUTexture;
  ms!: GPUTexture;
  sky!: GPUTexture;
  ap!: GPUTexture;
  shBuf!: GPUBuffer;
  noise!: GPUTexture;
  private pTrans!: GPUComputePipeline;
  private pMs!: GPUComputePipeline;
  private pSky!: GPUComputePipeline;
  private pAp!: GPUComputePipeline;
  private pSh!: GPUComputePipeline;
  private bgTrans!: GPUBindGroup;
  private bgMs!: GPUBindGroup;
  private bgSky!: GPUBindGroup;
  private bgAp!: GPUBindGroup;
  private bgSh!: GPUBindGroup;

  constructor(private ctx: Ctx) {}

  async init() {
    const { device } = this.ctx;
    const S = T.STORAGE_BINDING | T.TEXTURE_BINDING;
    this.trans = tex2d(device, 256, 64, 'rgba16float', S, 'transLUT');
    this.ms = tex2d(device, 32, 32, 'rgba16float', S, 'msLUT');
    this.sky = tex2d(device, 192, 108, 'rgba16float', S, 'skyLUT');
    this.ap = tex3d(device, 32, 32, 32, 'rgba16float', S, 'apLUT');
    this.shBuf = buffer(device, 9 * 16, U.STORAGE, 'ambientSH');
    const mod = await this.ctx.module(src, 'atmo_luts');
    const C = GPUShaderStage.COMPUTE;
    const lt = bgl(device, [L.stTex(3, 'rgba16float')]);
    const lm = bgl(device, [L.tex(0, 'float', '2d', C), L.stTex(4, 'rgba16float')]);
    const ls = bgl(device, [L.tex(0, 'float', '2d', C), L.tex(1, 'float', '2d', C), L.stTex(5, 'rgba16float')]);
    const la = bgl(device, [L.tex(0, 'float', '2d', C), L.tex(1, 'float', '2d', C), L.stTex(6, 'rgba16float', '3d')]);
    const lh = bgl(device, [L.tex(0, 'float', '2d', C), L.tex(2, 'float', '2d', C), L.storage(7)]);
    const fb = this.ctx.frameBGL;
    [this.pTrans, this.pMs, this.pSky, this.pAp, this.pSh] = await Promise.all([
      computePipe(device, mod, 'transmittanceLut', [fb, lt]),
      computePipe(device, mod, 'multiScatterLut', [fb, lm]),
      computePipe(device, mod, 'skyViewLut', [fb, ls]),
      computePipe(device, mod, 'aerialLut', [fb, la]),
      computePipe(device, mod, 'ambientSH', [fb, lh]),
    ]);
    const tv = this.trans.createView(), mv = this.ms.createView(), sv = this.sky.createView();
    this.bgTrans = device.createBindGroup({ layout: lt, entries: [{ binding: 3, resource: tv }] });
    this.bgMs = device.createBindGroup({ layout: lm, entries: [{ binding: 0, resource: tv }, { binding: 4, resource: mv }] });
    this.bgSky = device.createBindGroup({ layout: ls, entries: [{ binding: 0, resource: tv }, { binding: 1, resource: mv }, { binding: 5, resource: sv }] });
    this.bgAp = device.createBindGroup({ layout: la, entries: [{ binding: 0, resource: tv }, { binding: 1, resource: mv }, { binding: 6, resource: this.ap.createView() }] });
    this.bgSh = device.createBindGroup({ layout: lh, entries: [{ binding: 0, resource: tv }, { binding: 2, resource: sv }, { binding: 7, resource: { buffer: this.shBuf } }] });

    // tiling 3D noise for clouds / fog (generated once)
    this.noise = tex3d(device, 128, 128, 128, 'rgba8unorm', S, 'noise3d');
    const nmod = await this.ctx.module(noiseSrc, 'noise3d');
    const ln = bgl(device, [L.stTex(0, 'rgba8unorm', '3d')]);
    const pN = await computePipe(device, nmod, 'main', [fb, ln]);
    const bgN = device.createBindGroup({ layout: ln, entries: [{ binding: 0, resource: this.noise.createView() }] });
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setBindGroup(0, this.ctx.frameBG);
    pass.setPipeline(pN);
    pass.setBindGroup(1, bgN);
    pass.dispatchWorkgroups(32, 32, 32);
    pass.end();
    device.queue.submit([enc.finish()]);
  }

  // Sun is static: transmittance + multi-scattering computed once.
  precompute() {
    const enc = this.ctx.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setBindGroup(0, this.ctx.frameBG);
    pass.setPipeline(this.pTrans);
    pass.setBindGroup(1, this.bgTrans);
    pass.dispatchWorkgroups(32, 8);
    pass.setPipeline(this.pMs);
    pass.setBindGroup(1, this.bgMs);
    pass.dispatchWorkgroups(32, 32);
    pass.end();
    this.ctx.device.queue.submit([enc.finish()]);
  }

  encodeFrame(pass: GPUComputePassEncoder) {
    pass.setBindGroup(0, this.ctx.frameBG);
    pass.setPipeline(this.pSky);
    pass.setBindGroup(1, this.bgSky);
    pass.dispatchWorkgroups(24, 14);
    pass.setPipeline(this.pAp);
    pass.setBindGroup(1, this.bgAp);
    pass.dispatchWorkgroups(8, 8);
    pass.setPipeline(this.pSh);
    pass.setBindGroup(1, this.bgSh);
    pass.dispatchWorkgroups(1);
  }
}
