// Post-processing: composite (sky/AP/clouds/fog), TAA upsampling, bloom, tone mapping + overlay.
import compositeSrc from './shaders/composite.wgsl';
import taaSrc from './shaders/taa.wgsl';
import bloomSrc from './shaders/bloom.wgsl';
import finalSrc from './shaders/final.wgsl';
import expoSrc from './shaders/exposure.wgsl';
import { L, bgl, buffer, bufferWith, computePipe, tex2d, U, T } from './gpu';
import type { Ctx } from './ctx';
import { FONT_WORDS } from './font';

export const BLOOM_LEVELS = 6;
const C = GPUShaderStage.COMPUTE;

export interface Targets {
  fullW: number; fullH: number; outW: number; outH: number;
  color: GPUTexture; depth: GPUTexture; lit: GPUTexture; colorCopy: GPUTexture; depthCopy: GPUTexture;
  cloud: GPUTexture[]; fog: GPUTexture; volDepth: GPUTexture;
  hist: GPUTexture[]; bloomDown: GPUTexture; bloomUp: GPUTexture;
  fx: GPUTexture; // airborne particles (fx.ts), additive HDR at output resolution
}

export class Post {
  private pComp!: GPUComputePipeline;
  private pTaa!: GPUComputePipeline;
  private pDown!: GPUComputePipeline;
  private pUp!: GPUComputePipeline;
  private pFinal!: GPURenderPipeline;
  private pHist!: GPUComputePipeline;
  private pAdapt!: GPUComputePipeline;
  private lExpo!: GPUBindGroupLayout;
  private bgExpo!: GPUBindGroup;
  private histBuf!: GPUBuffer;
  private expoW = 1; private expoH = 1; // size of the metered bloom level
  private lComp!: GPUBindGroupLayout;
  private lTaa!: GPUBindGroupLayout;
  private lBloom!: GPUBindGroupLayout;
  private lFinal!: GPUBindGroupLayout;
  private bgComp: GPUBindGroup[] = [];
  private bgTaa: GPUBindGroup[] = [];
  private bgDown: GPUBindGroup[] = [];
  private bgUp: GPUBindGroup[] = [];
  private bgFinal: GPUBindGroup[] = [];
  taaU!: GPUBuffer;
  private bloomU: GPUBuffer[] = [];
  readonly taaData = new Float32Array(4);
  finalU!: GPUBuffer;
  readonly finalData = new Float32Array(4);
  fontBuf!: GPUBuffer;
  textBuf!: GPUBuffer;
  expoBuf!: GPUBuffer;
  private dims: Int32Array = new Int32Array(BLOOM_LEVELS * 2);
  readonly finalPassDesc: GPURenderPassDescriptor;
  private readonly finalAttachment: GPURenderPassColorAttachment;

  constructor(private ctx: Ctx) {
    this.finalAttachment = { view: undefined as unknown as GPUTextureView, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } };
    this.finalPassDesc = { colorAttachments: [this.finalAttachment] };
  }

  async init(worldBGL: GPUBindGroupLayout) {
    const { device } = this.ctx;
    const fb = this.ctx.frameBGL;
    this.lComp = bgl(device, [L.tex(0, 'unfilterable-float', '2d', C), L.depth(1, '2d', C), L.tex(2, 'float', '2d', C), L.tex(3, 'float', '2d', C), L.stTex(4, 'rgba16float'), L.tex(5, 'unfilterable-float', '2d', C)]);
    this.lTaa = bgl(device, [L.tex(0, 'unfilterable-float', '2d', C), L.depth(1, '2d', C), L.tex(2, 'float', '2d', C), L.stTex(3, 'rgba16float'), L.uniform(4, C)]);
    this.lBloom = bgl(device, [L.tex(0, 'float', '2d', C), L.stTex(1, 'rgba16float'), L.tex(2, 'float', '2d', C), L.uniform(3, C)]);
    const F = GPUShaderStage.FRAGMENT;
    this.lFinal = bgl(device, [L.tex(0, 'unfilterable-float', '2d', F), L.tex(1, 'float', '2d', F), L.storageRO(2, F), L.storageRO(3, F), L.uniform(4, F), L.storageRO(5, F), L.tex(6, 'unfilterable-float', '2d', F)]);
    this.lExpo = bgl(device, [L.tex(0, 'float', '2d', C), L.storage(1), L.storage(2), L.uniform(3, C)]);
    const [mc, mt, mb, mf, me] = await Promise.all([
      this.ctx.module(compositeSrc, 'composite'),
      this.ctx.module(taaSrc, 'taa'),
      this.ctx.module(bloomSrc, 'bloom'),
      this.ctx.module(finalSrc, 'final'),
      this.ctx.module(expoSrc, 'exposure'),
    ]);
    [this.pComp, this.pTaa, this.pDown, this.pUp, this.pHist, this.pAdapt, this.pFinal] = await Promise.all([
      computePipe(device, mc, 'main', [fb, worldBGL, this.lComp]),
      computePipe(device, mt, 'main', [fb, this.lTaa]),
      computePipe(device, mb, 'down', [fb, this.lBloom]),
      computePipe(device, mb, 'up', [fb, this.lBloom]),
      computePipe(device, me, 'hist', [fb, this.lExpo]),
      computePipe(device, me, 'adapt', [fb, this.lExpo]),
      device.createRenderPipelineAsync({
        label: 'final',
        layout: device.createPipelineLayout({ bindGroupLayouts: [fb, this.lFinal] }),
        vertex: { module: mf, entryPoint: 'vs' },
        fragment: { module: mf, entryPoint: 'fs', targets: [{ format: this.ctx.gpu.format }] },
        primitive: { topology: 'triangle-list' },
      }),
    ]);
    this.taaU = buffer(device, 16, U.UNIFORM | U.COPY_DST, 'taaU');
    this.finalU = buffer(device, 16, U.UNIFORM | U.COPY_DST, 'finalU');
    for (let i = 0; i < BLOOM_LEVELS * 2; i++) this.bloomU.push(buffer(device, 16, U.UNIFORM | U.COPY_DST));
    this.fontBuf = bufferWith(device, FONT_WORDS, U.STORAGE, 'font');
    this.textBuf = buffer(device, 64 * 10 * 4, U.STORAGE | U.COPY_DST, 'text');
    // exposure, EV, metered log2 luminance, set (exposure.wgsl); w = 0 makes the first frame snap
    this.expoBuf = buffer(device, 16, U.STORAGE | U.COPY_DST | U.COPY_SRC, 'exposure');
    device.queue.writeBuffer(this.expoBuf, 0, new Float32Array([1, 0, 0, 0]));
    this.histBuf = buffer(device, 128 * 4, U.STORAGE | U.COPY_DST, 'lumHist');
  }

  // Debug only: read back [exposure, EV, metered log2 luminance, set].
  async debugExposure(): Promise<number[]> {
    const { device } = this.ctx;
    const rb = buffer(device, 16, U.MAP_READ | U.COPY_DST);
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(this.expoBuf, 0, rb, 0, 16);
    device.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const a = Array.from(new Float32Array(rb.getMappedRange()));
    rb.unmap(); rb.destroy();
    return a;
  }

  createTargets(fullW: number, fullH: number, outW: number, outH: number): Targets {
    const { device } = this.ctx;
    const R = T.RENDER_ATTACHMENT, B = T.TEXTURE_BINDING, S = T.STORAGE_BINDING;
    const hw = Math.ceil(fullW / 2), hh = Math.ceil(fullH / 2);
    const bw = Math.ceil(outW / 2), bh = Math.ceil(outH / 2);
    const t: Targets = {
      fullW, fullH, outW, outH,
      color: tex2d(device, fullW, fullH, 'rgba16float', R | B | T.COPY_SRC, 'color'),
      depth: tex2d(device, fullW, fullH, 'depth32float', R | B | T.COPY_SRC, 'depth'),
      lit: tex2d(device, fullW, fullH, 'rgba16float', S | B | R, 'lit'),
      colorCopy: tex2d(device, fullW, fullH, 'rgba16float', B | T.COPY_DST, 'colorCopy'),
      depthCopy: tex2d(device, fullW, fullH, 'depth32float', B | T.COPY_DST, 'depthCopy'),
      cloud: [tex2d(device, hw, hh, 'rgba16float', S | B, 'cloud0'), tex2d(device, hw, hh, 'rgba16float', S | B, 'cloud1')],
      fog: tex2d(device, hw, hh, 'rgba16float', S | B, 'fog'),
      volDepth: tex2d(device, hw, hh, 'r32float', S | B, 'volDepth'),
      hist: [tex2d(device, outW, outH, 'rgba16float', S | B, 'hist0'), tex2d(device, outW, outH, 'rgba16float', S | B, 'hist1')],
      bloomDown: tex2d(device, bw, bh, 'rgba16float', S | B, 'bloomDown', BLOOM_LEVELS),
      bloomUp: tex2d(device, bw, bh, 'rgba16float', S | B, 'bloomUp', BLOOM_LEVELS),
      fx: tex2d(device, outW, outH, 'rgba16float', R | B, 'fx'),
    };
    for (let i = 0; i < BLOOM_LEVELS; i++) {
      this.dims[i * 2] = Math.max(1, bw >> i);
      this.dims[i * 2 + 1] = Math.max(1, bh >> i);
    }
    return t;
  }

  bindTargets(t: Targets) {
    const { device } = this.ctx;
    this.bgComp = [0, 1].map((i) => device.createBindGroup({ layout: this.lComp, entries: [
      { binding: 0, resource: t.color.createView() },
      { binding: 1, resource: t.depth.createView() },
      { binding: 2, resource: t.cloud[i].createView() },
      { binding: 3, resource: t.fog.createView() },
      { binding: 4, resource: t.lit.createView() },
      { binding: 5, resource: t.volDepth.createView() },
    ] }));
    this.bgTaa = [0, 1].map((i) => device.createBindGroup({ layout: this.lTaa, entries: [
      { binding: 0, resource: t.lit.createView() },
      { binding: 1, resource: t.depth.createView() },
      { binding: 2, resource: t.hist[1 - i].createView() },
      { binding: 3, resource: t.hist[i].createView() },
      { binding: 4, resource: { buffer: this.taaU } },
    ] }));
    const mv = (tex: GPUTexture, m: number) => tex.createView({ baseMipLevel: m, mipLevelCount: 1 });
    // downsample: hist[i] (+ particles) -> down0 ; down(k-1) -> down(k)
    this.bgDown = [];
    for (let h = 0; h < 2; h++) {
      for (let k = 0; k < BLOOM_LEVELS; k++) {
        const u = this.bloomU[k];
        device.queue.writeBuffer(u, 0, new Float32Array([k === 0 ? 1 : 0, 1, 1, 0]));
        this.bgDown.push(device.createBindGroup({ layout: this.lBloom, entries: [
          { binding: 0, resource: k === 0 ? t.hist[h].createView() : mv(t.bloomDown, k - 1) },
          { binding: 1, resource: mv(t.bloomDown, k) },
          { binding: 2, resource: k === 0 ? t.fx.createView() : mv(t.bloomUp, k) },
          { binding: 3, resource: { buffer: u } },
        ] }));
      }
    }
    // upsample: up(k) = down(k) + tent(up(k+1)) ; smallest: up(L-1) = down(L-1) + tent(down(L-1))*0
    this.bgUp = [];
    for (let k = BLOOM_LEVELS - 2; k >= 0; k--) {
      const u = this.bloomU[BLOOM_LEVELS + k];
      device.queue.writeBuffer(u, 0, new Float32Array([0, 1, 1.0, 0]));
      const srcView = k === BLOOM_LEVELS - 2 ? mv(t.bloomDown, BLOOM_LEVELS - 1) : mv(t.bloomUp, k + 1);
      this.bgUp.push(device.createBindGroup({ layout: this.lBloom, entries: [
        { binding: 0, resource: srcView },
        { binding: 1, resource: mv(t.bloomUp, k) },
        { binding: 2, resource: mv(t.bloomDown, k) },
        { binding: 3, resource: { buffer: u } },
      ] }));
    }
    // eye adaptation meters bloom down level 1 (a quarter of the output resolution)
    this.expoW = this.dims[2]; this.expoH = this.dims[3];
    this.bgExpo = device.createBindGroup({ layout: this.lExpo, entries: [
      { binding: 0, resource: mv(t.bloomDown, 1) },
      { binding: 1, resource: { buffer: this.histBuf } },
      { binding: 2, resource: { buffer: this.expoBuf } },
      { binding: 3, resource: { buffer: this.taaU } },
    ] });
    this.bgFinal = [0, 1].map((i) => device.createBindGroup({ layout: this.lFinal, entries: [
      { binding: 0, resource: t.hist[i].createView() },
      { binding: 1, resource: mv(t.bloomUp, 0) },
      { binding: 2, resource: { buffer: this.fontBuf } },
      { binding: 3, resource: { buffer: this.textBuf } },
      { binding: 4, resource: { buffer: this.finalU } },
      { binding: 5, resource: { buffer: this.expoBuf } },
      { binding: 6, resource: t.fx.createView() },
    ] }));
  }

  encodeComposite(pass: GPUComputePassEncoder, worldBG: GPUBindGroup, w: number, h: number, ci: number) {
    pass.setPipeline(this.pComp);
    pass.setBindGroup(0, this.ctx.frameBG);
    pass.setBindGroup(1, worldBG);
    pass.setBindGroup(2, this.bgComp[ci]);
    pass.dispatchWorkgroups(Math.ceil(w / 8), Math.ceil(h / 8));
  }

  encodeTaa(pass: GPUComputePassEncoder, hi: number, t: Targets) {
    pass.setPipeline(this.pTaa);
    pass.setBindGroup(0, this.ctx.frameBG);
    pass.setBindGroup(1, this.bgTaa[hi]);
    pass.dispatchWorkgroups(Math.ceil(t.outW / 8), Math.ceil(t.outH / 8));
  }

  encodeBloom(pass: GPUComputePassEncoder, hi: number) {
    pass.setBindGroup(0, this.ctx.frameBG);
    pass.setPipeline(this.pDown);
    for (let k = 0; k < BLOOM_LEVELS; k++) {
      pass.setBindGroup(1, this.bgDown[hi * BLOOM_LEVELS + k]);
      pass.dispatchWorkgroups(Math.ceil(this.dims[k * 2] / 8), Math.ceil(this.dims[k * 2 + 1] / 8));
    }
    // eye adaptation (needs down level 1; the final pass reads the result)
    pass.setBindGroup(1, this.bgExpo);
    pass.setPipeline(this.pHist);
    pass.dispatchWorkgroups(Math.ceil(this.expoW / 16), Math.ceil(this.expoH / 16));
    pass.setPipeline(this.pAdapt);
    pass.dispatchWorkgroups(1);
    pass.setPipeline(this.pUp);
    for (let i = 0; i < this.bgUp.length; i++) {
      const k = BLOOM_LEVELS - 2 - i;
      pass.setBindGroup(1, this.bgUp[i]);
      pass.dispatchWorkgroups(Math.ceil(this.dims[k * 2] / 8), Math.ceil(this.dims[k * 2 + 1] / 8));
    }
  }

  // view: the swap-chain texture itself where the browser accepts textures as attachments
  encodeFinal(enc: GPUCommandEncoder, view: GPUTexture | GPUTextureView, hi: number, tsw: GPURenderPassTimestampWrites | undefined) {
    this.finalAttachment.view = view;
    this.finalPassDesc.timestampWrites = tsw;
    const pass = enc.beginRenderPass(this.finalPassDesc);
    pass.setPipeline(this.pFinal);
    pass.setBindGroup(0, this.ctx.frameBG);
    pass.setBindGroup(1, this.bgFinal[hi]);
    pass.draw(3);
    pass.end();
  }
}
