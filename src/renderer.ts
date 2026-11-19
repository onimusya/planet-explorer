// Frame orchestration: per-frame uniforms, culling, pass encoding with persistent descriptors.
import type { Ctx } from './ctx';
import { Terrain } from './terrain';
import { Atmosphere } from './atmo';
import { Post, type Targets } from './post';
import { Metrics } from './metrics';
import { Water } from './water';
import { Vegetation } from './vegetation';
import { buildMarks } from './landmarks';
import { Grass } from './grass';
import { Volume } from './volume';
import { Fauna } from './fauna';
import { Fx } from './fx';
import { Camera } from './camera';
import { FO } from './frame';
import { tex2d, T } from './gpu';
import { m4, m4mul, m4invert, m4perspRevInf, m4viewBasis, m4ortho, halton, v3, v3cross, v3norm, v3set } from './math';
import { SUN_DIR, WORLD_L } from './world';
import { clock, DT, TIME } from './clock';

const NEAR = 0.15; // near plane (m)

export const SHADOW_SIZE = 2048;
export const CASCADES = 4;

export class Renderer {
  terrain: Terrain;
  atmo: Atmosphere;
  post: Post;
  water: Water;
  veg: Vegetation;
  grass: Grass;
  vol: Volume;
  fauna: Fauna;
  fx: Fx;
  private prevIW = 1;
  private prevIH = 1;
  private volDesc: GPUComputePassDescriptor = {};
  grassScale = 1;
  private shadowDescs: GPURenderPassDescriptor[] = [];
  private shadowBundles: GPURenderBundle[][] = [];
  private splits = new Float64Array([24, 90, 360, 1500]);
  private lv = m4();
  private lp = m4();
  private lm = m4();
  // light basis (the sun is static)
  private lr = v3();
  private lu = v3();
  private lf = v3();
  vegScale = 1;
  // per-frame scalars that only change on resize, and the TAA jitter sequence (Halton 2,3)
  private aspect = 1;
  private tanHalfFov = 1;
  private jitter = new Float64Array(32);
  private wcam = v3(); // camera position wrapped into the world tile
  metrics: Metrics;
  private waterBundles: GPURenderBundle[] = [];
  private waterColorAtt: GPURenderPassColorAttachment;
  private waterDepthAtt: GPURenderPassDepthStencilAttachment;
  private waterDesc: GPURenderPassDescriptor;
  private cpyColorSrc: GPUTexelCopyTextureInfo = { texture: undefined as unknown as GPUTexture };
  private cpyColorDst: GPUTexelCopyTextureInfo = { texture: undefined as unknown as GPUTexture };
  private cpyDepthSrc: GPUTexelCopyTextureInfo = { texture: undefined as unknown as GPUTexture };
  private cpyDepthDst: GPUTexelCopyTextureInfo = { texture: undefined as unknown as GPUTexture };
  private cpySize = [1, 1, 1];
  t!: Targets;
  worldBG!: GPUBindGroup;
  lightBG!: GPUBindGroup;
  shadowTex!: GPUTexture;
  private mainBundle!: GPURenderBundle;
  private bundles: GPURenderBundle[] = [];
  private mainColorAtt: GPURenderPassColorAttachment;
  private mainDepthAtt: GPURenderPassDepthStencilAttachment;
  private mainDesc: GPURenderPassDescriptor;
  private computeDesc: GPUComputePassDescriptor = {};
  private cmd: GPUCommandBuffer[] = [];
  private postDesc: GPUComputePassDescriptor = {};
  // matrices
  private view = m4();
  private proj = m4();
  private projNJ = m4();
  private vp = m4();
  private vpNJ = m4();
  private ivp = m4();
  private prevVP = m4();
  private hasPrev = false;
  frame = 0;
  hist = 0;
  renderScale = 1;
  quality = 2;
  iw = 1; ih = 1;
  resetHistory = true;
  exposure = 1;
  overlayOn = false;
  time = 0;

  constructor(private ctx: Ctx, public camera: Camera) {
    this.terrain = new Terrain(ctx);
    this.atmo = new Atmosphere(ctx);
    this.post = new Post(ctx);
    this.water = new Water(ctx);
    this.veg = new Vegetation(ctx);
    this.grass = new Grass(ctx);
    this.vol = new Volume(ctx);
    this.fauna = new Fauna(ctx);
    this.fx = new Fx(ctx);
    this.waterColorAtt = { view: undefined as unknown as GPUTextureView, loadOp: 'load', storeOp: 'store' };
    this.waterDepthAtt = { view: undefined as unknown as GPUTextureView, depthLoadOp: 'load', depthStoreOp: 'store' };
    this.waterDesc = { colorAttachments: [this.waterColorAtt], depthStencilAttachment: this.waterDepthAtt };
    this.metrics = new Metrics(ctx.device, ctx.gpu.timestamps);
    this.mainColorAtt = { view: undefined as unknown as GPUTextureView, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 0 } };
    this.mainDepthAtt = { view: undefined as unknown as GPUTextureView, depthClearValue: 0, depthLoadOp: 'clear', depthStoreOp: 'store' };
    this.mainDesc = { colorAttachments: [this.mainColorAtt], depthStencilAttachment: this.mainDepthAtt };
    for (let i = 0; i < 16; i++) {
      this.jitter[i * 2] = halton(i + 1, 2) - 0.5;
      this.jitter[i * 2 + 1] = halton(i + 1, 3) - 0.5;
    }
    v3set(this.lf, -SUN_DIR[0], -SUN_DIR[1], -SUN_DIR[2]);
    v3cross(this.lr, this.lf, v3(0, 1, 0)); v3norm(this.lr, this.lr);
    v3cross(this.lu, this.lr, this.lf);
  }

  async init(progress: (f: number, m: string) => void) {
    const { device } = this.ctx;
    this.writeStaticFrame();
    // A first visit spends most of its load compiling shaders (the browser caches them for later
    // visits), and the GPU process compiles pipelines on worker threads. So every subsystem builds
    // its pipelines at once, while the terrain generates; only plant placement waits for the
    // terrain's heights (measured on a cold shader cache: 3.6 s for all of it, 15 s when run one
    // after another). Progress runs to 0.86 (main.ts adds the warm-up frames); each task weighs
    // the time its completion added in that measurement (s), and the label is the first
    // unfinished task's.
    const W = [0.42, 1.14, 1.58, 0.31, 0.19];
    const label = ['Compiling shaders', 'Shaping continents', 'Gathering clouds', 'Growing flora', 'Waking the wildlife'];
    const frac = [0, 0, 0, 0, 0];
    const report = () => {
      let all = 0, got = 0, m = 'Warming up';
      for (let i = W.length - 1; i >= 0; i--) {
        all += W[i];
        got += W[i] * frac[i];
        if (frac[i] < 1) m = label[i];
      }
      progress(0.02 + (0.84 * got) / all, m);
    };
    const task = (i: number, p: Promise<unknown>) => p.then(() => { frac[i] = 1; report(); });
    report();
    const terrain = this.terrain.generate((f, m) => { frac[1] = Math.min(f / 0.92, 0.99); label[1] = m; report(); });
    await Promise.all([
      task(0, this.atmo.init().then(() => this.atmo.precompute())),
      task(1, terrain),
      task(2, Promise.all([this.terrain.initRender('rgba16float'), this.post.init(this.ctx.worldBGL), this.water.init(), this.vol.init()])),
      task(3, Promise.all([this.veg.init(terrain.then(() => buildMarks((x, z) => this.terrain.heightAt(x, z)))), this.grass.init()])),
      task(4, Promise.all([this.fauna.init(), this.fx.init()])),
    ]);
    this.shadowTex = tex2d(device, SHADOW_SIZE, SHADOW_SIZE, 'depth32float', T.RENDER_ATTACHMENT | T.TEXTURE_BINDING, 'shadow', 1, CASCADES);
    // clear shadow layers once (until casters exist)
    {
      const enc = device.createCommandEncoder();
      for (let i = 0; i < CASCADES; i++) {
        const p = enc.beginRenderPass({ colorAttachments: [], depthStencilAttachment: {
          view: this.shadowTex.createView({ dimension: '2d', baseArrayLayer: i, arrayLayerCount: 1 }),
          depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' } });
        p.end();
      }
      device.queue.submit([enc.finish()]);
    }
    this.worldBG = device.createBindGroup({ layout: this.ctx.worldBGL, entries: [
      { binding: 0, resource: this.terrain.hTex.createView({ baseMipLevel: 0, mipLevelCount: 1 }) },
      { binding: 1, resource: this.terrain.gTex.createView() },
      { binding: 2, resource: this.terrain.aTex.createView() },
      { binding: 3, resource: this.atmo.trans.createView() },
      { binding: 4, resource: this.atmo.sky.createView() },
      { binding: 5, resource: this.atmo.ap.createView() },
      { binding: 6, resource: { buffer: this.atmo.shBuf } },
      { binding: 7, resource: this.atmo.noise.createView() },
      { binding: 8, resource: this.atmo.weather.createView() },
      { binding: 9, resource: this.terrain.fTex.createView() },
      { binding: 10, resource: this.terrain.tTex.createView() },
    ] });
    {
      const enc = device.createCommandEncoder();
      this.fauna.encodeTable(enc, this.worldBG);
      device.queue.submit([enc.finish()]);
    }
    this.lightBG = device.createBindGroup({ layout: this.ctx.lightBGL, entries: [
      { binding: 0, resource: this.shadowTex.createView({ dimension: '2d-array' }) },
    ] });
    for (let i = 0; i < CASCADES; i++) {
      this.shadowDescs.push({ colorAttachments: [], depthStencilAttachment: {
        view: this.shadowTex.createView({ dimension: '2d', baseArrayLayer: i, arrayLayerCount: 1 }),
        depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' } });
    }
    this.recordBundles();
  }

  private recordBundles() {
    const { device } = this.ctx;
    const be = device.createRenderBundleEncoder({ colorFormats: ['rgba16float'], depthStencilFormat: 'depth32float' });
    be.setBindGroup(0, this.ctx.frameBG);
    be.setBindGroup(1, this.worldBG);
    be.setBindGroup(2, this.lightBG);
    this.terrain.recordDraw(be);
    this.veg.recordMain(be);
    this.fauna.recordMain(be);
    this.grass.record(be);
    this.mainBundle = be.finish();
    this.bundles = [this.mainBundle];
    this.shadowBundles = [];
    for (let c = 0; c < CASCADES; c++) {
      const sb = device.createRenderBundleEncoder({ colorFormats: [], depthStencilFormat: 'depth32float' });
      this.veg.recordShadow(sb, c);
      this.fauna.recordShadow(sb, c);
      this.shadowBundles.push([sb.finish()]);
    }
  }

  resize(outW: number, outH: number) {
    // internal resolution capped at 1080p-equivalent pixel count
    const maxPix = 1920 * 1080;
    const s = Math.min(1, Math.sqrt(maxPix / (outW * outH)));
    const fw = Math.max(16, Math.round(outW * s)), fh = Math.max(16, Math.round(outH * s));
    if (this.t) {
      for (const k of ['color', 'depth', 'lit', 'colorCopy', 'depthCopy', 'fog', 'volDepth', 'bloomDown', 'bloomUp', 'fx'] as const) this.t[k].destroy();
      this.t.hist[0].destroy(); this.t.hist[1].destroy();
      this.t.cloud[0].destroy(); this.t.cloud[1].destroy();
    }
    this.t = this.post.createTargets(fw, fh, outW, outH);
    this.post.bindTargets(this.t);
    this.vol.bind(this.t);
    this.fx.bind(this.t, this.worldBG);
    this.mainColorAtt.view = this.t.color.createView();
    this.mainDepthAtt.view = this.t.depth.createView();
    this.waterColorAtt.view = this.mainColorAtt.view;
    this.waterDepthAtt.view = this.mainDepthAtt.view;
    this.cpyColorSrc.texture = this.t.color; this.cpyColorDst.texture = this.t.colorCopy;
    this.cpyDepthSrc.texture = this.t.depth; this.cpyDepthDst.texture = this.t.depthCopy;
    this.cpySize[0] = this.t.fullW; this.cpySize[1] = this.t.fullH;
    this.aspect = outW / outH;
    this.tanHalfFov = Math.tan(this.camera.fovY / 2);
    m4perspRevInf(this.projNJ, this.camera.fovY, this.aspect, NEAR, 0, 0);
    this.water.bind(this.t.colorCopy, this.t.depthCopy);
    {
      const be = this.ctx.device.createRenderBundleEncoder({ colorFormats: ['rgba16float'], depthStencilFormat: 'depth32float' });
      be.setBindGroup(0, this.ctx.frameBG);
      be.setBindGroup(1, this.worldBG);
      be.setBindGroup(2, this.lightBG);
      this.water.record(be);
      this.waterBundles = [be.finish()];
    }
    this.resetHistory = true;
  }


  private writeStaticFrame() {
    const f = this.ctx.frameData;
    f[FO.sunDir] = SUN_DIR[0]; f[FO.sunDir + 1] = SUN_DIR[1]; f[FO.sunDir + 2] = SUN_DIR[2];
    f[FO.sunDir + 3] = Math.cos(0.0125);
    f[FO.sunColor] = 1.0 * 14; f[FO.sunColor + 1] = 0.96 * 14; f[FO.sunColor + 2] = 0.9 * 14; f[FO.sunColor + 3] = 1;
    f[FO.shadowInfo] = SHADOW_SIZE; f[FO.shadowInfo + 1] = 1 / SHADOW_SIZE; f[FO.shadowInfo + 2] = CASCADES;
    this.ctx.device.queue.writeBuffer(this.ctx.frameBuf, 0, f);
  }


  // Stable cascaded shadow maps (bounding spheres, texel-snapped in world space).
  private computeCascades() {
    const cam = this.camera;
    const f = this.ctx.frameData;
    const tanY = this.tanHalfFov, tanX = tanY * this.aspect;
    const k2 = tanX * tanX + tanY * tanY;
    const lf = this.lf, lr = this.lr, lu = this.lu;
    for (let c = 0; c < CASCADES; c++) {
      const near = c === 0 ? 0.2 : this.splits[c - 1];
      const far = this.splits[c];
      const dC = Math.min(far, 0.5 * (near + far) * (1 + k2));
      let radius = Math.sqrt((far - dC) * (far - dC) + far * far * k2) * 1.03;
      radius = Math.ceil(radius / 8) * 8;
      let cx = cam.fwd[0] * dC, cy = cam.fwd[1] * dC, cz = cam.fwd[2] * dC;
      const texel = (2 * radius) / SHADOW_SIZE;
      const wx = cam.pos[0] + cx, wy = cam.pos[1] + cy, wz = cam.pos[2] + cz;
      const pr = wx * lr[0] + wy * lr[1] + wz * lr[2];
      const pu = wx * lu[0] + wy * lu[1] + wz * lu[2];
      const dr = Math.round(pr / texel) * texel - pr;
      const du = Math.round(pu / texel) * texel - pu;
      cx += lr[0] * dr + lu[0] * du; cy += lr[1] * dr + lu[1] * du; cz += lr[2] * dr + lu[2] * du;
      const extend = c < 2 ? 500 : 1800;
      const ex = cx - lf[0] * (radius + extend), ey = cy - lf[1] * (radius + extend), ez = cz - lf[2] * (radius + extend);
      m4viewBasis(this.lv, lr, lu, lf);
      this.lv[12] = -(lr[0] * ex + lr[1] * ey + lr[2] * ez);
      this.lv[13] = -(lu[0] * ex + lu[1] * ey + lu[2] * ez);
      this.lv[14] = lf[0] * ex + lf[1] * ey + lf[2] * ez;
      m4ortho(this.lp, -radius, radius, -radius, radius, 0, 2 * radius + extend);
      m4mul(this.lm, this.lp, this.lv);
      f.set(this.lm, FO.shadowMats + c * 16);
      f[FO.cascadeSplits + c] = far;
    }
  }
  // ------------------------------------------------------------------------------------------
  // Encodes and submits one frame. Allocation-free apart from the encoders WebGPU requires: time
  // comes from clock.ts, and no fractional number is passed through a call.
  render(swap: GPUTexture | GPUTextureView) {
    const { device } = this.ctx;
    const cam = this.camera;
    const f = this.ctx.frameData;
    const t = this.t;
    const dt = clock[DT], time = clock[TIME];
    this.time = time;
    this.iw = Math.max(16, Math.round(t.fullW * this.renderScale));
    this.ih = Math.max(16, Math.round(t.fullH * this.renderScale));
    const iw = this.iw, ih = this.ih;
    const aspect = this.aspect;
    // jitter, in NDC units of the internal resolution
    const hi = (this.frame % 16) * 2;
    const jx = (this.jitter[hi] * 2) / iw;
    const jy = (this.jitter[hi + 1] * 2) / ih;
    m4viewBasis(this.view, cam.right, cam.up, cam.fwd);
    // projNJ is built on resize; the jittered projection only differs in two entries
    this.proj.set(this.projNJ);
    this.proj[8] = -jx; this.proj[9] = -jy;
    m4mul(this.vp, this.proj, this.view);
    m4mul(this.vpNJ, this.projNJ, this.view);
    m4invert(this.ivp, this.vp);
    if (!this.hasPrev) { this.prevVP.set(this.vpNJ); this.hasPrev = true; }
    const px = cam.pos[0], py = cam.pos[1], pz = cam.pos[2];
    const wx = px - Math.floor(px / WORLD_L) * WORLD_L;
    const wz = pz - Math.floor(pz / WORLD_L) * WORLD_L;
    f.set(this.vp, FO.viewProj);
    f.set(this.ivp, FO.invViewProj);
    f.set(this.prevVP, FO.prevViewProj);
    f.set(this.vpNJ, FO.viewProjNJ);
    f[FO.camPos] = wx; f[FO.camPos + 1] = py; f[FO.camPos + 2] = wz; f[FO.camPos + 3] = time;
    f[FO.camDelta] = px - cam.prevPos[0]; f[FO.camDelta + 1] = py - cam.prevPos[1]; f[FO.camDelta + 2] = pz - cam.prevPos[2]; f[FO.camDelta + 3] = dt;
    f[FO.camFwd] = cam.fwd[0]; f[FO.camFwd + 1] = cam.fwd[1]; f[FO.camFwd + 2] = cam.fwd[2]; f[FO.camFwd + 3] = this.tanHalfFov;
    f[FO.camRight] = cam.right[0]; f[FO.camRight + 1] = cam.right[1]; f[FO.camRight + 2] = cam.right[2]; f[FO.camRight + 3] = aspect;
    f[FO.camUp] = cam.up[0]; f[FO.camUp + 1] = cam.up[1]; f[FO.camUp + 2] = cam.up[2]; f[FO.camUp + 3] = NEAR;
    f[FO.jitter + 2] = f[FO.jitter]; f[FO.jitter + 3] = f[FO.jitter + 1];
    f[FO.jitter] = jx; f[FO.jitter + 1] = jy;
    f[FO.screen] = iw; f[FO.screen + 1] = ih; f[FO.screen + 2] = 1 / iw; f[FO.screen + 3] = 1 / ih;
    f[FO.full] = t.fullW; f[FO.full + 1] = t.fullH; f[FO.full + 2] = 1 / t.fullW; f[FO.full + 3] = 1 / t.fullH;
    f[FO.outSize] = t.outW; f[FO.outSize + 1] = t.outH; f[FO.outSize + 2] = 1 / t.outW; f[FO.outSize + 3] = 1 / t.outH;
    f[FO.misc] = this.frame; f[FO.misc + 1] = this.renderScale; f[FO.misc + 2] = this.quality; f[FO.misc + 3] = 1;
    f[FO.sunColor + 3] = this.exposure;
    const tq = this.terrain.q;
    tq[0] = px; tq[1] = pz;
    this.terrain.heightQ();
    f[FO.extra] = py - tq[2];
    this.computeCascades();
    device.queue.writeBuffer(this.ctx.frameBuf, 0, f);
    const wc = this.wcam;
    wc[0] = wx; wc[1] = py; wc[2] = wz;
    this.terrain.select(wc, this.vpNJ);
    this.veg.setPlanes(this.terrain.planes, this.vegScale);
    this.grass.update(this.terrain.planes, this.grassScale, 1);
    this.post.taaData[0] = this.resetHistory ? 1 : 0;
    device.queue.writeBuffer(this.post.taaU, 0, this.post.taaData);
    this.vol.update(this.resetHistory || this.prevIW !== iw || this.prevIH !== ih, this.prevIW, this.prevIH);
    this.post.finalData[0] = 0.045; this.post.finalData[1] = this.overlayOn ? 1 : 0; this.post.finalData[3] = 0.35;
    device.queue.writeBuffer(this.post.finalU, 0, this.post.finalData);

    const m = this.metrics;
    const enc = device.createCommandEncoder();
    // 1. atmosphere LUTs
    this.computeDesc.timestampWrites = m.ts(0);
    let cp = enc.beginComputePass(this.computeDesc);
    this.atmo.encodeFrame(cp);
    this.veg.encodeCull(cp, this.worldBG);
    this.grass.encodeCull(cp, this.worldBG);
    this.fauna.encodeUpdate(cp, this.worldBG);
    cp.end();
    // 1b. shadow cascades
    for (let c = 0; c < CASCADES; c++) {
      const d = this.shadowDescs[c];
      d.timestampWrites = c === 0 ? m.tsBegin(1) : c === CASCADES - 1 ? m.tsEnd(1) : undefined;
      const sp = enc.beginRenderPass(d);
      sp.executeBundles(this.shadowBundles[c]);
      sp.end();
    }
    // 2. main opaque pass
    this.mainDesc.timestampWrites = m.ts(2);
    const rp = enc.beginRenderPass(this.mainDesc);
    rp.setViewport(0, 0, iw, ih, 0, 1);
    rp.setScissorRect(0, 0, iw, ih);
    rp.executeBundles(this.bundles);
    rp.end();
    // 2b. water (needs a copy of the opaque colour/depth for refraction + SSR)
    enc.copyTextureToTexture(this.cpyColorSrc, this.cpyColorDst, this.cpySize);
    enc.copyTextureToTexture(this.cpyDepthSrc, this.cpyDepthDst, this.cpySize);
    this.waterDesc.timestampWrites = m.ts(3);
    const wp = enc.beginRenderPass(this.waterDesc);
    wp.setViewport(0, 0, iw, ih, 0, 1);
    wp.setScissorRect(0, 0, iw, ih);
    wp.executeBundles(this.waterBundles);
    wp.end();
    // 2c. airborne particles into their own layer (needs the final scene depth)
    this.fx.encode(enc, m.ts(7));
    // 3. volumetric clouds + fog (half resolution)
    const ci = this.frame & 1;
    this.volDesc.timestampWrites = m.ts(4);
    cp = enc.beginComputePass(this.volDesc);
    this.vol.encode(cp, this.worldBG, ci, iw, ih);
    cp.end();
    // 4. composite + TAA + bloom
    this.postDesc.timestampWrites = m.ts(5);
    cp = enc.beginComputePass(this.postDesc);
    this.post.encodeComposite(cp, this.worldBG, iw, ih, ci);
    this.post.encodeTaa(cp, this.hist, t);
    this.post.encodeBloom(cp, this.hist);
    cp.end();
    // 5. final
    this.post.encodeFinal(enc, swap, this.hist, m.ts(6));
    m.resolve(enc, this.frame);
    this.cmd[0] = enc.finish();
    device.queue.submit(this.cmd);
    m.poll();
    this.prevVP.set(this.vpNJ);
    this.prevIW = iw; this.prevIH = ih;
    this.hist = 1 - this.hist;
    this.frame++;
    this.resetHistory = false;
  }
}
