// Shared GPU context: device, common bind group layouts, samplers, frame uniform.
import { L, bgl, buffer, shader, U, type Gpu } from './gpu';
import { FRAME_BYTES, FO } from './frame';
import { wgslPrelude } from './world';

export interface Ctx {
  gpu: Gpu;
  device: GPUDevice;
  frameBuf: GPUBuffer;
  frameData: Float32Array;
  frameBGL: GPUBindGroupLayout;
  frameBG: GPUBindGroup;
  worldBGL: GPUBindGroupLayout;
  lightBGL: GPUBindGroupLayout;
  emptyBGL: GPUBindGroupLayout;
  emptyBG: GPUBindGroup;
  module: (code: string, label: string) => Promise<GPUShaderModule>;
}

export function createCtx(gpu: Gpu): Ctx {
  const device = gpu.device;
  const frameBuf = buffer(device, FRAME_BYTES, U.UNIFORM | U.COPY_DST, 'frame');
  const frameData = new Float32Array(FO.SIZE);
  const frameBGL = bgl(device, [
    L.uniform(0), L.sampler(1), L.sampler(2), L.sampler(3, 'non-filtering'), L.sampler(4, 'comparison'),
  ], 'frameBGL');
  const sLinRep = device.createSampler({ magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear', addressModeU: 'repeat', addressModeV: 'repeat', addressModeW: 'repeat' });
  const sLinClamp = device.createSampler({ magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear' });
  const sPoint = device.createSampler({ magFilter: 'nearest', minFilter: 'nearest' });
  const sShadow = device.createSampler({ compare: 'less', magFilter: 'linear', minFilter: 'linear' });
  const frameBG = device.createBindGroup({
    layout: frameBGL,
    entries: [
      { binding: 0, resource: { buffer: frameBuf } },
      { binding: 1, resource: sLinRep },
      { binding: 2, resource: sLinClamp },
      { binding: 3, resource: sPoint },
      { binding: 4, resource: sShadow },
    ],
  });
  const worldBGL = bgl(device, [
    L.tex(0, gpu.hfilter ? 'float' : 'unfilterable-float'), L.tex(1), L.tex(2), L.tex(3), L.tex(4), L.tex(5, 'float', '3d'),
    L.storageRO(6), L.tex(7, 'float', '3d'), L.tex(8), L.tex(9),
  ], 'worldBGL');
  const lightBGL = bgl(device, [L.depth(0, '2d-array')], 'lightBGL');
  const emptyBGL = bgl(device, [], 'empty');
  const emptyBG = device.createBindGroup({ layout: emptyBGL, entries: [] });
  const prelude = wgslPrelude();
  const cache = new Map<string, Promise<GPUShaderModule>>();
  const module = (code: string, label: string) => {
    let m = cache.get(label);
    if (!m) {
      // feature blocks: #if HFILTER ... #else ... #endif (survive WGSL minification)
      const src = code.replace(/#if HFILTER([\s\S]*?)#else([\s\S]*?)#endif/g, (_s, a: string, b: string) => (gpu.hfilter ? a : b));
      m = shader(device, prelude + src, label);
      cache.set(label, m);
    }
    return m;
  };
  return { gpu, device, frameBuf, frameData, frameBGL, frameBG, worldBGL, lightBGL, emptyBGL, emptyBG, module };
}
