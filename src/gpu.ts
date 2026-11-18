// WebGPU device bootstrap and small creation helpers (startup-time only).

export interface Gpu {
  device: GPUDevice;
  adapter: GPUAdapter;
  context: GPUCanvasContext;
  format: GPUTextureFormat;
  canvas: HTMLCanvasElement;
  timestamps: boolean;
  hfilter: boolean;
  textureViews: boolean; // a texture can stand in for its default view in a render pass
  adapterName: string;
}

// Texture-view shorthand (a recent WebGPU addition): passing the swap-chain texture as the
// attachment saves a createView() call, and its allocation, every frame.
async function detectTextureViews(device: GPUDevice): Promise<boolean> {
  const t = device.createTexture({ size: [1, 1], format: 'rgba8unorm', usage: GPUTextureUsage.RENDER_ATTACHMENT });
  device.pushErrorScope('validation');
  let ok = true;
  try {
    const e = device.createCommandEncoder();
    e.beginRenderPass({ colorAttachments: [{ view: t, loadOp: 'clear', storeOp: 'store' }] }).end();
    e.finish();
  } catch {
    ok = false; // older implementations reject a texture here with a TypeError
  }
  if (await device.popErrorScope()) ok = false;
  t.destroy();
  return ok;
}

export async function initGpu(canvas: HTMLCanvasElement): Promise<Gpu> {
  if (!navigator.gpu) throw new Error('WebGPU is not available in this browser.');
  let adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('No WebGPU adapter found.');
  const want: GPUFeatureName[] = [];
  const timestamps = adapter.features.has('timestamp-query');
  if (timestamps) want.push('timestamp-query');
  const hfilter = adapter.features.has('float32-filterable');
  if (hfilter) want.push('float32-filterable');
  const lim = adapter.limits;
  const device = await adapter.requestDevice({
    requiredFeatures: want,
    requiredLimits: {
      maxStorageBufferBindingSize: Math.min(lim.maxStorageBufferBindingSize, 1 << 30),
      maxBufferSize: Math.min(lim.maxBufferSize, 1 << 30),
      maxStorageTexturesPerShaderStage: Math.min(lim.maxStorageTexturesPerShaderStage, 8),
      maxSampledTexturesPerShaderStage: Math.min(lim.maxSampledTexturesPerShaderStage, 24),
      maxComputeWorkgroupStorageSize: Math.min(lim.maxComputeWorkgroupStorageSize, 32768),
      maxComputeInvocationsPerWorkgroup: Math.min(lim.maxComputeInvocationsPerWorkgroup, 256),
      maxVertexAttributes: Math.min(lim.maxVertexAttributes, 16),
      maxStorageBuffersPerShaderStage: Math.min(lim.maxStorageBuffersPerShaderStage, 10),
    },
  });
  device.lost.then((info) => {
    console.error('WebGPU device lost:', info.message);
    const el = document.getElementById('err');
    if (el) { el.textContent = 'GPU device lost: ' + info.message; el.style.display = 'block'; }
  });
  device.addEventListener('uncapturederror', (e) => {
    console.error('WebGPU error:', (e as GPUUncapturedErrorEvent).error.message);
  });
  const context = canvas.getContext('webgpu') as GPUCanvasContext;
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'opaque' });
  const info = adapter.info;
  const adapterName = info ? `${info.vendor} ${info.architecture || info.description || ''}`.trim() : 'unknown';
  const textureViews = await detectTextureViews(device);
  return { device, adapter, context, format, canvas, timestamps, hfilter, textureViews, adapterName };
}

export const U = GPUBufferUsage;
export const T = GPUTextureUsage;

export function buffer(device: GPUDevice, size: number, usage: number, label?: string): GPUBuffer {
  return device.createBuffer({ size: Math.ceil(size / 4) * 4, usage, label });
}

export function bufferWith(device: GPUDevice, data: ArrayBufferView, usage: number, label?: string): GPUBuffer {
  const b = device.createBuffer({ size: Math.ceil(data.byteLength / 4) * 4, usage: usage | U.COPY_DST, label });
  device.queue.writeBuffer(b, 0, data.buffer, data.byteOffset, data.byteLength);
  return b;
}

export function tex2d(device: GPUDevice, w: number, h: number, format: GPUTextureFormat, usage: number, label?: string, mips = 1, layers = 1): GPUTexture {
  return device.createTexture({ size: [w, h, layers], format, usage, label, mipLevelCount: mips });
}

export function tex3d(device: GPUDevice, w: number, h: number, d: number, format: GPUTextureFormat, usage: number, label?: string): GPUTexture {
  return device.createTexture({ size: [w, h, d], format, usage, dimension: '3d', label });
}

// Shader module creation with readable compile diagnostics.
export async function shader(device: GPUDevice, code: string, label: string): Promise<GPUShaderModule> {
  const mod = device.createShaderModule({ code, label });
  const info = await mod.getCompilationInfo();
  let hasErr = false;
  for (const m of info.messages) {
    if (m.type === 'error') hasErr = true;
    const lines = code.split('\n');
    const ctx = lines.slice(Math.max(0, m.lineNum - 3), m.lineNum + 1).join('\n');
    (m.type === 'error' ? console.error : console.warn)(`[${label}] ${m.type} ${m.lineNum}:${m.linePos} ${m.message}\n${ctx}`);
  }
  if (hasErr) throw new Error(`Shader compile failed: ${label}`);
  return mod;
}

// Bind group layout entry helpers ------------------------------------------------------------
const ALL = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE;
const VF = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT;
export const vis = { ALL, VF, C: GPUShaderStage.COMPUTE, F: GPUShaderStage.FRAGMENT, V: GPUShaderStage.VERTEX };

export const L = {
  uniform: (binding: number, visibility = ALL): GPUBindGroupLayoutEntry => ({ binding, visibility, buffer: { type: 'uniform' } }),
  storageRO: (binding: number, visibility = ALL): GPUBindGroupLayoutEntry => ({ binding, visibility, buffer: { type: 'read-only-storage' } }),
  storage: (binding: number, visibility = GPUShaderStage.COMPUTE): GPUBindGroupLayoutEntry => ({ binding, visibility, buffer: { type: 'storage' } }),
  tex: (binding: number, sampleType: GPUTextureSampleType = 'float', viewDimension: GPUTextureViewDimension = '2d', visibility = ALL): GPUBindGroupLayoutEntry => ({ binding, visibility, texture: { sampleType, viewDimension } }),
  depth: (binding: number, viewDimension: GPUTextureViewDimension = '2d', visibility = ALL): GPUBindGroupLayoutEntry => ({ binding, visibility, texture: { sampleType: 'depth', viewDimension } }),
  stTex: (binding: number, format: GPUTextureFormat, viewDimension: GPUTextureViewDimension = '2d', access: GPUStorageTextureAccess = 'write-only'): GPUBindGroupLayoutEntry => ({ binding, visibility: GPUShaderStage.COMPUTE, storageTexture: { format, viewDimension, access } }),
  sampler: (binding: number, type: GPUSamplerBindingType = 'filtering', visibility = ALL): GPUBindGroupLayoutEntry => ({ binding, visibility, sampler: { type } }),
};

export function bgl(device: GPUDevice, entries: GPUBindGroupLayoutEntry[], label?: string): GPUBindGroupLayout {
  return device.createBindGroupLayout({ entries, label });
}

export function bg(device: GPUDevice, layout: GPUBindGroupLayout, resources: (GPUBindingResource)[], label?: string): GPUBindGroup {
  return device.createBindGroup({ layout, label, entries: resources.map((resource, binding) => ({ binding, resource })) });
}

export function computePipe(device: GPUDevice, module: GPUShaderModule, entryPoint: string, layouts: GPUBindGroupLayout[], label?: string): Promise<GPUComputePipeline> {
  return device.createComputePipelineAsync({
    label: label ?? entryPoint,
    layout: device.createPipelineLayout({ bindGroupLayouts: layouts }),
    compute: { module, entryPoint },
  });
}

export const nextFrame = () => new Promise<void>((r) => requestAnimationFrame(() => r()));
