// Samples JS heap allocations in the running app through the Chrome DevTools Protocol, including
// short-lived objects the GC has already collected, and prints bytes per frame by function.
//
// Usage: node scripts/heap-sample.mjs <cdp-ws-url> [seconds=5] [page-url-filter=localhost] [--probe]
//   --probe  instead measures what single browser API calls allocate (bytes per call), to
//            account for what remains in the frame loop.
// Get the URL with `agent-browser --session planet get cdp-url`, or start Chrome with
// --remote-debugging-port=9222 and read webSocketDebuggerUrl from /json/version.
//   --interval=N  mean bytes between samples (default 256; smaller is less noisy and slower)
const args = process.argv.slice(2);
const probe = args.includes('--probe') || args.includes('--app');
const interval = Number(args.find((a) => a.startsWith('--interval='))?.slice(11) ?? 256);
const [wsUrl, secsArg = '5', filter = 'localhost'] = args.filter((a) => !a.startsWith('--'));
if (!wsUrl) {
  console.error('usage: node scripts/heap-sample.mjs <cdp-ws-url> [seconds] [page-url-filter] [--probe]');
  process.exit(1);
}
const secs = Number(secsArg);
// never hang: give up if the browser stops answering
setTimeout(() => { console.error('timed out waiting for the browser'); process.exit(2); }, (secs + 30) * 1000).unref();

const ws = new WebSocket(wsUrl);
let nextId = 0;
const pending = new Map();
const send = (method, params = {}, sessionId) =>
  new Promise((resolve, reject) => {
    const id = ++nextId;
    pending.set(id, { resolve, reject });
    ws.send(JSON.stringify({ id, method, params, sessionId }));
  });
ws.onmessage = (ev) => {
  const m = JSON.parse(ev.data);
  const p = m.id && pending.get(m.id);
  if (!p) return;
  pending.delete(m.id);
  if (m.error) p.reject(new Error(`${m.error.message}`));
  else p.resolve(m.result);
};
await new Promise((resolve, reject) => { ws.onopen = resolve; ws.onerror = reject; });

let sessionId;
if (wsUrl.includes('/devtools/browser/')) {
  const { targetInfos } = await send('Target.getTargets');
  const page = targetInfos.find((t) => t.type === 'page' && t.url.includes(filter));
  if (!page) throw new Error(`no page whose URL contains "${filter}"`);
  ({ sessionId } = await send('Target.attachToTarget', { targetId: page.targetId, flatten: true }));
}
const cdp = (method, params) => send(method, params, sessionId);
const evaluate = async (expression, awaitPromise = false) =>
  (await cdp('Runtime.evaluate', { expression, returnByValue: true, awaitPromise })).result.value;

// API probes: each named function makes N calls of one kind; its allocations divided by N give the
// bytes per call. A separate device keeps the app's own device untouched.
const N = 4000;
const PROBES = `(async () => {
  window.__probeGo = 0;
  const ad = await navigator.gpu.requestAdapter();
  const ts = ad.features.has('timestamp-query');
  const d = await ad.requestDevice({ requiredFeatures: ts ? ['timestamp-query'] : [] });
  const tex = d.createTexture({ size: [4, 4], format: 'rgba8unorm', usage: GPUTextureUsage.RENDER_ATTACHMENT });
  const view = tex.createView();
  const rpd = { colorAttachments: [{ view, loadOp: 'clear', storeOp: 'store' }] };
  const cpd = {};
  const keep = [null], out = new Float64Array(1);
  const N = ${N};
  function probeCommandEncoder() { for (let i = 0; i < N; i++) keep[0] = d.createCommandEncoder(); }
  function probeComputePass(e) { for (let i = 0; i < N; i++) e.beginComputePass(cpd).end(); }
  function probeRenderPass(e) { for (let i = 0; i < N; i++) e.beginRenderPass(rpd).end(); }
  function probeFinish() { for (let i = 0; i < N; i++) { const e = d.createCommandEncoder(); keep[0] = e.finish(); } }
  function probeCreateView() { for (let i = 0; i < N; i++) keep[0] = tex.createView(); }
  const td = { size: [1, 1], format: 'rgba8unorm', usage: GPUTextureUsage.RENDER_ATTACHMENT };
  function probeTexture() { for (let i = 0; i < N; i++) { const t = d.createTexture(td); t.destroy(); } }
  function probePerformanceNow() { for (let i = 0; i < N; i++) out[0] = performance.now(); }
  // argument conversions used every frame: arrays as GPUExtent3D / sequences, typed-array uploads
  const src = { texture: d.createTexture({ size: [4, 4], format: 'rgba8unorm', usage: GPUTextureUsage.COPY_SRC }) };
  const dst = { texture: d.createTexture({ size: [4, 4], format: 'rgba8unorm', usage: GPUTextureUsage.COPY_DST }) };
  const extArr = [4, 4, 1], extDict = { width: 4, height: 4 };
  function probeCopyExtentArray(e) { for (let i = 0; i < N; i++) e.copyTextureToTexture(src, dst, extArr); }
  function probeCopyExtentDict(e) { for (let i = 0; i < N; i++) e.copyTextureToTexture(src, dst, extDict); }
  const ub = d.createBuffer({ size: 256, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }), ud = new Float32Array(64);
  function probeWriteBuffer() { for (let i = 0; i < N; i++) d.queue.writeBuffer(ub, 0, ud); }
  const be = d.createRenderBundleEncoder({ colorFormats: ['rgba8unorm'] }), bundles = [be.finish()];
  function probeExecuteBundles(p) { for (let i = 0; i < N; i++) p.executeBundles(bundles); }
  const none = [];
  function probeSubmitEmpty() { for (let i = 0; i < N; i++) d.queue.submit(none); }
  // pass descriptors shaped like the app's: depth-only with a begin timestamp (shadow cascade),
  // colour + depth + timestamps (main pass), compute with timestamps
  const qs = ts ? d.createQuerySet({ type: 'timestamp', count: 4 }) : null;
  const dtex = d.createTexture({ size: [4, 4], format: 'depth32float', usage: GPUTextureUsage.RENDER_ATTACHMENT });
  const dview = dtex.createView();
  const shd = { colorAttachments: [], depthStencilAttachment: { view: dview, depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' } };
  const mainD = { colorAttachments: [{ view, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 0 } }],
    depthStencilAttachment: { view: dview, depthClearValue: 0, depthLoadOp: 'clear', depthStoreOp: 'store' } };
  const compD = {};
  if (qs) {
    shd.timestampWrites = { querySet: qs, beginningOfPassWriteIndex: 0 };
    mainD.timestampWrites = { querySet: qs, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 };
    compD.timestampWrites = { querySet: qs, beginningOfPassWriteIndex: 2, endOfPassWriteIndex: 3 };
  }
  const texD = { colorAttachments: [{ view: tex, loadOp: 'clear', storeOp: 'store' }] };
  function probeTextureAsView(e) { for (let i = 0; i < N; i++) e.beginRenderPass(texD).end(); }
  function probeViewportScissor(p) { for (let i = 0; i < N; i++) { p.setViewport(0, 0, 4, 4, 0, 1); p.setScissorRect(0, 0, 4, 4); } }
  const bgl = d.createBindGroupLayout({ entries: [] }), bg = d.createBindGroup({ layout: bgl, entries: [] });
  const cpipe = d.createComputePipeline({ layout: d.createPipelineLayout({ bindGroupLayouts: [bgl] }),
    compute: { module: d.createShaderModule({ code: '@compute @workgroup_size(1) fn main() {}' }), entryPoint: 'main' } });
  const qres = qs ? d.createBuffer({ size: 32, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }) : null;
  const qdst = d.createBuffer({ size: 32, usage: GPUBufferUsage.COPY_DST });
  function probeResolve(e) { if (qs) for (let i = 0; i < N; i++) { e.resolveQuerySet(qs, 0, 4, qres, 0); e.copyBufferToBuffer(qres, 0, qdst, 0, 32); } }
  function probeSetPipeline(p) { for (let i = 0; i < N; i++) p.setPipeline(cpipe); }
  function probeSetBindGroup(p) { for (let i = 0; i < N; i++) p.setBindGroup(0, bg); }
  function probeDispatch(p) { for (let i = 0; i < N; i++) p.dispatchWorkgroups(8, 1); }
  // GPU timing readback: one map cycle (start, then read and unmap), and polling mapState
  const rb = d.createBuffer({ size: 1792, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const words = [null];
  function probeMapStart() { rb.mapAsync(GPUMapMode.READ); }
  function probeMapRead() { words[0] = new Uint32Array(rb.getMappedRange()); rb.unmap(); }
  function probeMapState() { let n = 0; for (let i = 0; i < N; i++) if (rb.mapState === 'mapped') n++; return n; }
  // one submit (or write) per task, as in a frame loop: lets the browser schedule its flush each time
  function probeSubmitPerTask() { cbs[0] = d.createCommandEncoder().finish(); d.queue.submit(cbs); }
  function probeWritePerTask() { d.queue.writeBuffer(ub, 0, ud); }
  function probeWrite6PerTask() { for (let j = 0; j < 6; j++) d.queue.writeBuffer(ub, 0, ud); }
  // the app's per-frame sequence of WebGPU calls, once per task: 1 encoder, 3 compute passes,
  // 6 render passes, 2 copies, 4 buffer writes, finish and submit (11 API objects)
  function probeFrameCallsPerTask() {
    for (let j = 0; j < 4; j++) d.queue.writeBuffer(ub, 0, ud);
    const e = d.createCommandEncoder();
    e.beginComputePass(cpd).end();
    for (let j = 0; j < 4; j++) e.beginRenderPass(shd).end();
    e.beginRenderPass(mainD).end();
    e.copyTextureToTexture(src, dst, extArr); e.copyTextureToTexture(src, dst, extArr);
    e.beginRenderPass(rpd).end();
    e.beginComputePass(cpd).end();
    e.beginComputePass(cpd).end();
    cbs[0] = e.finish();
    d.queue.submit(cbs);
  }
  async function perTask() {
    for (let k = 0; k < 400; k++) { probeFrameCallsPerTask(); await new Promise((r) => setTimeout(r, 0)); }
    for (let k = 0; k < 400; k++) { probeSubmitPerTask(); await new Promise((r) => setTimeout(r, 0)); }
    for (let k = 0; k < 400; k++) { probeWritePerTask(); await new Promise((r) => setTimeout(r, 0)); }
    for (let k = 0; k < 400; k++) { probeWrite6PerTask(); await new Promise((r) => setTimeout(r, 0)); }
  }
  const MAPS = 200;
  async function mapCycles() {
    for (let k = 0; k < MAPS; k++) {
      probeMapStart();
      while (rb.mapState !== 'mapped') await new Promise((r) => setTimeout(r, 0));
      if (k === 0) probeMapState();
      probeMapRead();
    }
  }
  const cbs = [null];
  function probeSubmitOne() { for (let i = 0; i < N; i++) { cbs[0] = d.createCommandEncoder().finish(); d.queue.submit(cbs); } }
  function probeShadowPass(e) { for (let i = 0; i < N; i++) e.beginRenderPass(shd).end(); }
  function probeMainPass(e) { for (let i = 0; i < N; i++) e.beginRenderPass(mainD).end(); }
  function probeComputePassTS(e) { for (let i = 0; i < N; i++) e.beginComputePass(compD).end(); }
  for (let r = 0; r < 3; r++) { // warm up, then measure the last round
    probeCommandEncoder();
    const e1 = d.createCommandEncoder(); probeComputePass(e1); e1.finish();
    const e2 = d.createCommandEncoder(); probeRenderPass(e2); e2.finish();
    probeFinish(); probeCreateView(); probeTexture(); probePerformanceNow();
    const e3 = d.createCommandEncoder(); probeCopyExtentArray(e3); probeCopyExtentDict(e3); e3.finish();
    probeWriteBuffer(); probeSubmitEmpty();
    const e4 = d.createCommandEncoder(), p4 = e4.beginRenderPass(rpd); probeExecuteBundles(p4); p4.end(); e4.finish();
    const e5 = d.createCommandEncoder(); probeShadowPass(e5); probeMainPass(e5); probeComputePassTS(e5); e5.finish();
    const e6 = d.createCommandEncoder(); probeTextureAsView(e6); probeResolve(e6);
    const p6 = e6.beginRenderPass(rpd); probeViewportScissor(p6); p6.end();
    const c6 = e6.beginComputePass(); probeSetPipeline(c6); probeSetBindGroup(c6); probeDispatch(c6); c6.end(); e6.finish();
    probeSubmitOne();
    await mapCycles();
    await perTask();
    if (r === 1) window.__probeGo = 1, await new Promise((res) => (window.__probeStart = res));
  }
  d.destroy();
  return 1;
})()`;

// --app: the same per-call measurement on the app's own device and pass descriptors
const APP_PROBES = `(async () => {
  window.__probeGo = 0;
  const r = window.__pe.renderer, d = r.ctx.device, N = ${N};
  const keep = [null];
  d.pushErrorScope('validation');
  function probeAppEncoder() { for (let i = 0; i < N; i++) keep[0] = d.createCommandEncoder(); }
  function probeAppComputePass(e) { for (let i = 0; i < N; i++) e.beginComputePass(r.computeDesc).end(); }
  function probeAppShadowPass(e) { for (let i = 0; i < N; i++) e.beginRenderPass(r.shadowDescs[0]).end(); }
  function probeAppMainPass(e) { for (let i = 0; i < N; i++) e.beginRenderPass(r.mainDesc).end(); }
  function probeAppWaterPass(e) { for (let i = 0; i < N; i++) e.beginRenderPass(r.waterDesc).end(); }
  for (let k = 0; k < 3; k++) {
    probeAppEncoder();
    const e = d.createCommandEncoder();
    probeAppComputePass(e); probeAppShadowPass(e); probeAppMainPass(e); probeAppWaterPass(e);
    if (k === 1) window.__probeGo = 1, await new Promise((res) => (window.__probeStart = res));
  }
  await d.popErrorScope();
  return 1;
})()`;
const appProbe = args.includes('--app');

let f0 = 0, f1 = 0, t0 = 0, t1 = 0, profile;
const frameCount = 'window.__pe && window.__pe.frames ? window.__pe.frames() : -1';
await cdp('HeapProfiler.enable');
await cdp('HeapProfiler.collectGarbage');
if (probe) {
  // run two warm-up rounds, then start sampling and let the last round run
  let failed = '';
  const done = cdp('Runtime.evaluate', { expression: appProbe ? APP_PROBES : PROBES, awaitPromise: true, returnByValue: true }).then((r) => {
    if (r.exceptionDetails) failed = r.exceptionDetails.exception?.description ?? r.exceptionDetails.text;
  });
  while (!(await evaluate('window.__probeGo === 1'))) {
    if (failed) { console.error(`probe failed: ${failed}`); process.exit(3); }
    await new Promise((r) => setTimeout(r, 50));
  }
  await cdp('HeapProfiler.startSampling', { samplingInterval: 64, includeObjectsCollectedByMajorGC: true, includeObjectsCollectedByMinorGC: true });
  await evaluate('window.__probeStart()');
  await done;
  ({ profile } = await cdp('HeapProfiler.stopSampling'));
} else {
  await cdp('HeapProfiler.startSampling', { samplingInterval: interval, includeObjectsCollectedByMajorGC: true, includeObjectsCollectedByMinorGC: true });
  f0 = await evaluate(frameCount);
  t0 = performance.now();
  await new Promise((r) => setTimeout(r, secs * 1000));
  f1 = await evaluate(frameCount);
  t1 = performance.now();
  ({ profile } = await cdp('HeapProfiler.stopSampling'));
}
ws.close();

// self bytes per function (allocations with no script URL go to the nearest script frame)
const sites = new Map();
let total = 0;
const walk = (node, stack) => {
  const cf = node.callFrame;
  const file = cf.url ? cf.url.replace(/^https?:\/\/[^/]+\//, '').replace(/\?.*$/, '') : '';
  const name = `${cf.functionName || '(anonymous)'}${file ? ` ${file}:${cf.lineNumber + 1}` : ''}`;
  const here = [...stack, name];
  if (node.selfSize > 0) {
    let key = name;
    if (!file) for (let i = here.length - 2; i >= 0; i--) if (/:\d+$/.test(here[i])) { key = `${name} <- ${here[i]}`; break; }
    sites.set(key, (sites.get(key) ?? 0) + node.selfSize);
    total += node.selfSize;
  }
  for (const c of node.children ?? []) walk(c, here);
};
walk(profile.head, []);

const sampleSizes = () => {
  // object sizes sampled per function, as "size x samples" (each sample is reported as size x its
  // scale factor, e.g. 80 = one 16 B object and 72 = one 12 B object at a 64 B interval)
  const nodeName = new Map();
  const index = (node) => { nodeName.set(node.id, node.callFrame.functionName || '(anonymous)'); for (const c of node.children ?? []) index(c); };
  index(profile.head);
  const bySite = new Map();
  for (const s of profile.samples ?? []) {
    const name = nodeName.get(s.nodeId);
    if (!bySite.has(name)) bySite.set(name, new Map());
    const h = bySite.get(name);
    h.set(s.size, (h.get(s.size) ?? 0) + 1);
  }
  for (const [name, h] of bySite) {
    const parts = [...h].sort((a, b) => b[1] - a[1]).slice(0, 8).map(([sz, n]) => `${sz}x${n}`);
    console.log(`sizes ${name}: ${parts.join(' ')}`);
  }
};
if (probe) {
  if (args.includes('--sizes')) sampleSizes();
  console.log(`bytes per call (${N} calls each; map cycle: per cycle of ${200}; sampled every 64 B on average):`);
  for (const [k, v] of [...sites].filter(([k]) => k.startsWith('probe')).sort()) {
    const name = k.split(' ')[0];
    const per = name === 'probeMapStart' || name === 'probeMapRead' ? 200 : name.endsWith('PerTask') ? 400 : N;
    console.log(`${(v / per).toFixed(1).padStart(8)}  ${name}`);
  }
} else {
  const frames = Math.max(1, f1 - f0);
  const seconds = (t1 - t0) / 1000;
  const kb = (b) => (b / 1024).toFixed(1);
  console.log(`sampled ${seconds.toFixed(1)} s, ${frames} frames (${(frames / seconds).toFixed(0)} fps)`);
  console.log(`allocated ${kb(total)} KiB total = ${(total / frames).toFixed(0)} B/frame, ${kb(total / seconds)} KiB/s`);
  console.log('allocation sites (B/frame):');
  for (const [k, v] of [...sites].sort((a, b) => b[1] - a[1]).slice(0, 18)) {
    console.log(`${(v / frames).toFixed(1).padStart(9)}  ${k}`);
  }
  if (args.includes('--objects')) {
    // samples carry size x scaled count; at this interval 80 = one 16 B object, 72 = one 12 B object
    const perObj = (s) => [16, 12, 20, 24, 28, 32].find((o) => Math.round(1 / (1 - Math.exp(-o / interval))) * o === s);
    const nodeName = new Map();
    const index = (node) => { nodeName.set(node.id, node.callFrame.functionName || '(anonymous)'); for (const c of node.children ?? []) index(c); };
    index(profile.head);
    const by = new Map();
    for (const s of profile.samples ?? []) {
      const o = perObj(s.size);
      const key = `${nodeName.get(s.nodeId)} ${o ? o + ' B' : '?' + s.size}`;
      by.set(key, (by.get(key) ?? 0) + Math.round(1 / (1 - Math.exp(-(o ?? s.size) / interval))));
    }
    for (const [k, n] of [...by].sort((a, b) => b[1] - a[1]).slice(0, 10)) console.log(`objects/frame ${(n / frames).toFixed(2).padStart(6)}  ${k}`);
  }
  if (args.includes('--sizes')) sampleSizes();
}
