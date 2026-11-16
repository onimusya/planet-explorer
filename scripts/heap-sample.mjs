// Samples JS heap allocations in the running app through the Chrome DevTools Protocol, including
// short-lived objects that the GC has already collected, and prints the top allocation sites.
// Usage: node scripts/heap-sample.mjs <cdp-ws-url> [seconds=5] [page-url-filter=localhost]
// Get the URL with `agent-browser --session planet get cdp-url` (or chrome://inspect for a manual run).
const [wsUrl, secsArg = '5', filter = 'localhost'] = process.argv.slice(2);
if (!wsUrl) {
  console.error('usage: node scripts/heap-sample.mjs <cdp-ws-url> [seconds] [page-url-filter]');
  process.exit(1);
}
const secs = Number(secsArg);

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
const evaluate = async (expression) => (await cdp('Runtime.evaluate', { expression, returnByValue: true })).result.value;

// frame counter (a reused named callback: no per-frame closure)
await evaluate(`window.__fc = 0; if (!window.__fcOn) { window.__fcOn = 1; const f = () => { window.__fc++; requestAnimationFrame(f); }; requestAnimationFrame(f); } 1`);
await cdp('HeapProfiler.enable');
await cdp('HeapProfiler.collectGarbage');
await cdp('HeapProfiler.startSampling', {
  samplingInterval: 256,
  includeObjectsCollectedByMajorGC: true,
  includeObjectsCollectedByMinorGC: true,
});
const f0 = await evaluate('window.__fc');
const t0 = performance.now();
await new Promise((r) => setTimeout(r, secs * 1000));
const f1 = await evaluate('window.__fc');
const t1 = performance.now();
const { profile } = await cdp('HeapProfiler.stopSampling');
ws.close();

const frames = Math.max(1, f1 - f0);
const seconds = (t1 - t0) / 1000;
const sites = new Map();
let total = 0;
const walk = (node, stack) => {
  const cf = node.callFrame;
  const file = cf.url ? cf.url.replace(/^https?:\/\/[^/]+\//, '').replace(/\?.*$/, '') : '';
  const name = `${cf.functionName || '(anonymous)'}${file ? ` ${file}:${cf.lineNumber + 1}` : ''}`;
  const here = [...stack, name];
  if (node.selfSize > 0) {
    // attribute native allocations ("(V8 API)", builtins without a URL) to the nearest script frame
    let key = name;
    if (!file) for (let i = here.length - 2; i >= 0; i--) if (/:\d+$/.test(here[i])) { key = `${name} <- ${here[i]}`; break; }
    sites.set(key, (sites.get(key) ?? 0) + node.selfSize);
    total += node.selfSize;
  }
  for (const c of node.children ?? []) walk(c, here);
};
walk(profile.head, []);

const kb = (b) => (b / 1024).toFixed(1);
console.log(`sampled ${seconds.toFixed(1)} s, ${frames} frames (${(frames / seconds).toFixed(0)} fps)`);
console.log(`allocated ${kb(total)} KiB total = ${(total / frames).toFixed(0)} B/frame, ${kb(total / seconds)} KiB/s`);
console.log('top allocation sites (B/frame):');
for (const [k, v] of [...sites].sort((a, b) => b[1] - a[1]).slice(0, 18)) {
  console.log(`${(v / frames).toFixed(1).padStart(9)}  ${k}`);
}
