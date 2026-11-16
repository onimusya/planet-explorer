// Entry point: bootstrap, destinations, adaptive quality, metrics HUD, benchmark, frame loop.
import { initGpu } from './gpu';
import { createCtx } from './ctx';
import { Renderer } from './renderer';
import { Camera } from './camera';
import { Input } from './input';
import { clamp } from './math';
import {
  PLANET_NAME, SEED, VISTA_P, VISTA_DIR, VALLEY_A, VALLEY_B, HIGHLANDS_C, LAKE_C, WORLD_L,
} from './world';

const canvas = document.getElementById('c') as HTMLCanvasElement;
const loadEl = document.getElementById('load') as HTMLDivElement;
const msgEl = document.getElementById('msg') as HTMLDivElement;
const barEl = document.querySelector('#bar i') as HTMLElement;
const helpEl = document.getElementById('help') as HTMLDivElement;
const errEl = document.getElementById('err') as HTMLDivElement;

function fail(e: unknown) {
  console.error(e);
  errEl.textContent = String((e as Error)?.message ?? e);
  errEl.style.display = 'block';
  msgEl.textContent = 'FAILED';
}

// Destination poses: x, y, z, dirX, dirY, dirZ (heights resolved after terrain generation)
const POSES = new Float64Array(4 * 6);

async function main() {
  const gpu = await initGpu(canvas);
  const ctx = createCtx(gpu);
  const camera = new Camera();
  const input = new Input(canvas);
  const r = new Renderer(ctx, camera);
  await r.init((f, m) => { barEl.style.width = (f * 100).toFixed(0) + '%'; msgEl.textContent = m.toUpperCase(); });
  const ground = (x: number, z: number) => Math.max(r.terrain.heightAt(x, z), 0);
  camera.groundAt = ground;

  // --- destinations ---
  const setPose = (i: number, x: number, y: number, z: number, dx: number, dy: number, dz: number) => {
    const l = Math.hypot(dx, dy, dz);
    POSES.set([x, y, z, dx / l, dy / l, dz / l], i * 6);
  };
  {
    const x = VISTA_P[0] - VISTA_DIR[0] * 150, z = VISTA_P[1] - VISTA_DIR[1] * 150;
    setPose(0, x, ground(x, z) + 24, z, VISTA_DIR[0], -0.16, VISTA_DIR[1]);
    const ax = VALLEY_B[0] - VALLEY_A[0], az = VALLEY_B[1] - VALLEY_A[1];
    const vx = VALLEY_A[0] + ax * 0.3, vz = VALLEY_A[1] + az * 0.3;
    setPose(1, vx, ground(vx, vz) + 14, vz, ax, 0.08 * Math.hypot(ax, az), az);
    const hx = HIGHLANDS_C[0] + 900, hz = HIGHLANDS_C[1] + 1400;
    setPose(2, hx, ground(hx, hz) + 140, hz, HIGHLANDS_C[0] - hx, -0.12 * 1600, HIGHLANDS_C[1] - hz);
    void LAKE_C;
  }
  const goPose = (i: number, fly: boolean) => {
    const o = i * 6;
    if (fly) camera.flyTo(POSES[o], POSES[o + 1], POSES[o + 2], POSES[o + 3], POSES[o + 4], POSES[o + 5]);
    else camera.setPose(POSES[o], POSES[o + 1], POSES[o + 2], POSES[o + 3], POSES[o + 4], POSES[o + 5]);
    r.resetHistory = !fly;
  };
  goPose(0, false);

  // --- benchmark path through all destinations ---
  const benchKeys = new Float64Array([
    ...POSES.subarray(0, 6),
    POSES[0] + 900, POSES[1] + 80, POSES[2] - 1500, 0.4, -0.15, -0.9,
    POSES[6] - 600, POSES[7] + 260, POSES[8] - 400, 0.7, -0.25, 0.3,
    ...POSES.subarray(6, 12),
    POSES[6] + (VALLEY_B[0] - VALLEY_A[0]) * 0.4, POSES[7] + 4, POSES[8] + (VALLEY_B[1] - VALLEY_A[1]) * 0.4, VALLEY_B[0] - VALLEY_A[0], 0, VALLEY_B[1] - VALLEY_A[1],
    POSES[12] + 600, POSES[13] + 300, POSES[14] + 600, -0.5, -0.2, -0.7,
    ...POSES.subarray(12, 18),
    POSES[12] - 500, POSES[13] - 60, POSES[14] - 1200, -0.3, -0.1, -0.9,
    POSES[0], POSES[1] + 400, POSES[2], VISTA_DIR[0], -0.2, VISTA_DIR[1],
  ]);
  const bench = { running: false, start: 0, frames: 0, warm: 0 };
  const benchStart = () => {
    r.metrics.clearGpu();
    r.metrics.gpuEnabled = r.metrics.hasGpu;
    bench.running = true;
    bench.warm = 30;
    bench.frames = 0;
    camera.playPath(benchKeys, 5.0);
  };
  camera.onPathDone = () => {
    if (!bench.running) return;
    bench.running = false;
    r.metrics.summarize(bench.frames);
    const S = r.metrics.sum;
    (window as unknown as Record<string, unknown>).__bench = {
      frames: bench.frames, seconds: (performance.now() - bench.start) / 1000,
      avgFps: S[0], low1Fps: S[1], avgMs: S[2], p95Ms: S[3], p99Ms: S[4], maxMs: S[7],
      cpuAvgMs: S[5], cpuP99Ms: S[11], gpuAvgMs: S[6], gpuP95Ms: S[8], gpuP99Ms: S[9],
      renderScaleEnd: r.renderScale, internal: `${r.iw}x${r.ih}`, output: `${r.t.outW}x${r.t.outH}`,
      adapter: gpu.adapterName, timestamps: gpu.timestamps,
    };
    benchDone = true;
  };
  let benchDone = false;

  // --- resize ---
  let wantW = 0, wantH = 0;
  const onResize = () => {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    wantW = Math.max(64, Math.round(canvas.clientWidth * dpr));
    wantH = Math.max(64, Math.round(canvas.clientHeight * dpr));
    const s = Math.min(1, Math.sqrt((2560 * 1440) / (wantW * wantH)));
    wantW = Math.round(wantW * s); wantH = Math.round(wantH * s);
  };
  new ResizeObserver(onResize).observe(canvas);
  onResize();

  // --- adaptive quality ---
  const urlp = new URLSearchParams(location.search);
  let adaptive = urlp.get('adaptive') !== '0';
  const fixedScale = urlp.get('scale');
  if (fixedScale) { r.renderScale = clamp(parseFloat(fixedScale), 0.4, 1); adaptive = false; }
  let qTimer = 0;
  let hudTimer = 0;
  const m = r.metrics;
  m.clearText();

  const writeHud = () => {
    m.summarize(240);
    const S = m.sum;
    m.clearText();
    let c = m.put(0, 0, PLANET_NAME);
    c = m.put(0, c, '   SEED ');
    m.hex(0, c, SEED);
    c = m.put(1, 0, 'FPS ');
    c = m.num(1, c, S[0], 1);
    c = m.put(1, c, '   1% LOW ');
    c = m.num(1, c, S[1], 1);
    c = m.put(1, c, '   FRAME ');
    c = m.num(1, c, S[2], 2);
    m.put(1, c, ' MS');
    c = m.put(2, 0, 'P95 ');
    c = m.num(2, c, S[3], 2);
    c = m.put(2, c, '  P99 ');
    c = m.num(2, c, S[4], 2);
    c = m.put(2, c, ' MS   CPU ');
    c = m.num(2, c, S[5], 2);
    m.put(2, c, ' MS');
    c = m.put(3, 0, 'GPU ');
    if (m.hasGpu) {
      c = m.num(3, c, S[6], 2);
      c = m.put(3, c, ' MS  P99 ');
      c = m.num(3, c, S[9], 2);
      c = m.put(3, c, '  ');
      for (let p = 0; p < m.passMs.length; p++) {
        if (m.passMs[p] <= 0) continue;
        c = m.put(3, c, p === 0 ? 'PRE ' : p === 1 ? 'SHD ' : p === 2 ? 'MAIN ' : p === 3 ? 'WATR ' : p === 4 ? 'POST ' : 'FIN ');
        c = m.num(3, c, m.passMs[p], 1);
        c = m.put(3, c, ' ');
      }
    } else m.put(3, c, 'N/A (NO TIMESTAMP QUERY)');
    c = m.put(4, 0, 'RES ');
    c = m.num(4, c, r.iw, 0);
    c = m.put(4, c, 'X');
    c = m.num(4, c, r.ih, 0);
    c = m.put(4, c, ' (');
    c = m.num(4, c, r.renderScale * 100, 0);
    c = m.put(4, c, '%)  OUT ');
    c = m.num(4, c, r.t.outW, 0);
    c = m.put(4, c, 'X');
    c = m.num(4, c, r.t.outH, 0);
    c = m.put(4, c, adaptive ? '  ADAPTIVE' : '  FIXED');
    c = m.put(5, 0, 'QUALITY ');
    c = m.put(5, c, r.quality >= 2 ? 'HIGH' : r.quality >= 1 ? 'MEDIUM' : 'LOW');
    c = m.put(5, c, '   NODES ');
    c = m.num(5, c, r.terrain.nodeCount, 0);
    c = m.put(5, c, '   ALT ');
    c = m.num(5, c, camera.pos[1], 0);
    c = m.put(5, c, ' M   SPEED ');
    m.num(5, c, camera.speed, 0);
    if (bench.running) m.put(6, 0, 'BENCHMARK RUNNING...');
    else if (benchDone) {
      c = m.put(6, 0, 'BENCH  AVG ');
      const b = (window as unknown as { __bench: Record<string, number> }).__bench;
      c = m.num(6, c, b.avgFps, 1);
      c = m.put(6, c, ' FPS  1% LOW ');
      c = m.num(6, c, b.low1Fps, 1);
      c = m.put(6, c, '  P99 ');
      c = m.num(6, c, b.p99Ms, 2);
      c = m.put(6, c, ' MS  GPU ');
      m.num(6, c, b.gpuAvgMs, 2);
    }
    ctx.device.queue.writeBuffer(r.post.textBuf, 0, m.text);
  };

  let last = performance.now();
  let started = false;
  let frameStart = 0;
  const loop = (now: number) => {
    requestAnimationFrame(loop);
    frameStart = performance.now();
    const dtMs = now - last;
    last = now;
    const dt = clamp(dtMs / 1000, 0, 0.1);
    if (!started) { started = true; loadEl.style.opacity = '0'; }
    if (canvas.width !== wantW || canvas.height !== wantH) {
      canvas.width = wantW; canvas.height = wantH;
      r.resize(wantW, wantH);
    }
    // actions
    const a = input.actions;
    if (a[0]) { a[0] = 0; goPose(0, false); }
    if (a[1]) { a[1] = 0; goPose(0, true); }
    if (a[2]) { a[2] = 0; goPose(1, true); }
    if (a[3]) { a[3] = 0; goPose(2, true); }
    if (a[4]) { a[4] = 0; r.overlayOn = !r.overlayOn; }
    if (a[5]) { a[5] = 0; helpEl.style.opacity = helpEl.style.opacity === '0' ? '1' : '0'; }
    if (a[6]) { a[6] = 0; benchStart(); bench.start = performance.now(); }
    if (a[7]) { a[7] = 0; adaptive = !adaptive; }
    camera.update(dt, input);
    // GPU timing: every frame while HUD/benchmark active, else sampled every 8th frame for adaptivity
    m.gpuEnabled = m.hasGpu && (r.overlayOn || bench.running || (r.frame & 7) === 0);
    r.render(dt, now / 1000, ctx.gpu.context.getCurrentTexture().createView());
    const cpu = performance.now() - frameStart;
    if (bench.warm > 0) bench.warm--;
    else if (bench.running) bench.frames++;
    m.record(dtMs, cpu);
    // adaptive render scale from GPU time (hysteresis)
    qTimer += dt;
    if (adaptive && qTimer > 0.5 && m.hasGpu) {
      qTimer = 0;
      const g = m.lastGpu;
      if (g > 14.8) r.renderScale = clamp(r.renderScale - (g > 18 ? 0.1 : 0.05), 0.5, 1);
      else if (g > 0 && g < 11.5) r.renderScale = clamp(r.renderScale + 0.05, 0.5, 1);
    }
    hudTimer += dt;
    if (r.overlayOn && hudTimer > 0.25) { hudTimer = 0; writeHud(); }
  };
  requestAnimationFrame(loop);

  // --- automation / debug API (allocations only happen when called externally) ---
  const api = {
    ready: true,
    view: (i: number) => goPose(i, false),
    pose: (x: number, y: number, z: number, dx: number, dy: number, dz: number) => { camera.setPose(x, y, z, dx, dy, dz); r.resetHistory = true; },
    ground: (x: number, z: number) => r.terrain.heightAt(x, z),
    bench: () => { benchDone = false; benchStart(); bench.start = performance.now(); },
    benchDone: () => benchDone,
    overlay: (on: boolean) => { r.overlayOn = on; },
    scale: (s: number) => { r.renderScale = clamp(s, 0.4, 1); adaptive = false; },
    adaptive: (on: boolean) => { adaptive = on; },
    stats: () => { m.summarize(600); return Array.from(m.sum); },
    cam: () => [camera.pos[0], camera.pos[1], camera.pos[2], camera.fwd[0], camera.fwd[1], camera.fwd[2]],
    vegCounts: () => r.veg.debugCounts(),
    vegStats: () => r.veg.stats,
    info: () => ({ adapter: gpu.adapterName, timestamps: gpu.timestamps, internal: [r.iw, r.ih], out: [r.t.outW, r.t.outH], worldL: WORLD_L }),
  };
  (window as unknown as Record<string, unknown>).__pe = api;
  if (urlp.get('bench') === '1') setTimeout(() => api.bench(), 2500);
}

main().catch(fail);
