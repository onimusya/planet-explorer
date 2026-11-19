// Entry point: bootstrap, destinations, adaptive quality, metrics HUD, benchmark, frame loop.
import { initGpu } from './gpu';
import { createCtx } from './ctx';
import { Renderer } from './renderer';
import { Camera } from './camera';
import { Input } from './input';
import { clamp } from './math';
import { PASS_NAMES } from './metrics';
import { clock, DT, TIME, FRAME_MS, CPU_MS } from './clock';
import { Music } from './music';
import {
  PLANET_NAME, SEED, VISTA_P, VISTA_DIR, VALLEY_A, VALLEY_B, HIGH_CAM, LAKE_C, WORLD_L, valleyFraming,
} from './world';

const canvas = document.getElementById('c') as HTMLCanvasElement;
const loadEl = document.getElementById('load') as HTMLDivElement;
const progEl = document.getElementById('prog') as HTMLElement;
const msgEl = document.getElementById('msg') as HTMLElement;
const pctEl = document.getElementById('pct') as HTMLElement;
const barEl = document.querySelector('#bar i') as HTMLElement;
const goEl = document.getElementById('go') as HTMLButtonElement;
const tipBox = document.getElementById('tip') as HTMLElement;
const tipEl = document.querySelector('#tip span') as HTMLElement;
const helpEl = document.getElementById('help') as HTMLElement;
const errEl = document.getElementById('err') as HTMLDivElement;

function fail(e: unknown) {
  console.error(e);
  errEl.textContent = String((e as Error)?.message ?? e);
  errEl.style.display = 'block';
  msgEl.textContent = 'FAILED';
}

// --- preloader (index.html): real progress from the renderer's init steps, rotating tips ---
const loadLog: [number, number, string][] = []; // ms since navigation, percent, step (debug API)
const progress = (f: number, m: string) => {
  const pc = Math.round(f * 100);
  loadLog.push([Math.round(performance.now()), pc, m]);
  barEl.style.width = pc + '%';
  pctEl.textContent = pc + '%';
  progEl.setAttribute('aria-valuenow', String(pc));
  msgEl.textContent = m.toUpperCase();
};
const TIPS = [
  'Press 1, 2 or 3 to fly between the Arrival Vista, the Bioluminescent Valley and the Monumental Highlands.',
  'Click to capture the mouse, then roam with W A S D. Hold Shift to boost, use the wheel to set your speed.',
  'Starships cross the sky about once a minute, trailing their engine glow. Look up.',
  'In the Valley a glowing herd grazes from clearing to clearing, with sky jellies drifting overhead.',
  'Long-necked titans and stone colossi walk the mesa tops of the Highlands.',
  'Everything you see is generated in your browser. No textures or models are downloaded.',
  'Press N to switch the music off or on, M for the performance overlay.',
];
let tipI = Math.floor(Math.random() * TIPS.length);
tipEl.textContent = TIPS[tipI];
const tipTimer = setInterval(() => {
  tipBox.style.opacity = '0';
  setTimeout(() => {
    tipI = (tipI + 1) % TIPS.length;
    tipEl.textContent = TIPS[tipI];
    tipBox.style.opacity = '1';
  }, 650);
}, 5600);
// The world is revealed only after it has rendered behind the preloader for a moment (shadows,
// temporal filters and adaptive resolution settled), so the first frame seen is a finished one.
const WARM_FRAMES = 45;
const WARM_MS = 800;

// Destination poses: x, y, z, dirX, dirY, dirZ (heights resolved after terrain generation)
const POSES = new Float64Array(4 * 6);

async function main() {
  const music = new Music();
  music.load();
  // Gate: 0 building the world, 1 warming up behind the preloader, 2 waiting for the click or key
  // press that reveals it, 3 revealed. Until then the game sees no keys.
  let gate = 0;
  const enter = (gesture: boolean) => {
    if (gate === 3) return;
    gate = 3;
    if (gesture) music.unlock(); else music.arm();
    clearInterval(tipTimer);
    loadEl.classList.add('out');
    // the faded preloader must not stay layered over the canvas
    const hide = () => { loadEl.style.display = 'none'; };
    loadEl.addEventListener('transitionend', (e) => { if (e.target === loadEl) hide(); });
    setTimeout(hide, 2000);
  };
  loadEl.addEventListener('click', () => { if (gate === 2) enter(true); });
  addEventListener('keydown', (e) => {
    if (gate === 3) return;
    e.stopPropagation();
    // keys that count as a user gesture for audio (not Esc, modifiers or function keys)
    if (gate === 2 && !e.repeat && (e.key.length === 1 || e.key === 'Enter')) enter(true);
  }, true);

  const gpu = await initGpu(canvas);
  const ctx = createCtx(gpu);
  const camera = new Camera();
  const input = new Input(canvas);
  const r = new Renderer(ctx, camera);
  await r.init(progress);
  progress(0.86, 'Warming up');
  gate = 1;
  const ground = (x: number, z: number) => Math.max(r.terrain.heightAt(x, z), 0);
  camera.ground = r.terrain;

  // --- destinations ---
  const setPose = (i: number, x: number, y: number, z: number, dx: number, dy: number, dz: number) => {
    const l = Math.hypot(dx, dy, dz);
    POSES.set([x, y, z, dx / l, dy / l, dz / l], i * 6);
  };
  const D2R = Math.PI / 180;
  const dirAz = (az: number, pitch: number): [number, number, number] => [
    Math.sin(az * D2R) * Math.cos(pitch * D2R), Math.sin(pitch * D2R), -Math.cos(az * D2R) * Math.cos(pitch * D2R)];
  // B. Bioluminescent Valley at time t: standing in the herd's glowing clearing, looking back at
  // the herd as it walks its trail, with the mushroom forest behind it (world.ts valleyFraming)
  const VF = new Float64Array(4);
  const valleyPose = (t: number) => {
    valleyFraming(t, VF);
    const vy = ground(VF[0], VF[1]) + 5;
    // aim a little over the animals' backs: the herd sits low in the frame, under the canopy and
    // the kettle that circles behind it
    setPose(1, VF[0], vy, VF[1], VF[2] - VF[0], ground(VF[2], VF[3]) + 3.5 - vy, VF[3] - VF[1]);
  };
  {
    // A. Arrival Vista: on the cliff lip; sun just off the left edge, gas giant right of centre over the lake
    const x = VISTA_P[0] + VISTA_DIR[0] * 30, z = VISTA_P[1] + VISTA_DIR[1] * 30;
    setPose(0, x, ground(x, z) + 4.5, z, ...dirAz(18, -4));
    // B. Bioluminescent Valley: see valleyPose()
    valleyPose(0);
    // C. Monumental Highlands: above a mesa prow on the west rim, looking east-north-east along the canyon
    const [hx, hz] = HIGH_CAM;
    setPose(2, hx, Math.max(600, ground(hx, hz) + 40), hz, ...dirAz(75, -12));
    void LAKE_C;
  }
  const goPose = (i: number, fly: boolean) => {
    const o = i * 6;
    // the herd walks a fixed rhythm, so frame it where it will be when the camera arrives
    if (i === 1) valleyPose(clock[TIME] + (fly ? camera.flyDuration(POSES[o], POSES[o + 1], POSES[o + 2]) : 0));
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
  const benchSum = new Float64Array(12); // copy of Metrics.sum at the end of the last run
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
    benchSum.set(S);
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
  const m = r.metrics;
  m.clearText();

  const writeHud = () => {
    m.summarize(240);
    const S = m.sum;
    m.clearText();
    let c = m.put(0, 0, PLANET_NAME);
    c = m.put(0, c, '   SEED ');
    m.hex(0, c, SEED);
    // values are passed as (array, index) pairs: see Metrics.num
    c = m.put(1, 0, 'FPS ');
    c = m.num(1, c, S, 0, 1);
    c = m.put(1, c, '   1% LOW ');
    c = m.num(1, c, S, 1, 1);
    c = m.put(1, c, '   FRAME ');
    c = m.num(1, c, S, 2, 2);
    m.put(1, c, ' MS');
    c = m.put(2, 0, 'P95 ');
    c = m.num(2, c, S, 3, 2);
    c = m.put(2, c, '  P99 ');
    c = m.num(2, c, S, 4, 2);
    c = m.put(2, c, ' MS   CPU ');
    c = m.num(2, c, S, 5, 2);
    m.put(2, c, ' MS');
    c = m.put(3, 0, 'GPU ');
    if (m.hasGpu) {
      c = m.num(3, c, S, 6, 2);
      c = m.put(3, c, ' MS   P99 ');
      c = m.num(3, c, S, 9, 2);
      m.put(3, c, ' MS');
      c = 0;
      for (let p = 0; p < m.passMs.length; p++) {
        if (m.passMs[p] <= 0) continue;
        c = m.put(4, c, PASS_NAMES[p]);
        c = m.put(4, c, ' ');
        c = m.num(4, c, m.passMs, p, 1);
        c = m.put(4, c, ' ');
      }
    } else m.put(3, c, 'N/A (NO TIMESTAMP QUERY)');
    c = m.put(5, 0, 'RES ');
    c = m.int(5, c, r.iw);
    c = m.put(5, c, 'X');
    c = m.int(5, c, r.ih);
    c = m.put(5, c, ' (');
    m.v[0] = r.renderScale * 100;
    c = m.num(5, c, m.v, 0, 0);
    c = m.put(5, c, '%)  OUT ');
    c = m.int(5, c, r.t.outW);
    c = m.put(5, c, 'X');
    c = m.int(5, c, r.t.outH);
    c = m.put(5, c, adaptive ? '  ADAPTIVE' : '  FIXED');
    c = m.put(6, 0, 'QUALITY ');
    c = m.put(6, c, r.quality >= 2 ? 'HIGH' : r.quality >= 1 ? 'MEDIUM' : 'LOW');
    c = m.put(6, c, '   NODES ');
    c = m.int(6, c, r.terrain.nodeCount);
    c = m.put(6, c, '   ALT ');
    c = m.num(6, c, camera.pos, 1, 0);
    c = m.put(6, c, ' M   SPEED ');
    m.v[0] = camera.speed;
    m.num(6, c, m.v, 0, 0);
    if (bench.running) m.put(7, 0, 'BENCHMARK RUNNING...');
    else if (benchDone) {
      c = m.put(7, 0, 'BENCH  AVG ');
      c = m.num(7, c, benchSum, 0, 1);
      c = m.put(7, c, ' FPS  1% LOW ');
      c = m.num(7, c, benchSum, 1, 1);
      c = m.put(7, c, '  P99 ');
      c = m.num(7, c, benchSum, 4, 2);
      c = m.put(7, c, ' MS  GPU ');
      m.num(7, c, benchSum, 6, 2);
    }
    ctx.device.queue.writeBuffer(r.post.textBuf, 0, m.text);
  };

  // Frame-loop scalars live in typed arrays, not closure variables: a fractional number stored
  // in a closure variable is boxed into a new heap object (see clock.ts).
  const lt = new Float64Array(5); // previous rAF timestamp, frame start, quality timer, HUD timer, warm-up start
  lt[0] = performance.now();
  let warm = 0; // frames rendered behind the preloader
  const loop = (now: number) => {
    requestAnimationFrame(loop);
    // CPU time is taken only while the overlay or the benchmark reports it: every
    // performance.now() call returns a new heap number. Frame time uses the rAF timestamp.
    const cpuTimed = r.overlayOn || bench.running;
    if (cpuTimed) lt[1] = performance.now();
    clock[FRAME_MS] = now - lt[0];
    lt[0] = now;
    clock[DT] = Math.min(Math.max(clock[FRAME_MS] / 1000, 0), 0.1);
    clock[TIME] = now / 1000;
    const dt = clock[DT];
    if (gate === 1) {
      if (warm === 0) lt[4] = now;
      warm++;
      if (warm === 20) progress(0.93, 'Warming up');
      if (warm >= WARM_FRAMES && now - lt[4] >= WARM_MS) {
        gate = 2;
        progress(1, 'Ready');
        loadEl.classList.add('ready');
        goEl.focus({ preventScroll: true });
      }
    }
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
    if (a[8]) { a[8] = 0; music.toggle(); }
    camera.update(input);
    music.update(camera.pos[0], camera.pos[2]);
    // GPU timing runs while something reads it (read back in batches, see Metrics)
    m.gpuEnabled = m.hasGpu && (r.overlayOn || bench.running || adaptive);
    const swap = ctx.gpu.context.getCurrentTexture();
    r.render(ctx.gpu.textureViews ? swap : swap.createView());
    clock[CPU_MS] = cpuTimed ? performance.now() - lt[1] : NaN;
    if (bench.warm > 0) bench.warm--;
    else if (bench.running) bench.frames++;
    m.record();
    // adaptive render scale from GPU time (hysteresis)
    lt[2] += dt;
    if (adaptive && lt[2] > 0.5 && m.hasGpu) {
      lt[2] = 0;
      const g = m.lastGpu;
      if (g > 14.8) r.renderScale = Math.min(Math.max(r.renderScale - (g > 18 ? 0.1 : 0.05), 0.5), 1);
      else if (g > 0 && g < 11.5) r.renderScale = Math.min(Math.max(r.renderScale + 0.05, 0.5), 1);
    }
    lt[3] += dt;
    if (r.overlayOn && lt[3] > 0.25) { lt[3] = 0; writeHud(); }
  };
  requestAnimationFrame(loop);

  // --- automation / debug API (allocations only happen when called externally) ---
  const api = {
    ready: true,
    view: (i: number) => goPose(i, false),
    pose: (x: number, y: number, z: number, dx: number, dy: number, dz: number) => { camera.setPose(x, y, z, dx, dy, dz); r.resetHistory = true; },
    ground: (x: number, z: number) => r.terrain.heightAt(x, z),
    hprobe: (x: number, z: number) => r.terrain.probe(x, z).then((a) => Array.from(a)),
    bench: () => { benchDone = false; benchStart(); bench.start = performance.now(); },
    benchDone: () => benchDone,
    overlay: (on: boolean) => { r.overlayOn = on; },
    scale: (s: number) => { r.renderScale = clamp(s, 0.4, 1); adaptive = false; },
    adaptive: (on: boolean) => { adaptive = on; },
    stats: () => { m.summarize(600); return Array.from(m.sum); },
    cam: () => [camera.pos[0], camera.pos[1], camera.pos[2], camera.fwd[0], camera.fwd[1], camera.fwd[2]],
    vegCounts: () => r.veg.debugCounts(),
    vegStats: () => r.veg.stats,
    fauna: () => r.fauna.debugPoses(),
    faunaStats: () => r.fauna.stats,
    volDebug: (m: number) => { r.vol.debug = m; },
    passMs: () => Array.from(r.metrics.passMs),
    music: () => music.state(),
    enter: () => enter(false), // dismiss the preloader (audio then waits for the next gesture)
    gate: () => gate,
    loadLog: () => loadLog,
    frames: () => r.frame,
    renderer: r, // for scripts/heap-sample.mjs --app (probes the app's own pass descriptors)
    info: () => ({ adapter: gpu.adapterName, timestamps: gpu.timestamps, textureViews: gpu.textureViews, internal: [r.iw, r.ih], out: [r.t.outW, r.t.outH], worldL: WORLD_L }),
  };
  (window as unknown as Record<string, unknown>).__pe = api;
  // ?bench=1 (or ?gate=0) skips the preloader's click
  if (urlp.get('bench') === '1' || urlp.get('gate') === '0') enter(false);
  if (urlp.get('bench') === '1') setTimeout(() => api.bench(), 2500);
}

main().catch(fail);
