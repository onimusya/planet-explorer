// Frame-time statistics, GPU timestamp queries and the GPU text overlay buffer.
// All per-frame paths write into preallocated typed arrays.
import { clock, FRAME_MS, CPU_MS } from './clock';

export const RING = 2048;
export const PASS_NAMES = ['PRE', 'SHD', 'MAIN', 'WATR', 'VOL', 'POST', 'FIN', 'FX'] as const;
export const NPASS = PASS_NAMES.length;
// Frames of GPU timestamps per readback. Each readback allocates (mapAsync returns a Promise,
// every mapping a new ArrayBuffer, and reading it needs a view), so frames are read in batches.
export const BATCH = 16;
const QUERIES = BATCH * NPASS * 2;

export class Metrics {
  readonly frameMs = new Float32Array(RING).fill(NaN);
  readonly cpuMs = new Float32Array(RING).fill(NaN);
  readonly gpuMs = new Float32Array(RING).fill(NaN);
  private scratch = new Float32Array(RING);
  idx = 0;
  // summary
  readonly sum = new Float64Array(12); // fps, low1, avg, p95, p99, cpu, gpu, max, gpuP95, gpuP99, count, cpuP99
  readonly passMs = new Float32Array(NPASS); // mean time per pass over the latest batch (ms)
  private passAcc = new Float64Array(NPASS);
  // GPU timestamps: one query set holds BATCH frame slots of NPASS begin/end pairs
  querySet: GPUQuerySet | null = null;
  private resolveBuf: GPUBuffer | null = null;
  private readBufs: GPUBuffer[] = [];
  private readState = new Uint8Array(3); // 0 free, 1 copy queued, 2 mapping
  private readFrame = new Int32Array(3); // frame index of the batch's first slot
  private readMask = new Uint32Array(3); // slots of the batch that carry timestamps
  private ringSlot = 0;
  private slot = 0; // this frame's slot in the current batch
  private mask = 0; // slots of the current batch that carry timestamps
  gpuEnabled = false;
  private readonly tsw: GPUComputePassTimestampWrites[] = [];
  private readonly tswBegin: GPUComputePassTimestampWrites[] = [];
  private readonly tswEnd: GPUComputePassTimestampWrites[] = [];
  readonly text = new Uint32Array(64 * 10);
  readonly v = new Float64Array(1); // value register for num()
  lastGpu = 0; // mean GPU frame time of the latest batch (ms)

  constructor(device: GPUDevice, timestamps: boolean) {
    if (!timestamps) return;
    const querySet = device.createQuerySet({ type: 'timestamp', count: QUERIES });
    this.querySet = querySet;
    this.resolveBuf = device.createBuffer({ size: QUERIES * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    for (let i = 0; i < 3; i++) this.readBufs.push(device.createBuffer({ size: QUERIES * 8, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }));
    for (let f = 0; f < BATCH; f++) for (let p = 0; p < NPASS; p++) {
      const b = (f * NPASS + p) * 2;
      this.tsw.push({ querySet, beginningOfPassWriteIndex: b, endOfPassWriteIndex: b + 1 });
      this.tswBegin.push({ querySet, beginningOfPassWriteIndex: b });
      this.tswEnd.push({ querySet, endOfPassWriteIndex: b + 1 });
    }
  }

  get hasGpu() { return this.querySet !== null; }

  // Timestamp writes for pass p in this frame (undefined while timing is off). tsBegin and tsEnd
  // serve a pass that spans several encoder passes (the shadow cascades).
  ts(p: number) { return this.gpuEnabled && this.querySet ? this.tsw[this.slot * NPASS + p] : undefined; }
  tsBegin(p: number) { return this.gpuEnabled && this.querySet ? this.tswBegin[this.slot * NPASS + p] : undefined; }
  tsEnd(p: number) { return this.gpuEnabled && this.querySet ? this.tswEnd[this.slot * NPASS + p] : undefined; }

  // Called after all passes are encoded (before finish).
  // At the end of each batch, copies its timestamps into a free readback buffer (the batch is
  // dropped if all three are still being read).
  resolve(enc: GPUCommandEncoder, frame: number) {
    if (!this.querySet || !this.resolveBuf) return;
    if (this.gpuEnabled) this.mask |= 1 << this.slot;
    if (this.slot === BATCH - 1) {
      for (let i = 0; i < 3 && this.mask !== 0; i++) {
        const s = (this.ringSlot + i) % 3;
        if (this.readState[s] !== 0) continue;
        enc.resolveQuerySet(this.querySet, 0, QUERIES, this.resolveBuf, 0);
        enc.copyBufferToBuffer(this.resolveBuf, 0, this.readBufs[s], 0, QUERIES * 8);
        this.readState[s] = 1;
        this.readFrame[s] = frame - (BATCH - 1);
        this.readMask[s] = this.mask;
        this.ringSlot = (s + 1) % 3;
        break;
      }
      this.mask = 0;
    }
    this.slot = (this.slot + 1) % BATCH;
  }

  // Called after submit. Starts mapping copies and polls finished maps (no promise callbacks).
  poll() {
    if (!this.querySet) return;
    for (let s = 0; s < 3; s++) {
      const b = this.readBufs[s];
      if (this.readState[s] === 1) {
        b.mapAsync(GPUMapMode.READ); // the Promise is not awaited: mapState is polled instead
        this.readState[s] = 2;
      } else if (this.readState[s] === 2 && b.mapState === 'mapped') {
        // every mapping returns a new ArrayBuffer, so the view cannot be kept
        this.readBatch(new Uint32Array(b.getMappedRange()), this.readFrame[s], this.readMask[s]);
        b.unmap();
        this.readState[s] = 0;
      }
    }
  }

  // Frame and pass times from one batch. Timestamps are 64-bit nanoseconds, read as pairs of
  // 32-bit words because BigInt values would allocate.
  private readBatch(t: Uint32Array, f0: number, mask: number) {
    const acc = this.passAcc;
    acc.fill(0);
    let frames = 0, total = 0;
    for (let k = 0; k < BATCH; k++) {
      if ((mask & (1 << k)) === 0) continue;
      let first = Infinity, last = 0;
      for (let p = 0; p < NPASS; p++) {
        const w = (k * NPASS + p) * 4; // word index of the pass's begin timestamp
        const a = t[w + 1] * 4294967296 + t[w], e = t[w + 3] * 4294967296 + t[w + 2];
        if (a > 0 && e >= a) {
          acc[p] += e - a;
          if (a < first) first = a;
          if (e > last) last = e;
        }
      }
      if (last > first) {
        const g = (last - first) / 1e6;
        this.gpuMs[(f0 + k) & (RING - 1)] = g;
        total += g;
        frames++;
      }
    }
    if (frames === 0) return;
    this.lastGpu = total / frames;
    for (let p = 0; p < NPASS; p++) this.passMs[p] = acc[p] / frames / 1e6;
  }

  // Records the frame described by clock[FRAME_MS] and clock[CPU_MS].
  record() {
    const i = this.idx & (RING - 1);
    this.frameMs[i] = clock[FRAME_MS];
    this.cpuMs[i] = clock[CPU_MS];
    this.idx++;
  }

  // Percentile statistics over the ring (in-place numeric sort, NaN sorts last).
  summarize(window: number) {
    const n = Math.min(window, this.idx, RING);
    const s = this.scratch;
    // copy the most recent n frames
    s.fill(NaN);
    for (let k = 0; k < n; k++) s[k] = this.frameMs[(this.idx - 1 - k) & (RING - 1)];
    s.sort();
    let tot = 0;
    for (let k = 0; k < n; k++) tot += s[k];
    const avg = tot / Math.max(n, 1);
    const worst = Math.max(1, Math.floor(n * 0.01));
    let wt = 0;
    for (let k = n - worst; k < n; k++) wt += s[k];
    const S = this.sum;
    S[0] = 1000 / avg;
    S[1] = 1000 / (wt / worst);
    S[2] = avg;
    S[3] = s[Math.min(n - 1, Math.floor(n * 0.95))];
    S[4] = s[Math.min(n - 1, Math.floor(n * 0.99))];
    S[7] = s[n - 1];
    S[10] = n;
    // cpu (only timed while the overlay or the benchmark runs, so it may have gaps)
    s.fill(NaN);
    let cn = 0, ct = 0;
    for (let k = 0; k < n; k++) {
      const v = this.cpuMs[(this.idx - 1 - k) & (RING - 1)];
      if (v === v) { s[cn++] = v; ct += v; }
    }
    s.sort();
    S[5] = cn ? ct / cn : NaN;
    S[11] = cn ? s[Math.min(cn - 1, Math.floor(cn * 0.99))] : NaN;
    // gpu (may have gaps)
    s.fill(NaN);
    let gn = 0, gt = 0;
    for (let k = 0; k < n; k++) {
      const v = this.gpuMs[(this.idx - 1 - k) & (RING - 1)];
      if (v === v) { s[gn++] = v; gt += v; }
    }
    s.sort();
    S[6] = gn ? gt / gn : NaN;
    S[8] = gn ? s[Math.min(gn - 1, Math.floor(gn * 0.95))] : NaN;
    S[9] = gn ? s[Math.min(gn - 1, Math.floor(gn * 0.99))] : NaN;
  }

  clearGpu() { this.gpuMs.fill(NaN); }

  // --- text overlay helpers (write char codes; no string allocation) ---
  clearText() { this.text.fill(32); }
  put(row: number, col: number, s: string): number {
    const base = row * 64;
    for (let i = 0; i < s.length && col < 64; i++, col++) this.text[base + col] = s.charCodeAt(i);
    return col;
  }
  // Writes src[i] with dec decimals. The value comes in an array so that no fractional number is
  // passed through the call (see clock.ts); computed values go through this.v.
  num(row: number, col: number, src: Float32Array | Float64Array, i: number, dec: number, minInt = 1): number {
    let v = src[i];
    if (v !== v) return this.put(row, col, '--');
    const base = row * 64;
    if (v < 0) { if (col < 64) this.text[base + col++] = 45; v = -v; }
    const m = Math.pow(10, dec);
    let iv = Math.round(v * m);
    const ip = Math.floor(iv / m);
    let fp = iv - ip * m;
    // integer digits
    let digits = 1, t = ip;
    while (t >= 10) { t = Math.floor(t / 10); digits++; }
    if (digits < minInt) digits = minInt;
    for (let d = digits - 1; d >= 0; d--) {
      const p = Math.floor(ip / Math.pow(10, d)) % 10;
      if (col < 64) this.text[base + col++] = 48 + p;
    }
    if (dec > 0) {
      if (col < 64) this.text[base + col++] = 46;
      for (let d = dec - 1; d >= 0; d--) {
        const p = Math.floor(fp / Math.pow(10, d)) % 10;
        if (col < 64) this.text[base + col++] = 48 + p;
      }
    }
    iv = 0; fp = 0;
    return col;
  }
  // Whole numbers pass through calls as small integers, without boxing.
  int(row: number, col: number, v: number): number {
    this.v[0] = v;
    return this.num(row, col, this.v, 0, 0);
  }
  hex(row: number, col: number, v: number): number {
    const base = row * 64;
    for (let d = 5; d >= 0; d--) {
      const h = (v >>> (d * 4)) & 15;
      if (col < 64) this.text[base + col++] = h < 10 ? 48 + h : 55 + h;
    }
    return col;
  }
}
