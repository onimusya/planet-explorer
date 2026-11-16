// Frame-time statistics, GPU timestamp queries and the GPU text overlay buffer.
// All per-frame paths write into preallocated typed arrays.

export const RING = 2048;
export const PASS_NAMES = ['PRE', 'SHD', 'MAIN', 'WATR', 'POST', 'FIN'] as const;
export const NPASS = PASS_NAMES.length;

export class Metrics {
  readonly frameMs = new Float32Array(RING).fill(NaN);
  readonly cpuMs = new Float32Array(RING).fill(NaN);
  readonly gpuMs = new Float32Array(RING).fill(NaN);
  private scratch = new Float32Array(RING);
  idx = 0;
  // summary
  readonly sum = new Float64Array(12); // fps, low1, avg, p95, p99, cpu, gpu, max, gpuP95, gpuP99, count, cpuP99
  readonly passMs = new Float32Array(NPASS);
  // GPU timestamps
  querySet: GPUQuerySet | null = null;
  private resolveBuf: GPUBuffer | null = null;
  private readBufs: GPUBuffer[] = [];
  private readState = new Uint8Array(3); // 0 free, 1 copy queued, 2 mapping
  private readFrame = new Int32Array(3);
  private ringSlot = 0;
  gpuEnabled = false;
  readonly tsw: GPUComputePassTimestampWrites[] = [];
  readonly text = new Uint32Array(64 * 10);
  lastGpu = 0;

  constructor(device: GPUDevice, timestamps: boolean) {
    if (timestamps) {
      this.querySet = device.createQuerySet({ type: 'timestamp', count: NPASS * 2 });
      this.resolveBuf = device.createBuffer({ size: NPASS * 2 * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
      for (let i = 0; i < 3; i++) this.readBufs.push(device.createBuffer({ size: NPASS * 2 * 8, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }));
      for (let i = 0; i < NPASS; i++) this.tsw.push({ querySet: this.querySet, beginningOfPassWriteIndex: i * 2, endOfPassWriteIndex: i * 2 + 1 });
    }
  }

  get hasGpu() { return this.querySet !== null; }

  // Called after all passes are encoded (before finish).
  resolve(enc: GPUCommandEncoder, frame: number) {
    if (!this.gpuEnabled || !this.querySet || !this.resolveBuf) return;
    let slot = -1;
    for (let i = 0; i < 3; i++) {
      const s = (this.ringSlot + i) % 3;
      if (this.readState[s] === 0) { slot = s; break; }
    }
    if (slot < 0) return;
    enc.resolveQuerySet(this.querySet, 0, NPASS * 2, this.resolveBuf, 0);
    enc.copyBufferToBuffer(this.resolveBuf, 0, this.readBufs[slot], 0, NPASS * 2 * 8);
    this.readState[slot] = 1;
    this.readFrame[slot] = frame;
    this.ringSlot = (slot + 1) % 3;
  }

  // Called after submit. Starts mapping copies and polls finished maps (no promise callbacks).
  poll() {
    if (!this.gpuEnabled) return;
    for (let s = 0; s < 3; s++) {
      const b = this.readBufs[s];
      if (this.readState[s] === 1) {
        b.mapAsync(GPUMapMode.READ); // WebGPU API returns a Promise (unavoidable); state is polled
        this.readState[s] = 2;
      } else if (this.readState[s] === 2 && b.mapState === 'mapped') {
        const t = new BigUint64Array(b.getMappedRange());
        let first = Infinity, last = 0;
        for (let p = 0; p < NPASS; p++) {
          const a = Number(t[p * 2]), e = Number(t[p * 2 + 1]);
          if (a > 0 && e >= a) {
            this.passMs[p] = (e - a) / 1e6;
            if (a < first) first = a;
            if (e > last) last = e;
          } else this.passMs[p] = 0;
        }
        b.unmap();
        this.readState[s] = 0;
        if (last > first) {
          this.lastGpu = (last - first) / 1e6;
          this.gpuMs[(this.readFrame[s] & (RING - 1))] = this.lastGpu;
        }
      }
    }
  }

  record(frameMs: number, cpuMs: number) {
    const i = this.idx & (RING - 1);
    this.frameMs[i] = frameMs;
    this.cpuMs[i] = cpuMs;
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
    // cpu
    s.fill(NaN);
    let ct = 0;
    for (let k = 0; k < n; k++) { const v = this.cpuMs[(this.idx - 1 - k) & (RING - 1)]; s[k] = v; ct += v; }
    S[5] = ct / Math.max(n, 1);
    s.sort();
    S[11] = s[Math.min(n - 1, Math.floor(n * 0.99))];
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
  num(row: number, col: number, v: number, dec: number, minInt = 1): number {
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
  hex(row: number, col: number, v: number): number {
    const base = row * 64;
    for (let d = 5; d >= 0; d--) {
      const h = (v >>> (d * 4)) & 15;
      if (col < 64) this.text[base + col++] = h < 10 ? 48 + h : 55 + h;
    }
    return col;
  }
}
