// Ambient soundtrack: one looping track per destination region (Arrival Vista, which also covers
// the open country between destinations, Bioluminescent Valley, Monumental Highlands), crossfaded
// as the camera crosses region borders. The tracks are ElevenLabs-generated MP3s in public/audio
// (scripts/music.mjs). They are fetched after the first frame, so they never delay it, and each
// is decoded the first time its region is entered (off the main thread), made loopable in small
// slices between frames, and kept for later visits.
// Browsers keep audio off until the first click or key press; missing files just mean silence.
// update() runs every frame and allocates nothing: Web Audio is touched only on region changes.
import { VALLEY_A, VALLEY_B, HIGHLANDS_C, WORLD_L } from './world';

const TRACKS = ['vista', 'valley', 'highlands'];
const SEAM = 6; // s: crossfade that joins each track's end back to its start
const FADE = 1.6; // s: time constant of region crossfades (~95% after 5 s)
const VOLUME = 0.55;
const EDGE = 20; // s: how far into each end to look for a faded intro or outro to trim

const wrap = (d: number) => d - WORLD_L * Math.round(d / WORLD_L);
const smooth = (a: number, b: number, x: number) => {
  const t = Math.min(Math.max((x - a) / (b - a), 0), 1);
  return t * t * (3 - 2 * t);
};
// Region weights, the same masks the shaders use (world.wgsl valleyMask, highlandMask).
const abx = wrap(VALLEY_B[0] - VALLEY_A[0]), abz = wrap(VALLEY_B[1] - VALLEY_A[1]);
function valleyW(x: number, z: number) {
  const px = wrap(x - VALLEY_A[0]), pz = wrap(z - VALLEY_A[1]);
  const t = Math.min(Math.max((px * abx + pz * abz) / (abx * abx + abz * abz), 0), 1);
  const dx = px - abx * t, dz = pz - abz * t;
  return 1 - smooth(500, 1300, Math.sqrt(dx * dx + dz * dz));
}
function highW(x: number, z: number) {
  const dx = wrap(x - HIGHLANDS_C[0]), dz = wrap(z - HIGHLANDS_C[1]);
  return 1 - smooth(2000, 3600, Math.sqrt(dx * dx + dz * dz));
}

interface Voice { src: AudioBufferSourceNode; gain: GainNode }

export class Music {
  private ac: AudioContext | null = null;
  private master: GainNode | null = null;
  private readonly enc: (ArrayBuffer | null)[] = [null, null, null];
  // Decoded tracks made loopable, kept once prepared (about 44 MiB each at 48 kHz), so later
  // visits only start a source; and the loop points of each.
  private readonly ready: (Promise<AudioBuffer> | null)[] = [null, null, null];
  private readonly loops: [number, number][] = [[0, 0], [0, 0], [0, 0]];
  private voice: Voice | null = null;
  private voices = 0; // sources alive, including ones still fading out
  private region = 0; // region the camera is in (with hysteresis)
  private playing = -1; // region whose track plays or is being prepared
  private loop: number[] = []; // loop start, end and track length (s) of the playing track
  on = true;

  // Fetch the tracks in the background and arm the audio unlock on the first user gesture.
  load() {
    TRACKS.forEach((name, i) => {
      fetch(`audio/${name}.mp3`)
        .then((r) => (r.ok ? r.arrayBuffer() : null))
        .then((b) => { this.enc[i] = b; })
        .catch(() => {});
    });
    const unlock = () => {
      removeEventListener('pointerdown', unlock, true);
      removeEventListener('keydown', unlock, true);
      const ac = new AudioContext();
      this.master = ac.createGain();
      this.master.gain.value = this.on ? VOLUME : 0;
      this.master.connect(ac.destination);
      this.ac = ac;
    };
    addEventListener('pointerdown', unlock, true);
    addEventListener('keydown', unlock, true);
    document.addEventListener('visibilitychange', () => {
      if (!this.ac) return;
      if (document.hidden) void this.ac.suspend();
      else if (this.on) void this.ac.resume();
    });
  }

  toggle() {
    this.on = !this.on;
    const ac = this.ac;
    if (!ac || !this.master) return;
    if (this.on) void ac.resume();
    this.master.gain.setTargetAtTime(this.on ? VOLUME : 0, ac.currentTime, 0.25);
    // once faded out, stop the audio clock too (no mixing while muted)
    if (!this.on) setTimeout(() => { if (!this.on) void ac.suspend(); }, 1500);
  }

  // Debug API: region, region whose track plays, context state, which tracks have loaded.
  state() {
    return { region: TRACKS[this.region], playing: TRACKS[this.playing] ?? null, audio: this.ac?.state ?? 'locked', on: this.on, loaded: this.enc.map((b) => (b ? b.byteLength : 0)), loop: this.loop, voices: this.voices };
  }

  // Per frame, with the camera position: pick the region (switching only well past a border, so
  // hovering on it does not flip the music) and start a crossfade when its track should change.
  update(x: number, z: number) {
    const wv = valleyW(x, z), wh = highW(x, z);
    const w0 = 1 - Math.max(wv, wh);
    const cur = this.region === 0 ? w0 : this.region === 1 ? wv : wh;
    const best = wv > w0 && wv >= wh ? 1 : wh > w0 && wh > wv ? 2 : 0;
    const wb = best === 0 ? w0 : best === 1 ? wv : wh;
    if (best !== this.region && wb > cur + 0.2) this.region = best;
    // while muted no new track is decoded; unmuting starts the region's track
    if (this.on && this.region !== this.playing && this.ac && this.enc[this.region]) this.play(this.region);
  }

  private prepare(i: number): Promise<AudioBuffer> {
    let p = this.ready[i];
    if (!p) {
      p = this.ac!.decodeAudioData(this.enc[i]!.slice(0)).then(async (buf) => {
        this.loops[i] = await loopable(buf);
        return buf;
      });
      this.ready[i] = p;
    }
    return p;
  }

  private play(i: number) {
    const ac = this.ac!;
    this.playing = i;
    this.prepare(i).then((buf) => {
      if (this.playing !== i) return; // the camera moved on while this track was prepared
      const [a, b] = this.loops[i];
      this.loop = [a, b, buf.duration];
      const gain = ac.createGain();
      gain.gain.value = 0;
      gain.connect(this.master!);
      const src = ac.createBufferSource();
      src.buffer = buf;
      src.loop = true;
      src.loopStart = a;
      src.loopEnd = b;
      src.connect(gain);
      // enter at a random point of the loop, so the same opening is not heard on every visit
      src.start(0, a + Math.random() * (b - a));
      const now = ac.currentTime;
      gain.gain.setTargetAtTime(1, now, FADE);
      const old = this.voice;
      if (old) {
        old.gain.gain.cancelScheduledValues(now);
        old.gain.gain.setTargetAtTime(0, now, FADE);
        old.src.stop(now + FADE * 6);
        old.src.onended = () => { old.src.disconnect(); old.gain.disconnect(); this.voices--; };
      }
      this.voice = { src, gain };
      this.voices++;
    }).catch(() => {});
  }
}

// Sample work per task: ~1 ms. Preparing a track touches up to about 2.5 million samples, so it
// is cut into slices with a yield to the event loop between them, and frames keep rendering.
const SLICE = 150000;
const yieldTask = () => new Promise<void>((r) => setTimeout(r, 0));

// Make a decoded track loop seamlessly: trim a faded intro and outro, then crossfade the last SEAM
// seconds into the first ones (equal power, in place). Resolves to the loop start and end (s).
async function loopable(buf: AudioBuffer): Promise<[number, number]> {
  const sr = buf.sampleRate, n = buf.length, ch = buf.numberOfChannels;
  const data: Float32Array[] = [];
  for (let c = 0; c < ch; c++) data.push(buf.getChannelData(c));
  const hop = Math.round(sr * 0.1);
  let work = 0;
  const rms = (s: number, len: number) => {
    let e = 0;
    for (let c = 0; c < ch; c++) {
      const d = data[c];
      for (let k = s; k < s + len; k++) e += d[k] * d[k];
    }
    work += len * ch;
    return Math.sqrt(e / (len * ch));
  };
  const pace = async () => { if (work > SLICE) { work = 0; await yieldTask(); } };
  const edge = Math.min(Math.round(EDGE * sr), Math.floor(n / 4));
  // body level over 2 s just inside both edges; the loop runs from where the intro reaches half
  // of it (-6 dB) to where the outro falls below that, so the seam joins two full-level passages
  const body = Math.min(hop * 20, Math.floor(n / 8));
  const r0 = rms(edge, body);
  await pace();
  const ref = Math.max(r0, rms(n - edge - body, body));
  await pace();
  let head = 0, tail = n;
  for (let s = 0; s + hop <= edge; s += hop) {
    if (rms(s, hop) > ref / 2) { head = s; break; }
    await pace();
  }
  for (let s = n - hop; s >= n - edge; s -= hop) {
    if (rms(s, hop) > ref / 2) { tail = s + hop; break; }
    await pace();
  }
  // equal-power crossfade, sin/cos stepped by rotation (one exact pair per slice)
  const m = Math.min(Math.round(SEAM * sr), Math.floor((tail - head) / 4));
  const dt = (Math.PI * 0.5) / m, cd = Math.cos(dt), sd = Math.sin(dt);
  const step = Math.max(1, Math.floor(SLICE / ch));
  for (let j0 = 0; j0 < m; j0 += step) {
    const j1 = Math.min(m, j0 + step);
    for (let c = 0; c < ch; c++) {
      const d = data[c];
      let s = Math.sin(j0 * dt), co = Math.cos(j0 * dt);
      for (let j = j0; j < j1; j++) {
        d[head + j] = d[head + j] * s + d[tail - m + j] * co;
        const s2 = s * cd + co * sd;
        co = co * cd - s * sd;
        s = s2;
      }
    }
    await yieldTask();
  }
  return [head / sr, (tail - m) / sr];
}
