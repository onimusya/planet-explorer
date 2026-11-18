// Generates the ambient soundtrack with the ElevenLabs Music API: one instrumental track per
// destination, written to public/audio/<name>.mp3 (served next to index.html, not bundled).
// Usage: node scripts/music.mjs [vista] [valley] [highlands] [--force] [--check]
//   --check  print the account tier and remaining credits, generate nothing
//   --force  regenerate tracks that already exist
// The API key comes from .env.local, ELEVENLABS_API_KEY or ~/.media-gen/.env (never printed).
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const OUT = join(ROOT, 'public', 'audio');
const API = 'https://api.elevenlabs.io/v1';
const MODEL = 'music_v2_5';
const FORMAT = 'mp3_44100_128';
const LENGTH_MS = 120000;

// Each prompt answers genre, mood, instruments, tempo and production, and bans what would break a
// seamless ambient loop (vocals, drums, build-ups, intros and endings).
const TRACKS = {
  vista:
    'Instrumental only. Expansive cinematic ambient post-rock for arriving on a sunlit alien world: wonder ' +
    'and discovery on a pink grassland under a giant ringed planet and floating islands. 72 BPM, D major. ' +
    'Warm analog synth pads swelling slowly, shimmering delayed electric guitar harmonics, soft felt piano ' +
    'motifs, distant airy choir pads, gentle tape saturation, wide plate reverb, modern indie game soundtrack ' +
    'production. One slowly evolving texture at a steady level from start to end. No drums, no vocals, no ' +
    'build-ups, no fade-in, no fade-out, no ending.',
  valley:
    'Instrumental only. Mysterious, dreamy ambient for a bioluminescent mushroom forest at twilight, glowing ' +
    'spores drifting and gentle alien creatures grazing. 80 BPM, F sharp minor. Glassy bell tones and kalimba ' +
    'plucks with long echoes, soft pulsing sub bass, warm granular synth pads, delicate wind chimes, distant ' +
    'water droplet textures, a faint organic shaker pulse, spacious and calm, magical. One slowly evolving ' +
    'texture at a steady level from start to end. No vocals, no heavy drums, no build-ups, no fade-in, no ' +
    'fade-out, no ending.',
  highlands:
    'Instrumental only. Vast, monumental ambient for towering striped mesas and deep canyons under an enormous ' +
    'sky: scale, solitude and awe. 60 BPM, C minor. Deep sustained drones, slow swelling string-like synth ' +
    'pads, distant low brass-like tones, high-altitude wind texture, sparse soft timpani like far-off thunder, ' +
    'cavernous reverb, cinematic but restrained. One slowly evolving texture at a steady level from start to ' +
    'end. No vocals, no fast percussion, no build-ups, no fade-in, no fade-out, no ending.',
};

// Key lookup order: the project's .env.local (git-ignored), the environment, ~/.media-gen/.env.
// Accepts the usual variable names; the value is only ever sent to the API.
const KEY_NAMES = ['ELEVENLABS_API_KEY', 'ELEVEN_LABS_API_KEY', 'ELEVENLABS_KEY', 'XI_API_KEY', 'VITE_ELEVENLABS_API_KEY'];
function keyFromFile(f) {
  if (!existsSync(f)) return null;
  const text = readFileSync(f, 'utf8');
  for (const name of KEY_NAMES) {
    const m = text.match(new RegExp(`^\\s*(?:export\\s+)?${name}\\s*=\\s*["']?([^"'\\r\\n#]+)["']?`, 'm'));
    if (m && m[1].trim()) return m[1].trim();
  }
  return null;
}
function apiKey() {
  const local = keyFromFile(join(ROOT, '.env.local'));
  if (local) { console.log('API key from .env.local'); return local; }
  for (const name of KEY_NAMES) if (process.env[name]) { console.log('API key from the environment'); return process.env[name].trim(); }
  const home = keyFromFile(join(homedir(), '.media-gen', '.env'));
  if (home) { console.log('API key from ~/.media-gen/.env'); return home; }
  throw new Error('No ElevenLabs key: add ELEVENLABS_API_KEY=... to .env.local');
}

const args = process.argv.slice(2);
const key = apiKey();
if (args.includes('--check')) {
  const r = await fetch(`${API}/user/subscription`, { headers: { 'xi-api-key': key }, signal: AbortSignal.timeout(30000) });
  const s = await r.json();
  console.log(JSON.stringify({ status: r.status, tier: s.tier, used: s.character_count, limit: s.character_limit, resets: s.next_character_count_reset_unix }));
  process.exit(0);
}
const names = args.filter((a) => !a.startsWith('--'));
const todo = names.length ? names : Object.keys(TRACKS);
mkdirSync(OUT, { recursive: true });
for (const name of todo) {
  const prompt = TRACKS[name];
  if (!prompt) throw new Error(`unknown track ${name}`);
  const file = join(OUT, `${name}.mp3`);
  if (existsSync(file) && !args.includes('--force')) { console.log(`${name}: exists, skipped (use --force)`); continue; }
  const t0 = Date.now();
  const r = await fetch(`${API}/music?output_format=${FORMAT}`, {
    method: 'POST',
    headers: { 'xi-api-key': key, 'Content-Type': 'application/json', Accept: 'audio/mpeg' },
    body: JSON.stringify({ prompt, music_length_ms: LENGTH_MS, model_id: MODEL, force_instrumental: true }),
    signal: AbortSignal.timeout(600000),
  });
  if (!r.ok) {
    console.log(`${name}: HTTP ${r.status} ${(await r.text()).slice(0, 400)}`);
    process.exitCode = 1;
    if (r.status === 401 || r.status === 402 || r.status === 403) break; // account problem: the other tracks would fail too
    continue;
  }
  const buf = Buffer.from(await r.arrayBuffer());
  writeFileSync(file, buf);
  const ids = [...r.headers].filter(([k]) => /song|request-id|character/i.test(k)).map(([k, v]) => `${k}=${v}`).join(' ');
  console.log(`${name}: ${(buf.length / 1024).toFixed(0)} KiB in ${((Date.now() - t0) / 1000).toFixed(0)} s ${ids}`);
}
