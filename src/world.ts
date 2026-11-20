// World design constants shared by TypeScript and WGSL (injected as a WGSL prelude).

export const PLANET_NAME = 'ITHRA VAEL';
export const SEED = 0x7a11e5;
export const WORLD_L = 16384; // world tiles seamlessly every WORLD_L metres
export const HMAP_N = 4096; // heightmap resolution (4 m texels)
export const PLANET_R = 150000; // planet radius (m) used for curvature + atmosphere
export const WATER_Y = 0;
export const CLOUD_BOT = 2300;
export const CLOUD_TOP = 4300;
export const CLOUD_COVER = -0.02;

const D2R = Math.PI / 180;
export const SUN_AZ = -30 * D2R; // azimuth from north (-Z) toward east (+X)
export const SUN_EL = 7.5 * D2R;
export const SUN_DIR = [Math.sin(SUN_AZ) * Math.cos(SUN_EL), Math.sin(SUN_EL), -Math.cos(SUN_AZ) * Math.cos(SUN_EL)];
// Direction to the ringed gas giant (sky.wgsl): azimuth 30 deg, 10 deg up, 9.5 deg in radius.
export const GIANT_DIR = [0.492, 0.174, -0.853];

// Times of day (T cycles them; ?time=day|sunset|night picks one at load).
// key: the light that shades the world, as azimuth / elevation (deg) and illuminance at the top
//   of the atmosphere. By day and at sunset it is the sun; at night it is the gas giant's
//   reflected sunlight, from the giant's direction.
// sun: the real sun, which lights the gas giant and the moon and draws its disk; body: the
//   brightness of the gas giant and the moon (lower at night, so they read as lit bodies and not
//   as white disks once the view has adapted to the dark).
// night: strength of the star field; glow: extra emission of the luminous flora.
export interface TimeOfDay {
  name: string; keyAz: number; keyEl: number; key: [number, number, number];
  sunAz: number; sunEl: number; body: number; night: number; glow: number;
}
export const TIMES: TimeOfDay[] = [
  { name: 'DAY', keyAz: -30, keyEl: 7.5, key: [14, 13.44, 12.6], sunAz: -30, sunEl: 7.5, body: 14, night: 0, glow: 0 },
  { name: 'SUNSET', keyAz: -30, keyEl: 3.0, key: [14, 10.6, 7.4], sunAz: -30, sunEl: 3.0, body: 14, night: 0, glow: 0.4 },
  { name: 'NIGHT', keyAz: 29.97, keyEl: 10.02, key: [0.3, 0.27, 0.25], sunAz: -110, sunEl: -20, body: 0.5, night: 1, glow: 0.8 },
];
export function dirOf(azDeg: number, elDeg: number, out: Float64Array | number[]) {
  const a = azDeg * D2R, e = elDeg * D2R;
  out[0] = Math.sin(a) * Math.cos(e); out[1] = Math.sin(e); out[2] = -Math.cos(a) * Math.cos(e);
}

const az = (deg: number, dist: number, ox: number, oz: number): [number, number] => [
  ox + Math.sin(deg * D2R) * dist,
  oz - Math.cos(deg * D2R) * dist,
];

export const VISTA_P: [number, number] = [6000, 9300];
export const VISTA_AZ = 10;
export const VISTA_DIR: [number, number] = [Math.sin(VISTA_AZ * D2R), -Math.cos(VISTA_AZ * D2R)];
// Camera spot of the Arrival Vista and the level shelf cut into the cliff face around it.
export const VISTA_CAM: [number, number] = [VISTA_P[0] + VISTA_DIR[0] * 30, VISTA_P[1] + VISTA_DIR[1] * 30];
export const VISTA_LEDGE = 175;
export const LAKE_C = az(30, 1800, ...VISTA_P);
export const HIGHLANDS_C = az(5, 4800, ...VISTA_P);
const VALLEY_C = az(78, 3600, ...VISTA_P);
const vAxis: [number, number] = [Math.cos(SUN_AZ), Math.sin(SUN_AZ)]; // perpendicular to sun azimuth (NE)
export const VALLEY_A: [number, number] = [VALLEY_C[0] - vAxis[0] * 1600, VALLEY_C[1] - vAxis[1] * 1600];
export const VALLEY_B: [number, number] = [VALLEY_C[0] + vAxis[0] * 1600, VALLEY_C[1] + vAxis[1] * 1600];
const RANGE_C = az(20, 8400, ...VISTA_P);
const rAxis: [number, number] = [Math.cos(VISTA_AZ * D2R), Math.sin(VISTA_AZ * D2R)];
export const RANGE_A: [number, number] = [RANGE_C[0] - rAxis[0] * 3800, RANGE_C[1] - rAxis[1] * 3800];
export const RANGE_B: [number, number] = [RANGE_C[0] + rAxis[0] * 3800, RANGE_C[1] + rAxis[1] * 3800];
export const VALLEY_CENTER = VALLEY_C;
// Cloud-layer point that shadows the vista (kept clear so the first frame is sunlit).
const tSun = (0.5 * (CLOUD_BOT + CLOUD_TOP) - 420) / SUN_DIR[1];
export const VISTA_SUNCLEAR: [number, number] = [VISTA_P[0] + SUN_DIR[0] * tSun, VISTA_P[1] + SUN_DIR[2] * tSun];
export const FOG_MAX = 14000; // ground-fog march distance (m)

// Fauna. The herd walks a looped trail on the valley floor, defined in valley coordinates
// (u = metres along VALLEY_A -> VALLEY_B, v = metres to the right of the meandering centreline,
// see shaders/valley.wgsl): a stadium of half length HERD_HL and turn radius HERD_R centred at
// (HERD_U, HERD_V), clear of the stream that meanders within about 55 m of the centreline.
export const FAUNA_NC = 10; // herd animals (the last three are young)
export const FAUNA_BPF = 14; // birds per flock
export const FAUNA_NB = FAUNA_BPF * 3;
export const HERD = [640, 82, 40, 13] as const; // HERD_U, HERD_V, HERD_HL, HERD_R
const vLen = Math.hypot(VALLEY_B[0] - VALLEY_A[0], VALLEY_B[1] - VALLEY_A[1]);
const vAx: [number, number] = [(VALLEY_B[0] - VALLEY_A[0]) / vLen, (VALLEY_B[1] - VALLEY_A[1]) / vLen];
const valleyPt = (u: number, v: number): [number, number] => [
  VALLEY_A[0] + vAx[0] * u - vAx[1] * v, VALLEY_A[1] + vAx[1] * u + vAx[0] * v];
// Trail mask area (the warp moves the trail up to ~180 m from its unwarped position).
export const TRAIL_C = valleyPt(HERD[0], HERD[1]);
// Valley coordinates -> world near the herd's clearing: a linear fit of valleyWorld() (valley.wgsl)
// through (u, v) = (560, 82) and (700, 82), good to a metre or two around the clearing.
const herdPt = (u: number, v: number): [number, number] => [
  8670.6 + (u - 560) * 0.8979 + (v - 82) * 0.4171, 9177.7 - (u - 560) * 0.4171 + (v - 82) * 0.8979];
// Herd rhythm (fauna.wgsl): walking bouts of HERD_WALK metres at HERD_CRUISE m/s with smooth starts
// and stops (HERD_RAMP of each bout), then HERD_GRAZE seconds of grazing. The leader is HERD_S0
// metres along the trail at time 0.
export const HERD_WALK = 34, HERD_RAMP = 0.22, HERD_CRUISE = 1.4, HERD_GRAZE = 20, HERD_S0 = 236;

// Valley coordinates (u, v) and unit direction at arc length s along the trail loop (valley.wgsl herdPath).
function herdPath(s: number): [number, number, number, number] {
  const [hu, hv, hl, hr] = HERD;
  const len = 4 * hl + 2 * Math.PI * hr;
  let x = s - len * Math.floor(s / len);
  if (x < 2 * hl) return [hu - hl + x, hv - hr, 1, 0];
  x -= 2 * hl;
  if (x < Math.PI * hr) { const a = x / hr; return [hu + hl + hr * Math.sin(a), hv - hr * Math.cos(a), Math.cos(a), Math.sin(a)]; }
  x -= Math.PI * hr;
  if (x < 2 * hl) return [hu + hl - x, hv + hr, -1, 0];
  const a = (x - 2 * hl) / hr;
  return [hu - hl - hr * Math.sin(a), hv + hr * Math.cos(a), -Math.cos(a), -Math.sin(a)];
}

// Valley destination framing at time t: a camera standing in the clearing inside the herd's loop,
// a few metres past the spot where the herd's walking bout in progress ends (or where it grazes
// now, unless that stop is nearly over), looking back along the trail. The herd walks in, stops
// and grazes in front of the camera with the mushroom forest that rims the clearing behind it.
// Writes camera x, z and look-at x, z into out.
export function valleyFraming(t: number, out: Float64Array) {
  const period = HERD_WALK / (HERD_CRUISE * (1 - HERD_RAMP)) + HERD_GRAZE;
  const k = Math.floor(t / period) + (t % period > period - 8 ? 1 : 0);
  const lead = HERD_S0 + (k + 1) * HERD_WALK; // leader's position at that stop (fauna.wgsl herdTravel)
  const c = herdPath(lead - 15); // middle of the formation (adults walk 5 m apart)
  const a = herdPath(lead + 7);
  const cam = herdPt(a[0] - a[3] * 12, a[1] + a[2] * 12); // inward normal of the anticlockwise loop
  const look = herdPt(c[0], c[1]);
  out[0] = cam[0]; out[1] = cam[1]; out[2] = look[0]; out[3] = look[1];
}
export const TRAIL_HS = 320; // half size (m)
export const TRAIL_N = 320; // texels across (2 m each)
// Flocks: kettles of gliders circling in view of the three destination cameras (main.ts): centre
// x, z, altitude (m), kettle radius (m). FLOCK_SCALE sizes the birds (2.6 m wingspan at 1).
const vistaF = az(16, 190, ...VISTA_CAM); // ~50 m above the vista camera, just left of centre
// The valley kettle follows the herd along its trail (fauna.wgsl), ~36 m above the clearing's
// floor and clear of the mushroom caps; its x, z here is only the middle of the trail.
const valleyF = herdPt(HERD[0], HERD[1]);
// Monumental Highlands destination camera (main.ts): above a mesa prow on the west rim, 600 m up.
export const HIGH_CAM: [number, number] = [HIGHLANDS_C[0] - 1200, HIGHLANDS_C[1] + 250];
const highCam = HIGH_CAM;
const highF = az(72, 230, ...highCam); // ~25 m above the highland camera
export const FLOCKS = [
  [vistaF[0], vistaF[1], 232, 45],
  [valleyF[0], valleyF[1], 70, 38],
  [highF[0], highF[1], 625, 40],
] as const;
export const FLOCK_SCALE = [2.6, 1.4, 2.2] as const;

// More life (fauna.wgsl). Pose slots follow the herd and the flocks, in this order:
export const FAUNA_NT = 3; // titans: long-necked giants grazing the Highlands mesa
export const FAUNA_NH = 4; // travellers in exosuits: two on the Vista shelf, two following the herd
export const FAUNA_NK = 2; // colossi: giant humanoid walkers on the Highlands mesa
export const FAUNA_NJ = 14; // sky jellies drifting over the herd's clearing
export const FAUNA_NR = 2; // great sky rays soaring over the plain below the Vista
export const FAUNA_NS = 9; // starships: three per destination (SHIP_LANES)
export const FAUNA_OT = FAUNA_NC + FAUNA_NB;
export const FAUNA_OH = FAUNA_OT + FAUNA_NT;
export const FAUNA_OK = FAUNA_OH + FAUNA_NH;
export const FAUNA_OJ = FAUNA_OK + FAUNA_NK;
export const FAUNA_OR = FAUNA_OJ + FAUNA_NJ;
export const FAUNA_OS = FAUNA_OR + FAUNA_NR;
export const FAUNA_N = FAUNA_OS + FAUNA_NS;
// The herd trail's world positions, tabulated once at startup (fauna.wgsl trailTable): samples
// along the loop (~0.5 m apart), each with points across it every 2 m from 6 m right to 6 m left.
export const HERD_TAB_NS = 480, HERD_TAB_NL = 7;
// Walking loops are stadiums (like the herd trail): centre x, z, half length, turn radius, with
// the straight lanes along `across` (perpendicular to the camera's line of sight to the centre).
const D = (deg: number): [number, number] => [Math.cos(deg * D2R), Math.sin(deg * D2R)];
export const TITAN_LOOP = [...az(84, 470, ...highCam), 80, 40] as const; // mesa top at ~466 m
export const TITAN_AX = D(84);
export const COLOSSUS_LOOP = [...az(88, 590, ...highCam), 60, 25] as const;
export const COLOSSUS_AX = D(88);
// Travellers on the Vista shelf (175 m): one standing near the edge, one pacing along it.
export const TRAVELLER_SPOTS = [6001.3, 9261.5, 6005.2, 9261.0, 6011.7, 9262.2] as const;
// Sky jellies drift over the herd's clearing; sky rays circle above the plain below the Vista.
export const JELLY_C = herdPt(HERD[0], HERD[1]);
export const RAY_C = [...az(14, 290, ...VISTA_CAM), 280, 125] as const; // x, z, altitude, radius
// Starship lanes: a destination camera's x, y, z and view azimuth (deg). Every pass crosses the
// view of its lane's camera: it descends from high up, flies over, then climbs out of sight.
export const SHIP_LANES = [
  [VISTA_CAM[0], 179, VISTA_CAM[1], 18],
  [JELLY_C[0], 34, JELLY_C[1], Math.atan2(0.8979, 0.4171) / D2R], // along the valley (herdPt's u)
  [highCam[0], 600, highCam[1], 75],
] as const;
// Airborne particles: luminous spores over the valley floor, sunlit pollen motes elsewhere.
export const FX_SPORES = 16384;
export const FX_MOTES = 8192;
export const FX_TRAIL_SEG = 40; // segments per starship engine trail (fx.wgsl vsTrail)

const f = (x: number) => (Number.isInteger(x) ? x.toFixed(1) : String(x));
const v2 = (a: readonly number[]) => `vec2f(${f(a[0])}, ${f(a[1])})`;
const v4 = (a: readonly number[]) => `vec4f(${a.map(f).join(', ')})`;

export function wgslPrelude(): string {
  return [
    `const WORLD_L: f32 = ${f(WORLD_L)};`,
    `const HMAP_N: i32 = ${HMAP_N};`,
    `const PLANET_R: f32 = ${f(PLANET_R)};`,
    `const WATER_Y: f32 = ${f(WATER_Y)};`,
    `const SEED_U: u32 = ${SEED}u;`,
    `const CLOUD_BOT: f32 = ${f(CLOUD_BOT)};`,
    `const CLOUD_TOP: f32 = ${f(CLOUD_TOP)};`,
    `const CLOUD_COVER: f32 = ${f(CLOUD_COVER)};`,
    `const VISTA_P: vec2f = ${v2(VISTA_P)};`,
    `const VISTA_DIR: vec2f = ${v2(VISTA_DIR)};`,
    `const VISTA_CAM: vec2f = ${v2(VISTA_CAM)};`,
    `const VISTA_LEDGE: f32 = ${f(VISTA_LEDGE)};`,
    `const LAKE_C: vec2f = ${v2(LAKE_C)};`,
    `const HIGHLANDS_C: vec2f = ${v2(HIGHLANDS_C)};`,
    `const VALLEY_A: vec2f = ${v2(VALLEY_A)};`,
    `const VALLEY_B: vec2f = ${v2(VALLEY_B)};`,
    `const RANGE_A: vec2f = ${v2(RANGE_A)};`,
    `const RANGE_B: vec2f = ${v2(RANGE_B)};`,
    `const VISTA_SUNCLEAR: vec2f = ${v2(VISTA_SUNCLEAR)};`,
    `const GIANT_DIR: vec3f = vec3f(${GIANT_DIR.map(f).join(', ')});`,
    `const FOG_MAX: f32 = ${f(FOG_MAX)};`,
    `const FAUNA_NC: u32 = ${FAUNA_NC}u;`,
    `const FAUNA_BPF: u32 = ${FAUNA_BPF}u;`,
    `const FAUNA_NB: u32 = ${FAUNA_NB}u;`,
    `const HERD: vec4f = ${v4(HERD)};`,
    `const HERD_WALK: f32 = ${f(HERD_WALK)};`,
    `const HERD_RAMP: f32 = ${f(HERD_RAMP)};`,
    `const HERD_CRUISE: f32 = ${f(HERD_CRUISE)};`,
    `const HERD_GRAZE: f32 = ${f(HERD_GRAZE)};`,
    `const HERD_S0: f32 = ${f(HERD_S0)};`,
    `const TRAIL_C: vec2f = ${v2(TRAIL_C)};`,
    `const TRAIL_HS: f32 = ${f(TRAIL_HS)};`,
    `const TRAIL_N: u32 = ${TRAIL_N}u;`,
    `const FLOCK0: vec4f = ${v4(FLOCKS[0])};`,
    `const FLOCK1: vec4f = ${v4(FLOCKS[1])};`,
    `const FLOCK2: vec4f = ${v4(FLOCKS[2])};`,
    `const FX_SPORES: u32 = ${FX_SPORES}u;`,
    `const FX_MOTES: u32 = ${FX_MOTES}u;`,
    `const FX_TRAIL_SEG: u32 = ${FX_TRAIL_SEG}u;`,
    `const FLOCK_SCALE: vec3f = vec3f(${FLOCK_SCALE.map(f).join(', ')});`,
    `const FAUNA_NT: u32 = ${FAUNA_NT}u;`,
    `const FAUNA_NH: u32 = ${FAUNA_NH}u;`,
    `const FAUNA_NK: u32 = ${FAUNA_NK}u;`,
    `const FAUNA_NJ: u32 = ${FAUNA_NJ}u;`,
    `const FAUNA_NR: u32 = ${FAUNA_NR}u;`,
    `const FAUNA_NS: u32 = ${FAUNA_NS}u;`,
    `const FAUNA_OT: u32 = ${FAUNA_OT}u;`,
    `const FAUNA_OH: u32 = ${FAUNA_OH}u;`,
    `const FAUNA_OK: u32 = ${FAUNA_OK}u;`,
    `const FAUNA_OJ: u32 = ${FAUNA_OJ}u;`,
    `const FAUNA_OR: u32 = ${FAUNA_OR}u;`,
    `const FAUNA_OS: u32 = ${FAUNA_OS}u;`,
    `const FAUNA_N: u32 = ${FAUNA_N}u;`,
    `const HERD_TAB_NS: u32 = ${HERD_TAB_NS}u;`,
    `const HERD_TAB_NL: u32 = ${HERD_TAB_NL}u;`,
    `const TITAN_LOOP: vec4f = ${v4(TITAN_LOOP)};`,
    `const TITAN_AX: vec2f = ${v2(TITAN_AX)};`,
    `const COLOSSUS_LOOP: vec4f = ${v4(COLOSSUS_LOOP)};`,
    `const COLOSSUS_AX: vec2f = ${v2(COLOSSUS_AX)};`,
    `const TRAVELLER_A: vec4f = ${v4(TRAVELLER_SPOTS.slice(0, 4))};`,
    `const TRAVELLER_B: vec2f = ${v2(TRAVELLER_SPOTS.slice(4, 6))};`,
    `const JELLY_C: vec2f = ${v2(JELLY_C)};`,
    `const RAY_C: vec4f = ${v4(RAY_C)};`,
    ...SHIP_LANES.map((l, i) => `const SHIP_LANE${i}: vec4f = ${v4(l)};`),
    '',
  ].join('\n');
}
