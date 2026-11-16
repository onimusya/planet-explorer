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

const f = (x: number) => (Number.isInteger(x) ? x.toFixed(1) : String(x));
const v2 = (a: readonly number[]) => `vec2f(${f(a[0])}, ${f(a[1])})`;

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
    `const FOG_MAX: f32 = ${f(FOG_MAX)};`,
    '',
  ].join('\n');
}
