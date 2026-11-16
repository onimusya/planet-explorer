// CPU mirror of the WGSL `Frame` uniform (common.wgsl). Offsets are in float32 units.
export const FO = {
  viewProj: 0,
  invViewProj: 16,
  prevViewProj: 32,
  viewProjNJ: 48,
  camPos: 64,
  camDelta: 68,
  camFwd: 72,
  camRight: 76,
  camUp: 80,
  sunDir: 84,
  sunColor: 88,
  jitter: 92,
  screen: 96,
  full: 100,
  outSize: 104,
  misc: 108,
  cascadeSplits: 112,
  shadowMats: 116,
  shadowInfo: 180,
  extra: 184,
  SIZE: 188,
} as const;

export const FRAME_BYTES = FO.SIZE * 4;
