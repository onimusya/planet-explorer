// Eye adaptation, all on the GPU (nothing is read back): a centre-weighted luminance histogram of
// a quarter-resolution level of the bloom chain, then the exposure adapted toward the target that
// histogram implies. final.wgsl multiplies the scene by expo[0].
//#include "common.wgsl"

@group(1) @binding(0) var src: texture_2d<f32>;                        // bloom down level 1
@group(1) @binding(1) var<storage, read_write> bins: array<atomic<u32>, 128>;
@group(1) @binding(2) var<storage, read_write> expo: array<f32, 4>;     // exposure, EV, metered log2 luminance, set
@group(1) @binding(3) var<uniform> TA: vec4f;                           // TAA parameters: x = history reset

const NB: u32 = 128u;
const LMIN: f32 = -16.0;   // log2 luminance at the bottom of bin 1 (bin 0: darker, ignored)
const LRANGE: f32 = 22.0;  // bins 1..127 span LMIN .. LMIN + LRANGE
// The metered value is the mean log2 luminance between these weighted percentiles: the darkest
// half (deep shade, night ground) and the brightest 5 % (sun disk, glows, sky) do not move it.
const P_LO: f32 = 0.5;
const P_HI: f32 = 0.95;
// Target exposure (EV) = STRENGTH * (KEY - metered), within EV_MIN..EV_MAX. KEY is the metered
// value of the Arrival Vista view, so it keeps the exposure it was graded at (EV 0). STRENGTH < 1
// adapts only part of the way: dark places stay darker than bright ones, as they should.
const KEY: f32 = -1.05;
const STRENGTH: f32 = 0.5;
const EV_MIN: f32 = -1.5;
const EV_MAX: f32 = 3.5;
// Adaptation rates (1/s): toward a lower exposure (a brighter view) the eye is quicker.
const RATE_DARKER: f32 = 2.2;
const RATE_BRIGHTER: f32 = 1.1;

var<workgroup> wb: array<atomic<u32>, 128>;

@compute @workgroup_size(16, 16)
fn hist(@builtin(global_invocation_id) id: vec3u, @builtin(local_invocation_index) li: u32) {
  if (li < NB) { atomicStore(&wb[li], 0u); }
  workgroupBarrier();
  let ds = textureDimensions(src);
  if (id.x < ds.x && id.y < ds.y) {
    let l = luma(textureLoad(src, id.xy, 0).rgb);
    var b = 0u;
    if (l > exp2(LMIN)) { b = 1u + min(u32((log2(l) - LMIN) / LRANGE * f32(NB - 1u)), NB - 2u); }
    // centre-weighted: the middle of the view counts four times as much as the corners
    let uv = (vec2f(id.xy) + 0.5) / vec2f(ds) - 0.5;
    let r = length(uv * vec2f(f32(ds.x) / f32(ds.y), 1.0));
    atomicAdd(&wb[b], u32(mix(16.0, 4.0, smoothstep(0.15, 0.75, r))));
  }
  workgroupBarrier();
  if (li < NB) {
    let c = atomicLoad(&wb[li]);
    if (c > 0u) { atomicAdd(&bins[li], c); }
  }
}

var<workgroup> sb: array<u32, 128>;

@compute @workgroup_size(128)
fn adapt(@builtin(local_invocation_index) li: u32) {
  sb[li] = atomicLoad(&bins[li]);
  atomicStore(&bins[li], 0u);
  workgroupBarrier();
  if (li != 0u) { return; }
  var total = 0.0;
  for (var i = 1u; i < NB; i++) { total += f32(sb[i]); }
  if (total <= 0.0) { return; }
  let lo = total * P_LO;
  let hi = total * P_HI;
  var cum = 0.0;
  var sum = 0.0;
  var wsum = 0.0;
  for (var i = 1u; i < NB; i++) {
    let c = f32(sb[i]);
    let w = max(min(cum + c, hi) - max(cum, lo), 0.0);
    sum += w * (LMIN + (f32(i - 1u) + 0.5) / f32(NB - 1u) * LRANGE);
    wsum += w;
    cum += c;
  }
  let metered = sum / max(wsum, 1e-6);
  let tgt = clamp(STRENGTH * (KEY - metered), EV_MIN, EV_MAX);
  var ev = expo[1];
  if (expo[3] < 0.5 || TA.x > 0.5) {
    ev = tgt; // first frame, or a jump: no adaptation from the old view
  } else {
    let rate = select(RATE_DARKER, RATE_BRIGHTER, tgt > ev);
    ev += (tgt - ev) * (1.0 - exp(-F.camDelta.w * rate));
  }
  expo[0] = exp2(ev);
  expo[1] = ev;
  expo[2] = metered;
  expo[3] = 1.0;
}
