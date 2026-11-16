// Final: bloom combine, exposure, filmic tone mapping, grading, vignette, grain, text overlay.
//#include "common.wgsl"
//#include "noise.wgsl"

@group(1) @binding(0) var hdrTex: texture_2d<f32>;
@group(1) @binding(1) var bloomTex: texture_2d<f32>;
@group(1) @binding(2) var<storage, read> font: array<u32>;   // 2 words per glyph (ASCII 32..95)
@group(1) @binding(3) var<storage, read> textBuf: array<u32>; // TEXT_COLS x TEXT_ROWS char codes
@group(1) @binding(4) var<uniform> FU: vec4f;                 // x bloom strength, y overlay on, z sun screen vis, w sharpen
@group(1) @binding(5) var<storage, read> expo: array<f32>;    // [0] = adapted exposure

const TEXT_COLS: u32 = 64u;
const TEXT_ROWS: u32 = 10u;

struct VO { @builtin(position) pos: vec4f };

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VO {
  var o: VO;
  let x = f32((i << 1u) & 2u);
  let y = f32(i & 2u);
  o.pos = vec4f(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
  return o;
}

fn acesFit(x: vec3f) -> vec3f {
  let mi = mat3x3f(vec3f(0.59719, 0.07600, 0.02840), vec3f(0.35458, 0.90834, 0.13383), vec3f(0.04823, 0.01566, 0.83777));
  let mo = mat3x3f(vec3f(1.60475, -0.10208, -0.00327), vec3f(-0.53108, 1.10813, -0.07276), vec3f(-0.07367, -0.00605, 1.07602));
  let v = mi * x;
  let a = v * (v + 0.0245786) - 0.000090537;
  let b = v * (0.983729 * v + 0.4329510) + 0.238081;
  return sat3(mo * (a / b));
}

fn toSrgb(c: vec3f) -> vec3f {
  let lo = c * 12.92;
  let hi = 1.055 * pow(c, vec3f(1.0 / 2.4)) - 0.055;
  return select(hi, lo, c <= vec3f(0.0031308));
}

fn glyph(ch: u32, gx: u32, gy: u32) -> bool {
  if (ch < 32u || ch > 95u || gx >= 5u || gy >= 7u) { return false; }
  let gi = (ch - 32u) * 2u;
  var col: u32;
  if (gx < 4u) { col = (font[gi] >> (gx * 8u)) & 0xffu; } else { col = font[gi + 1u] & 0xffu; }
  return ((col >> gy) & 1u) == 1u;
}

// Returns (text coverage, shadow coverage)
fn overlay(px: vec2f) -> vec2f {
  let scale = max(1.0, floor(F.outSize.y / 720.0 + 0.5));
  let origin = vec2f(14.0, 12.0) * scale;
  let cell = vec2f(6.0, 9.0) * scale;
  let q = (px - origin) / cell;
  if (q.x < 0.0 || q.y < 0.0) { return vec2f(0.0); }
  let c = vec2u(q);
  if (c.x >= TEXT_COLS || c.y >= TEXT_ROWS) { return vec2f(0.0); }
  let ch = textBuf[c.y * TEXT_COLS + c.x];
  let g = vec2u(fract(q) * cell / scale);
  let on = glyph(ch, g.x, g.y);
  let gs = vec2u((fract(q) * cell - scale) / scale);
  let sh = glyph(ch, gs.x, gs.y) && all(fract(q) * cell >= vec2f(scale));
  return vec2f(select(0.0, 1.0, on), select(0.0, 1.0, sh));
}

@fragment
fn fs(in: VO) -> @location(0) vec4f {
  let px = in.pos.xy;
  let ip = vec2i(px);
  let uv = px * F.outSize.zw;
  var c = textureLoad(hdrTex, ip, 0).rgb;
  // light sharpening to counter TAA softness
  if (FU.w > 0.0) {
    let n = textureLoad(hdrTex, ip + vec2i(0, -1), 0).rgb + textureLoad(hdrTex, ip + vec2i(0, 1), 0).rgb
          + textureLoad(hdrTex, ip + vec2i(-1, 0), 0).rgb + textureLoad(hdrTex, ip + vec2i(1, 0), 0).rgb;
    let lc = luma(c);
    let sharp = (lc * 4.0 - luma(n)) * FU.w;
    c *= max(1.0 + sharp / max(lc, 1e-4) * 0.25, 0.5);
  }
  let b = textureSampleLevel(bloomTex, sLinClamp, uv, 0.0).rgb;
  c = mix(c, b, FU.x);
  c *= expo[0];
  // pre-tonemap grading: gentle warm/cool split and saturation
  let l = luma(c);
  c = mix(vec3f(l), c, 1.12);
  c = max(c, vec3f(0.0));
  var t = acesFit(c);
  // vignette
  let d = (uv - 0.5) * vec2f(F.outSize.x * F.outSize.w, 1.0);
  t *= 1.0 - 0.28 * smoothstep(0.35, 1.05, length(d));
  // shadows toward teal-violet, highlights toward warm
  let tl = luma(t);
  t += (vec3f(-0.010, 0.004, 0.020) * (1.0 - tl) + vec3f(0.012, 0.004, -0.010) * tl) * 0.8;
  var s = toSrgb(sat3(t));
  // film grain / dither
  let gn = ign(px, F.misc.x) - 0.5;
  s += gn * (1.5 / 255.0);
  if (FU.y > 0.5) {
    let o = overlay(px);
    s = mix(s, s * 0.35, o.y * 0.85);
    s = mix(s, vec3f(0.92, 0.97, 1.0), o.x);
  }
  return vec4f(s, 1.0);
}
