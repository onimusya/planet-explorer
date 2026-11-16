// Temporal anti-aliasing with upsampling (internal render resolution -> output resolution).
//#include "common.wgsl"

@group(1) @binding(0) var curTex: texture_2d<f32>;
@group(1) @binding(1) var depthTex: texture_depth_2d;
@group(1) @binding(2) var histTex: texture_2d<f32>;
@group(1) @binding(3) var outTex: texture_storage_2d<rgba16float, write>;
@group(1) @binding(4) var<uniform> TA: vec4f; // x = reset (1 = no history), y = sharpness

fn rgb2ycocg(c: vec3f) -> vec3f {
  return vec3f(c.r * 0.25 + c.g * 0.5 + c.b * 0.25, c.r * 0.5 - c.b * 0.5, -c.r * 0.25 + c.g * 0.5 - c.b * 0.25);
}
fn ycocg2rgb(c: vec3f) -> vec3f {
  return vec3f(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z);
}
// Tonemap-weighted space to suppress fireflies
fn tw(c: vec3f) -> vec3f { return c / (1.0 + luma(c)); }
fn itw(c: vec3f) -> vec3f { return c / max(1.0 - luma(c), 1e-4); }

fn loadCur(p: vec2i) -> vec3f {
  let q = clamp(p, vec2i(0), vec2i(F.screen.xy) - 1);
  return tw(textureLoad(curTex, q, 0).rgb);
}

// 9-tap Catmull-Rom history fetch (bilinear-optimised to 5 taps)
fn sampleHistory(uv: vec2f) -> vec3f {
  let size = F.outSize.xy;
  let sp = uv * size;
  let tp = floor(sp - 0.5) + 0.5;
  let f = sp - tp;
  let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
  let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
  let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
  let w3 = f * f * (-0.5 + 0.5 * f);
  let w12 = w1 + w2;
  let o12 = w2 / w12;
  let t0 = (tp - 1.0) / size;
  let t3 = (tp + 2.0) / size;
  let t12 = (tp + o12) / size;
  var r = vec3f(0.0);
  r += textureSampleLevel(histTex, sLinClamp, vec2f(t12.x, t0.y), 0.0).rgb * (w12.x * w0.y);
  r += textureSampleLevel(histTex, sLinClamp, vec2f(t0.x, t12.y), 0.0).rgb * (w0.x * w12.y);
  r += textureSampleLevel(histTex, sLinClamp, vec2f(t12.x, t12.y), 0.0).rgb * (w12.x * w12.y);
  r += textureSampleLevel(histTex, sLinClamp, vec2f(t3.x, t12.y), 0.0).rgb * (w3.x * w12.y);
  r += textureSampleLevel(histTex, sLinClamp, vec2f(t12.x, t3.y), 0.0).rgb * (w12.x * w3.y);
  let wsum = w12.x * w0.y + w0.x * w12.y + w12.x * w12.y + w3.x * w12.y + w12.x * w3.y;
  return max(r / wsum, vec3f(0.0));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let op = vec2i(id.xy);
  if (op.x >= i32(F.outSize.x) || op.y >= i32(F.outSize.y)) { return; }
  let uv = (vec2f(op) + 0.5) * F.outSize.zw;
  // position of this output pixel in the jittered internal image
  let jit = F.jitter.xy * vec2f(0.5, -0.5);
  let sp = (uv + jit) * F.screen.xy - 0.5;
  let ip = vec2i(floor(sp + 0.5));
  // neighbourhood statistics + closest depth
  var m1 = vec3f(0.0);
  var m2 = vec3f(0.0);
  var mn = vec3f(1e9);
  var mx = vec3f(-1e9);
  var center = vec3f(0.0);
  var wsum = 0.0;
  var bestD = 0.0;
  var bestP = ip;
  for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
      let q = ip + vec2i(x, y);
      let c = rgb2ycocg(loadCur(q));
      m1 += c;
      m2 += c * c;
      mn = min(mn, c);
      mx = max(mx, c);
      // reconstruction filter (Blackman-Harris approx) around the exact sample position
      let d = vec2f(q) - sp;
      let w = exp(-2.29 * dot(d, d));
      center += c * w;
      wsum += w;
      let qd = textureLoad(depthTex, clamp(q, vec2i(0), vec2i(F.screen.xy) - 1), 0);
      if (qd > bestD) { bestD = qd; bestP = q; }
    }
  }
  center /= wsum;
  let mean = m1 / 9.0;
  let sigma = sqrt(max(m2 / 9.0 - mean * mean, vec3f(0.0)));
  // reprojection using closest depth (camera motion only)
  let duv = (vec2f(bestP) + 0.5) * F.screen.zw;
  let rel = relFromDepth(duv, max(bestD, 1e-6));
  let prevClip = F.prevViewProj * vec4f(rel + F.camDelta.xyz, 1.0);
  var prevUV = prevClip.xy / prevClip.w * vec2f(0.5, -0.5) + 0.5;
  let mv = prevUV - (duv - jit);
  let motion = mv * F.outSize.xy;
  prevUV = uv + mv;
  var alpha = 0.08;
  let offscreen = any(prevUV < vec2f(0.0)) || any(prevUV > vec2f(1.0)) || prevClip.w <= 0.0;
  var res: vec3f;
  if (TA.x > 0.5 || offscreen) {
    res = center;
  } else {
    var hist = rgb2ycocg(tw(sampleHistory(prevUV)));
    // variance clipping
    let g = 1.25;
    let bmin = max(mn, mean - sigma * g);
    let bmax = min(mx, mean + sigma * g);
    let cc = 0.5 * (bmax + bmin);
    let ex = 0.5 * (bmax - bmin) + 1e-5;
    let v = hist - cc;
    let ua = abs(v / ex);
    let ma = max(ua.x, max(ua.y, ua.z));
    if (ma > 1.0) { hist = cc + v / ma; }
    let speed = length(motion);
    alpha = mix(0.06, 0.25, sat(speed / 40.0));
    // lower internal resolution needs more accumulation
    alpha *= mix(0.75, 1.0, F.misc.y);
    res = mix(hist, center, alpha);
  }
  let outc = itw(ycocg2rgb(res));
  textureStore(outTex, op, vec4f(max(outc, vec3f(0.0)), 1.0));
}
