// Physically based atmosphere model (Hillaire 2020 style), units in kilometres.
// Alien parameters: dense, strongly green/blue Rayleigh scattering (teal sky) with
// warm dust Mie haze and a thin green-absorbing layer (magenta twilight band).

const ATM_BOTTOM: f32 = PLANET_R * 0.001;
const ATM_TOP: f32 = PLANET_R * 0.001 + 60.0;
const RAY_SCAT: vec3f = vec3f(11.0, 31.0, 50.0) * 1e-3;
const RAY_H: f32 = 7.0;
const MIE_SCAT: vec3f = vec3f(26.0, 22.0, 19.0) * 1e-3;
const MIE_ABS: vec3f = vec3f(0.6, 0.8, 1.1) * 1e-3;
const MIE_H: f32 = 1.4;
const MIE_G: f32 = 0.78;
const OZ_ABS: vec3f = vec3f(0.35, 1.1, 0.05) * 1e-3;
const GROUND_ALBEDO: vec3f = vec3f(0.30, 0.22, 0.20);

const TRANS_W: f32 = 256.0;
const TRANS_H: f32 = 64.0;
const MS_RES: f32 = 32.0;
const SKY_W: f32 = 192.0;
const SKY_H: f32 = 108.0;
const AP_RES: f32 = 32.0;
const AP_MAX_KM: f32 = 48.0;

struct Medium {
  scat: vec3f,
  ext: vec3f,
  rayS: vec3f,
  mieS: vec3f,
};

fn medium(h: f32) -> Medium {
  let dr = exp(-max(h, 0.0) / RAY_H);
  let dm = exp(-max(h, 0.0) / MIE_H);
  let doz = max(0.0, 1.0 - abs(h - 22.0) / 14.0);
  var m: Medium;
  m.rayS = RAY_SCAT * dr;
  m.mieS = MIE_SCAT * dm;
  m.scat = m.rayS + m.mieS;
  m.ext = m.scat + MIE_ABS * dm + OZ_ABS * doz;
  return m;
}

// Distance to sphere of radius R from origin along ray (ro, rd); returns -1 when no hit.
fn raySphere(ro: vec3f, rd: vec3f, R: f32) -> vec2f {
  let b = dot(ro, rd);
  let c = dot(ro, ro) - R * R;
  let d = b * b - c;
  if (d < 0.0) { return vec2f(-1.0); }
  let s = sqrt(d);
  return vec2f(-b - s, -b + s);
}

fn rayleighPhase(c: f32) -> f32 { return 3.0 / (16.0 * PI) * (1.0 + c * c); }
fn hgPhase(c: f32, g: f32) -> f32 {
  let g2 = g * g;
  return (1.0 - g2) / (4.0 * PI * pow(max(1.0 + g2 - 2.0 * g * c, 1e-4), 1.5));
}
// Cornette-Shanks style Mie with stronger forward peak
fn miePhase(c: f32) -> f32 {
  let g = MIE_G;
  let k = 3.0 / (8.0 * PI) * (1.0 - g * g) / (2.0 + g * g);
  return k * (1.0 + c * c) / pow(max(1.0 + g * g - 2.0 * g * c, 1e-4), 1.5);
}

fn transUV(r: f32, mu: f32) -> vec2f {
  let H = sqrt(ATM_TOP * ATM_TOP - ATM_BOTTOM * ATM_BOTTOM);
  let rho = sqrt(max(r * r - ATM_BOTTOM * ATM_BOTTOM, 0.0));
  let disc = r * r * (mu * mu - 1.0) + ATM_TOP * ATM_TOP;
  let d = max(0.0, -r * mu + sqrt(max(disc, 0.0)));
  let dmin = ATM_TOP - r;
  let dmax = rho + H;
  return vec2f((d - dmin) / max(dmax - dmin, 1e-5), rho / H);
}

fn transParams(uv: vec2f) -> vec2f {
  let H = sqrt(ATM_TOP * ATM_TOP - ATM_BOTTOM * ATM_BOTTOM);
  let rho = H * uv.y;
  let r = sqrt(rho * rho + ATM_BOTTOM * ATM_BOTTOM);
  let dmin = ATM_TOP - r;
  let dmax = rho + H;
  let d = dmin + uv.x * (dmax - dmin);
  var mu = 1.0;
  if (d > 0.0) { mu = (H * H - rho * rho - d * d) / (2.0 * r * d); }
  return vec2f(r, clamp(mu, -1.0, 1.0));
}

fn sampleTrans(lut: texture_2d<f32>, s: sampler, r: f32, mu: f32) -> vec3f {
  return textureSampleLevel(lut, s, transUV(r, mu), 0.0).rgb;
}

// Transmittance from a world-space altitude (metres) toward the sun direction.
fn sunTransmittance(lut: texture_2d<f32>, s: sampler, altM: f32, sunY: f32) -> vec3f {
  let r = ATM_BOTTOM + max(altM, 0.0) * 0.001 + 0.002;
  return sampleTrans(lut, s, r, sunY);
}

fn msUV(r: f32, muS: f32) -> vec2f {
  return vec2f(muS * 0.5 + 0.5, (r - ATM_BOTTOM) / (ATM_TOP - ATM_BOTTOM));
}

fn skyViewUV(r: f32, viewZenithCos: f32, lightViewCos: f32, hitsGround: bool) -> vec2f {
  let vh = sqrt(max(r * r - ATM_BOTTOM * ATM_BOTTOM, 0.0));
  let cosBeta = vh / r;
  let beta = acos(clamp(cosBeta, -1.0, 1.0));
  let zha = PI - beta;
  var uv: vec2f;
  if (!hitsGround) {
    var c = acos(clamp(viewZenithCos, -1.0, 1.0)) / zha;
    c = 1.0 - c;
    c = sqrt(max(c, 0.0));
    c = 1.0 - c;
    uv.y = c * 0.5;
  } else {
    var c = (acos(clamp(viewZenithCos, -1.0, 1.0)) - zha) / beta;
    c = sqrt(max(c, 0.0));
    uv.y = c * 0.5 + 0.5;
  }
  uv.x = sqrt(max(-lightViewCos * 0.5 + 0.5, 0.0));
  uv.x = (uv.x + 0.5 / SKY_W) * (SKY_W / (SKY_W + 1.0));
  uv.y = (uv.y + 0.5 / SKY_H) * (SKY_H / (SKY_H + 1.0));
  return uv;
}

// Sky radiance (per unit solar illuminance) for a world direction from camera altitude.
fn skyRadiance(lut: texture_2d<f32>, s: sampler, camAltM: f32, dir: vec3f, sunDir: vec3f) -> vec3f {
  let r = ATM_BOTTOM + max(camAltM, 1.0) * 0.001;
  let vzc = dir.y;
  let hl = length(dir.xz);
  let sl = length(sunDir.xz);
  var lvc = 1.0;
  if (hl > 1e-4 && sl > 1e-4) { lvc = dot(dir.xz, sunDir.xz) / (hl * sl); }
  let ro = vec3f(0.0, r, 0.0);
  let hit = raySphere(ro, dir, ATM_BOTTOM);
  let ground = hit.x > 0.0;
  return textureSampleLevel(lut, s, skyViewUV(r, vzc, lvc, ground), 0.0).rgb;
}
