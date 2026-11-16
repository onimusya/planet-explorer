// Ecological distribution rules for each species (density 0..1 and scale multiplier).
// Species: 0 tree, 1 mushroom, 2 crystal, 3 flower, 4 glowpod, 5 shrub, 6 boulder.

struct Eco { dens: f32, scale: f32 };

fn ecology(s: u32, p: vec2f, h: f32) -> Eco {
  var o: Eco;
  o.dens = 0.0;
  o.scale = 1.0;
  if (h < WATER_Y + 0.7) { return o; }
  let a = attrAt(p); // crease, lush, ao, rock
  let g = gradAt(p);
  let slope = length(g);
  let valley = valleyMask(p);
  let high = highlandMask(p);
  let grove = fbm(p / 256.0, vec2i(64), 3, 0.5);
  let patchN = fbm(p / 64.0, vec2i(256), 2, 0.5);
  let flat = 1.0 - smoothstep(0.28, 0.55, slope);
  let lush = a.y;
  let alt = 1.0 - smoothstep(650.0, 950.0, h);
  switch (s) {
    case 0u: {
      // groves in lush lowlands and on the vista plateau rim; sparse in valley/highlands
      let gr = smoothstep(-0.02, 0.22, grove + lush * 0.15 - 0.08);
      o.dens = gr * smoothstep(0.2, 0.55, lush + 0.15) * flat * alt * (1.0 - valley * 0.75) * (1.0 - high * 0.85) * 0.85;
      o.scale = 0.85 + 0.35 * gr;
    }
    case 1u: {
      // giant bioluminescent mushrooms dominate the valley floor
      let vf = valley * (1.0 - smoothstep(0.25, 0.5, slope));
      o.dens = vf * (0.55 + 0.45 * smoothstep(-0.2, 0.2, grove)) + lush * flat * 0.03 * (1.0 - high);
      o.scale = 0.8 + valley * 0.9 * smoothstep(-0.1, 0.3, grove);
    }
    case 2u: {
      // crystal outcrops: a few large clusters on highland ledges, never on steep walls
      let rocky = smoothstep(0.1, 0.45, slope + a.w * 0.6);
      let steepCut = 1.0 - smoothstep(0.5, 0.75, slope);
      let clump = smoothstep(0.14, 0.32, patchN + grove * 0.25);
      o.dens = (high * (0.18 + 0.4 * rocky) * clump + a.w * 0.012 + valley * 0.025 * rocky) * steepCut;
      o.scale = (1.15 + high * 0.75 * clump) * mix(0.8, 1.2, rocky);
    }
    case 3u: {
      // sunbloom flowers in open meadows, clustered
      let cl = smoothstep(0.05, 0.25, patchN);
      o.dens = cl * flat * smoothstep(0.3, 0.6, lush + 0.1) * (1.0 - valley) * (1.0 - high * 0.7) * alt * 0.5;
      o.scale = 0.8 + 0.4 * cl;
    }
    case 4u: {
      // glowing pods: valley floor carpets and water margins
      let wet = 1.0 - smoothstep(0.8, 5.0, h - WATER_Y);
      o.dens = valley * flat * 0.9 + wet * 0.35 * flat + lush * 0.03;
    }
    case 5u: {
      // understory fronds wherever ground is lush
      o.dens = smoothstep(0.25, 0.6, lush + patchN * 0.4) * flat * 0.7 * alt + valley * 0.25;
    }
    default: {
      // boulders: rocky ground, cliff bases, scattered on meadows
      // (talus collects where cliffs flatten out; never stuck to the walls themselves)
      o.dens = (0.04 + a.w * 0.35 + smoothstep(0.2, 0.5, slope) * 0.25 + high * 0.15) * (1.0 - smoothstep(0.7, 1.0, slope));
      o.scale = 0.7 + a.w * 0.8;
    }
  }
  return o;
}
