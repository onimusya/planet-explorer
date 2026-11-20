# Ithra Vael: a procedural WebGPU planet explorer

A free-flight explorer for one alien planet, written in TypeScript and WGSL on native WebGPU. It uses no engine. Terrain, sky, clouds, water, plants, animals, people, starships, rocks and landmarks are all generated in code, at startup or every frame. The code is one JavaScript file and `index.html`, 121 KiB gzipped. The only files it loads are the soundtrack: three ElevenLabs-generated music tracks, one per destination (5.5 MiB of MP3), fetched in the background while the world is built.

**Live:** https://onimusya.github.io/planet-explorer/ (needs a WebGPU browser, see [Requirements](#requirements)).

The three destinations are part of one continuous world, so you fly between them with no loading screens:

- **A. Arrival Vista** (start position): a grassy ledge above a lake and a stone arch, with buttes, striped peaks, floating islands and a ringed gas giant low in the sky. Two explorers in exosuits are on the ledge, one gazing out from the edge and one pacing along it, and two great sky rays soar over the plain below.
- **B. Bioluminescent Valley**: a 3.2 km canyon filled with giant glowing mushrooms, glowing ground cover and valley fog. A herd of ten grazers walks a trail through it, followed by two explorers, under 14 drifting sky jellies and luminous spores.
- **C. Monumental Highlands**: striped mesas cut by river canyons, with peaks, rock pillars, arches and a moon. Three long-necked titans and two stone colossi walk the mesa top.

Every destination also has a circling flock of flyers and three starships that pass overhead, leaving glowing engine trails.

Press T to switch between day, sunset and night. At night the gas giant lights the land, a star field and a galactic band fill the sky, and the glowing plants shine brighter. The exposure adapts as you fly between bright and dark places. The camera collides with the terrain and with solid things: trees, mushrooms, crystals, boulders, arches, pillars, floating islands, the grazers, titans, colossi, sky jellies and explorers.

A loading screen with a progress bar covers startup. When the world is ready, a click or a key press reveals it and starts the music.

## Requirements

- A browser with WebGPU: Chrome or Edge 113 or later on Windows, macOS or ChromeOS. Tested on Chrome 157 and 158 on Windows. Other WebGPU browsers are untested.
- A discrete GPU is recommended. The app asks for the high-performance adapter. On dual-GPU Windows laptops Chrome can still pick the integrated GPU; set Chrome to "High performance" in Windows graphics settings or start it with `--force_high_performance_gpu`.
- Optional GPU features: `timestamp-query` enables GPU timings and the adaptive resolution, and `float32-filterable` enables hardware-filtered height sampling. The app runs without them.
- Node.js `^20.19.0` or `>=22.12.0` (required by Vite 8) to run the dev server or build.

## Run

```sh
npm install
npm run dev
```

Open http://localhost:5173 and wait for **CLICK TO EXPLORE** on the loading screen. Click it, or press Enter or any key that types a character, to enter the world; the same click or key press starts the music. Then click the view to capture the mouse.

## Build

```sh
npm run build     # type-check, then bundle into dist/
npm run preview   # serve dist/ at http://localhost:4173
npm run size      # minified and gzip sizes of the files in dist/
```

`dist/` is a static site that any web server can host. Its URLs are relative (`base: './'` in `vite.config.ts`), so it also works from a subpath. WebGPU needs a secure context, so serve it over HTTPS or from `localhost`.

### GitHub Pages

Every push to `main` builds the site and publishes it to https://onimusya.github.io/planet-explorer/ through `.github/workflows/deploy-pages.yml`. You can also run the workflow by hand from the repository's Actions tab. For a fork, set Settings > Pages > Source to "GitHub Actions"; the site then appears at `https://<user>.github.io/<repo>/`.

## Controls

| Input | Action |
|---|---|
| Click | Capture the mouse (pointer lock) |
| Mouse | Look |
| W A S D | Move |
| Space / Ctrl (or C) | Up / down |
| Q / E | Roll |
| Shift | Boost (6×) |
| Wheel | Flight speed (×1.3 per step, 1 to 4000 m/s) |
| Esc | Release the mouse |
| 1 / 2 / 3 | Fly to the Vista / Valley / Highlands |
| R | Jump back to the Vista |
| T | Next time of day (day, sunset, night) |
| M | Metrics overlay |
| B | Run the benchmark |
| G | Adaptive resolution on/off |
| N | Music on/off |
| H | Hide the controls panel |

Speed also scales with height above the ground, from 0.55× near the surface up to 5×.

The camera stays at least 1.2 m above the terrain and keeps 0.6 m from solid things. When it hits something it slides along the surface; a head-on hit stops it. Leaves, grass, flowers, glow pods and shrubs are passable; trunks and branches are not. The flyers, sky rays and starships are not solid either. The flights on 1, 2 and 3 and the benchmark ignore collisions.

Switching the time of day fades the view to black, re-traces the terrain's shadows for the new light and fades back in. It took 1.87 to 2.07 s per switch.

### URL parameters

| Parameter | Effect |
|---|---|
| `?scale=0.75` | Fixed internal render scale (0.4 to 1). Turns adaptive resolution off. |
| `?adaptive=0` | Turns adaptive resolution off. |
| `?bench=1` | Starts the benchmark 2.5 s after loading. Skips the click-to-explore step, like `?gate=0`. |
| `?gate=0` | Skips the warm-up and the click-to-explore step: the loading screen fades out as soon as the world is built. The music then starts on the first click or key press. |
| `?time=sunset` | Starts at a time of day: `day` (the default), `sunset` or `night`. |
| `?collide=0` | Turns camera collisions off. Only the 1.2 m clearance above the terrain remains. |

### Metrics overlay (M)

The overlay shows the following, computed over the last 240 frames:
- the planet name and seed;
- FPS, 1% low FPS and average frame time;
- P95 and P99 frame time, and CPU time;
- GPU time, GPU P99 and the GPU time of each pass (the mean over the latest 16-frame readback);
- internal and output resolution, render scale and adaptive state;
- quality tier, terrain node count, altitude and flight speed;
- the last benchmark result.

The final shader draws it with a bitmap font, so it causes no DOM updates.

## Performance

### How it is measured

- **Frame time**: the time between `requestAnimationFrame` callbacks.
- **CPU time**: JavaScript time from the start of the frame callback until after `queue.submit`. It is measured only while the overlay or the benchmark runs, because each `performance.now()` call allocates.
- **GPU time**: timestamp queries around every pass. The frame value is the span from the first pass start to the last pass end. Timestamps for 16 frames are collected before each readback, into a ring of three readback buffers, so the CPU never waits for them.
- **1% low**: 1000 divided by the mean of the slowest 1% of frame times.
- **Benchmark (B)**: a scripted 40 s flight through all three destinations (9 keyframes, 5 s per segment). The first 30 frames are excluded. Results appear in the overlay and in `window.__bench`.

### Test machine

- **GPU:** NVIDIA GeForce RTX 3060 Laptop GPU (WebGPU adapter "nvidia ampere").
- **CPU and memory:** AMD Ryzen 7 5800H, 28 GB RAM.
- **Browser:** Chrome 157 and 158 on Windows.
- **Display:** canvas at 1920×1080 with device pixel ratio 1, with vsync. The panel runs at 165 Hz (165.8 Hz measured with `requestAnimationFrame` just before benchmark runs 2 and 3).
- **Build:** the production build served by `npm run preview`.

### Flythrough benchmark at 1920×1080

Five runs in one session, with adaptive resolution on (the default): the current build at each time of day, and the previous commit (`0fdc1bc`, before auto exposure, times of day and collision) for comparison. The internal resolution was 1920×1080 at the end of every run.

| | Day | Day, again | Sunset | Night | `0fdc1bc` |
|---|---|---|---|---|---|
| Frames measured | 4,538 | 4,527 | 4,526 | 4,461 | 4,533 |
| Average FPS | 119.2 | 119.1 | 118.8 | 118.0 | 119.3 |
| 1% low FPS | 62.6 | 62.2 | 62.6 | 62.3 | 62.3 |
| Frame time avg / P95 / P99 / max (ms) | 8.39 / 15.0 / 15.7 / 16.4 | 8.40 / 14.9 / 16.0 / 16.5 | 8.42 / 14.9 / 15.7 / 16.2 | 8.47 / 15.0 / 15.9 / 16.5 | 8.38 / 15.0 / 15.9 / 16.4 |
| CPU time avg / P99 (ms) | 0.23 / 0.40 | 0.23 / 0.40 | 0.23 / 0.40 | 0.23 / 0.40 | 0.23 / 0.40 |
| GPU time avg / P95 / P99 (ms) | 8.32 / 9.90 / 10.20 | 8.33 / 9.91 / 10.24 | 8.35 / 9.91 / 10.20 | 8.41 / 9.94 / 10.15 | 8.32 / 9.90 / 10.18 |

How to read these numbers:
- **The new features cost under 0.1 ms of GPU time.** By day the current build averaged 8.32 and 8.33 ms against 8.32 ms for `0fdc1bc`. Sunset added 0.02 ms and night 0.09 ms. At night the POST pass, which composites the sky and its stars, took up to 0.08 ms more in the per-pass figures below.
- **Every frame met 60 FPS in these runs.** The slowest frame took 16.5 ms, under the 16.7 ms budget, and GPU P99 was 10.2 ms.
- **Run-to-run variance is large.** Earlier the same day the machine was slower: the same comparison measured 10.67 ms average GPU time for the current build (10.68 ms at sunset) against 10.62 ms for `0fdc1bc`, with a GPU P99 of 14.5 to 14.6 ms. In that state some frames took three refreshes at 165 Hz (18.2 ms frame-time P99), so the 1% low fell to 54.6 to 54.8 FPS. The code was identical, so the gap comes from the machine's state (power mode, GPU clocks or temperature).
- **The CPU needs under 1 ms per frame.**

### GPU time per pass at each destination

These figures were taken on the production build at a fixed 100% scale (1920×1080), in the same session as the benchmark runs above. Each value is the mean of two 3 s samples by day, with timestamps on every frame. All values are in ms.

| Pass | Work | Vista | Valley | Highlands |
|---|---|---|---|---|
| PRE | Sky-view and aerial-perspective LUTs, sky ambient, plant and grass culling, creature and ship poses, collision gather (compute) | 0.25 | 0.27 | 0.20 |
| SHD | Four shadow cascades | 1.61 | 1.03 | 0.27 |
| MAIN | Terrain, plants, rocks, landmarks, grass, creatures, ships | 5.93 | 7.45 | 4.25 |
| WATR | Water | 0.05 | 0.01 | 0.03 |
| FX | Spores, pollen and engine trails | 0.02 | 0.02 | 0.02 |
| VOL | Clouds and fog at half resolution (compute) | 2.27 | 0.87 | 1.90 |
| POST | Composite, TAA upsampling, bloom, exposure histogram (compute) | 0.86 | 0.75 | 0.85 |
| FIN | Tone mapping, grading, overlay | 0.14 | 0.14 | 0.14 |
| **Sum** | | **11.11** | **10.52** | **7.66** |

- The sum leaves out the colour and depth copies before the water pass and any idle time between passes.
- **Sunset and night** summed 11.09 / 10.42 / 7.66 ms and 11.15 / 10.59 / 7.72 ms at the three views.
- **Against `0fdc1bc`:** in a same-session comparison (in the slower machine state, 4 s samples), the sums were 0.12 ms higher at the Vista, 0.10 ms higher in the Valley and 0.01 ms lower in the Highlands.

### Adaptive resolution

Every 0.5 s the app compares the mean GPU time of the latest 16-frame readback against two thresholds:
- Above 14.8 ms, the render scale drops by 5%, or by 10% when GPU time is above 18 ms.
- Below 11.5 ms, it rises by 5%.

The scale stays between 50% and 100%, and TAA rebuilds the output resolution from the jittered internal image. Other limits:
- The internal resolution is capped at the pixel count of 1920×1080.
- The output is capped at the pixel count of 2560×1440.
- The device pixel ratio is capped at 2.

Adaptive resolution needs `timestamp-query`. Without it, the scale stays fixed.

## Bundle size

| File | Minified | gzip -9 |
|---|---|---|
| `assets/index-[hash].js` | 412.0 KiB | 116.6 KiB |
| `index.html` | 10.3 KiB | 3.9 KiB |
| **Code total** | **422.3 KiB** | **120.5 KiB** |
| `audio/vista.mp3`, `audio/valley.mp3`, `audio/highlands.mp3` | 1875.6 KiB each | (MP3 does not compress further) |
| **Total with audio** | **6049.0 KiB** (6,194,149 B) | **5724.0 KiB** (5,861,413 B) |

- **How measured:** `npm run size` after `npm run build`. Vite's own build report uses kB (1,000 bytes) and its own gzip settings, so it prints 421.88 kB and 122.51 kB for the JS file.
- **Growth:** auto exposure, times of day and collision added 26.2 KiB minified and 9.5 KiB gzipped to the JS file. The collision worker is inlined into it.
- **`index.html`** holds the loading screen: its markup, CSS and a small script that draws the star field.
- **Audio:** 120 s per track at 128 kbit/s. The tracks are fetched while the world is built, and nothing waits for them; the explorer runs silent if they are missing.
- **Nothing else downloads:** no textures, meshes or fonts.

## Architecture

### World

- The world is a 16,384 m square that tiles seamlessly, because every noise function is periodic.
- A 150 km planet radius curves the terrain toward the horizon and sizes the atmosphere.
- The light comes from one of three fixed times of day (`TIMES` in `src/world.ts`):
  - **Day:** the sun at 7.5° elevation.
  - **Sunset:** the sun at 3°, with warmer light.
  - **Night:** the sun is 20° below the horizon. The land is lit by sunlight reflected from the gas giant, about 2% as bright as daylight and coming from the giant's direction. The gas giant and the moon stay lit by the real sun. The sky's share of the ambient light is multiplied by four, to stand in for starlight and the giant's glow.
- Terrain ambient occlusion and the atmosphere's transmittance and multi-scattering LUTs do not depend on the light direction, so they are computed once at startup. The terrain's sun shadows are traced at startup and again after each switch.
- `src/world.ts` holds the design constants: seed, times of day, destination positions and the cloud layer at 2.3 to 4.3 km. They are injected into every shader as a WGSL prelude.

### Startup

The loading screen is plain HTML and CSS in `index.html`, so it shows before the script has loaded. Its progress bar follows the steps below, weighted by their usual duration.

1. `gpu.ts` requests the adapter and device. `ctx.ts` creates the shared bind group layouts, samplers and the frame uniform. `music.ts` starts fetching the soundtrack.
2. Every subsystem starts creating its shader modules and pipelines at once, with `create*PipelineAsync`, while the terrain is generated. Only the steps that need the terrain wait for it: landmark placement and the herd's trail table.
3. The terrain is generated on the GPU in 256-row slices:
   - the 4096² height field, including analytic erosion gullies;
   - a max-height mip chain;
   - gradient and attribute maps (crease, lushness, AO, rock);
   - sun-shadow heights;
   - a 16 m CPU height grid for camera clearance and placement;
   - min/max bounds for every 64×64-texel block;
   - a coarse 1024² field for the volumetrics.
4. The atmosphere transmittance and multi-scattering LUTs, a 128³ cloud noise volume and a 512² weather map are computed once.
5. `plants.ts` builds 84 meshes on the CPU from seeded generators: 10 species, 28 variants and 3 LODs, with 326k triangles and 24-byte vertices in total. A worker (`sdf.worker.ts`) then turns the solid parts of the LOD 0 meshes into signed distance fields for collision: 21 fields with 1.14 million voxels (4.3 MiB), built in 0.58 s while the main thread goes on.
6. `landmarks.ts` places arches, floating islands and rock pillars around the destinations using the CPU height grid.
7. `fauna.ts` builds the meshes of the creatures, explorers and ships on the CPU. A compute pass then tabulates the herd's trail through the valley once (480 × 7 points), so the per-frame pose update looks positions up instead of evaluating the valley's terrain noise.
8. The render bundles for the main and shadow passes are recorded.
9. The world renders behind the loading screen for at least 45 frames and 0.8 s, so that shadows, temporal filters and the render scale have settled before anyone sees it. Then the loading screen shows CLICK TO EXPLORE.

Startup times on the test machine, production build:
- **Warm shader cache**, eight loads: the world started rendering 1.12 to 1.49 s after navigation and was ready to enter 1.93 to 2.30 s after it. In the same session `0fdc1bc` was ready after 2.20 to 2.24 s (four loads), so the new features add no measurable startup time.
- **Right after loading the other build**, when Chrome's shader cache missed: 4.60 s for the current build and 4.78 s for `0fdc1bc`.
- **No shader cache** (Chrome started with `--disable-gpu-shader-disk-cache`, measured before this round's features): building the world took 3.6 s. It took 15 s before the subsystems created their pipelines in parallel.

### Frame

Each frame the CPU updates the frame uniform and a few small parameter blocks, walks the terrain quadtree, writes the node list and indirect arguments, and encodes the passes below (`renderer.ts`):

1. **PRE** (compute): sky-view LUT (192×108), aerial-perspective LUT (32³), L2 spherical-harmonics sky ambient, plant and grass culling, then the pose of every creature, explorer and ship. Every 4th frame while the camera moves, the collision gather follows (see the Collision row below).
2. **SHD**: four 2048² shadow cascades, each replayed from a render bundle.
3. **MAIN**: terrain, plants, rocks, landmarks, grass, creatures, explorers and ships. They render into `rgba16float` colour and `depth32float` depth at the internal resolution, with TAA jitter and reversed-Z with an infinite far plane.
4. **WATR**: copies colour and depth, then draws the water with refraction and screen-space reflections.
5. **FX**: spores, pollen and the ships' engine trails, tested against the scene depth and drawn into their own HDR layer at the output resolution.
6. **VOL** (compute, half resolution): clouds and fog.
7. **POST** (compute):
   - composite of sky, aerial perspective, clouds and fog over the scene;
   - TAA upsampling to the output resolution;
   - bloom, which also takes in the FX layer;
   - the exposure histogram and adaptation.
8. **FIN**: bloom combine, the FX layer, exposure, tone mapping, grading and the text overlay, written to the swap chain.

After a time-of-day switch, while the view is black, an extra compute pass before PRE re-traces 128 rows of the terrain's sun shadows per frame. After all 4096 rows (32 frames) it rebuilds the coarse field for the volumetrics.

### Subsystems

| Area | Files | Notes |
|---|---|---|
| Terrain | `terrain.ts`, `terrain_gen.wgsl`, `terrain.wgsl`, `materials.wgsl` | Heights use 4 m texels. Rendering is CDLOD with 11 LODs: a CPU quadtree with per-block bounds and curvature-aware frustum culling, geomorphing, and five indirect draws from a render bundle. Height sampling is bicubic near the camera. Rock strata, cliff detail and meadow detail are added per pixel. |
| Sky | `atmo.ts`, `atmosphere.wgsl`, `atmo_luts.wgsl`, `sky.wgsl` | Physically based atmosphere in the style of Hillaire (2020), with dense teal Rayleigh scattering. Draws the sun disk, a ringed gas giant and a moon. At night: a star field with at most one star per cell of a grid over the six faces of a cube, denser along a galactic band whose glow comes from two lookups into a noise texture. |
| Clouds and fog | `volume.ts`, `volume.wgsl`, `noise3d.wgsl`, `weather.wgsl` | Clouds: 64 ray-march steps through Perlin-Worley noise shaped by the weather map, accumulated over frames. Fog: 20 steps of height and valley fog, with light shafts shadowed by terrain and clouds. |
| Plants, rocks, landmarks | `vegetation.ts`, `plants.ts`, `landmarks.ts`, `ecology.wgsl`, `veg_cull.wgsl`, `veg.wgsl` | Species: tree, mushroom, crystal, flower, glow pod, shrub, boulder, arch, floating island and rock pillar. A compute pass scatters candidates on a grid per species around the camera, with 4–16 m cells out to 0.19–1.28 km. Per-species ecology rules then set density and scale. The pass culls each candidate, picks its LOD and whether it casts shadows, and appends it to per-(species, variant, LOD) indirect draw lists. Landmarks come from a fixed placement buffer and go through the same lists. `veg.wgsl` adds wind animation and per-part materials (bark, leaf, cap, gill, petal, crystal, bulb, rock, frond, stem). |
| Grass | `grass.ts`, `grass.wgsl` | 2 m tiles in a 192 m window around the camera, up to 4096 visible tiles. Each tile has 160, 56 or 18 blades depending on distance (18 m, 42 m, 92 m). The vertex shader builds each blade from its index (7 vertices). Three indirect draws. |
| Water | `water.ts`, `water.wgsl` | A camera-centred radial grid at sea level (72 rings × 128 segments) covers the lakes, canyon rivers and the valley stream. Per-pixel waves, refraction with depth absorption, 28-step screen-space reflections that fall back to sky and planet, sun glint and foam. |
| Creatures, explorers and ships | `fauna.ts`, `fauna.wgsl`, `ships.wgsl`, `veg.wgsl` | 86 in all: a herd of 10 grazers (3 of them young), 3 flocks of 14 flyers, 3 long-necked titans, 2 stone colossi, 4 explorers in exosuits, 14 sky jellies, 2 sky rays and 9 starships (6 fighters and 3 haulers, three over each destination). The meshes are built once in a rest pose, with one bone and a blend weight per vertex. Each frame a compute pass writes every pose from the time alone, and the vertex shaders pose the meshes: legs by two-bone IK with planted feet, necks and tails by rotations that grow along their length, wings by shoulder and elbow flaps. They share the plants' fragment shading, so they sit in the same light and shadow. |
| Particles and trails | `fx.ts`, `fx.wgsl` | Luminous spores in the valley and sunlit pollen elsewhere (16,384 and 8,192 particles). Each ship trails one camera-facing ribbon per engine, 40 segments long, rebuilt every frame from its flight path; the two plumes drift together and merge within a few seconds. Both are instanced draws replayed from one render bundle. |
| Loading screen | `index.html`, `main.ts` | A ringed planet with a moon, a star field and a progress bar, in HTML and CSS. The world renders behind it while it warms up. Until the world is revealed, key presses do not reach the explorer. A click, Enter or a key that types a character reveals it, and the same gesture unlocks the audio. |
| Shadows | `renderer.ts`, `lighting.wgsl` | Plants, rocks, landmarks, animals and explorers render into four cascades that end at 24, 90, 360 and 1500 m, with dithered blending between cascades. The jellies and ships cast no shadows. Terrain self-shadowing uses the precomputed sun-shadow heights instead. |
| Post | `post.ts`, `composite.wgsl`, `taa.wgsl`, `bloom.wgsl`, `final.wgsl` | TAA: Halton jitter, a Catmull-Rom history fetch and history rejection from neighbourhood statistics, with upsampling to the output resolution. Bloom: 6 levels, 13-tap downsample with a Karis average, tent upsample. Tone mapping: an ACES fit, plus colour grading, vignette, film grain and sharpening. At night, dim tones lose colour toward blue. |
| Auto exposure | `exposure.wgsl`, `post.ts` | A 128-bin histogram of log luminance over a quarter-resolution bloom level, weighted four times more at the centre of the view than at the corners. The metered value is the mean between the 50th and 95th percentiles, so deep shade and the brightest highlights do not move it. The exposure makes up half of the difference between that value and the Vista's, within −1.5 to +3.5 EV, so dark places stay darker than bright ones and the Vista keeps the exposure it was graded at. It adapts with a time constant of 0.45 s when the view gets brighter and 0.9 s when it gets darker, and jumps straight to the target when the temporal history resets (a jump to a destination or a time-of-day switch). All of it runs on the GPU and nothing is read back. |
| Times of day | `world.ts`, `main.ts`, `terrain.ts`, `terrain_gen.wgsl` | Three presets of light direction and colour, sun position, star-field strength and extra glow for the luminous plants. A switch fades to black in 0.45 s, re-traces the terrain shadows, resets the temporal history and fades back in over 1.1 s. |
| Collision | `collide.ts`, `sdf.ts`, `sdf.worker.ts`, `scatter.wgsl`, `veg_cull.wgsl`, `camera.ts` | Every 4th frame while the camera moves, compute entry points in `veg_cull.wgsl` gather what is solid within 24 to 60 m of a point just ahead of the camera: the plant and rock instances (from the same scatter code as the drawn ones, in `scatter.wgsl`), the landmarks, the animal poses and a 32×32 window of exact terrain heights. The CPU reads that back through a ring of three buffers without waiting. Each frame `resolve()` checks the camera's move in steps of at most 0.35 m (up to 32 steps) against the distance fields of the plants, rocks and landmarks, capsules for the animals and explorers, and the terrain heights (bicubic in the window, the 16 m grid outside it). Each contact pushes the camera out along the surface normal and removes the velocity into the surface; a head-on contact also cuts the remaining speed by up to 75%. The distance fields have 32 voxels along the longest side of a plant or rock and 48 along that of a landmark. A morphological closing makes open-ended shells such as mushroom stems solid; tree crowns get a smaller closing, so the gaps between branches stay open. |
| Frame, camera, input | `renderer.ts`, `camera.ts`, `input.ts`, `metrics.ts`, `main.ts` | Pass encoding; the free-flight camera with smoothing, fly-to and scripted paths; statistics and GPU timers; destination poses and adaptive resolution. |
| Music | `music.ts`, `scripts/music.mjs`, `public/audio/` | One instrumental track per region: the Vista track also covers the open country between destinations; the Valley and Highlands tracks start where the shaders' `valleyMask` and `highlandMask` pass about 0.6, and stop again below 0.4, so hovering on a border does not flip the music. Region changes crossfade over about 5 s. Each track is decoded the first time its region is entered, off the main thread. Its faded intro and outro are trimmed and its end crossfaded into its start for a seamless loop, in slices between frames, then it is kept for later visits (about 44 MiB of PCM per track at 48 kHz). Each visit enters the loop at a random point. Audio starts with the click or key press that enters the world, as browsers require; with `?gate=0`, on the first click or key press. `N` mutes and suspends it; hidden tabs suspend it too. |
| Build | `vite.config.ts`, `ctx.ts` | A WGSL loader that supports `//#include` (each file once per module) and minifies WGSL at the token level for production. `ctx.ts` resolves `#if HFILTER` blocks at startup, depending on `float32-filterable`. |

### Performance techniques

- **GPU-driven scene:** compute shaders cull plants, rocks, landmarks and grass and pick their LODs, and indirect draws render them. The CPU only walks the terrain quadtree.
- **Recorded command lists:** the main and shadow passes replay render bundles. Pass descriptors, bind groups and timestamp descriptors are created once.
- **Static sun:** terrain shadows and AO and the atmosphere transmittance and multi-scattering LUTs are computed once.
- **Shadow maps for objects only:** the cascades hold only plants, rocks and landmarks. The terrain uses the precomputed shadow heights.
- **Reduced resolution:** clouds and fog render at half resolution with temporal accumulation. TAA upsampling lets the adaptive scale lower the internal resolution.
- **Depth:** reversed-Z with an infinite far plane and a 32-bit depth buffer covers the 0.15 m near plane and the distant horizon in one buffer.
- **GPU timers:** timestamps are written every frame while the overlay, the benchmark or adaptive resolution needs them. They are read back once per 16 frames.
- **Parallel startup:** every subsystem creates its pipelines at the same time, while the terrain is generated. With no shader cache this took 3.6 s, against 15 s when the subsystems took turns.
- **One compile of the shared creature shader:** the creature pipelines share one fragment shader. Creating them all at once took 14 s with no shader cache, apparently because each pipeline compiled that shader again. `fauna.ts` creates the first pipeline alone, so the shader is cached before the others start; that took 2.6 s.
- **Stateless creature animation:** every pose follows from the time alone, so the pose update keeps no state and reads nothing back. It runs one 64-thread workgroup per group (herd, flyers, titans, explorers, colossi, jellies, rays, ships) in a single dispatch. The herd's trail comes from the table built at startup, and each walker samples the ground 5 times (body and 4 feet). It measured 0.14 ms for all 86 poses, against 0.36 ms with one dispatch per group.
- **No per-frame DOM work:** the overlay is drawn by the final shader.
- **Collision without stalls:** the GPU already knows where every plant, landmark and animal is, so it gathers the nearby solid ones; the CPU never repeats the scatter. The gather is skipped during scripted flights and, while the camera is at rest, runs only once per 60 frames. `resolve()` measured about 0.04 ms of CPU time per frame.
- **Exposure without readback:** the histogram, the metering and the adaptation all run in compute shaders, and the final pass reads the exposure from a storage buffer.
- **Collision shapes off the main thread:** building the distance fields on the main thread took about 0.6 s and delayed startup by about 1 s; the worker removed that delay.

### Memory and allocations

**Targets:**
- no JavaScript heap allocation per frame from application code;
- no per-frame GPU allocation;
- no shader compilation after startup;
- no blocking readbacks.

**Result:** the current build shows no per-frame allocations from application code. Everything the frame loop allocates comes from WebGPU and browser calls that create objects by design. Each readback also needs one typed-array view over the mapped data, which is the only way to read it: GPU timings are read back every 16 frames, and the collision gather every 4 frames while the camera moves.

The 27 B per frame that the frame loop of `0fdc1bc` allocated beyond the swap-chain texture is gone. It came from fractional numbers passed to the music's region-weight helpers in `music.update`, which V8 boxed; the weights are now computed inline.

The measurements are below.

**How the code meets the targets:**
- **Per-frame state** lives in preallocated typed arrays. These hold:
  - the frame uniform mirror, input state, terrain node lists, indirect arguments and the metrics rings;
  - the camera and matrix math (`math.ts` writes into arrays that the caller passes in).

  Uploads use `queue.writeBuffer` from those arrays.
- **No fractional numbers cross function calls.** V8 stores a fractional number in a new heap object ("boxes" it) whenever it is passed to or returned from a call the JIT did not inline, or stored in a closure variable. Per-frame code avoids all three:
  - The time step, animation time, frame time and CPU time live in `clock.ts`, a shared `Float64Array`.
  - Height queries use `Terrain.q`: write x and z, call `heightQ()`, read the height. Nothing passes through the call.
  - The terrain quadtree walk is iterative. Its helpers take stack indices. Patch coordinates are whole metres, which V8 passes as small integers without boxing.
  - The HUD's number formatter reads values from (array, index) pairs.
  - Per-frame math helpers take vectors: `qlookDirV`, and `qslerp`, whose parameter is passed in an array.
  - The camera's look rotation reads its angles from an array.
  - Constant inputs (projection, jitter sequence, sun basis) are built at startup or on resize.
- **No hidden allocations from built-ins.**
  - `Math.hypot` is not used per frame: V8's implementation allocates a temporary array.
  - GPU timestamps are read as pairs of 32-bit words instead of BigInts.
  - The height query no longer creates a closure per call.
- **Fewer API objects.**
  - The swap-chain texture goes straight into the final pass. `createView()` is skipped when the browser accepts a texture as an attachment, which `gpu.ts` detects at startup.
  - GPU timestamps for 16 frames are collected per readback.
  - CPU time, which needs two `performance.now()` calls, is measured only while the overlay or the benchmark shows it.
- **GPU resources** (buffers, textures, bind groups) are created at startup, or again when the window is resized.
- **Pipelines:** all shader modules and pipelines are created at startup with `create*PipelineAsync`. There are no synchronous pipeline calls.
- **No blocking readbacks:** GPU timings and the collision gather each come back through `mapAsync` into a ring of three buffers. The frame loop polls `mapState` and never waits. The exposure is never read back.
- **Collision** keeps its scalars in typed arrays, as above: `resolve()` passes no fractional number through a call.

**How it was measured.** `scripts/heap-sample.mjs` drives Chrome's sampling heap profiler. It also counts short-lived objects the garbage collector has already freed. It reports bytes per frame by function and, with `--objects`, object counts by size.

**Production build:** the current build was sampled twice for 20 s, with a sampling interval of 64 B, starting 40 s after loading, with the overlay hidden:
- **At rest** at the Vista: 1,656 frames (83 FPS while profiled), **300 bytes per frame**.
- **Flying** forward with collisions on: 1,539 frames (77 FPS), **344 bytes per frame**.

`0fdc1bc` measured 308 B per frame at rest (2,082 frames at 104 FPS).

| At rest (B/frame) | Flying | At `0fdc1bc` | Where | What it is |
|---|---|---|---|---|
| 221 | 219 | 221 | `Renderer.render` | WebGPU objects: 1 command encoder, the compute and render pass encoders, 1 command buffer, plus the browser's bookkeeping for `writeBuffer` and `submit` |
| 16 | 16 | 16 | `Post.encodeFinal` | the final render pass encoder |
| 16 | 17 | 43 | the frame loop | the swap-chain texture from `getCurrentTexture()` (16 B); at `0fdc1bc` also the 27 B from `music.update` |
| 21 | 78 | 15 | `Metrics.poll` and `Collider.poll` (reported together) | per readback: the `mapAsync` Promise, the `ArrayBuffer` from `getMappedRange()` and the typed-array view over it, about 250 B in all. GPU timings come every 16 frames. The collision gather comes every 4 frames while the camera moves and once per 60 frames at rest. |
| 12 | 0 | – | `Metrics.readBatch` | not yet optimized by V8 in the first sample: it runs only once per 16 frames. It allocated 0.3 B per frame in the second sample. |
| 13 | 11 | 12 | no JavaScript on the stack | one 12 B object per frame created by the browser outside JavaScript, most likely the timestamp it passes to the `requestAnimationFrame` callback |
| 2 | 2 | 2 | V8 API | browser-side |

**Checking that `render` contains only API objects** (at `8732b71`). The `--probe` mode replays the same WebGPU calls in a separate named function, once per event-loop task, as in a frame. The calls are:
- 1 encoder;
- 3 compute passes and 6 render passes;
- 2 texture copies;
- 4 buffer writes;
- `finish` and `submit`.

Measured this way, the calls alone allocate 207 bytes, against 202 bytes for `render` with all its logic. Single calls measured the same way cost about 16 bytes per encoder, pass encoder, command buffer, texture or texture view. Buffer writes add 20 to 30 bytes per task in total, whether there is one write or six. `submit` adds nothing measurable.

**Other situations:**
- **Overlay on:** frames also allocate two 12 B numbers for the CPU timer. The HUD text update, 4 times a second, added about 6 B per frame in a sample taken right after it was switched on.
- **Benchmark flight** (at `0fdc1bc`): a second run, started right after the first, allocated 284 B per frame. That is the same list plus the CPU timer. The first flight after loading also showed about 190 B per frame in the camera path code. That code had not run before, and the cost disappears once V8 optimizes it.
- **The first seconds after loading:** V8 runs new code unoptimized at first, which boxes every fractional intermediate. This is why early samples are much higher. One sample of the previous version 13 s after load showed 6.3 KB per frame, most of it in the recursive terrain walk.

**Before `8732b71`:** the version before it measured 725 B per frame after the same warm-up, on the dev server. The extra 460 B came from:
- boxed numbers in the frame loop, `Renderer.render`, `Camera.update` and `qnorm`;
- `Math.hypot`;
- `Terrain.heightAt` and its per-call closure;
- BigInt timestamp reads and a typed-array view every 8 frames;
- the swap-chain texture view;
- two `performance.now()` calls per frame.

In the samples above that is 24.3 KiB/s of short-lived objects at rest and 25.8 KiB/s in flight.

**To repeat the measurement:**
1. Open the app in Chrome.
2. Get a DevTools WebSocket URL: use `agent-browser get cdp-url`, or start Chrome with `--remote-debugging-port=9222` and read `webSocketDebuggerUrl` from http://127.0.0.1:9222/json/version.
3. Keep the window visible: Chrome pauses `requestAnimationFrame` in hidden windows.
4. Run:

```sh
node scripts/heap-sample.mjs <ws-url> 20 localhost:4173 --interval=64 --objects   # frame loop
node scripts/heap-sample.mjs <ws-url> 60 localhost:4173 --probe                    # single API calls
node scripts/heap-sample.mjs <ws-url> 60 localhost:4173 --app                      # the app's own pass descriptors
```

The debug API below allocates when you call it. For example, `hprobe` creates a readback buffer per call. None of it runs unless called.

## Debugging and automation

`window.__pe` exposes:

| Call | Effect |
|---|---|
| `view(i)` | Jump to destination `i` (0, 1, 2) |
| `pose(x, y, z, dx, dy, dz)` | Set the camera position and direction |
| `cam()` | The camera position and forward vector |
| `bench()`, `benchDone()` | Start the benchmark, check whether it finished |
| `stats()` | Frame statistics over the last 600 frames |
| `passMs()` | GPU time of each pass, averaged over the latest 16-frame readback |
| `frames()` | Number of frames rendered |
| `info()` | Adapter, timestamp support, texture-as-attachment support, internal and output resolution |
| `renderer` | The renderer object, for the `--app` allocation probe |
| `overlay(on)`, `scale(s)`, `adaptive(on)` | Overlay, fixed render scale, adaptive resolution |
| `ground(x, z)` | Height from the 16 m CPU grid |
| `hprobe(x, z)` | 64×64 block of exact GPU heights from (x, z), read back once |
| `vegCounts()`, `vegStats()` | Drawn instances per species, mesh totals |
| `volDebug(mask)` | Volumetrics debug views |
| `music()` | Music region, the track playing, audio state, loaded track sizes, loop points |
| `enter()` | Reveal the world, as the click on the loading screen does. The audio then unlocks on the next click or key press. |
| `gate()` | Loading state: 0 building the world, 1 warming up, 2 waiting for the click, 3 revealed |
| `loadLog()` | Startup progress as `[ms since navigation, percent, step]` entries |
| `fauna()` | Promise of the pose buffer of every creature, explorer and ship (20 floats each), read back once |
| `expo()` | Promise of the exposure state: [exposure, EV, metered log2 luminance, set], read back once |
| `time(i)`, `timeBusy()` | Switch to time of day `i` (0 day, 1 sunset, 2 night) with the fade, as T does; whether a switch is still running. `time()` returns the current index. |
| `collide(on)` | Turn camera collisions on or off; returns the current state |
| `collideStats()` | Number of distance fields, their voxels and the time the worker took to build them |
| `collideDebug(x, y, z)` | The latest gather (kind, variant, position and scale of each instance), plus the distance and normal to the nearest solid surface from (x, y, z) |
| `collideMove(a, b)` | Where a move from point `a` to point `b` ends, against the latest gather |
| `hold(key, ms)` | Hold a movement key (`'W'`, `'A'`, `'S'`, `'D'`, `'UP'`, `'DOWN'`, `'SHIFT'`) for `ms` milliseconds, for scripted collision tests |

Scripts in `scripts/`:
- `size.mjs`: bundle sizes.
- `music.mjs`: generates the soundtrack (one instrumental track per destination) with the ElevenLabs Music API into `public/audio/`. It reads `ELEVENLABS_API_KEY` from the environment or `~/.media-gen/.env`; `--check` shows the account's credits, `--force` regenerates existing tracks. Without the files the explorer runs silent.
- `heap-sample.mjs`: the allocation measurement above. Options:
  - `--interval=N`: sampling interval in bytes;
  - `--objects`: object counts by size;
  - `--sizes`: raw sample sizes;
  - `--probe`: single browser API calls;
  - `--app`: the app's own pass descriptors.
- PowerShell tools for Windows. They drive Chrome through the `agent-browser` CLI with a session named `planet`:
  - `shot.ps1`: screenshots;
  - `ev.ps1`: evaluate JavaScript in the page;
  - `perf.ps1`: per-pass GPU timings;
  - `crop.ps1`: crop screenshots;
  - `front.ps1`: restore and focus the app's window, so the page stays visible during measurements.

## Screenshots

`scripts/shot.ps1` saves screenshots to `shots/`, which git ignores. The current set:
- `v040_vista.png`, `v040_valley.png` and `v040_high.png` show the three destinations as they open.
- `v040_right.png` and `v040_hump.png` are extra Vista angles.
- `v041_help.png` shows the Vista with the controls panel.

Example:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\shot.ps1 -Out vista -Reload -View 0 -Wait 6
```

## Project layout

```
index.html           page, loading screen, controls panel
vite.config.ts       WGSL loader and minifier
src/*.ts             one module per subsystem (see the table above)
src/shaders/*.wgsl   shaders, combined through //#include
public/audio/        soundtrack (generated by scripts/music.mjs, copied into dist/ as is)
scripts/             measurement, screenshot and soundtrack tools
.github/workflows/   GitHub Pages deployment
```

## Known gaps

- The times of day are three fixed presets. The light never moves between them, so there is no continuous day/night cycle: each switch re-traces the terrain's shadows behind a fade to black.
- At high speed collisions cover only the terrain, from its 16 m grid. Objects are checked only within 48 m of the latest gather, which lags a few frames behind the camera, and moves longer than 11.2 m per frame are checked in steps longer than 0.35 m.
- A gather holds at most 256 solid instances; any beyond that are dropped.
- The adaptive system changes only the render scale. The quality tier stays at High.
- The destinations are tuned for the default seed (`0x7a11e5` in `src/world.ts`).
