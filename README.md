# Ithra Vael: a procedural WebGPU planet explorer

A free-flight explorer for one alien planet, written in TypeScript and WGSL on native WebGPU. It uses no engine and loads no assets. Terrain, sky, clouds, water, plants, rocks and landmarks are all generated in code, at startup or every frame. The production build is one JavaScript file and `index.html`, 77 KiB gzipped.

The three destinations are part of one continuous world, so you fly between them with no loading screens:

- **A. Arrival Vista** (start position): a grassy ledge above a lake and a stone arch, with buttes, striped peaks, floating islands and a ringed gas giant low in the sky.
- **B. Bioluminescent Valley**: a 3.2 km canyon filled with giant glowing mushrooms, glowing ground cover and valley fog.
- **C. Monumental Highlands**: striped mesas cut by river canyons, with peaks, rock pillars, arches and a moon.

## Requirements

- A browser with WebGPU: Chrome or Edge 113 or later on Windows, macOS or ChromeOS. Tested on Chrome 157 on Windows. Other WebGPU browsers are untested.
- A discrete GPU is recommended. The app asks for the high-performance adapter. On dual-GPU Windows laptops Chrome can still pick the integrated GPU; set Chrome to "High performance" in Windows graphics settings or start it with `--force_high_performance_gpu`.
- Optional GPU features: `timestamp-query` enables GPU timings and the adaptive resolution, and `float32-filterable` enables hardware-filtered height sampling. The app runs without them.
- Node.js `^20.19.0` or `>=22.12.0` (required by Vite 8) to run the dev server or build.

## Run

```sh
npm install
npm run dev
```

Open http://localhost:5173, wait for the loading bar, then click the view to capture the mouse.

## Build

```sh
npm run build     # type-check, then bundle into dist/
npm run preview   # serve dist/ at http://localhost:4173
npm run size      # minified and gzip sizes of the files in dist/
```

`dist/` is a static site that any web server can host. WebGPU needs a secure context, so serve it over HTTPS or from `localhost`.

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
| M | Metrics overlay |
| B | Run the benchmark |
| G | Adaptive resolution on/off |
| H | Hide the controls panel |

Speed also scales with height above the ground, from 0.55× near the surface up to 5×. The camera stays at least 1.2 m above the terrain.

### URL parameters

| Parameter | Effect |
|---|---|
| `?scale=0.75` | Fixed internal render scale (0.4 to 1). Turns adaptive resolution off. |
| `?adaptive=0` | Turns adaptive resolution off. |
| `?bench=1` | Starts the benchmark 2.5 s after loading. |

### Metrics overlay (M)

The overlay shows the following, computed over the last 240 frames:
- the planet name and seed;
- FPS, 1% low FPS and average frame time;
- P95 and P99 frame time, and CPU time;
- GPU time, GPU P99 and the GPU time of each pass;
- internal and output resolution, render scale and adaptive state;
- quality tier, terrain node count, altitude and flight speed;
- the last benchmark result.

The final shader draws it with a bitmap font, so it causes no DOM updates.

## Performance

### How it is measured

- **Frame time**: the time between `requestAnimationFrame` callbacks.
- **CPU time**: JavaScript time from the start of the frame callback until after `queue.submit`.
- **GPU time**: timestamp queries around every pass. The frame value is the span from the first pass start to the last pass end. Results come back through a ring of three readback buffers, so the CPU never waits for them.
- **1% low**: 1000 divided by the mean of the slowest 1% of frame times.
- **Benchmark (B)**: a scripted 40 s flight through all three destinations (9 keyframes, 5 s per segment). The first 30 frames are excluded. Results appear in the overlay and in `window.__bench`.

### Test machine

- **GPU:** NVIDIA GeForce RTX 3060 Laptop GPU (WebGPU adapter "nvidia ampere").
- **CPU and memory:** AMD Ryzen 7 5800H, 28 GB RAM.
- **Browser:** Chrome 157 on Windows.
- **Display:** canvas at 1920×1080 with device pixel ratio 1, on a 165 Hz display with vsync.
- **Build:** the production build served by `npm run preview`.

### Flythrough benchmark at 1920×1080

| | Adaptive resolution (default) | Fixed 100% scale |
|---|---|---|
| Frames measured | 3,379 | 3,279 |
| Average FPS | 87.3 | 86.5 |
| 1% low FPS | 54.4 | 54.4 |
| Frame time avg / P95 / P99 / max | 11.46 / 18.20 / 18.30 / 18.80 ms | 11.56 / 18.20 / 18.30 / 18.70 ms |
| CPU time avg / P99 | 0.38 / 0.70 ms | 0.37 / 0.60 ms |
| GPU time avg / P95 / P99 | 11.57 / 15.28 / 15.87 ms | 11.91 / 15.83 / 16.85 ms |
| Internal resolution | 1920×1080 at the end of the run | 1920×1080 |

How to read these numbers:
- Frame times snap to multiples of the display's 6.06 ms refresh interval. The P95 and P99 frame times of 18.2 ms are frames that took three refreshes, and the 1% low of 54.4 FPS comes from the same frames.
- GPU time decides whether 60 FPS holds:
  - With adaptive resolution, GPU P99 is 15.9 ms, inside the 16.7 ms budget.
  - At a fixed 100% scale, GPU P99 is 16.9 ms, so about 1% of frames go over budget.
- The CPU needs under 1 ms per frame.

### GPU time per pass at each destination

These figures were taken at a fixed 100% scale (1920×1080) and averaged over 4 s with timestamps on every frame (`scripts/perf.ps1`). All values are in ms.

| Pass | Work | Vista | Valley | Highlands |
|---|---|---|---|---|
| PRE | Sky-view and aerial-perspective LUTs, sky ambient, plant and grass culling (compute) | 0.30 | 0.24 | 0.25 |
| SHD | Four shadow cascades | 1.57 | 0.87 | 0.22 |
| MAIN | Terrain, plants, rocks, landmarks, grass | 7.08 | 8.17 | 5.18 |
| WATR | Water | 0.05 | 0.01 | 0.03 |
| VOL | Clouds and fog at half resolution (compute) | 2.39 | 0.90 | 2.13 |
| POST | Composite, TAA upsampling, bloom (compute) | 0.89 | 0.72 | 0.96 |
| FIN | Tone mapping, grading, overlay | 0.12 | 0.13 | 0.13 |
| **Sum** | | **12.40** | **11.04** | **8.89** |

The sum leaves out the colour and depth copies before the water pass and any idle time between passes.

### Adaptive resolution

Every 0.5 s the app compares the latest GPU time against two thresholds:
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
| `assets/index-[hash].js` | 283.5 KiB | 75.4 KiB |
| `index.html` | 3.3 KiB | 1.4 KiB |
| **Total** | **286.7 KiB** (293,616 B) | **76.9 KiB** (78,719 B) |

- **How measured:** `npm run size` after `npm run build`. Vite's own build report uses kB (1,000 bytes) and its own gzip settings, so it prints 290.28 kB and 79.24 kB for the JS file.
- **WGSL share:** about 195 KiB of the minified JS is WGSL source, which compresses to 36 KiB on its own. Shared WGSL includes are repeated in each shader module that uses them.
- **Nothing else downloads:** no textures, meshes, fonts or audio.

## Architecture

### World

- The world is a 16,384 m square that tiles seamlessly, because every noise function is periodic.
- A 150 km planet radius curves the terrain toward the horizon and sizes the atmosphere.
- The sun is fixed at 7.5° elevation. Because it never moves, terrain shadows, terrain ambient occlusion and atmosphere transmittance are computed once at startup.
- `src/world.ts` holds the design constants: seed, sun, destination positions and the cloud layer at 2.3 to 4.3 km. They are injected into every shader as a WGSL prelude.

### Startup

1. `gpu.ts` requests the adapter and device. `ctx.ts` creates the shared bind group layouts, samplers and the frame uniform.
2. The terrain is generated on the GPU in 256-row slices:
   - the 4096² height field, including analytic erosion gullies;
   - a max-height mip chain;
   - gradient and attribute maps (crease, lushness, AO, rock);
   - sun-shadow heights;
   - a 16 m CPU height grid for camera clearance and placement;
   - min/max bounds for every 64×64-texel block;
   - a coarse 1024² field for the volumetrics.
3. The atmosphere transmittance and multi-scattering LUTs, a 128³ cloud noise volume and a 512² weather map are computed once.
4. `plants.ts` builds 84 meshes on the CPU from seeded generators: 10 species, 28 variants and 3 LODs, with 326k triangles and 24-byte vertices in total.
5. `landmarks.ts` places arches, floating islands and rock pillars around the destinations using the CPU height grid.
6. All shader modules and pipelines are created asynchronously, and the render bundles for the main and shadow passes are recorded.

On the test machine the first frame appears about 7 s after the page loads.

### Frame

Each frame the CPU updates the frame uniform and a few small parameter blocks, walks the terrain quadtree, writes the node list and indirect arguments, and encodes the passes below (`renderer.ts`):

1. **PRE** (compute): sky-view LUT (192×108), aerial-perspective LUT (32³), L2 spherical-harmonics sky ambient, then plant and grass culling.
2. **SHD**: four 2048² shadow cascades, each replayed from a render bundle.
3. **MAIN**: terrain, plants, rocks, landmarks and grass. They render into `rgba16float` colour and `depth32float` depth at the internal resolution, with TAA jitter and reversed-Z with an infinite far plane.
4. **WATR**: copies colour and depth, then draws the water with refraction and screen-space reflections.
5. **VOL** (compute, half resolution): clouds and fog.
6. **POST** (compute):
   - composite of sky, aerial perspective, clouds and fog over the scene;
   - TAA upsampling to the output resolution;
   - bloom.
7. **FIN**: bloom combine, exposure, tone mapping, grading and the text overlay, written to the swap chain.

### Subsystems

| Area | Files | Notes |
|---|---|---|
| Terrain | `terrain.ts`, `terrain_gen.wgsl`, `terrain.wgsl`, `materials.wgsl` | Heights use 4 m texels. Rendering is CDLOD with 11 LODs: a CPU quadtree with per-block bounds and curvature-aware frustum culling, geomorphing, and five indirect draws from a render bundle. Height sampling is bicubic near the camera. Rock strata, cliff detail and meadow detail are added per pixel. |
| Sky | `atmo.ts`, `atmosphere.wgsl`, `atmo_luts.wgsl`, `sky.wgsl` | Physically based atmosphere in the style of Hillaire (2020), with dense teal Rayleigh scattering. Draws the sun disk, a ringed gas giant and a moon. |
| Clouds and fog | `volume.ts`, `volume.wgsl`, `noise3d.wgsl`, `weather.wgsl` | Clouds: 64 ray-march steps through Perlin-Worley noise shaped by the weather map, accumulated over frames. Fog: 20 steps of height and valley fog, with light shafts shadowed by terrain and clouds. |
| Plants, rocks, landmarks | `vegetation.ts`, `plants.ts`, `landmarks.ts`, `ecology.wgsl`, `veg_cull.wgsl`, `veg.wgsl` | Species: tree, mushroom, crystal, flower, glow pod, shrub, boulder, arch, floating island and rock pillar. A compute pass scatters candidates on a grid per species around the camera, with 4–16 m cells out to 0.19–1.28 km. Per-species ecology rules then set density and scale. The pass culls each candidate, picks its LOD and whether it casts shadows, and appends it to per-(species, variant, LOD) indirect draw lists. Landmarks come from a fixed placement buffer and go through the same lists. `veg.wgsl` adds wind animation and per-part materials (bark, leaf, cap, gill, petal, crystal, bulb, rock, frond, stem). |
| Grass | `grass.ts`, `grass.wgsl` | 2 m tiles in a 192 m window around the camera, up to 4096 visible tiles. Each tile has 160, 56 or 18 blades depending on distance (18 m, 42 m, 92 m). The vertex shader builds each blade from its index (7 vertices). Three indirect draws. |
| Water | `water.ts`, `water.wgsl` | A camera-centred radial grid at sea level (72 rings × 128 segments) covers the lakes, canyon rivers and the valley stream. Per-pixel waves, refraction with depth absorption, 28-step screen-space reflections that fall back to sky and planet, sun glint and foam. |
| Shadows | `renderer.ts`, `lighting.wgsl` | Plants, rocks and landmarks render into four cascades that end at 24, 90, 360 and 1500 m, with dithered blending between cascades. Terrain self-shadowing uses the precomputed sun-shadow heights instead. |
| Post | `post.ts`, `composite.wgsl`, `taa.wgsl`, `bloom.wgsl`, `final.wgsl` | TAA: Halton jitter, a Catmull-Rom history fetch and history rejection from neighbourhood statistics, with upsampling to the output resolution. Bloom: 6 levels, 13-tap downsample with a Karis average, tent upsample. Tone mapping: an ACES fit, plus colour grading, vignette, film grain and sharpening. Exposure is fixed. |
| Frame, camera, input | `renderer.ts`, `camera.ts`, `input.ts`, `metrics.ts`, `main.ts` | Pass encoding; the free-flight camera with smoothing, fly-to and scripted paths; statistics and GPU timers; destination poses and adaptive resolution. |
| Build | `vite.config.ts`, `ctx.ts` | A WGSL loader that supports `//#include` (each file once per module) and minifies WGSL at the token level for production. `ctx.ts` resolves `#if HFILTER` blocks at startup, depending on `float32-filterable`. |

### Performance techniques

- **GPU-driven scene:** compute shaders cull plants, rocks, landmarks and grass and pick their LODs, and indirect draws render them. The CPU only walks the terrain quadtree.
- **Recorded command lists:** the main and shadow passes replay render bundles. Pass descriptors, bind groups and timestamp descriptors are created once.
- **Static sun:** terrain shadows and AO and the atmosphere transmittance and multi-scattering LUTs are computed once.
- **Shadow maps for objects only:** the cascades hold only plants, rocks and landmarks. The terrain uses the precomputed shadow heights.
- **Reduced resolution:** clouds and fog render at half resolution with temporal accumulation. TAA upsampling lets the adaptive scale lower the internal resolution.
- **Depth:** reversed-Z with an infinite far plane and a 32-bit depth buffer covers the 0.15 m near plane and the distant horizon in one buffer.
- **GPU timers:** timestamps are read every frame only while the overlay or the benchmark runs. Otherwise they are read every 8th frame, for adaptive resolution.
- **No per-frame DOM work:** the overlay is drawn by the final shader.

### Memory and allocations

**Targets:**
- no JavaScript heap allocation per frame from application code;
- no per-frame GPU allocation;
- no shader compilation after startup;
- no blocking readbacks.

**How the code meets them:**
- **Per-frame state:** it lives in preallocated typed arrays. These include the frame uniform mirror, the camera and matrix math (`math.ts` writes into arrays that the caller passes in), input state, terrain node lists, indirect arguments and the metrics rings. Uploads use `queue.writeBuffer` from those arrays.
- **GPU resources:** buffers, textures and bind groups are created at startup, or again when the window is resized.
- **Pipelines:** all shader modules and pipelines are created at startup with `create*PipelineAsync`. There are no synchronous pipeline calls.
- **GPU timings:** they come back through `mapAsync` into a ring of three buffers. The frame loop checks `mapState` and never waits.

**What the WebGPU API allocates every frame anyway:**
- one command encoder, ten pass encoders and one command buffer;
- the swap-chain texture and its view (`getCurrentTexture().createView()`);
- for each GPU timing readback, the `mapAsync` Promise and the `ArrayBuffer` that `getMappedRange()` returns.

**Measured result: the target is not met yet.** `scripts/heap-sample.mjs` runs the Chrome DevTools sampling heap profiler for 6 s. It also counts objects the garbage collector has already freed. The run was at the Vista with the overlay hidden, on the dev server, with 443 frames at 74 FPS. It recorded 6,305 bytes per frame (453 KiB/s):

| Bytes per frame | Where |
|---|---|
| 5,588 | `Terrain.node` (the CDLOD quadtree walk) |
| 307 | `Renderer.render` (includes the WebGPU encoders above) |
| 104 | the frame loop in `main.ts` (includes the swap-chain texture and view) |
| 95 | `Metrics.poll` (timestamp readback, including a `BigUint64Array` view and BigInt reads) |
| 87 | `qnorm` in `math.ts` (`Math.hypot` with four arguments) |
| 37 | `Terrain.heightAt` |
| 36 | `Camera.update` |
| 18 | `Post.encodeFinal` |
| 33 | other (idle, V8 internals) |

`Terrain.node` creates no objects in the source. The likely cause is V8 storing floating-point values as heap objects when they are passed to and returned from the helper methods it calls (`bounds`, `boxDist2`, `inFrustum`). This has not been confirmed yet.

To repeat the measurement, open the app in Chrome and get a DevTools WebSocket URL. Use `agent-browser get cdp-url`, or start Chrome with `--remote-debugging-port=9222` and read `webSocketDebuggerUrl` from http://127.0.0.1:9222/json/version. Then run:

```sh
node scripts/heap-sample.mjs <ws-url> 6 localhost:5173
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
| `passMs()` | The latest GPU time of each pass |
| `info()` | Adapter, timestamp support, internal and output resolution |
| `overlay(on)`, `scale(s)`, `adaptive(on)` | Overlay, fixed render scale, adaptive resolution |
| `ground(x, z)` | Height from the 16 m CPU grid |
| `hprobe(x, z)` | 64×64 block of exact GPU heights from (x, z), read back once |
| `vegCounts()`, `vegStats()` | Drawn instances per species, mesh totals |
| `volDebug(mask)` | Volumetrics debug views |

Scripts in `scripts/`:
- `size.mjs`: bundle sizes.
- `heap-sample.mjs`: the allocation measurement above.
- PowerShell tools for Windows. They drive Chrome through the `agent-browser` CLI with a session named `planet`:
  - `shot.ps1`: screenshots;
  - `ev.ps1`: evaluate JavaScript in the page;
  - `perf.ps1`: per-pass GPU timings;
  - `crop.ps1`: crop screenshots.

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
scripts/             measurement and screenshot tools
```

## Known gaps

- Glowing spores and other airborne particles are not implemented yet.
- Exposure is fixed. There is no eye adaptation.
- The sun never moves, so there is no day/night cycle. Several precomputed fields depend on this.
- Per-frame JavaScript allocations remain (see Memory and allocations).
- The adaptive system changes only the render scale. The quality tier stays at High.
- The destinations are tuned for the default seed (`0x7a11e5` in `src/world.ts`).
