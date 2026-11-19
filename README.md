# Ithra Vael: a procedural WebGPU planet explorer

A free-flight explorer for one alien planet, written in TypeScript and WGSL on native WebGPU. It uses no engine. Terrain, sky, clouds, water, plants, animals, rocks and landmarks are all generated in code, at startup or every frame. The code is one JavaScript file and `index.html`, 92 KiB gzipped. The only files it loads are the soundtrack: three ElevenLabs-generated music tracks, one per destination (5.5 MiB of MP3), fetched after the first frame.

**Live:** https://onimusya.github.io/planet-explorer/ (needs a WebGPU browser, see [Requirements](#requirements)).

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
| M | Metrics overlay |
| B | Run the benchmark |
| G | Adaptive resolution on/off |
| N | Music on/off |
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
- **Browser:** Chrome 157 on Windows.
- **Display:** canvas at 1920×1080 with device pixel ratio 1, on a 165 Hz display with vsync.
- **Build:** the production build served by `npm run preview`.

### Flythrough benchmark at 1920×1080

Three runs of the current build in one session:

| | Adaptive (run 1) | Adaptive (run 2) | Fixed 100% scale |
|---|---|---|---|
| Frames measured | 4,143 | not recorded | 4,154 |
| Average FPS | 108.0 | 109.9 | 112.1 |
| 1% low FPS | 57.8 | 54.8 | 66.8 |
| Frame time avg / P95 / P99 / max | 9.26 / 12.20 / 12.30 / 18.30 ms | P99 18.20 ms | 8.92 / 12.20 / 12.30 / 18.30 ms |
| CPU time avg / P99 | 0.27 / 0.50 ms | not recorded | 0.26 / 0.50 ms |
| GPU time avg / P95 / P99 | 9.19 / 11.49 / 13.27 ms | 9.02 / – / 14.47 ms | 8.85 / 10.93 / 12.49 ms |
| Internal resolution at the end | 1920×1080 | 1920×1080 | 1920×1080 |

How to read these numbers:
- Frame times snap to multiples of the display's 6.06 ms refresh interval:
  - A P99 of 12.3 ms is two refreshes. A P99 of 18.2 ms means at least 1% of frames took three.
  - The 1% low comes from those slowest frames.
- GPU time decides whether 60 FPS holds. GPU P99 stayed between 12.5 and 14.5 ms in every run, inside the 16.7 ms budget.
- In the adaptive runs the scale stayed at or returned to 100%, so they measure the same work as the fixed run.
- The CPU needs under 1 ms per frame.
- **Run-to-run variance is large.** The same session also benchmarked the previous build twice and got 91.3 and 105.5 FPS average, against 108.0 and 109.9 here. The GPU work is identical between the two builds, so most of that gap is likely laptop GPU clock and thermal behaviour, not the code changes.
- The previous README measured 87.3 FPS on the previous build, in an earlier session.

### GPU time per pass at each destination

These figures were taken at a fixed 100% scale (1920×1080) and averaged over 4 s with timestamps on every frame (`scripts/perf.ps1`). All values are in ms.

| Pass | Work | Vista | Valley | Highlands |
|---|---|---|---|---|
| PRE | Sky-view and aerial-perspective LUTs, sky ambient, plant and grass culling (compute) | 0.28 | 0.27 | 0.29 |
| SHD | Four shadow cascades | 1.58 | 0.88 | 0.22 |
| MAIN | Terrain, plants, rocks, landmarks, grass | 6.24 | 7.50 | 4.52 |
| WATR | Water | 0.05 | 0.01 | 0.03 |
| VOL | Clouds and fog at half resolution (compute) | 2.53 | 0.92 | 2.07 |
| POST | Composite, TAA upsampling, bloom (compute) | 0.87 | 0.75 | 0.84 |
| FIN | Tone mapping, grading, overlay | 0.13 | 0.13 | 0.13 |
| **Sum** | | **11.67** | **10.47** | **8.11** |

The sum leaves out the colour and depth copies before the water pass and any idle time between passes.

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
| `assets/index-[hash].js` | 348.7 KiB | 90.3 KiB |
| `index.html` | 3.3 KiB | 1.4 KiB |
| **Code total** | **352.0 KiB** | **91.7 KiB** |
| `audio/vista.mp3`, `audio/valley.mp3`, `audio/highlands.mp3` | 1875.6 KiB each | (MP3 does not compress further) |
| **Total with audio** | **5978.7 KiB** (6,122,139 B) | **5695.3 KiB** (5,831,953 B) |

- **How measured:** `npm run size` after `npm run build`. Vite's own build report uses kB (1,000 bytes) and its own gzip settings, so it prints 357.06 kB and 96.65 kB for the JS file.
- **Audio:** 120 s per track at 128 kbit/s. The tracks are fetched after the first frame and never block it; the explorer runs silent if they are missing.
- **Nothing else downloads:** no textures, meshes or fonts.

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
| Music | `music.ts`, `scripts/music.mjs`, `public/audio/` | One instrumental track per region: the Vista track also covers the open country between destinations; the Valley and Highlands tracks start where the shaders' `valleyMask` and `highlandMask` pass about 0.6, and stop again below 0.4, so hovering on a border does not flip the music. Region changes crossfade over about 5 s. Each track is decoded the first time its region is entered, off the main thread. Its faded intro and outro are trimmed and its end crossfaded into its start for a seamless loop, in slices between frames, then it is kept for later visits (about 44 MiB of PCM per track at 48 kHz). Each visit enters the loop at a random point. Audio starts on the first click or key press, as browsers require. `N` mutes and suspends it; hidden tabs suspend it too. |
| Build | `vite.config.ts`, `ctx.ts` | A WGSL loader that supports `//#include` (each file once per module) and minifies WGSL at the token level for production. `ctx.ts` resolves `#if HFILTER` blocks at startup, depending on `float32-filterable`. |

### Performance techniques

- **GPU-driven scene:** compute shaders cull plants, rocks, landmarks and grass and pick their LODs, and indirect draws render them. The CPU only walks the terrain quadtree.
- **Recorded command lists:** the main and shadow passes replay render bundles. Pass descriptors, bind groups and timestamp descriptors are created once.
- **Static sun:** terrain shadows and AO and the atmosphere transmittance and multi-scattering LUTs are computed once.
- **Shadow maps for objects only:** the cascades hold only plants, rocks and landmarks. The terrain uses the precomputed shadow heights.
- **Reduced resolution:** clouds and fog render at half resolution with temporal accumulation. TAA upsampling lets the adaptive scale lower the internal resolution.
- **Depth:** reversed-Z with an infinite far plane and a 32-bit depth buffer covers the 0.15 m near plane and the distant horizon in one buffer.
- **GPU timers:** timestamps are written every frame while the overlay, the benchmark or adaptive resolution needs them. They are read back once per 16 frames.
- **No per-frame DOM work:** the overlay is drawn by the final shader.

### Memory and allocations

**Targets:**
- no JavaScript heap allocation per frame from application code;
- no per-frame GPU allocation;
- no shader compilation after startup;
- no blocking readbacks.

**Result:** application code makes no per-frame allocations. Everything the frame loop still allocates comes from WebGPU and browser calls that create objects by design. The only app-created object is one typed-array view per 16-frame GPU timing readback, which is the only way to read the mapped data. The measurements behind this are below.

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
- **No blocking readbacks:** GPU timings come back through `mapAsync` into a ring of three buffers. The frame loop polls `mapState` and never waits.

**How it was measured.** `scripts/heap-sample.mjs` drives Chrome's sampling heap profiler. It also counts short-lived objects the garbage collector has already freed. It reports bytes per frame by function and, with `--objects`, object counts by size.

**Production build:** sampled at the Vista with the overlay hidden, after 60 s of warm-up. Over 20 s it saw 1,605 frames at 80 FPS, with a sampling interval of 64 B. It measured **262 bytes per frame** in total:

| Bytes per frame | Where | What it is |
|---|---|---|
| 202 | `Renderer.render` | WebGPU objects: 1 command encoder, 3 compute and 6 render pass encoders, 1 command buffer, plus the browser's bookkeeping for `writeBuffer` and `submit` |
| 16 | `Post.encodeFinal` | the final render pass encoder |
| 16 | the frame loop | the swap-chain texture from `getCurrentTexture()` |
| 15 | `Metrics.poll` | once per 16 frames: the `mapAsync` Promise, the `ArrayBuffer` from `getMappedRange()`, and the `Uint32Array` view over it |
| 11 | no JavaScript on the stack | one 12 B object per frame created by the browser outside JavaScript, most likely the timestamp it passes to the `requestAnimationFrame` callback |
| 2 | V8 API | browser-side |

**Checking that `render` contains only API objects.** The `--probe` mode replays the same WebGPU calls in a separate named function, once per event-loop task, as in a frame. The calls are:
- 1 encoder;
- 3 compute passes and 6 render passes;
- 2 texture copies;
- 4 buffer writes;
- `finish` and `submit`.

Measured this way, the calls alone allocate 207 bytes, against 202 bytes for `render` with all its logic. Single calls measured the same way cost about 16 bytes per encoder, pass encoder, command buffer, texture or texture view. Buffer writes add 20 to 30 bytes per task in total, whether there is one write or six. `submit` adds nothing measurable.

**Other situations:**
- **Overlay on:** frames also allocate two 12 B numbers for the CPU timer. The HUD text update, 4 times a second, added about 6 B per frame in a sample taken right after it was switched on.
- **Benchmark flight:** a second run, started right after the first, allocated 284 B per frame. That is the same list plus the CPU timer. The first flight after loading also showed about 190 B per frame in the camera path code. That code had not run before, and the cost disappears once V8 optimizes it.
- **The first seconds after loading:** V8 runs new code unoptimized at first, which boxes every fractional intermediate. This is why early samples are much higher. One sample of the previous version 13 s after load showed 6.3 KB per frame, most of it in the recursive terrain walk.

**Before these changes:** the previous version measured 725 B per frame after the same warm-up, on the dev server. The extra 460 B came from:
- boxed numbers in the frame loop, `Renderer.render`, `Camera.update` and `qnorm`;
- `Math.hypot`;
- `Terrain.heightAt` and its per-call closure;
- BigInt timestamp reads and a typed-array view every 8 frames;
- the swap-chain texture view;
- two `performance.now()` calls per frame.

At 80 FPS the remaining 262 B per frame is about 21 KiB/s of short-lived objects, all created inside browser APIs.

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

- Exposure is fixed. There is no eye adaptation.
- The sun never moves, so there is no day/night cycle. Several precomputed fields depend on this.
- The adaptive system changes only the render scale. The quality tier stays at High.
- The destinations are tuned for the default seed (`0x7a11e5` in `src/world.ts`).
