# Average GPU pass timings (ms) over a short window at a given view/scale/debug setting.
# Usage: scripts\perf.ps1 [-View 0] [-Scale 1] [-Vol 0] [-Pose "x,y,z,dx,dy,dz"] [-Secs 2]
param([int]$View = 0, [double]$Scale = 1, [int]$Vol = 0, [string]$Pose = "", [double]$Secs = 2, [string]$Eval = "")
$S = "planet"
$pose = if ($Pose -ne "") { "p.pose($Pose);" } else { "p.view($View);" }
$extra = if ($Eval -ne "") { "$Eval;" } else { "" }
$js = @"
(async () => {
  const p = window.__pe; if (p.enter) p.enter(); p.overlay(true); $pose p.scale($Scale); p.volDebug($Vol); $extra
  await new Promise(r => setTimeout(r, 1200));
  const acc = new Float64Array(8); let n = 0;
  const t0 = performance.now();
  while (performance.now() - t0 < $Secs * 1000) {
    await new Promise(r => setTimeout(r, 50));
    const m = p.passMs(); for (let i = 0; i < m.length; i++) acc[i] += m[i]; n++;
  }
  const names = ['PRE','SHD','MAIN','WATR','VOL','POST','FIN','FX'];
  const o = {}; let tot = 0;
  for (let i = 0; i < 8; i++) { o[names[i]] = +(acc[i] / n).toFixed(2); tot += acc[i] / n; }
  o.SUM = +tot.toFixed(2); o.res = p.info().internal.join('x');
  p.volDebug(0); p.overlay(false);
  return JSON.stringify(o);
})()
"@
agent-browser --session $S eval $js 2>&1 | Select-Object -Last 1
