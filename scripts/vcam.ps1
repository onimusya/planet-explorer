# Valley camera experiments: place the camera at valley coordinates (U, V) metres, -H above the
# ground, looking at valley point (LU, LV) -LH above its ground, then screenshot. Valley -> world
# uses a local linear fit around the herd trail (from valleyWorld at u = 560 and 700, v = 82).
param([string]$Out = "vcam", [double]$U = 575, [double]$V = 100, [double]$H = 10, [double]$LU = 700, [double]$LV = 75, [double]$LH = 2, [double]$Wait = 3)
$js = "const A = [8670.6, 9177.7], eu = [0.8979, -0.4171], ev = [0.4171, 0.8979]; " +
  "const w = (u, v) => [A[0] + (u - 560) * eu[0] + (v - 82) * ev[0], A[1] + (u - 560) * eu[1] + (v - 82) * ev[1]]; " +
  "const c = w($U, $V), l = w($LU, $LV); const cy = Math.max(p.ground(c[0], c[1]), 0) + $H, ly = Math.max(p.ground(l[0], l[1]), 0) + $LH; " +
  "p.pose(c[0], cy, c[1], l[0] - c[0], ly - cy, l[1] - c[1]); return [c[0], cy, c[1], l[0] - c[0], ly - cy, l[1] - c[1]].map(x => x.toFixed(2)).join(',')"
$r = powershell -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "ev.ps1") -Js $js -Shot $Out -Wait $Wait
Write-Output $r
