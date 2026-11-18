# Point the camera at a group of animals (live poses read back from the GPU) and take a screenshot.
# Groups: 0 = valley herd, 1/2/3 = vista/valley/highland flocks. The camera sits -Ahead metres in
# front of the group's centre along its mean heading, -Side metres to its right and -Up metres above
# the ground (herd) or above the group's centre (flocks), looking at the centre (+ -Look metres up).
# Usage: scripts\herdcam.ps1 -Out v044_herd [-Group 0] [-Ahead 22] [-Side 10] [-Up 2] [-Look 1] [-Wait 2]
param([string]$Out = "herd", [int]$Group = 0, [double]$Ahead = 22, [double]$Side = 10, [double]$Up = 2, [double]$Look = 1, [double]$Wait = 2)
$start = if ($Group -eq 0) { 0 } else { 10 + ($Group - 1) * 14 }
$count = if ($Group -eq 0) { 10 } else { 14 }
$js = "return p.fauna().then(a => { let cx = 0, cy = 0, cz = 0, fx = 0, fz = 0; const g = $Group; " +
  "for (let i = $start; i < $start + $count; i++) { const b = i * 20; cx += a[b]; cy += a[b + 1]; cz += a[b + 2]; " +
  "if (g === 0) { fx += Math.sin(a[b + 3]); fz += Math.cos(a[b + 3]); } else { fx += a[b + 4]; fz += a[b + 6]; } } " +
  "cx /= $count; cy /= $count; cz /= $count; const l = Math.sqrt(fx * fx + fz * fz); fx /= l; fz /= l; " +
  "const px = cx + fx * $Ahead + fz * $Side, pz = cz + fz * $Ahead - fx * $Side; " +
  "const py = g === 0 ? Math.max(p.ground(px, pz), 30) + $Up : cy + $Up; " +
  "p.pose(px, py, pz, cx - px, cy + $Look - py, cz - pz); " +
  "return [cx, cy, cz, fx, fz, px, py, pz].map(v => v.toFixed(2)).join(','); })"
$r = powershell -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "ev.ps1") -Js $js -Shot $Out -Wait $Wait
Write-Output $r
