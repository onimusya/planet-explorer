# Capture a screenshot of the running app via agent-browser (headed Chrome, real GPU).
# Usage: scripts\shot.ps1 -Out v010 [-Reload] [-View 0] [-Pose "x,y,z,dx,dy,dz"] [-Wait 3] [-Hud] [-Eval "js"]
param(
  [string]$Out = "shot",
  [switch]$Reload,
  [int]$View = -1,
  [string]$Pose = "",
  [double]$Wait = 3,
  [switch]$Hud,
  [string]$Eval = "",
  [string]$Url = "http://localhost:5173/"
)
$root = Split-Path -Parent $PSScriptRoot
$shots = Join-Path $root "shots"
New-Item -ItemType Directory -Force $shots | Out-Null
$S = "planet"
if ($Reload) {
  $j = Start-Job -ScriptBlock { param($u) agent-browser --session planet open $u 2>&1 } -ArgumentList $Url
  Wait-Job $j -Timeout 30 | Out-Null
  Remove-Job $j -Force
  for ($i = 0; $i -lt 60; $i++) {
    $r = agent-browser --session $S eval "typeof window.__pe" 2>&1 | Select-Object -Last 1
    if ($r -match "object") { break }
    Start-Sleep -Milliseconds 500
  }
  agent-browser --session $S set viewport 1920 1080 2>&1 | Out-Null
  Start-Sleep -Seconds 1
}
$js = "(() => { const p = window.__pe; p.overlay(" + ($(if ($Hud) { "true" } else { "false" })) + ");"
if ($View -ge 0) { $js += " p.view($View);" }
if ($Pose -ne "") { $js += " p.pose($Pose);" }
if ($Eval -ne "") { $js += " $Eval;" }
$js += " return 1; })()"
agent-browser --session $S eval $js 2>&1 | Out-Null
Start-Sleep -Milliseconds ([int]($Wait * 1000))
$path = Join-Path $shots "$Out.png"
agent-browser --session $S screenshot $path 2>&1 | Out-Null
if ($Hud) {
  & powershell -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "crop.ps1") -In $path -Out (Join-Path $shots "$Out`_hud.png") -X 0 -Y 20 -W 760 -H 130
}
$vis = agent-browser --session $S eval "document.visibilityState" 2>&1 | Select-Object -Last 1
Write-Output "saved $path (page $vis)"
