# Evaluate JS in the running app (debug API available as `p`), optionally screenshot afterwards.
# Usage: scripts\ev.ps1 -Js "p.view(0); return p.cam()" [-Shot NAME] [-Wait 3]
param([string]$Js = "return 1", [string]$Shot = "", [double]$Wait = 3)
$root = Split-Path -Parent $PSScriptRoot
$code = "(() => { const p = window.__pe; if (p.enter) p.enter(); p.overlay(false); document.getElementById('help').style.opacity = '0'; $Js })()"
# Windows PowerShell 5 drops embedded double quotes when calling native executables
$code = $code.Replace('"', '\"')
$r = agent-browser --session planet eval $code 2>&1 | Select-Object -Last 1
Write-Output $r
if ($Shot -ne "") {
  Start-Sleep -Milliseconds ([int]($Wait * 1000))
  $path = Join-Path (Join-Path $root "shots") "$Shot.png"
  agent-browser --session planet screenshot $path 2>&1 | Out-Null
  Write-Output "saved $path"
}
