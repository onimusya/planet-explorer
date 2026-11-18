# Restores the app's Chrome window (if minimized) and brings it to the front, so that the page
# stays visible and requestAnimationFrame keeps running during measurements.
# Usage: scripts\front.ps1 [-Title "Ithra Vael"]
param([string]$Title = "Ithra Vael")
Add-Type -Namespace W -Name U -MemberDefinition @'
[DllImport("user32.dll")] public static extern bool ShowWindow(System.IntPtr h, int c);
[DllImport("user32.dll")] public static extern bool SetForegroundWindow(System.IntPtr h);
[DllImport("user32.dll")] public static extern bool IsIconic(System.IntPtr h);
'@ -ErrorAction SilentlyContinue
$p = Get-Process chrome -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowTitle -like "$Title*" } | Select-Object -First 1
if (-not $p) { Write-Output "window not found"; exit 1 }
if ([W.U]::IsIconic($p.MainWindowHandle)) { [W.U]::ShowWindow($p.MainWindowHandle, 9) | Out-Null } # SW_RESTORE
[W.U]::SetForegroundWindow($p.MainWindowHandle) | Out-Null
Write-Output "front"
