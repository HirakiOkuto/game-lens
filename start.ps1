# Opens game-lens. It starts the translation server and the local model by itself (and stops them when it quits),
# so nothing else needs to be running. Running this again just brings the window back.
#   .\start.ps1          open
#   .\start.ps1 -Quit    close game-lens and what it started
param([switch]$Quit)
$py = Join-Path $PSScriptRoot ".venv\Scripts\pythonw.exe"
$a = @("-m", "gamelens")
if ($Quit) { $a += "--quit" }
Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $PSScriptRoot
