param(
  [Parameter(Mandatory = $true)]
  [string]$TexPath
)

$root = Split-Path -Parent $PSScriptRoot
$fontConfig = Join-Path $root "config\\fonts.conf"

if (-not (Test-Path $fontConfig)) {
  throw "Fontconfig file not found: $fontConfig"
}

$env:FONTCONFIG_FILE = $fontConfig

$texFullPath = Resolve-Path $TexPath
$workDir = Split-Path $texFullPath -Parent
$texFile = Split-Path $texFullPath -Leaf

Push-Location $workDir
try {
  pdflatex --keep-logs --keep-intermediates -p $texFile
} finally {
  Pop-Location
}
