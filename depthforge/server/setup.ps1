# DepthForge setup for Windows 10/11 (PowerShell 5.1+).
#
#   powershell -ExecutionPolicy Bypass -File setup.ps1              # server + photogrammetry tools
#   powershell -ExecutionPolicy Bypass -File setup.ps1 -LocalAI     # also the local TripoSR engine (~3 GB)
#
# Installs into this folder only: .venv\ (Python packages) and tools\ (COLMAP, OpenMVS, TripoSR).
param(
    [switch]$LocalAI,
    [switch]$SkipTools
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'   # makes Invoke-WebRequest much faster
Set-Location $PSScriptRoot
$tools = Join-Path $PSScriptRoot 'tools'
New-Item -ItemType Directory -Force $tools | Out-Null

function Step($msg) { Write-Host "`n== $msg" -ForegroundColor Cyan }

# ---------------------------------------------------------------- Python
Step 'Python'
$py = $null
foreach ($v in '3.11', '3.10', '3.12') {
    try { & py -$v -c "import sys" 2>$null; if ($LASTEXITCODE -eq 0) { $py = @('py', "-$v"); break } } catch {}
}
if (-not $py) {
    try { & python -c "import sys; assert sys.version_info >= (3, 10)" 2>$null; if ($LASTEXITCODE -eq 0) { $py = @('python') } } catch {}
}
if (-not $py) { throw 'Python 3.10-3.12 not found. Install Python 3.11 from https://www.python.org/downloads/ (tick "Add to PATH") and run this again.' }
Write-Host "Using: $($py -join ' ')"
if (-not (Test-Path '.venv\Scripts\python.exe')) {
    if ($py.Count -gt 1) { & $py[0] $py[1] -m venv .venv } else { & $py[0] -m venv .venv }
}
$vpy = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
& $vpy -m pip install --upgrade pip | Out-Null
& $vpy -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'pip install failed' }

# ---------------------------------------------------------------- helpers
function Get-LatestAsset($repo, $pattern) {
    $rel = Invoke-RestMethod "https://api.github.com/repos/$repo/releases/latest" -Headers @{ 'User-Agent' = 'depthforge-setup' }
    $asset = $rel.assets | Where-Object { $_.name -like $pattern } | Select-Object -First 1
    if (-not $asset) { throw "No release file matching '$pattern' in $repo $($rel.tag_name)" }
    return $asset
}
function Expand-Any($archive, $dest) {
    New-Item -ItemType Directory -Force $dest | Out-Null
    if ($archive -like '*.zip') { Expand-Archive -Force $archive $dest; return }
    $sevenZip = @("$env:ProgramFiles\7-Zip\7z.exe", "${env:ProgramFiles(x86)}\7-Zip\7z.exe") | Where-Object { Test-Path $_ } | Select-Object -First 1
    if ($sevenZip) { & $sevenZip x -y "-o$dest" $archive | Out-Null; return }
    & tar -xf $archive -C $dest   # Windows' built-in tar can read .7z on recent builds
    if ($LASTEXITCODE -ne 0) { throw "Could not extract $archive. Install 7-Zip (https://www.7-zip.org) and run setup again." }
}
function Install-Tool($name, $repo, $pattern, $exe) {
    $dest = Join-Path $tools $name
    if (Get-ChildItem $dest -Recurse -Filter $exe -ErrorAction SilentlyContinue | Select-Object -First 1) {
        Write-Host "$name already installed."; return
    }
    $asset = Get-LatestAsset $repo $pattern
    $file = Join-Path $env:TEMP $asset.name
    Write-Host "Downloading $($asset.name) ($([math]::Round($asset.size / 1MB)) MB)..."
    Invoke-WebRequest $asset.browser_download_url -OutFile $file
    Expand-Any $file $dest
    Remove-Item $file -ErrorAction SilentlyContinue
    if (-not (Get-ChildItem $dest -Recurse -Filter $exe | Select-Object -First 1)) { throw "$exe not found after extracting $name" }
    Write-Host "$name installed."
}

# ---------------------------------------------------------------- photogrammetry tools
if (-not $SkipTools) {
    Step 'COLMAP (camera alignment)'
    # CPU build: your GPU is too old for current CUDA builds.
    Install-Tool 'colmap' 'colmap/colmap' '*windows*no*cuda*.zip' 'colmap.exe'
    Step 'OpenMVS (dense mesh and texture)'
    try { Install-Tool 'openmvs' 'cdcseacave/openMVS' '*Windows*' 'DensifyPointCloud.exe' }
    catch {
        Write-Warning $_
        Write-Host 'Download OpenMVS for Windows manually from https://github.com/cdcseacave/openMVS/releases and extract it into' $tools'\openmvs'
    }
}

# ---------------------------------------------------------------- local AI (optional)
if ($LocalAI) {
    Step 'Local TripoSR engine (CPU)'
    & $vpy -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    & $vpy -m pip install -r requirements-local-ai.txt
    if ($LASTEXITCODE -ne 0) { throw 'pip install for the local AI engine failed' }
    $tsr = Join-Path $tools 'TripoSR'
    if (-not (Test-Path (Join-Path $tsr 'tsr\system.py'))) {
        $zip = Join-Path $env:TEMP 'TripoSR.zip'
        Invoke-WebRequest 'https://github.com/VAST-AI-Research/TripoSR/archive/refs/heads/main.zip' -OutFile $zip
        Expand-Archive -Force $zip $tools
        if (Test-Path $tsr) { Remove-Item -Recurse -Force $tsr }
        Rename-Item (Join-Path $tools 'TripoSR-main') 'TripoSR'
        Remove-Item $zip
    }
    Write-Host 'TripoSR installed. The first run downloads ~1.7 GB of model weights.'
}

# ---------------------------------------------------------------- config
if (-not (Test-Path 'config.json')) { Copy-Item 'config.example.json' 'config.json' }
Step 'Done'
Write-Host 'Start DepthForge with start.bat, then open http://127.0.0.1:8765'
Write-Host 'To use the cloud engines, put your fal.ai key in config.json ("fal_key").'
