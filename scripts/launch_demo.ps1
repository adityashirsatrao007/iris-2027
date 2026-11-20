param(
    [switch]$Share,
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "auto"
)

$ErrorActionPreference = "Stop"
$python = Join-Path $PSScriptRoot "..\.venv-cuda\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
}
if (-not (Test-Path $python)) {
    throw "Create .venv-cuda or .venv first."
}

$args = @("scripts/demo_app.py", "--device", $Device)
if ($Share) { $args += "--share" }
& $python @args
