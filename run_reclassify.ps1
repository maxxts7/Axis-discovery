# Run the LLM-based reclassification of cross-axis capping results.
#
# Usage:
#   .\run_reclassify.ps1                         # process all CSVs
#   .\run_reclassify.ps1 -Resume                 # resume interrupted run
#   .\run_reclassify.ps1 -SummaryOnly            # just print stats
#   .\run_reclassify.ps1 -Model "gpt-4.1"       # use a different model
#   .\run_reclassify.ps1 -Concurrency 20         # more parallel calls
#   .\run_reclassify.ps1 -Input "path\to\file.csv"

param(
    [string]$Input      = "",
    [string]$InputDir   = "cross_axis_full_results",
    [string]$Model      = "gpt-5.4-mini",
    [int]$Concurrency   = 5,
    [switch]$Resume,
    [switch]$SummaryOnly
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# Load .env
if (Test-Path ".env") {
    Get-Content ".env" | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]+)=(.+)$') {
            [Environment]::SetEnvironmentVariable($Matches[1].Trim(), $Matches[2].Trim(), "Process")
        }
    }
    Write-Host "Loaded .env"
} else {
    Write-Host "WARNING: No .env file found. Make sure OPENAI_API_KEY is set." -ForegroundColor Yellow
}

if (-not $env:OPENAI_API_KEY) {
    Write-Host "ERROR: OPENAI_API_KEY not set." -ForegroundColor Red
    exit 1
}

# Build args
$pyArgs = @()

if ($Input)       { $pyArgs += "--input", $Input }
if ($InputDir)    { $pyArgs += "--input-dir", $InputDir }
if ($Model)       { $pyArgs += "--model", $Model }
if ($Concurrency) { $pyArgs += "--concurrency", $Concurrency }
if ($Resume)      { $pyArgs += "--resume" }
if ($SummaryOnly) { $pyArgs += "--summary-only" }

Write-Host "Running: python reclassify_refusals.py $($pyArgs -join ' ')"
python reclassify_refusals.py @pyArgs
exit $LASTEXITCODE
