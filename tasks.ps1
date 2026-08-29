<#
.SYNOPSIS
  Dev task runner for carbon-crunch-ocr.
.EXAMPLE
  .\tasks.ps1 check          # ruff + pytest (the pre-commit gate)
  .\tasks.ps1 test-all       # include the slow end-to-end test
  .\tasks.ps1 batch --limit 20
#>
param(
    [Parameter(Position = 0)][string]$Task = "check",
    [Parameter(ValueFromRemainingArguments = $true)]$Rest
)

$ErrorActionPreference = "Stop"
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$env:PYTHONUTF8 = "1"

switch ($Task) {
    "test"     { & $py -m pytest -q }
    "test-all" { & $py -m pytest -q -m "" }
    "lint"     { & $py -m ruff check src tests scripts }
    "fix"      { & $py -m ruff check --fix src tests scripts }
    "coverage" { & $py scripts/coverage_report.py }
    "check"    { & $py -m ruff check src tests scripts; if ($LASTEXITCODE -eq 0) { & $py -m pytest -q } }
    "batch"    { & $py -m src.cli batch --input data/receipts --output outputs @Rest }
    "summary"  { & $py -m src.cli summary --json-dir outputs/json --output outputs }
    default    { Write-Host "tasks: test | test-all | lint | fix | coverage | check | batch | summary" }
}
