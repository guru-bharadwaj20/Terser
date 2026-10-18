<#
.SYNOPSIS
    Terser task runner (Windows PowerShell 5.1+). Mirror of the Makefile.
.EXAMPLE
    .\tasks.ps1 test
    .\tasks.ps1 mlflow-ui -DryRun
#>
param(
    [Parameter(Position = 0)]
    [string]$Task = "help",
    # Print the commands that would run without executing them.
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$VenvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (Test-Path $VenvPython) { $Py = $VenvPython } else { $Py = "python" }

$Tasks = [ordered]@{
    "help"            = "List available tasks (default)"
    "install"         = "Create .venv if missing and install the CI extras into it"
    "test"            = "Run CPU tests (not gpu, not api)"
    "test-all"        = "Run the full test suite"
    "lint"            = "ruff check + black --check"
    "format"          = "ruff check --fix + black"
    "typecheck"       = "mypy src"
    "hooks"           = "pre-commit run --all-files"
    "data"            = "dvc repro (if dvc.yaml exists)"
    "train-classical" = "Train classical baselines (Phase 3)"
    "eval"            = "Run evaluation (Phase 3)"
    "mlflow-ui"       = "Start the MLflow UI on ./mlruns"
    "gpu-queue"       = "Dry-run the GPU job queue"
    "hello"           = "Run the hello smoke module"
    "clean"           = "Remove caches (never data/ or mlruns/)"
}

function Invoke-Step {
    param([string]$Exe, [string[]]$Arguments)
    Write-Host ("> " + $Exe + " " + ($Arguments -join " "))
    if ($DryRun) { return }
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

switch ($Task) {
    "help" {
        Write-Host "Usage: .\tasks.ps1 <task> [-DryRun]`n"
        foreach ($k in $Tasks.Keys) { Write-Host ("  {0,-16} {1}" -f $k, $Tasks[$k]) }
    }
    "install" {
        if (-not (Test-Path $VenvPython)) {
            Invoke-Step "python" @("-m", "venv", ".venv")
            $Py = $VenvPython
        }
        Invoke-Step $Py @("-m", "pip", "install", "-e", ".[dev,hpo,serve,mlops,orchestrate]")
    }
    "test"      { Invoke-Step $Py @("-m", "pytest", "-m", "not gpu and not api") }
    "test-all"  { Invoke-Step $Py @("-m", "pytest") }
    "lint" {
        Invoke-Step $Py @("-m", "ruff", "check", ".")
        Invoke-Step $Py @("-m", "black", "--check", ".")
    }
    "format" {
        Invoke-Step $Py @("-m", "ruff", "check", "--fix", ".")
        Invoke-Step $Py @("-m", "black", ".")
    }
    "typecheck" { Invoke-Step $Py @("-m", "mypy", "src") }
    "hooks"     { Invoke-Step $Py @("-m", "pre_commit", "run", "--all-files") }
    "data" {
        if (Test-Path "dvc.yaml") {
            Invoke-Step $Py @("-m", "dvc", "repro")
        } else {
            Write-Host "No dvc.yaml pipeline yet; nothing to reproduce."
        }
    }
    "train-classical" { Invoke-Step $Py @("-m", "terser.phase3.studies", "--help") }
    "eval"            { Write-Host "eval: the final evaluation is python -m terser.phase12.results --split test ... (see its --help); it needs the row 12.1 unlock." }
    "mlflow-ui" {
        Invoke-Step $Py @("-m", "mlflow", "ui", "--backend-store-uri", "./mlruns")
    }
    "gpu-queue" { Invoke-Step $Py @("-m", "terser.gpu_queue", "dry-run") }
    "hello"     { Invoke-Step $Py @("-m", "terser.hello") }
    "clean" {
        $names = @("__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "htmlcov", "*.egg-info")
        $skip = @(".venv", ".git", "data", "mlruns")
        # Top-level matches plus nested ones outside skipped roots.
        $targets = @()
        foreach ($pat in $names) {
            $targets += Get-ChildItem -Path . -Directory -Force -Filter $pat -ErrorAction SilentlyContinue
        }
        foreach ($dir in (Get-ChildItem -Path . -Directory -Force | Where-Object { $skip -notcontains $_.Name })) {
            foreach ($pat in $names) {
                $targets += Get-ChildItem -Path $dir.FullName -Directory -Force -Recurse -Filter $pat -ErrorAction SilentlyContinue
            }
        }
        if (Test-Path ".coverage") { $targets += Get-Item ".coverage" -Force }
        foreach ($t in ($targets | Sort-Object FullName -Unique)) {
            Write-Host ("removing " + $t.FullName)
            if ((-not $DryRun) -and (Test-Path -LiteralPath $t.FullName)) { Remove-Item -Recurse -Force -LiteralPath $t.FullName }
        }
    }
    default {
        Write-Host "Unknown task '$Task'. Run .\tasks.ps1 help"
        exit 2
    }
}
exit 0
