param(
    [switch]$UseCondaRun,
    [switch]$DryRun,
    [switch]$Force,
    [int[]]$Seeds = @(2026, 2027, 2028),
    [string[]]$MixedVersions = @("v1a", "v1b", "v1c"),
    [string[]]$PlanAVersions = @("tsif_v1"),
    [string]$CondaEnv = "granite_ft",
    [string]$ProjectRoot = "."
)

$ErrorActionPreference = "Stop"

[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$ProjectRoot = (Resolve-Path $ProjectRoot).Path
Set-Location $ProjectRoot

function Invoke-Aggregate {
    param(
        [string]$EvalDir,
        [string[]]$Versions,
        [int[]]$Seeds,
        [string]$OutputFile
    )

    $absEvalDir = Join-Path $ProjectRoot $EvalDir
    if (-not (Test-Path $absEvalDir)) {
        Write-Host "[SKIP] Missing eval dir: $EvalDir"
        return
    }

    if ((-not $Force) -and (Test-Path $OutputFile)) {
        Write-Host "[SKIP] Aggregate exists: $OutputFile (use -Force to overwrite)"
        return
    }

    $args = @(
        "scripts/aggregate_qwen3_strict_3seed.py",
        "--eval-dir", $EvalDir,
        "--output", $OutputFile
    )

    if ($Versions -and $Versions.Count -gt 0) {
        $args += "--versions"
        $args += $Versions
    }

    if ($Seeds -and $Seeds.Count -gt 0) {
        $args += "--seeds"
        $args += ($Seeds | ForEach-Object { "$_" })
    }

    if ($UseCondaRun) {
        $cmd = "conda"
        $cmdArgs = @("run", "-n", $CondaEnv, "python") + $args
    } else {
        $cmd = "python"
        $cmdArgs = $args
    }

    Write-Host "------------------------------------------------------------"
    Write-Host ("Aggregate eval dir: {0}" -f $EvalDir)
    Write-Host ("Output: {0}" -f $OutputFile)
    Write-Host ("Versions: {0}" -f ($Versions -join ", "))
    Write-Host ("Seeds: {0}" -f ($Seeds -join ", "))
    Write-Host "Command:"
    Write-Host ("  {0} {1}" -f $cmd, ($cmdArgs -join " "))

    if ($DryRun) { return }

    & $cmd @cmdArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Aggregate failed for $EvalDir (exit=$LASTEXITCODE)"
    }
}

$jobs = @(
    [PSCustomObject]@{
        EvalDir = "evaluation/strict_3seed"
        Versions = $MixedVersions
        OutputFile = "evaluation/strict_3seed/aggregate_3seed_summary.json"
    },
    [PSCustomObject]@{
        EvalDir = "evaluation/strict_3seed_hybrid"
        Versions = $MixedVersions
        OutputFile = "evaluation/strict_3seed_hybrid/aggregate_3seed_summary.json"
    },
    [PSCustomObject]@{
        EvalDir = "evaluation/strict_planA"
        Versions = $PlanAVersions
        OutputFile = "evaluation/strict_planA/aggregate_tsif_3seed_summary.json"
    },
    [PSCustomObject]@{
        EvalDir = "evaluation/strict_planA_hybrid"
        Versions = $PlanAVersions
        OutputFile = "evaluation/strict_planA_hybrid/aggregate_tsif_3seed_summary.json"
    }
)

Write-Host "==================== QWEN3 STRICT AUTO AGGREGATE ===================="
Write-Host ("ProjectRoot: {0}" -f $ProjectRoot)
Write-Host ("Mode: {0}" -f $(if ($UseCondaRun) { "conda:$CondaEnv" } else { "current env" }))
Write-Host ("Force: {0} | DryRun: {1}" -f $Force, $DryRun)

foreach ($job in $jobs) {
    Invoke-Aggregate -EvalDir $job.EvalDir -Versions $job.Versions -Seeds $Seeds -OutputFile $job.OutputFile
}

Write-Host "==================== AUTO AGGREGATE DONE ===================="

