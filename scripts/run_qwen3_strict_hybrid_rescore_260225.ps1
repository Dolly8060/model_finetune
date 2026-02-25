param(
    [switch]$UseCondaRun,
    [switch]$DryRun,
    [switch]$ForceRerun,
    [switch]$StrictOnly,
    [string]$CondaEnv = "granite_ft",
    [string]$OutputRoot = "D:\AI_code\model_finetune\evaluation\performance\strict_hybrid_260225"
)

$ErrorActionPreference = "Stop"

# UTF-8 console / python IO to avoid Windows GBK issues
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$repoRoot = Split-Path $PSScriptRoot -Parent
$scoreScript = Join-Path $repoRoot "scripts\score.py"

if (-not (Test-Path $scoreScript)) {
    throw "score.py not found: $scoreScript"
}

$jobs = @(
    @{ Name="base_labeled";      Input=(Join-Path $repoRoot "evaluation\output_data\strict_base\strict_base_labeled.json");                Out=(Join-Path $OutputRoot "base_labeled") },
    @{ Name="base_if_unlabeled"; Input=(Join-Path $repoRoot "evaluation\output_data\strict_base\strict_base_if_unlabeled.json");           Out=(Join-Path $OutputRoot "base_if_unlabeled") },

    @{ Name="v1a_s2026_labeled";      Input=(Join-Path $repoRoot "evaluation\output_data\strict_3seed\v1a_s2026_labeled.json");           Out=(Join-Path $OutputRoot "v1a_s2026_labeled") },
    @{ Name="v1a_s2026_if_unlabeled"; Input=(Join-Path $repoRoot "evaluation\output_data\strict_3seed\v1a_s2026_if_unlabeled.json");      Out=(Join-Path $OutputRoot "v1a_s2026_if_unlabeled") },

    @{ Name="v1b_s2026_labeled";      Input=(Join-Path $repoRoot "evaluation\output_data\strict_3seed\v1b_s2026_labeled.json");           Out=(Join-Path $OutputRoot "v1b_s2026_labeled") },
    @{ Name="v1b_s2026_if_unlabeled"; Input=(Join-Path $repoRoot "evaluation\output_data\strict_3seed\v1b_s2026_if_unlabeled.json");      Out=(Join-Path $OutputRoot "v1b_s2026_if_unlabeled") },

    @{ Name="v1c_s2026_labeled";      Input=(Join-Path $repoRoot "evaluation\output_data\strict_3seed\v1c_s2026_labeled.json");           Out=(Join-Path $OutputRoot "v1c_s2026_labeled") },
    @{ Name="v1c_s2026_if_unlabeled"; Input=(Join-Path $repoRoot "evaluation\output_data\strict_3seed\v1c_s2026_if_unlabeled.json");      Out=(Join-Path $OutputRoot "v1c_s2026_if_unlabeled") },

    @{ Name="strict_planA_tsif_v1_labeled";      Input=(Join-Path $repoRoot "evaluation\output_data\planA_eval\tsif_v1_routed_labeled_merged.json"); Out=(Join-Path $OutputRoot "strict_planA_tsif_v1_labeled") },
    @{ Name="strict_planA_tsif_v1_if_unlabeled"; Input=(Join-Path $repoRoot "evaluation\output_data\planA_eval\tsif_v1_if_unlabeled.json");          Out=(Join-Path $OutputRoot "strict_planA_tsif_v1_if_unlabeled") }
)

function Invoke-ScoreJob {
    param(
        [Parameter(Mandatory=$true)]$Job
    )

    $inputFile = $Job.Input
    $outDir = $Job.Out
    $name = $Job.Name
    $evalJson = Join-Path $outDir "eval_results.json"

    if (-not (Test-Path $inputFile)) {
        Write-Warning "MISSING INPUT [$name] $inputFile"
        return [pscustomobject]@{ Job=$name; Status="MISSING_INPUT"; ExitCode=$null; Output=$outDir }
    }

    if ((Test-Path $evalJson) -and (-not $ForceRerun)) {
        Write-Host "SKIP [$name] -> $evalJson" -ForegroundColor Yellow
        return [pscustomobject]@{ Job=$name; Status="SKIPPED"; ExitCode=0; Output=$outDir }
    }

    New-Item -ItemType Directory -Force -Path $outDir | Out-Null

    $args = @($scoreScript, "--input-file", $inputFile, "--output-dir", $outDir)
    if (-not $StrictOnly) {
        $args += "--enable-lria-fallback"
    }

    if ($UseCondaRun) {
        $cmd = @("conda","run","-n",$CondaEnv,"python") + $args
    } else {
        $cmd = @("python") + $args
    }

    Write-Host ("RUN  [{0}] {1}" -f $name, ($cmd -join " ")) -ForegroundColor Cyan
    if ($DryRun) {
        return [pscustomobject]@{ Job=$name; Status="DRYRUN"; ExitCode=0; Output=$outDir }
    }

    & $cmd[0] $cmd[1..($cmd.Count-1)]
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        Write-Host "FAILED [$name] ExitCode=$exitCode" -ForegroundColor Red
        return [pscustomobject]@{ Job=$name; Status="FAILED"; ExitCode=$exitCode; Output=$outDir }
    }

    Write-Host "DONE  [$name]" -ForegroundColor Green
    return [pscustomobject]@{ Job=$name; Status="SUCCESS"; ExitCode=0; Output=$outDir }
}

Write-Host "==== QWEN3 Strict Re-Score (260225) ====" -ForegroundColor Green
Write-Host ("Mode: {0}" -f ($(if ($StrictOnly) { "STRICT" } else { "HYBRID (LRIA fallback enabled)" })))
Write-Host ("OutputRoot: {0}" -f $OutputRoot)
Write-Host ("Runner: {0}" -f ($(if ($UseCondaRun) { "conda run -n $CondaEnv python" } else { "python (current env)" })))

$results = @()
foreach ($job in $jobs) {
    $results += Invoke-ScoreJob -Job $job
}

Write-Host ""
Write-Host "==================== RESCORE SUMMARY ====================" -ForegroundColor Green
$results | Format-Table -AutoSize

$failed = @($results | Where-Object { $_.Status -eq "FAILED" -or $_.Status -eq "MISSING_INPUT" })
if ($failed.Count -gt 0) {
    exit 1
}
exit 0

