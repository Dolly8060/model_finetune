param(
    [switch]$UseCondaRun,
    [string]$CondaEnv = "granite_ft",
    [switch]$ContinueOnError,
    [switch]$ForceRerun
)

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$LogDir = Join-Path $ProjectRoot "logs"
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null

$Jobs = @(
    @{ Name = "v2a"; Config = "configs/finetune_qwen3_lora_v2a.yaml"; OutputDir = "outputs/qwen3-1.7B-lora-v2a"; TargetEpoch = 3.0 },
    @{ Name = "v2b"; Config = "configs/finetune_qwen3_lora_v2b.yaml"; OutputDir = "outputs/qwen3-1.7B-lora-v2b"; TargetEpoch = 3.0 },
    @{ Name = "v2c"; Config = "configs/finetune_qwen3_lora_v2c.yaml"; OutputDir = "outputs/qwen3-1.7B-lora-v2c"; TargetEpoch = 3.0 }
)

if ($UseCondaRun) {
    if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
        throw "conda command not found. Ensure conda is in PATH, or run without -UseCondaRun inside an activated env."
    }
}

$Results = @()

function Get-TrainingCompletionState {
    param(
        [string]$AbsOutputDir,
        [double]$TargetEpoch = 3.0
    )

    if (-not (Test-Path $AbsOutputDir)) {
        return @{ Completed = $false; Reason = "output dir missing" }
    }

    $TrainResultsPath = Join-Path $AbsOutputDir "train_results.json"
    if (Test-Path $TrainResultsPath) {
        try {
            $TrainResults = Get-Content $TrainResultsPath -Raw | ConvertFrom-Json
            if ($null -ne $TrainResults.epoch -and [double]$TrainResults.epoch -ge $TargetEpoch) {
                return @{ Completed = $true; Reason = "train_results.json epoch=$($TrainResults.epoch)" }
            }
        }
        catch {
            return @{ Completed = $false; Reason = "train_results.json parse failed" }
        }
    }

    $TrainerStatePath = Join-Path $AbsOutputDir "trainer_state.json"
    if (Test-Path $TrainerStatePath) {
        try {
            $TrainerState = Get-Content $TrainerStatePath -Raw | ConvertFrom-Json
            if ($null -ne $TrainerState.epoch -and [double]$TrainerState.epoch -ge $TargetEpoch) {
                return @{ Completed = $true; Reason = "trainer_state.json epoch=$($TrainerState.epoch)" }
            }
        }
        catch {
            return @{ Completed = $false; Reason = "trainer_state.json parse failed" }
        }
    }

    return @{ Completed = $false; Reason = "no completed epoch evidence" }
}

foreach ($Job in $Jobs) {
    $ConfigPath = Join-Path $ProjectRoot $Job.Config
    if (-not (Test-Path $ConfigPath)) {
        throw "Config file not found: $ConfigPath"
    }
    $AbsOutputDir = Join-Path $ProjectRoot $Job.OutputDir

    if (-not $ForceRerun) {
        $Completion = Get-TrainingCompletionState -AbsOutputDir $AbsOutputDir -TargetEpoch $Job.TargetEpoch
        if ($Completion.Completed) {
            Write-Host ""
            Write-Host "Skip training: $($Job.Name) ($($Completion.Reason))"
            $Results += [PSCustomObject]@{
                Job      = $Job.Name
                ExitCode = 0
                Status   = "SKIPPED"
                LogFile  = "-"
            }
            continue
        }
    }

    $Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $LogFile = Join-Path $LogDir "train_$($Job.Name)_$Stamp.log"

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "Start training: $($Job.Name)"
    Write-Host "Config: $ConfigPath"
    Write-Host "Log: $LogFile"
    Write-Host "Start time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    Write-Host "============================================================"

    # Native tools may write warnings to stderr; do not abort on stderr text.
    # We only use process exit code to decide success/failure.
    $PrevEAP = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        if ($UseCondaRun) {
            & conda run -n $CondaEnv llamafactory-cli train $ConfigPath 2>&1 | Tee-Object -FilePath $LogFile
        }
        else {
            & llamafactory-cli train $ConfigPath 2>&1 | Tee-Object -FilePath $LogFile
        }
    }
    finally {
        $ErrorActionPreference = $PrevEAP
    }

    $ExitCode = $LASTEXITCODE
    $Status = if ($ExitCode -eq 0) { "SUCCESS" } else { "FAILED" }

    $Results += [PSCustomObject]@{
        Job      = $Job.Name
        ExitCode = $ExitCode
        Status   = $Status
        LogFile  = $LogFile
    }

    if ($ExitCode -ne 0) {
        Write-Host ""
        Write-Host "Training failed: $($Job.Name), ExitCode=$ExitCode"
        if (-not $ContinueOnError) {
            break
        }
        Write-Host "ContinueOnError is enabled. Continue to next job."
    }
    else {
        Write-Host "Training finished: $($Job.Name)"
    }
}

Write-Host ""
Write-Host "======================= TRAIN SUMMARY ======================="
$Results | Format-Table -AutoSize
Write-Host "======================================================"

$HasFailure = $Results | Where-Object { $_.ExitCode -ne 0 }
if ($HasFailure) {
    exit 1
}

exit 0
