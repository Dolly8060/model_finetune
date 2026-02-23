param(
    [switch]$UseCondaRun,
    [string]$CondaEnv = "granite_ft",
    [switch]$ContinueOnError,
    [switch]$ForceRerun,
    [switch]$DryRun,
    [string[]]$Only = @()
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$LogDir = Join-Path $ProjectRoot "logs"
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null

$Jobs = @(
    @{ Name = "v3b1"; Config = "configs/finetune_qwen3_lora_v3b1.yaml"; OutputDir = "outputs/qwen3-1.7B-lora-v3b1"; TargetEpoch = 3.0 },
    @{ Name = "v3b2"; Config = "configs/finetune_qwen3_lora_v3b2.yaml"; OutputDir = "outputs/qwen3-1.7B-lora-v3b2"; TargetEpoch = 3.0 },
    @{ Name = "v3b3"; Config = "configs/finetune_qwen3_lora_v3b3.yaml"; OutputDir = "outputs/qwen3-1.7B-lora-v3b3"; TargetEpoch = 2.4 }
)

if ($Only.Count -gt 0) {
    $OnlySet = @{}
    foreach ($n in $Only) { $OnlySet[$n.ToLower()] = $true }
    $Jobs = @($Jobs | Where-Object { $OnlySet.ContainsKey($_.Name.ToLower()) })
}

if ($Jobs.Count -eq 0) {
    throw "No jobs selected. Check -Only values (valid: v3b1, v3b2, v3b3)."
}

if ($UseCondaRun -and -not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "conda command not found. Ensure conda is in PATH, or run without -UseCondaRun inside an activated env."
}

function Get-TrainingCompletionState {
    param([string]$AbsOutputDir, [double]$TargetEpoch)
    if (-not (Test-Path $AbsOutputDir)) { return @{ Completed = $false; Reason = "output dir missing" } }

    foreach ($fn in @("train_results.json","trainer_state.json")) {
        $p = Join-Path $AbsOutputDir $fn
        if (-not (Test-Path $p)) { continue }
        try {
            $obj = Get-Content $p -Raw | ConvertFrom-Json
            if ($null -ne $obj.epoch -and [double]$obj.epoch -ge $TargetEpoch) {
                return @{ Completed = $true; Reason = "$fn epoch=$($obj.epoch)" }
            }
        } catch {
            return @{ Completed = $false; Reason = "$fn parse failed" }
        }
    }
    return @{ Completed = $false; Reason = "no completed epoch evidence" }
}

$Results = @()

foreach ($Job in $Jobs) {
    $ConfigPath = Join-Path $ProjectRoot $Job.Config
    $AbsOutputDir = Join-Path $ProjectRoot $Job.OutputDir
    if (-not (Test-Path $ConfigPath)) { throw "Config file not found: $ConfigPath" }

    if (-not $ForceRerun) {
        $Completion = Get-TrainingCompletionState -AbsOutputDir $AbsOutputDir -TargetEpoch $Job.TargetEpoch
        if ($Completion.Completed) {
            Write-Host ""
            Write-Host "Skip training: $($Job.Name) ($($Completion.Reason))"
            $Results += [PSCustomObject]@{ Job=$Job.Name; ExitCode=0; Status="SKIPPED"; LogFile="-" }
            continue
        }
    }

    $Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $LogFile = Join-Path $LogDir "train_$($Job.Name)_$Stamp.log"

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "Start training: $($Job.Name)"
    Write-Host "Config: $ConfigPath"
    Write-Host "OutputDir: $AbsOutputDir"
    Write-Host "Log: $LogFile"
    Write-Host "============================================================"

    if ($DryRun) {
        Write-Host "DRYRUN > llamafactory-cli train $ConfigPath"
        $Results += [PSCustomObject]@{ Job=$Job.Name; ExitCode=0; Status="DRYRUN"; LogFile=$LogFile }
        continue
    }

    $PrevEAP = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        if ($UseCondaRun) {
            & conda run -n $CondaEnv llamafactory-cli train $ConfigPath 2>&1 | Tee-Object -FilePath $LogFile | Out-Host
        } else {
            & llamafactory-cli train $ConfigPath 2>&1 | Tee-Object -FilePath $LogFile | Out-Host
        }
    } finally {
        $ErrorActionPreference = $PrevEAP
    }

    $ExitCode = [int]$LASTEXITCODE
    $Status = if ($ExitCode -eq 0) { "SUCCESS" } else { "FAILED" }
    $Results += [PSCustomObject]@{ Job=$Job.Name; ExitCode=$ExitCode; Status=$Status; LogFile=$LogFile }

    if ($ExitCode -ne 0) {
        Write-Host "Training failed: $($Job.Name), ExitCode=$ExitCode"
        if (-not $ContinueOnError) { break }
        Write-Host "ContinueOnError is enabled. Continue to next job."
    } else {
        Write-Host "Training finished: $($Job.Name)"
    }
}

Write-Host ""
Write-Host "======================== V3 TRAIN SUMMARY ===================="
$Results | Format-Table -AutoSize
Write-Host "=============================================================="

$HasFailure = $Results | Where-Object { $_.ExitCode -ne 0 }
if ($HasFailure) { exit 1 }
exit 0
