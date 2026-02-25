param(
    [switch]$UseCondaRun,
    [string]$CondaEnv = "granite_ft",
    [switch]$ThreeSeed,
    [int[]]$Seeds = @(2026, 2027, 2028),
    [string[]]$OnlyAdapters = @(),
    [switch]$ContinueOnError,
    [switch]$ForceRerun,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$LogDir = Join-Path $ProjectRoot "logs"
$GeneratedConfigDir = Join-Path $ProjectRoot "configs/generated_strict_planA_3seed"
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
New-Item -ItemType Directory -Path $GeneratedConfigDir -Force | Out-Null

# Windows console encoding guard for Python/llamafactory sample printing.
try {
    [Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false)
    [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
    $OutputEncoding = [System.Text.UTF8Encoding]::new($false)
} catch {}
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$AdapterDefs = @(
    @{ Adapter = "ts"; Version = "ts_v1"; Config = "configs/finetune_qwen3_lora_strict_ts_v1.yaml"; OutputDir = "outputs/qwen3-1.7B-lora-strict-ts-v1"; TargetEpoch = 3.0 },
    @{ Adapter = "if"; Version = "if_v1"; Config = "configs/finetune_qwen3_lora_strict_if_v1.yaml"; OutputDir = "outputs/qwen3-1.7B-lora-strict-if-v1"; TargetEpoch = 3.0 }
)

if ($OnlyAdapters.Count -gt 0) {
    $OnlySet = @{}
    foreach ($a in $OnlyAdapters) { $OnlySet[$a.ToLower()] = $true }
    $AdapterDefs = @($AdapterDefs | Where-Object { $OnlySet.ContainsKey($_.Adapter.ToLower()) })
}
if ($AdapterDefs.Count -eq 0) {
    throw "No adapters selected. Valid: ts, if"
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

function New-SeededConfig {
    param(
        [string]$BaseConfigPath,
        [string]$OutConfigPath,
        [string]$SeededOutputDir,
        [int]$Seed
    )
    $lines = Get-Content $BaseConfigPath
    $updated = @()
    $FoundOutput = $false
    $FoundSeed = $false
    foreach ($line in $lines) {
        if ($line -match '^\s*output_dir\s*:') {
            $updated += ("output_dir: ./{0}" -f ($SeededOutputDir -replace '\\','/'))
            $FoundOutput = $true
            continue
        }
        if ($line -match '^\s*seed\s*:') {
            $updated += ("seed: {0}" -f $Seed)
            $FoundSeed = $true
            continue
        }
        $updated += $line
    }
    if (-not $FoundOutput) {
        throw "output_dir not found in base config: $BaseConfigPath"
    }
    if (-not $FoundSeed) {
        $updated += ""
        $updated += "### Reproducibility"
        $updated += ("seed: {0}" -f $Seed)
    }
    Set-Content -Path $OutConfigPath -Value $updated -Encoding UTF8
}

$Jobs = @()
if ($ThreeSeed) {
    foreach ($ad in $AdapterDefs) {
        foreach ($s in $Seeds) {
            $seededOutDir = "{0}-s{1}" -f $ad.OutputDir, $s
            $seededConfigName = [System.IO.Path]::GetFileNameWithoutExtension($ad.Config) + "_s$($s).yaml"
            $seededConfigRel = "configs/generated_strict_planA_3seed/$seededConfigName"
            $Jobs += @{
                Name = "$($ad.Adapter)_s$s"
                Adapter = $ad.Adapter
                Seed = $s
                BaseConfig = $ad.Config
                Config = $seededConfigRel
                OutputDir = $seededOutDir
                TargetEpoch = $ad.TargetEpoch
                Generated = $true
            }
        }
    }
} else {
    foreach ($ad in $AdapterDefs) {
        $Jobs += @{
            Name = "$($ad.Adapter)_single"
            Adapter = $ad.Adapter
            Seed = $null
            BaseConfig = $ad.Config
            Config = $ad.Config
            OutputDir = $ad.OutputDir
            TargetEpoch = $ad.TargetEpoch
            Generated = $false
        }
    }
}

$Results = @()

foreach ($Job in $Jobs) {
    $BaseConfigPath = Join-Path $ProjectRoot $Job.BaseConfig
    $ConfigPath = Join-Path $ProjectRoot $Job.Config
    $AbsOutputDir = Join-Path $ProjectRoot $Job.OutputDir
    if (-not (Test-Path $BaseConfigPath)) { throw "Base config file not found: $BaseConfigPath" }

    if (-not $ForceRerun) {
        $Completion = Get-TrainingCompletionState -AbsOutputDir $AbsOutputDir -TargetEpoch $Job.TargetEpoch
        if ($Completion.Completed) {
            Write-Host ""
            Write-Host "Skip training: $($Job.Name) ($($Completion.Reason))"
            $Results += [PSCustomObject]@{ Job=$Job.Name; ExitCode=0; Status="SKIPPED"; LogFile="-"; Config=$Job.Config }
            continue
        }
    }

    if ($Job.Generated) {
        New-SeededConfig -BaseConfigPath $BaseConfigPath -OutConfigPath $ConfigPath -SeededOutputDir $Job.OutputDir -Seed $Job.Seed
    }

    $Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $LogFile = Join-Path $LogDir "train_planA_$($Job.Name)_$Stamp.log"

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "Start Plan A training: $($Job.Name)"
    Write-Host "Mode: $(if($ThreeSeed){'3-seed'}else{'1-seed'})"
    Write-Host "Base config: $BaseConfigPath"
    if ($Job.Generated) { Write-Host "Seeded config: $ConfigPath" } else { Write-Host "Config: $ConfigPath" }
    Write-Host "OutputDir: $AbsOutputDir"
    Write-Host "Log: $LogFile"
    Write-Host "============================================================"

    if ($DryRun) {
        Write-Host "DRYRUN > llamafactory-cli train $ConfigPath"
        $Results += [PSCustomObject]@{ Job=$Job.Name; ExitCode=0; Status="DRYRUN"; LogFile=$LogFile; Config=$Job.Config }
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
    $Results += [PSCustomObject]@{ Job=$Job.Name; ExitCode=$ExitCode; Status=$Status; LogFile=$LogFile; Config=$Job.Config }

    if ($ExitCode -ne 0) {
        Write-Host "Training failed: $($Job.Name), ExitCode=$ExitCode"
        if (-not $ContinueOnError) { break }
        Write-Host "ContinueOnError is enabled. Continue to next job."
    } else {
        Write-Host "Training finished: $($Job.Name)"
    }
}

Write-Host ""
Write-Host "================= PLAN A TRAIN SUMMARY ================="
$Results | Format-Table -AutoSize
Write-Host "========================================================"

$HasFailure = $Results | Where-Object { $_.ExitCode -ne 0 }
if ($HasFailure) { exit 1 }
exit 0

