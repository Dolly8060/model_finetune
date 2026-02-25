param(
    [switch]$UseCondaRun,
    [string]$CondaEnv = "granite_ft",
    [int[]]$Seeds = @(2026, 2027, 2028),
    [string[]]$OnlyVersions = @(),
    [switch]$ContinueOnError,
    [switch]$ForceRerun,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$LogDir = Join-Path $ProjectRoot "logs"
$GeneratedConfigDir = Join-Path $ProjectRoot "configs/generated_strict_3seed"
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
New-Item -ItemType Directory -Path $GeneratedConfigDir -Force | Out-Null

# Windows console encoding guard for Python/llamafactory sample printing.
# Prevents UnicodeEncodeError when decoded examples contain non-GBK chars.
try {
    [Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false)
    [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
    $OutputEncoding = [System.Text.UTF8Encoding]::new($false)
} catch {}
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$VersionDefs = @(
    @{ Version = "v1a"; Config = "configs/finetune_qwen3_lora_strict_v1a.yaml"; OutputDir = "outputs/qwen3-1.7B-lora-strict-v1a"; TargetEpoch = 3.0 },
    @{ Version = "v1b"; Config = "configs/finetune_qwen3_lora_strict_v1b.yaml"; OutputDir = "outputs/qwen3-1.7B-lora-strict-v1b"; TargetEpoch = 3.0 },
    @{ Version = "v1c"; Config = "configs/finetune_qwen3_lora_strict_v1c.yaml"; OutputDir = "outputs/qwen3-1.7B-lora-strict-v1c"; TargetEpoch = 3.0 }
)

if ($OnlyVersions.Count -gt 0) {
    $OnlySet = @{}
    foreach ($v in $OnlyVersions) { $OnlySet[$v.ToLower()] = $true }
    $VersionDefs = @($VersionDefs | Where-Object { $OnlySet.ContainsKey($_.Version.ToLower()) })
}
if ($VersionDefs.Count -eq 0) {
    throw "No versions selected. Valid: v1a, v1b, v1c"
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
foreach ($vd in $VersionDefs) {
    foreach ($s in $Seeds) {
        $seededOutDir = "{0}-s{1}" -f $vd.OutputDir, $s
        $seededConfigName = [System.IO.Path]::GetFileNameWithoutExtension($vd.Config) + "_s$($s).yaml"
        $seededConfigRel = "configs/generated_strict_3seed/$seededConfigName"
        $Jobs += @{
            Name = "$($vd.Version)_s$s"
            Version = $vd.Version
            Seed = $s
            BaseConfig = $vd.Config
            Config = $seededConfigRel
            OutputDir = $seededOutDir
            TargetEpoch = $vd.TargetEpoch
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

    New-SeededConfig -BaseConfigPath $BaseConfigPath -OutConfigPath $ConfigPath -SeededOutputDir $Job.OutputDir -Seed $Job.Seed

    $Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $LogFile = Join-Path $LogDir "train_strict_$($Job.Name)_$Stamp.log"

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "Start training: $($Job.Name)"
    Write-Host "Base config: $BaseConfigPath"
    Write-Host "Seeded config: $ConfigPath"
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
Write-Host "==================== STRICT 3-SEED TRAIN SUMMARY ===================="
$Results | Format-Table -AutoSize
Write-Host "====================================================================="

$HasFailure = $Results | Where-Object { $_.ExitCode -ne 0 }
if ($HasFailure) { exit 1 }
exit 0
