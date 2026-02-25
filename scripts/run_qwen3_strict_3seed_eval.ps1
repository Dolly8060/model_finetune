param(
    [switch]$UseCondaRun,
    [string]$CondaEnv = "granite_ft",
    [string]$BaseModelPath = "D:/AI_code/models/Qwen3-1.7B",
    [string]$LabeledEvalFile = "data/qwen3_strict_test_labeled.json",
    [string]$IfUnlabeledEvalFile = "data/qwen3_strict_test_if_unlabeled.json",
    [string]$OutputDataDir = "evaluation/output_data/strict_3seed",
    [string]$EvalDir = "evaluation/strict_3seed",
    [switch]$HybridIF,
    [int]$MaxInputLength = 2048,
    [int]$MaxNewTokens = 512,
    [int[]]$Seeds = @(2026),
    [string[]]$OnlyVersions = @(),
    [switch]$ContinueOnError,
    [switch]$ForceRerun,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$LogDir = Join-Path $ProjectRoot "logs"
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null

# Windows console encoding guard for Python subprocess output.
try {
    [Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false)
    [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
    $OutputEncoding = [System.Text.UTF8Encoding]::new($false)
} catch {}
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$VersionDefs = @(
    @{ Version = "v1a"; AdapterBase = "outputs/qwen3-1.7B-lora-strict-v1a" },
    @{ Version = "v1b"; AdapterBase = "outputs/qwen3-1.7B-lora-strict-v1b" },
    @{ Version = "v1c"; AdapterBase = "outputs/qwen3-1.7B-lora-strict-v1c" }
)

if ($OnlyVersions.Count -gt 0) {
    $OnlySet = @{}
    foreach ($v in $OnlyVersions) { $OnlySet[$v.ToLower()] = $true }
    $VersionDefs = @($VersionDefs | Where-Object { $OnlySet.ContainsKey($_.Version.ToLower()) })
}
if ($VersionDefs.Count -eq 0) { throw "No versions selected. Valid: v1a, v1b, v1c" }

if ($UseCondaRun -and -not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "conda command not found. Ensure conda is in PATH, or run without -UseCondaRun inside an activated env."
}

function Ensure-Path {
    param([string]$PathToCheck, [string]$Label)
    if (-not (Test-Path $PathToCheck)) { throw "$Label not found: $PathToCheck" }
}

function Invoke-Step {
    param([string]$StepName, [string]$Cmd, [string[]]$CmdArgs, [string]$LogFile)
    Write-Host ""
    Write-Host "[$StepName] $Cmd $($CmdArgs -join ' ')"

    if ($DryRun) {
        Add-Content -Path $LogFile -Value ("DRYRUN > {0} {1}" -f $Cmd, ($CmdArgs -join ' '))
        return 0
    }

    $PrevEAP = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        if ($UseCondaRun) {
            & conda run -n $CondaEnv $Cmd @CmdArgs 2>&1 | Tee-Object -FilePath $LogFile -Append | Out-Host
        } else {
            & $Cmd @CmdArgs 2>&1 | Tee-Object -FilePath $LogFile -Append | Out-Host
        }
    } finally {
        $ErrorActionPreference = $PrevEAP
    }
    return [int]$LASTEXITCODE
}

function Get-EvalCompletionState {
    param([string]$AbsEvalDir)
    $p = Join-Path $AbsEvalDir "eval_results.json"
    if (Test-Path $p) { return @{ Completed = $true; Reason = "eval_results.json exists" } }
    return @{ Completed = $false; Reason = "eval_results.json missing" }
}

$AbsLabeledEvalFile = Join-Path $ProjectRoot $LabeledEvalFile
$AbsIfUnlabeledEvalFile = Join-Path $ProjectRoot $IfUnlabeledEvalFile
$AbsOutputDataDir = Join-Path $ProjectRoot $OutputDataDir
$ResolvedEvalDir = if ($HybridIF) { "$EvalDir`_hybrid" } else { $EvalDir }
$AbsEvalDir = Join-Path $ProjectRoot $ResolvedEvalDir
$ScoreExtraArgs = @()
if ($HybridIF) { $ScoreExtraArgs += "--enable-lria-fallback" }

Ensure-Path -PathToCheck $BaseModelPath -Label "Base model path"
Ensure-Path -PathToCheck $AbsLabeledEvalFile -Label "Labeled eval file"
Ensure-Path -PathToCheck $AbsIfUnlabeledEvalFile -Label "IF unlabeled eval file"
New-Item -ItemType Directory -Path $AbsOutputDataDir -Force | Out-Null
New-Item -ItemType Directory -Path $AbsEvalDir -Force | Out-Null

Write-Host ("IF scoring mode: {0}" -f ($(if ($HybridIF) { "HYBRID (LRIA fallback)" } else { "STRICT" })))
Write-Host ("Eval output dir: {0}" -f $ResolvedEvalDir)

$Jobs = @()
foreach ($vd in $VersionDefs) {
    foreach ($s in $Seeds) {
        $Jobs += @{
            Name = "$($vd.Version)_s$s"
            Version = $vd.Version
            Seed = $s
            Adapter = ("{0}-s{1}" -f $vd.AdapterBase, $s)
        }
    }
}

$Results = @()

foreach ($Job in $Jobs) {
    $Name = $Job.Name
    $AbsAdapter = Join-Path $ProjectRoot $Job.Adapter
    Ensure-Path -PathToCheck $AbsAdapter -Label "Adapter path for $Name"

    $LabeledOutputFile = Join-Path $AbsOutputDataDir "$Name`_labeled.json"
    $IfOutputFile = Join-Path $AbsOutputDataDir "$Name`_if_unlabeled.json"
    $LabeledOutDir = Join-Path $AbsEvalDir "$Name`_labeled"
    $IfOutDir = Join-Path $AbsEvalDir "$Name`_if_unlabeled"
    New-Item -ItemType Directory -Path $LabeledOutDir -Force | Out-Null
    New-Item -ItemType Directory -Path $IfOutDir -Force | Out-Null

    if (-not $ForceRerun) {
        $LDone = Get-EvalCompletionState -AbsEvalDir $LabeledOutDir
        $UDone = Get-EvalCompletionState -AbsEvalDir $IfOutDir
        if ($LDone.Completed -and $UDone.Completed) {
            Write-Host ""
            Write-Host "Skip eval: $Name ($($LDone.Reason); $($UDone.Reason))"
            $Results += [PSCustomObject]@{ Model=$Name; Step="all"; ExitCode=0; Status="SKIPPED"; LogFile="-" }
            continue
        }
    }

    $Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $LogFile = Join-Path $LogDir "eval_strict_$Name`_$Stamp.log"
    if ($DryRun -and -not (Test-Path $LogFile)) { New-Item -ItemType File -Path $LogFile | Out-Null }

    $ModelSpec = "FTQwen3_{0}:{1}:{2}" -f $Name.ToUpper(), $BaseModelPath, $AbsAdapter
    Write-Host ""
    Write-Host "============================================================"
    Write-Host "Start evaluation: $Name"
    Write-Host "ModelSpec: $ModelSpec"
    Write-Host "Log: $LogFile"
    Write-Host "============================================================"

    $Failed = $false

    $Code = Invoke-Step -StepName "$Name.generate_labeled" -Cmd "python" -CmdArgs @(
        "scripts/generate.py",
        "--models", $ModelSpec,
        "--eval-file", $AbsLabeledEvalFile,
        "--output-file", $LabeledOutputFile,
        "--max-input-length", "$MaxInputLength",
        "--max-new-tokens", "$MaxNewTokens"
    ) -LogFile $LogFile
    $Results += [PSCustomObject]@{ Model=$Name; Step="generate_labeled"; ExitCode=$Code; Status=$(if($Code -eq 0){"SUCCESS"}else{"FAILED"}); LogFile=$LogFile }
    if ($Code -ne 0) { $Failed = $true }

    if (-not $Failed) {
        $ScoreArgs = @(
            "scripts/score.py",
            "--input-file", $LabeledOutputFile,
            "--output-dir", $LabeledOutDir
        ) + $ScoreExtraArgs
        $Code = Invoke-Step -StepName "$Name.score_labeled" -Cmd "python" -CmdArgs $ScoreArgs -LogFile $LogFile
        $Results += [PSCustomObject]@{ Model=$Name; Step="score_labeled"; ExitCode=$Code; Status=$(if($Code -eq 0){"SUCCESS"}else{"FAILED"}); LogFile=$LogFile }
        if ($Code -ne 0) { $Failed = $true }
    }

    if (-not $Failed) {
        $Code = Invoke-Step -StepName "$Name.generate_if_unlabeled" -Cmd "python" -CmdArgs @(
            "scripts/generate.py",
            "--models", $ModelSpec,
            "--eval-file", $AbsIfUnlabeledEvalFile,
            "--output-file", $IfOutputFile,
            "--max-input-length", "$MaxInputLength",
            "--max-new-tokens", "$MaxNewTokens"
        ) -LogFile $LogFile
        $Results += [PSCustomObject]@{ Model=$Name; Step="generate_if_unlabeled"; ExitCode=$Code; Status=$(if($Code -eq 0){"SUCCESS"}else{"FAILED"}); LogFile=$LogFile }
        if ($Code -ne 0) { $Failed = $true }
    }

    if (-not $Failed) {
        $ScoreArgs = @(
            "scripts/score.py",
            "--input-file", $IfOutputFile,
            "--output-dir", $IfOutDir
        ) + $ScoreExtraArgs
        $Code = Invoke-Step -StepName "$Name.score_if_unlabeled" -Cmd "python" -CmdArgs $ScoreArgs -LogFile $LogFile
        $Results += [PSCustomObject]@{ Model=$Name; Step="score_if_unlabeled"; ExitCode=$Code; Status=$(if($Code -eq 0){"SUCCESS"}else{"FAILED"}); LogFile=$LogFile }
        if ($Code -ne 0) { $Failed = $true }
    }

    if ($Failed) {
        Write-Host "Evaluation failed for $Name."
        if (-not $ContinueOnError) { break }
        Write-Host "ContinueOnError is enabled. Continue to next model."
    } else {
        Write-Host "Evaluation finished for $Name."
    }
}

Write-Host ""
Write-Host "===================== STRICT 3-SEED EVAL SUMMARY ===================="
$Results | Format-Table -AutoSize
Write-Host "====================================================================="

$HasFailure = $Results | Where-Object { $_.ExitCode -ne 0 }
if ($HasFailure) { exit 1 }
exit 0
