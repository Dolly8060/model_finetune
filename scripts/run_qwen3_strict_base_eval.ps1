param(
    [switch]$UseCondaRun,
    [string]$CondaEnv = "granite_ft",
    [string]$BaseModelPath = "D:/AI_code/models/Qwen3-1.7B",
    [string]$LabeledEvalFile = "data/qwen3_strict_test_labeled.json",
    [string]$IfUnlabeledEvalFile = "data/qwen3_strict_test_if_unlabeled.json",
    [string]$OutputDataDir = "evaluation/output_data/strict_base",
    [string]$EvalDir = "evaluation/strict_3seed",
    [switch]$HybridIF,
    [int]$MaxInputLength = 2048,
    [int]$MaxNewTokens = 512,
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

$Name = "base"
$LabeledOutputFile = Join-Path $AbsOutputDataDir "strict_base_labeled.json"
$IfOutputFile = Join-Path $AbsOutputDataDir "strict_base_if_unlabeled.json"
$LabeledOutDir = Join-Path $AbsEvalDir "base_labeled"
$IfOutDir = Join-Path $AbsEvalDir "base_if_unlabeled"
New-Item -ItemType Directory -Path $LabeledOutDir -Force | Out-Null
New-Item -ItemType Directory -Path $IfOutDir -Force | Out-Null

if (-not $ForceRerun) {
    $LDone = Get-EvalCompletionState -AbsEvalDir $LabeledOutDir
    $UDone = Get-EvalCompletionState -AbsEvalDir $IfOutDir
    if ($LDone.Completed -and $UDone.Completed) {
        Write-Host "Skip eval: base ($($LDone.Reason); $($UDone.Reason))"
        exit 0
    }
}

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile = Join-Path $LogDir "eval_strict_base_$Stamp.log"
if ($DryRun -and -not (Test-Path $LogFile)) { New-Item -ItemType File -Path $LogFile | Out-Null }

$ModelSpec = "BaseQwen3:{0}" -f $BaseModelPath
Write-Host ""
Write-Host "============================================================"
Write-Host "Start strict base evaluation"
Write-Host "ModelSpec: $ModelSpec"
Write-Host "Log: $LogFile"
Write-Host "============================================================"

$Results = @()
$Failed = $false

$Code = Invoke-Step -StepName "base.generate_labeled" -Cmd "python" -CmdArgs @(
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
    $Code = Invoke-Step -StepName "base.score_labeled" -Cmd "python" -CmdArgs $ScoreArgs -LogFile $LogFile
    $Results += [PSCustomObject]@{ Model=$Name; Step="score_labeled"; ExitCode=$Code; Status=$(if($Code -eq 0){"SUCCESS"}else{"FAILED"}); LogFile=$LogFile }
    if ($Code -ne 0) { $Failed = $true }
}

if (-not $Failed) {
    $Code = Invoke-Step -StepName "base.generate_if_unlabeled" -Cmd "python" -CmdArgs @(
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
    $Code = Invoke-Step -StepName "base.score_if_unlabeled" -Cmd "python" -CmdArgs $ScoreArgs -LogFile $LogFile
    $Results += [PSCustomObject]@{ Model=$Name; Step="score_if_unlabeled"; ExitCode=$Code; Status=$(if($Code -eq 0){"SUCCESS"}else{"FAILED"}); LogFile=$LogFile }
    if ($Code -ne 0) { $Failed = $true }
}

Write-Host ""
Write-Host "====================== STRICT BASE EVAL SUMMARY ======================"
$Results | Format-Table -AutoSize
Write-Host "====================================================================="

if ($Failed) {
    if (-not $ContinueOnError) { exit 1 }
}
exit 0
