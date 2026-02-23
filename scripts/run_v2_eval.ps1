param(
    [switch]$UseCondaRun,
    [string]$CondaEnv = "granite_ft",
    [string]$BaseModelPath = "D:/AI_code/models/Qwen3-1.7B",
    [string]$LabeledEvalFile = "data/qwen3_rigorous_test_labeled.json",
    [string]$IfUnlabeledEvalFile = "data/qwen3_rigorous_test_if_unlabeled.json",
    [string]$OutputDataDir = "evaluation/output_data",
    [string]$RigorousDir = "evaluation/rigorous",
    [int]$MaxInputLength = 2048,
    [int]$MaxNewTokens = 512,
    [switch]$ContinueOnError
)

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$LogDir = Join-Path $ProjectRoot "logs"
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null

$Jobs = @(
    @{ Name = "v2a"; Adapter = "outputs/qwen3-1.7B-lora-v2a" },
    @{ Name = "v2b"; Adapter = "outputs/qwen3-1.7B-lora-v2b" },
    @{ Name = "v2c"; Adapter = "outputs/qwen3-1.7B-lora-v2c" }
)

if ($UseCondaRun) {
    if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
        throw "conda command not found. Ensure conda is in PATH, or run without -UseCondaRun inside an activated env."
    }
}

function Invoke-Step {
    param(
        [string]$StepName,
        [string]$Cmd,
        [string[]]$CmdArgs,
        [string]$LogFile
    )

    Write-Host ""
    Write-Host "[$StepName] $Cmd $($CmdArgs -join ' ')"

    # Native tools may write warnings to stderr; do not abort on stderr text.
    # We only use process exit code to decide success/failure.
    $PrevEAP = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        if ($UseCondaRun) {
            & conda run -n $CondaEnv $Cmd @CmdArgs 2>&1 | Tee-Object -FilePath $LogFile -Append | Out-Host
        }
        else {
            & $Cmd @CmdArgs 2>&1 | Tee-Object -FilePath $LogFile -Append | Out-Host
        }
    }
    finally {
        $ErrorActionPreference = $PrevEAP
    }

    return [int]$LASTEXITCODE
}

function Ensure-Path {
    param([string]$PathToCheck, [string]$Label)
    if (-not (Test-Path $PathToCheck)) {
        throw "$Label not found: $PathToCheck"
    }
}

$AbsLabeledEvalFile = Join-Path $ProjectRoot $LabeledEvalFile
$AbsIfUnlabeledEvalFile = Join-Path $ProjectRoot $IfUnlabeledEvalFile
Ensure-Path -PathToCheck $AbsLabeledEvalFile -Label "Labeled eval file"
Ensure-Path -PathToCheck $AbsIfUnlabeledEvalFile -Label "IF unlabeled eval file"
Ensure-Path -PathToCheck $BaseModelPath -Label "Base model path"

$AbsOutputDataDir = Join-Path $ProjectRoot $OutputDataDir
$AbsRigorousDir = Join-Path $ProjectRoot $RigorousDir
New-Item -ItemType Directory -Path $AbsOutputDataDir -Force | Out-Null
New-Item -ItemType Directory -Path $AbsRigorousDir -Force | Out-Null

$Results = @()

foreach ($Job in $Jobs) {
    $Name = $Job.Name
    $AbsAdapter = Join-Path $ProjectRoot $Job.Adapter
    Ensure-Path -PathToCheck $AbsAdapter -Label "Adapter path for $Name"

    $Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $LogFile = Join-Path $LogDir "eval_$Name`_$Stamp.log"

    $ModelSpec = "FTQwen3_{0}:{1}:{2}" -f $Name.ToUpper(), $BaseModelPath, $AbsAdapter
    $LabeledOutputFile = Join-Path $AbsOutputDataDir "$Name`_labeled.json"
    $IfOutputFile = Join-Path $AbsOutputDataDir "$Name`_if_unlabeled.json"
    $LabeledOutDir = Join-Path $AbsRigorousDir "$Name`_labeled"
    $IfOutDir = Join-Path $AbsRigorousDir "$Name`_if_unlabeled"

    New-Item -ItemType Directory -Path $LabeledOutDir -Force | Out-Null
    New-Item -ItemType Directory -Path $IfOutDir -Force | Out-Null

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
    $Results += [PSCustomObject]@{ Model = $Name; Step = "generate_labeled"; ExitCode = $Code; Status = $(if ($Code -eq 0) { "SUCCESS" } else { "FAILED" }); LogFile = $LogFile }
    if ($Code -ne 0) { $Failed = $true }

    if (-not $Failed) {
        $Code = Invoke-Step -StepName "$Name.score_labeled" -Cmd "python" -CmdArgs @(
            "scripts/score.py",
            "--input-file", $LabeledOutputFile,
            "--output-dir", $LabeledOutDir
        ) -LogFile $LogFile
        $Results += [PSCustomObject]@{ Model = $Name; Step = "score_labeled"; ExitCode = $Code; Status = $(if ($Code -eq 0) { "SUCCESS" } else { "FAILED" }); LogFile = $LogFile }
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
        $Results += [PSCustomObject]@{ Model = $Name; Step = "generate_if_unlabeled"; ExitCode = $Code; Status = $(if ($Code -eq 0) { "SUCCESS" } else { "FAILED" }); LogFile = $LogFile }
        if ($Code -ne 0) { $Failed = $true }
    }

    if (-not $Failed) {
        $Code = Invoke-Step -StepName "$Name.score_if_unlabeled" -Cmd "python" -CmdArgs @(
            "scripts/score.py",
            "--input-file", $IfOutputFile,
            "--output-dir", $IfOutDir
        ) -LogFile $LogFile
        $Results += [PSCustomObject]@{ Model = $Name; Step = "score_if_unlabeled"; ExitCode = $Code; Status = $(if ($Code -eq 0) { "SUCCESS" } else { "FAILED" }); LogFile = $LogFile }
        if ($Code -ne 0) { $Failed = $true }
    }

    if ($Failed) {
        Write-Host "Evaluation failed for $Name."
        if (-not $ContinueOnError) {
            break
        }
        Write-Host "ContinueOnError is enabled. Continue to next model."
    }
    else {
        Write-Host "Evaluation finished for $Name."
    }
}

Write-Host ""
Write-Host "======================== EVAL SUMMARY ========================"
$Results | Format-Table -AutoSize
Write-Host "=============================================================="

$HasFailure = $Results | Where-Object { $_.ExitCode -ne 0 }
if ($HasFailure) {
    exit 1
}

exit 0
