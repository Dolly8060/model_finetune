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
    @{ Name = "v3b1"; Adapter = "outputs/qwen3-1.7B-lora-v3b1" },
    @{ Name = "v3b2"; Adapter = "outputs/qwen3-1.7B-lora-v3b2" },
    @{ Name = "v3b3"; Adapter = "outputs/qwen3-1.7B-lora-v3b3" }
)

if ($Only.Count -gt 0) {
    $OnlySet = @{}
    foreach ($n in $Only) { $OnlySet[$n.ToLower()] = $true }
    $Jobs = @($Jobs | Where-Object { $OnlySet.ContainsKey($_.Name.ToLower()) })
}
if ($Jobs.Count -eq 0) { throw "No jobs selected. Check -Only values (valid: v3b1, v3b2, v3b3)." }

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
    $res = Join-Path $AbsEvalDir "eval_results.json"
    if (Test-Path $res) { return @{ Completed = $true; Reason = "eval_results.json exists" } }
    return @{ Completed = $false; Reason = "eval_results.json missing" }
}

$AbsLabeledEvalFile = Join-Path $ProjectRoot $LabeledEvalFile
$AbsIfUnlabeledEvalFile = Join-Path $ProjectRoot $IfUnlabeledEvalFile
$AbsOutputDataDir = Join-Path $ProjectRoot $OutputDataDir
$AbsRigorousDir = Join-Path $ProjectRoot $RigorousDir

Ensure-Path -PathToCheck $BaseModelPath -Label "Base model path"
Ensure-Path -PathToCheck $AbsLabeledEvalFile -Label "Labeled eval file"
Ensure-Path -PathToCheck $AbsIfUnlabeledEvalFile -Label "IF unlabeled eval file"
New-Item -ItemType Directory -Path $AbsOutputDataDir -Force | Out-Null
New-Item -ItemType Directory -Path $AbsRigorousDir -Force | Out-Null

$Results = @()

foreach ($Job in $Jobs) {
    $Name = $Job.Name
    $AbsAdapter = Join-Path $ProjectRoot $Job.Adapter
    Ensure-Path -PathToCheck $AbsAdapter -Label "Adapter path for $Name"

    $LabeledOutputFile = Join-Path $AbsOutputDataDir "$Name`_labeled.json"
    $IfOutputFile = Join-Path $AbsOutputDataDir "$Name`_if_unlabeled.json"
    $LabeledOutDir = Join-Path $AbsRigorousDir "$Name`_labeled"
    $IfOutDir = Join-Path $AbsRigorousDir "$Name`_if_unlabeled"
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
    $LogFile = Join-Path $LogDir "eval_$Name`_$Stamp.log"
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
        $Code = Invoke-Step -StepName "$Name.score_labeled" -Cmd "python" -CmdArgs @(
            "scripts/score.py",
            "--input-file", $LabeledOutputFile,
            "--output-dir", $LabeledOutDir
        ) -LogFile $LogFile
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
        $Code = Invoke-Step -StepName "$Name.score_if_unlabeled" -Cmd "python" -CmdArgs @(
            "scripts/score.py",
            "--input-file", $IfOutputFile,
            "--output-dir", $IfOutDir
        ) -LogFile $LogFile
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
Write-Host "======================== V3 EVAL SUMMARY ====================="
$Results | Format-Table -AutoSize
Write-Host "=============================================================="

$HasFailure = $Results | Where-Object { $_.ExitCode -ne 0 }
if ($HasFailure) { exit 1 }
exit 0
