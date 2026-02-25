param(
    [switch]$UseCondaRun,
    [string]$CondaEnv = "granite_ft",
    [switch]$ThreeSeed,
    [int[]]$Seeds = @(2026, 2027, 2028),
    [string]$VersionTag = "tsif_v1",
    [string]$BaseModelPath = "D:/AI_code/models/Qwen3-1.7B",
    [string]$TSAdapterBase = "outputs/qwen3-1.7B-lora-strict-ts-v1",
    [string]$IFAdapterBase = "outputs/qwen3-1.7B-lora-strict-if-v1",
    [string]$LabeledEvalFile = "data/qwen3_strict_test_labeled.json",
    [string]$IfUnlabeledEvalFile = "data/qwen3_strict_test_if_unlabeled.json",
    [string]$RoutingWorkDir = "evaluation/output_data/planA_routing",
    [string]$OutputDataDir = "evaluation/output_data/planA_eval",
    [string]$EvalDir = "evaluation/strict_planA",
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
$AbsRoutingWorkDir = Join-Path $ProjectRoot $RoutingWorkDir
$AbsOutputDataDir = Join-Path $ProjectRoot $OutputDataDir
$ResolvedEvalDir = if ($HybridIF) { "$EvalDir`_hybrid" } else { $EvalDir }
$AbsEvalDir = Join-Path $ProjectRoot $ResolvedEvalDir
$ScoreExtraArgs = @()
if ($HybridIF) { $ScoreExtraArgs += "--enable-lria-fallback" }

Ensure-Path -PathToCheck $BaseModelPath -Label "Base model path"
Ensure-Path -PathToCheck $AbsLabeledEvalFile -Label "Labeled eval file"
Ensure-Path -PathToCheck $AbsIfUnlabeledEvalFile -Label "IF unlabeled eval file"
New-Item -ItemType Directory -Path $AbsRoutingWorkDir -Force | Out-Null
New-Item -ItemType Directory -Path $AbsOutputDataDir -Force | Out-Null
New-Item -ItemType Directory -Path $AbsEvalDir -Force | Out-Null

Write-Host ("IF scoring mode: {0}" -f ($(if ($HybridIF) { "HYBRID (LRIA fallback)" } else { "STRICT" })))
Write-Host ("Eval output dir: {0}" -f $ResolvedEvalDir)

$Jobs = @()
if ($ThreeSeed) {
    foreach ($s in $Seeds) {
        $Jobs += @{
            Name = "{0}_s{1}" -f $VersionTag, $s
            Seed = $s
            TSAdapter = "{0}-s{1}" -f $TSAdapterBase, $s
            IFAdapter = "{0}-s{1}" -f $IFAdapterBase, $s
        }
    }
} else {
    $Jobs += @{
        Name = $VersionTag
        Seed = $null
        TSAdapter = $TSAdapterBase
        IFAdapter = $IFAdapterBase
    }
}

$Results = @()

foreach ($Job in $Jobs) {
    $Name = $Job.Name
    $AbsTSAdapter = Join-Path $ProjectRoot $Job.TSAdapter
    $AbsIFAdapter = Join-Path $ProjectRoot $Job.IFAdapter
    Ensure-Path -PathToCheck $AbsTSAdapter -Label "TS adapter path for $Name"
    Ensure-Path -PathToCheck $AbsIFAdapter -Label "IF adapter path for $Name"

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
    $LogFile = Join-Path $LogDir "eval_planA_$Name`_$Stamp.log"
    if ($DryRun -and -not (Test-Path $LogFile)) { New-Item -ItemType File -Path $LogFile | Out-Null }

    $RoutedTsEval = Join-Path $AbsRoutingWorkDir "qwen3_strict_test_labeled_ts.json"
    $RoutedIfEval = Join-Path $AbsRoutingWorkDir "qwen3_strict_test_labeled_if.json"
    $RoutedTsOutput = Join-Path $AbsOutputDataDir "$Name`_routed_ts_labeled.json"
    $RoutedIfOutput = Join-Path $AbsOutputDataDir "$Name`_routed_if_labeled.json"
    $MergedLabeledOutput = Join-Path $AbsOutputDataDir "$Name`_routed_labeled_merged.json"
    $IfUnlabeledOutput = Join-Path $AbsOutputDataDir "$Name`_if_unlabeled.json"

    $TsModelSpec = "FTQwen3_{0}_TS:{1}:{2}" -f $Name.ToUpper(), $BaseModelPath, $AbsTSAdapter
    $IfModelSpec = "FTQwen3_{0}_IF:{1}:{2}" -f $Name.ToUpper(), $BaseModelPath, $AbsIFAdapter

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "Start Plan A routed evaluation: $Name"
    Write-Host "TS Adapter: $AbsTSAdapter"
    Write-Host "IF Adapter: $AbsIFAdapter"
    Write-Host "Log: $LogFile"
    Write-Host "============================================================"

    $Failed = $false

    # 0) Build routed labeled eval subsets
    $Code = Invoke-Step -StepName "$Name.route_inputs" -Cmd "python" -CmdArgs @(
        "scripts/build_qwen3_strict_routed_eval_inputs.py",
        "--input-file", $LabeledEvalFile,
        "--output-dir", $RoutingWorkDir
    ) -LogFile $LogFile
    $Results += [PSCustomObject]@{ Model=$Name; Step="route_inputs"; ExitCode=$Code; Status=$(if($Code -eq 0){"SUCCESS"}else{"FAILED"}); LogFile=$LogFile }
    if ($Code -ne 0) { $Failed = $true }

    if (-not $Failed) {
        $Code = Invoke-Step -StepName "$Name.generate_labeled_ts" -Cmd "python" -CmdArgs @(
            "scripts/generate.py",
            "--models", $TsModelSpec,
            "--eval-file", $RoutedTsEval,
            "--output-file", $RoutedTsOutput,
            "--max-input-length", "$MaxInputLength",
            "--max-new-tokens", "$MaxNewTokens"
        ) -LogFile $LogFile
        $Results += [PSCustomObject]@{ Model=$Name; Step="generate_labeled_ts"; ExitCode=$Code; Status=$(if($Code -eq 0){"SUCCESS"}else{"FAILED"}); LogFile=$LogFile }
        if ($Code -ne 0) { $Failed = $true }
    }

    if (-not $Failed) {
        $Code = Invoke-Step -StepName "$Name.generate_labeled_if" -Cmd "python" -CmdArgs @(
            "scripts/generate.py",
            "--models", $IfModelSpec,
            "--eval-file", $RoutedIfEval,
            "--output-file", $RoutedIfOutput,
            "--max-input-length", "$MaxInputLength",
            "--max-new-tokens", "$MaxNewTokens"
        ) -LogFile $LogFile
        $Results += [PSCustomObject]@{ Model=$Name; Step="generate_labeled_if"; ExitCode=$Code; Status=$(if($Code -eq 0){"SUCCESS"}else{"FAILED"}); LogFile=$LogFile }
        if ($Code -ne 0) { $Failed = $true }
    }

    if (-not $Failed) {
        $Code = Invoke-Step -StepName "$Name.merge_labeled" -Cmd "python" -CmdArgs @(
            "scripts/merge_qwen3_strict_routed_labeled_outputs.py",
            "--original-eval-file", $LabeledEvalFile,
            "--ts-output", (Resolve-Path $RoutedTsOutput).Path,
            "--if-output", (Resolve-Path $RoutedIfOutput).Path,
            "--output-file", $MergedLabeledOutput,
            "--model-name", "PlanA_Routed_$Name",
            "--model-path", $BaseModelPath,
            "--adapter-path", ("TS={0};IF={1}" -f $AbsTSAdapter, $AbsIFAdapter)
        ) -LogFile $LogFile
        $Results += [PSCustomObject]@{ Model=$Name; Step="merge_labeled"; ExitCode=$Code; Status=$(if($Code -eq 0){"SUCCESS"}else{"FAILED"}); LogFile=$LogFile }
        if ($Code -ne 0) { $Failed = $true }
    }

    if (-not $Failed) {
        $ScoreArgs = @(
            "scripts/score.py",
            "--input-file", $MergedLabeledOutput,
            "--output-dir", $LabeledOutDir
        ) + $ScoreExtraArgs
        $Code = Invoke-Step -StepName "$Name.score_labeled" -Cmd "python" -CmdArgs $ScoreArgs -LogFile $LogFile
        $Results += [PSCustomObject]@{ Model=$Name; Step="score_labeled"; ExitCode=$Code; Status=$(if($Code -eq 0){"SUCCESS"}else{"FAILED"}); LogFile=$LogFile }
        if ($Code -ne 0) { $Failed = $true }
    }

    # IF adapter handles prompt-only IF unlabeled set directly
    if (-not $Failed) {
        $Code = Invoke-Step -StepName "$Name.generate_if_unlabeled" -Cmd "python" -CmdArgs @(
            "scripts/generate.py",
            "--models", $IfModelSpec,
            "--eval-file", $AbsIfUnlabeledEvalFile,
            "--output-file", $IfUnlabeledOutput,
            "--max-input-length", "$MaxInputLength",
            "--max-new-tokens", "$MaxNewTokens"
        ) -LogFile $LogFile
        $Results += [PSCustomObject]@{ Model=$Name; Step="generate_if_unlabeled"; ExitCode=$Code; Status=$(if($Code -eq 0){"SUCCESS"}else{"FAILED"}); LogFile=$LogFile }
        if ($Code -ne 0) { $Failed = $true }
    }

    if (-not $Failed) {
        $ScoreArgs = @(
            "scripts/score.py",
            "--input-file", $IfUnlabeledOutput,
            "--output-dir", $IfOutDir
        ) + $ScoreExtraArgs
        $Code = Invoke-Step -StepName "$Name.score_if_unlabeled" -Cmd "python" -CmdArgs $ScoreArgs -LogFile $LogFile
        $Results += [PSCustomObject]@{ Model=$Name; Step="score_if_unlabeled"; ExitCode=$Code; Status=$(if($Code -eq 0){"SUCCESS"}else{"FAILED"}); LogFile=$LogFile }
        if ($Code -ne 0) { $Failed = $true }
    }

    if ($Failed) {
        Write-Host "Plan A routed evaluation failed for $Name."
        if (-not $ContinueOnError) { break }
        Write-Host "ContinueOnError is enabled. Continue to next job."
    } else {
        Write-Host "Plan A routed evaluation finished for $Name."
    }
}

Write-Host ""
Write-Host "=================== PLAN A ROUTED EVAL SUMMARY ==================="
$Results | Format-Table -AutoSize
Write-Host "=================================================================="

$HasFailure = $Results | Where-Object { $_.ExitCode -ne 0 }
if ($HasFailure) { exit 1 }
exit 0
