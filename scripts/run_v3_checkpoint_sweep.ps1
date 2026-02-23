param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("v3b1","v3b2","v3b3")]
    [string]$Variant,

    [switch]$UseCondaRun,
    [string]$CondaEnv = "granite_ft",
    [string]$BaseModelPath = "D:/AI_code/models/Qwen3-1.7B",
    [string]$LabeledEvalFile = "data/qwen3_rigorous_test_labeled.json",
    [string]$IfUnlabeledEvalFile = "data/qwen3_rigorous_test_if_unlabeled.json",
    [string]$OutputDataDir = "evaluation/output_data",
    [string]$RigorousDir = "evaluation/rigorous",
    [int]$MaxInputLength = 2048,
    [int]$MaxNewTokens = 256,
    [switch]$IncludeRootBest,
    [switch]$ForceRerun,
    [switch]$ContinueOnError,
    [switch]$DryRun,
    [int[]]$OnlyCheckpoints = @()
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$LogDir = Join-Path $ProjectRoot "logs"
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null

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

function New-CheckpointJob {
    param([string]$Name, [string]$AdapterPath)
    return @{ Name = $Name; Adapter = $AdapterPath }
}

$AbsOutputDataDir = Join-Path $ProjectRoot $OutputDataDir
$AbsRigorousDir = Join-Path $ProjectRoot $RigorousDir
$AbsLabeledEvalFile = Join-Path $ProjectRoot $LabeledEvalFile
$AbsIfUnlabeledEvalFile = Join-Path $ProjectRoot $IfUnlabeledEvalFile
Ensure-Path -PathToCheck $BaseModelPath -Label "Base model path"
Ensure-Path -PathToCheck $AbsLabeledEvalFile -Label "Labeled eval file"
Ensure-Path -PathToCheck $AbsIfUnlabeledEvalFile -Label "IF unlabeled eval file"
New-Item -ItemType Directory -Path $AbsOutputDataDir -Force | Out-Null
New-Item -ItemType Directory -Path $AbsRigorousDir -Force | Out-Null

$ModelOutputDir = Join-Path $ProjectRoot ("outputs/qwen3-1.7B-lora-{0}" -f $Variant)
Ensure-Path -PathToCheck $ModelOutputDir -Label "Model output dir"

$SweepRoot = Join-Path $AbsRigorousDir ("sweeps/{0}" -f $Variant)
New-Item -ItemType Directory -Path $SweepRoot -Force | Out-Null

$Jobs = @()

if ($IncludeRootBest) {
    $RootBestName = "{0}_best_root" -f $Variant
    $Jobs += New-CheckpointJob -Name $RootBestName -AdapterPath $ModelOutputDir
}

$CkptDirs = @(Get-ChildItem -Path $ModelOutputDir -Directory -Filter "checkpoint-*" | Sort-Object Name)
foreach ($d in $CkptDirs) {
    if ($OnlyCheckpoints.Count -gt 0) {
        $num = [int]($d.Name -replace '^checkpoint-','')
        if ($OnlyCheckpoints -notcontains $num) { continue }
    }
    $Jobs += New-CheckpointJob -Name ("{0}_{1}" -f $Variant, $d.Name) -AdapterPath $d.FullName
}

if ($Jobs.Count -eq 0) {
    throw "No checkpoint jobs found. Use -IncludeRootBest and/or check outputs."
}

function Test-EvalDone {
    param([string]$Dir)
    return (Test-Path (Join-Path $Dir "eval_results.json"))
}

$Results = @()

foreach ($Job in $Jobs) {
    $Name = $Job.Name
    $Adapter = $Job.Adapter
    Ensure-Path -PathToCheck $Adapter -Label "Adapter path for $Name"

    $Slug = ($Name -replace '[^A-Za-z0-9._-]','_')
    $LabeledOutFile = Join-Path $AbsOutputDataDir ("{0}_labeled.json" -f $Slug)
    $IfOutFile = Join-Path $AbsOutputDataDir ("{0}_if_unlabeled.json" -f $Slug)
    $LabeledOutDir = Join-Path $SweepRoot ("{0}_labeled" -f $Slug)
    $IfOutDir = Join-Path $SweepRoot ("{0}_if_unlabeled" -f $Slug)
    New-Item -ItemType Directory -Path $LabeledOutDir -Force | Out-Null
    New-Item -ItemType Directory -Path $IfOutDir -Force | Out-Null

    if ((-not $ForceRerun) -and (Test-EvalDone -Dir $LabeledOutDir) -and (Test-EvalDone -Dir $IfOutDir)) {
        Write-Host ""
        Write-Host "Skip sweep eval: $Name (eval_results.json exists)"
        $Results += [PSCustomObject]@{ Candidate=$Name; Step="all"; ExitCode=0; Status="SKIPPED"; LogFile="-" }
        continue
    }

    $Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $LogFile = Join-Path $LogDir ("sweep_{0}_{1}.log" -f $Slug, $Stamp)
    if ($DryRun -and -not (Test-Path $LogFile)) { New-Item -ItemType File -Path $LogFile | Out-Null }

    $ModelSpec = "SWEEP_{0}:{1}:{2}" -f $Slug.ToUpper(), $BaseModelPath, $Adapter
    Write-Host ""
    Write-Host "============================================================"
    Write-Host "Sweep candidate: $Name"
    Write-Host "Adapter: $Adapter"
    Write-Host "Log: $LogFile"
    Write-Host "============================================================"

    $Failed = $false

    $Code = Invoke-Step -StepName "$Name.generate_labeled" -Cmd "python" -CmdArgs @(
        "scripts/generate.py",
        "--models", $ModelSpec,
        "--eval-file", $AbsLabeledEvalFile,
        "--output-file", $LabeledOutFile,
        "--max-input-length", "$MaxInputLength",
        "--max-new-tokens", "$MaxNewTokens"
    ) -LogFile $LogFile
    $Results += [PSCustomObject]@{ Candidate=$Name; Step="generate_labeled"; ExitCode=$Code; Status=$(if($Code -eq 0){"SUCCESS"}else{"FAILED"}); LogFile=$LogFile }
    if ($Code -ne 0) { $Failed = $true }

    if (-not $Failed) {
        $Code = Invoke-Step -StepName "$Name.score_labeled" -Cmd "python" -CmdArgs @(
            "scripts/score.py",
            "--input-file", $LabeledOutFile,
            "--output-dir", $LabeledOutDir
        ) -LogFile $LogFile
        $Results += [PSCustomObject]@{ Candidate=$Name; Step="score_labeled"; ExitCode=$Code; Status=$(if($Code -eq 0){"SUCCESS"}else{"FAILED"}); LogFile=$LogFile }
        if ($Code -ne 0) { $Failed = $true }
    }

    if (-not $Failed) {
        $Code = Invoke-Step -StepName "$Name.generate_if_unlabeled" -Cmd "python" -CmdArgs @(
            "scripts/generate.py",
            "--models", $ModelSpec,
            "--eval-file", $AbsIfUnlabeledEvalFile,
            "--output-file", $IfOutFile,
            "--max-input-length", "$MaxInputLength",
            "--max-new-tokens", "$MaxNewTokens"
        ) -LogFile $LogFile
        $Results += [PSCustomObject]@{ Candidate=$Name; Step="generate_if_unlabeled"; ExitCode=$Code; Status=$(if($Code -eq 0){"SUCCESS"}else{"FAILED"}); LogFile=$LogFile }
        if ($Code -ne 0) { $Failed = $true }
    }

    if (-not $Failed) {
        $Code = Invoke-Step -StepName "$Name.score_if_unlabeled" -Cmd "python" -CmdArgs @(
            "scripts/score.py",
            "--input-file", $IfOutFile,
            "--output-dir", $IfOutDir
        ) -LogFile $LogFile
        $Results += [PSCustomObject]@{ Candidate=$Name; Step="score_if_unlabeled"; ExitCode=$Code; Status=$(if($Code -eq 0){"SUCCESS"}else{"FAILED"}); LogFile=$LogFile }
        if ($Code -ne 0) { $Failed = $true }
    }

    if ($Failed) {
        Write-Host "Sweep failed for $Name."
        if (-not $ContinueOnError) { break }
        Write-Host "ContinueOnError is enabled. Continue to next candidate."
    } else {
        Write-Host "Sweep finished for $Name."
    }
}

Write-Host ""
Write-Host "===================== V3 CHECKPOINT SWEEP SUMMARY ====================="
$Results | Format-Table -AutoSize
Write-Host "======================================================================="

$HasFailure = $Results | Where-Object { $_.ExitCode -ne 0 }
if ($HasFailure) { exit 1 }
exit 0
