param(
    [string]$TaskName = "MiniMindQueryCompilerWatcher",
    [string]$TrainJobId,
    [string]$EvalJobId,
    [string]$LoraLabel,
    [string]$BaselineLabel,
    [string]$OutSubdir
)

$ErrorActionPreference = "Stop"

if (-not $TrainJobId -or -not $EvalJobId -or -not $LoraLabel -or -not $BaselineLabel -or -not $OutSubdir) {
    throw "TrainJobId, EvalJobId, LoraLabel, BaselineLabel, and OutSubdir are required."
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$labRoot = Split-Path $scriptDir -Parent
$outDir = Join-Path $labRoot ("reports\eval\" + $OutSubdir)
$monitorDir = Join-Path $labRoot "reports\monitoring"
$statusPath = Join-Path $monitorDir ($OutSubdir + "_status.json")
$logPath = Join-Path $monitorDir ($OutSubdir + ".log")
$auditJsonPath = Join-Path $outDir ($LoraLabel + "_audit_" + $EvalJobId + ".json")
$pullFlagPath = Join-Path $monitorDir ($OutSubdir + "_pull_complete.flag")

$pythonExe = "<local_path_removed>"
$auditScript = Join-Path $scriptDir "audit_query_compiler_eval_completion.py"

$plink = "<local_path_removed>"
$plinkCommon = @(
    "-batch",
    "-hostkey", "ssh-rsa 3072 SHA256:UsQxLMhhInxyp2gEfcYBUMBiJP2Pcfp+4LliI03orZ0",
    "-P", "22",
    "-l", "scv7sd2@BSCC-N26",
    "-pw", "Cyf123456",
    "ssh.cn-zhongwei-1.paracloud.com"
)

New-Item -ItemType Directory -Force -Path $outDir | Out-Null
New-Item -ItemType Directory -Force -Path $monitorDir | Out-Null

function Write-Log {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Add-Content -LiteralPath $logPath -Value $line -Encoding UTF8
}

function Invoke-Remote {
    param([string]$Command)
    $output = & $plink @plinkCommon $Command 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Remote command failed: $Command`n$output"
    }
    return ($output -join "`n")
}

function Pull-RemoteTextFile {
    param(
        [string]$RemoteCommand,
        [string]$LocalPath
    )
    $content = Invoke-Remote -Command $RemoteCommand
    [System.IO.File]::WriteAllText($LocalPath, $content, [System.Text.UTF8Encoding]::new($false))
}

function Disable-TaskSafe {
    param([string]$Name)
    try {
        Disable-ScheduledTask -TaskName $Name | Out-Null
        Write-Log "Disabled scheduled task: $Name"
    } catch {
        Write-Log "Failed to disable scheduled task $Name : $($_.Exception.Message)"
    }
}

try {
    $statusText = Invoke-Remote -Command ("sacct -j {0},{1} --format=JobID,State,Start,End,Elapsed -P -n" -f $TrainJobId, $EvalJobId)
    $statusLines = $statusText -split "`n" | Where-Object { $_.Trim() }

    $payload = [ordered]@{
        checked_at = (Get-Date).ToString("s")
        train_job = $TrainJobId
        eval_job = $EvalJobId
        jobs = $statusLines
    }
    $payload | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $statusPath -Encoding UTF8
    Write-Log "Polled remote jobs."

    $trainComplete = $statusText -match ($TrainJobId + "\|COMPLETED")
    $evalComplete = $statusText -match ($EvalJobId + "\|COMPLETED")

    if (-not ($trainComplete -and $evalComplete)) {
        return
    }

    Write-Log "Detected train+eval completion. Pulling artifacts."

    Pull-RemoteTextFile -RemoteCommand ("cat /data/run01/scv7sd2/minimind-fbtp-lab/reports/eval/{0}_query_compiler_{1}/{0}_raw.jsonl" -f $LoraLabel, $EvalJobId) -LocalPath (Join-Path $outDir ($LoraLabel + "_raw_" + $EvalJobId + ".jsonl"))
    Pull-RemoteTextFile -RemoteCommand ("cat /data/run01/scv7sd2/minimind-fbtp-lab/reports/eval/{0}_query_compiler_{1}/{0}_score.md" -f $LoraLabel, $EvalJobId) -LocalPath (Join-Path $outDir ($LoraLabel + "_score_" + $EvalJobId + ".md"))
    Pull-RemoteTextFile -RemoteCommand ("cat /data/run01/scv7sd2/minimind-fbtp-lab/reports/eval/{0}_query_compiler_{1}/{0}_score.md" -f $BaselineLabel, $EvalJobId) -LocalPath (Join-Path $outDir ($BaselineLabel + "_score_" + $EvalJobId + ".md"))

    & $pythonExe $auditScript --raw (Join-Path $outDir ($LoraLabel + "_raw_" + $EvalJobId + ".jsonl")) --output $auditJsonPath | Out-String | Add-Content -LiteralPath $logPath -Encoding UTF8
    if ($LASTEXITCODE -ne 0) {
        throw "Audit script failed with exit code $LASTEXITCODE"
    }

    Set-Content -LiteralPath $pullFlagPath -Value ((Get-Date).ToString("s")) -Encoding UTF8
    Write-Log "Artifacts pulled and audit completed."
    Disable-TaskSafe -Name $TaskName
} catch {
    Write-Log "Watcher error: $($_.Exception.Message)"
    throw
}
