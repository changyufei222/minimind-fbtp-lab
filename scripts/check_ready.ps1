$ErrorActionPreference = 'Stop'

$scriptPath = $MyInvocation.MyCommand.Path
$normalizedScriptPath = $scriptPath -replace '/', '\'
$repoRoot = $normalizedScriptPath -replace '\\scripts\\[^\\]+$', ''
$projectRoot = if ($repoRoot.EndsWith('\minimind-fbtp-lab')) { $repoRoot.Substring(0, $repoRoot.Length - '\minimind-fbtp-lab'.Length) } else { Split-Path -Path $repoRoot -Parent }
$upstreamRoot = "$projectRoot\upstream-minimind-full"
$weightPath = "$upstreamRoot\out\full_sft_512.pth"
$weight768Path = "$upstreamRoot\out\full_sft_768.pth"
$trainData = "$repoRoot\data\processed\fbtp_sft_seed_train.jsonl"
$devData = "$repoRoot\data\processed\fbtp_sft_seed_dev.jsonl"
$configPath = "$repoRoot\configs\lora_stage1_1x4090.json"
$localCovunibindSource = "$repoRoot\data\raw\covunibind_covabdab_binding_ingest.csv"
$stage2TrainData = "$repoRoot\data\processed\covunibind_stage2_train.jsonl"
$stage2DevData = "$repoRoot\data\processed\covunibind_stage2_dev.jsonl"
$stage2ConfigPath = "$repoRoot\configs\lora_stage2_covunibind_1x4090.json"
$stage3TrainData = "$repoRoot\data\processed\covunibind_stage3_short_schema_train.jsonl"
$stage3DevData = "$repoRoot\data\processed\covunibind_stage3_short_schema_dev.jsonl"
$stage3ConfigPath = "$repoRoot\configs\lora_stage3_covunibind_short_104m_1x4090.json"
$stage31TrainData = "$repoRoot\data\processed\covunibind_stage3_1_strict_schema_train.jsonl"
$stage31DevData = "$repoRoot\data\processed\covunibind_stage3_1_strict_schema_dev.jsonl"
$stage31ConfigPath = "$repoRoot\configs\lora_stage3_1_covunibind_strict_104m_1x4090.json"

Write-Host '=== MiniMind-FBTP Ready Check ==='

function Show-Check($label, $ok, $detail) {
    $status = if ($ok) { '[OK]' } else { '[MISSING]' }
    Write-Host "$status $label - $detail"
}

Show-Check 'Upstream root' (Test-Path $upstreamRoot) $upstreamRoot
Show-Check 'Base weight' (Test-Path $weightPath) $weightPath
Show-Check '104M base weight' (Test-Path $weight768Path) $weight768Path
Show-Check 'Train dataset' (Test-Path $trainData) $trainData
Show-Check 'Dev dataset' (Test-Path $devData) $devData
Show-Check 'LoRA config' (Test-Path $configPath) $configPath
Show-Check 'Local CoVUniBind source' (Test-Path $localCovunibindSource) $localCovunibindSource
Show-Check 'Stage-2 train dataset' (Test-Path $stage2TrainData) $stage2TrainData
Show-Check 'Stage-2 dev dataset' (Test-Path $stage2DevData) $stage2DevData
Show-Check 'Stage-2 LoRA config' (Test-Path $stage2ConfigPath) $stage2ConfigPath
Show-Check 'Stage-3 train dataset' (Test-Path $stage3TrainData) $stage3TrainData
Show-Check 'Stage-3 dev dataset' (Test-Path $stage3DevData) $stage3DevData
Show-Check 'Stage-3 LoRA config' (Test-Path $stage3ConfigPath) $stage3ConfigPath
Show-Check 'Stage-3.1 train dataset' (Test-Path $stage31TrainData) $stage31TrainData
Show-Check 'Stage-3.1 dev dataset' (Test-Path $stage31DevData) $stage31DevData
Show-Check 'Stage-3.1 LoRA config' (Test-Path $stage31ConfigPath) $stage31ConfigPath

$pythonOk = $false
try {
    python --version | Out-Null
    $pythonOk = $true
} catch {}
Show-Check 'Python on PATH' $pythonOk 'python'

$torchOk = $false
if ($pythonOk) {
    try {
        @'
import importlib.util
print(bool(importlib.util.find_spec("torch")))
'@ | python - | Tee-Object -Variable torchFlag | Out-Null
        $torchOk = ($torchFlag.Trim() -eq 'True')
    } catch {}
}
Show-Check 'Torch importable' $torchOk 'torch'

$gpuCmd = Get-Command nvidia-smi -ErrorAction SilentlyContinue
Show-Check 'nvidia-smi on PATH' ([bool]$gpuCmd) ($(if ($gpuCmd) { $gpuCmd.Source } else { 'not found in current shell' }))

Write-Host ''
Write-Host 'Recommended next command (on the GPU machine):'
Write-Host "  cd $repoRoot"
Write-Host '  powershell -ExecutionPolicy Bypass -File .\scripts\run_lora_stage3_1_covunibind_strict_104m.ps1'
