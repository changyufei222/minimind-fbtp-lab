$ErrorActionPreference = 'Stop'

$scriptPath = $MyInvocation.MyCommand.Path
$normalizedScriptPath = $scriptPath -replace '/', '\'
$repoRoot = $normalizedScriptPath -replace '\\scripts\\[^\\]+$', ''
$config = Join-Path $repoRoot 'configs/full_sft_stage1_1x4090.json'

python (Join-Path $repoRoot 'scripts/launch_training.py') --config $config
