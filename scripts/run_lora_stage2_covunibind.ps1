$ErrorActionPreference = "Stop"

$scriptPath = $MyInvocation.MyCommand.Path
$normalizedScriptPath = $scriptPath -replace '/', '\'
$repoRoot = $normalizedScriptPath -replace '\\scripts\\[^\\]+$', ''
python "$repoRoot/scripts/build_covunibind_stage2_dataset.py"
python "$repoRoot/scripts/launch_training.py" --config "$repoRoot/configs/lora_stage2_covunibind_1x4090.json"
