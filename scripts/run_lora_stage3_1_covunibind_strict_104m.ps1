$ErrorActionPreference = "Stop"

$scriptPath = $MyInvocation.MyCommand.Path
$normalizedScriptPath = $scriptPath -replace '/', '\'
$repoRoot = $normalizedScriptPath -replace '\\scripts\\[^\\]+$', ''
python "$repoRoot/scripts/build_covunibind_stage3_1_strict_schema_dataset.py"
python "$repoRoot/scripts/launch_training.py" --config "$repoRoot/configs/lora_stage3_1_covunibind_strict_104m_1x4090.json"
