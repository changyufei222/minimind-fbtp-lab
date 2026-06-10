param(
    [switch]$WhatIf,
    [string]$ConfigPath = "",
    [string]$DatasetBuildArgs = ""
)

$ErrorActionPreference = "Stop"

$scriptPath = $MyInvocation.MyCommand.Path
$normalizedScriptPath = $scriptPath -replace '/', '\'
$repoRoot = $normalizedScriptPath -replace '\\scripts\\[^\\]+$', ''
$resolvedConfigPath = if ($ConfigPath) { $ConfigPath } else { "$repoRoot/configs/lora_query_compiler_104m_1x4090.json" }
$buildCommand = "python `"$repoRoot/scripts/build_fbbp_query_compiler_dataset.py`""
if ($DatasetBuildArgs) {
    $buildCommand = "$buildCommand $DatasetBuildArgs"
}
$trainCommand = "python `"$repoRoot/scripts/launch_training.py`" --config `"$resolvedConfigPath`""

if ($WhatIf) {
    Write-Host $buildCommand
    Write-Host $trainCommand
    exit 0
}

Invoke-Expression $buildCommand
Invoke-Expression $trainCommand
