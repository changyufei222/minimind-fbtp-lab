param(
    [string]$ConfigPath = ".\configs\dpo_query_compiler_104m_smoke.json",
    [switch]$WhatIf
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$resolvedConfig = Resolve-Path -LiteralPath (Join-Path $repoRoot $ConfigPath)
$command = @("python", (Join-Path $repoRoot "scripts\launch_training.py"), "--config", $resolvedConfig.Path)

if ($WhatIf) {
    Write-Output ($command -join " ")
    exit 0
}

& $command[0] $command[1] $command[2] $command[3]
