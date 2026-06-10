#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python "$REPO_ROOT/scripts/build_covunibind_stage3_1_strict_schema_dataset.py"
python "$REPO_ROOT/scripts/launch_training.py" --config "$REPO_ROOT/configs/lora_stage3_1_covunibind_strict_104m_1x4090.json"
