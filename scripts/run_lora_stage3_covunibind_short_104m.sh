#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python "$REPO_ROOT/scripts/build_covunibind_stage3_short_schema_dataset.py"
python "$REPO_ROOT/scripts/launch_training.py" --config "$REPO_ROOT/configs/lora_stage3_covunibind_short_104m_1x4090.json"
