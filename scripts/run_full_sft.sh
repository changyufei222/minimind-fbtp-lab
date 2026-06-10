#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python "$REPO_ROOT/scripts/launch_training.py" --config "$REPO_ROOT/configs/full_sft_stage1_1x4090.json"

