#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-$REPO_ROOT/configs/dpo_query_compiler_104m_smoke.json}"
python "$REPO_ROOT/scripts/launch_training.py" --config "$CONFIG_PATH"
