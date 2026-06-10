#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TRAIN_CONFIG_PATH="${TRAIN_CONFIG_PATH:-./configs/lora_query_compiler_104m_1x4090.json}"
QUERY_COMPILER_DATASET_BUILD_ARGS="${QUERY_COMPILER_DATASET_BUILD_ARGS:-}"

cd "$LAB_ROOT"
python ./scripts/build_fbbp_query_compiler_dataset.py ${QUERY_COMPILER_DATASET_BUILD_ARGS}
python ./scripts/launch_training.py --config "$TRAIN_CONFIG_PATH"
