#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
UPSTREAM_ROOT="$PROJECT_ROOT/upstream-minimind-full"
WEIGHT_PATH="$UPSTREAM_ROOT/out/full_sft_512.pth"
WEIGHT_768_PATH="$UPSTREAM_ROOT/out/full_sft_768.pth"
TRAIN_DATA="$REPO_ROOT/data/processed/fbtp_sft_seed_train.jsonl"
DEV_DATA="$REPO_ROOT/data/processed/fbtp_sft_seed_dev.jsonl"
CONFIG_PATH="$REPO_ROOT/configs/lora_stage1_1x4090.json"
LOCAL_COVUNIBIND_SOURCE="$REPO_ROOT/data/raw/covunibind_covabdab_binding_ingest.csv"
STAGE2_TRAIN_DATA="$REPO_ROOT/data/processed/covunibind_stage2_train.jsonl"
STAGE2_DEV_DATA="$REPO_ROOT/data/processed/covunibind_stage2_dev.jsonl"
STAGE2_CONFIG_PATH="$REPO_ROOT/configs/lora_stage2_covunibind_1x4090.json"
STAGE3_TRAIN_DATA="$REPO_ROOT/data/processed/covunibind_stage3_short_schema_train.jsonl"
STAGE3_DEV_DATA="$REPO_ROOT/data/processed/covunibind_stage3_short_schema_dev.jsonl"
STAGE3_CONFIG_PATH="$REPO_ROOT/configs/lora_stage3_covunibind_short_104m_1x4090.json"
STAGE3_1_TRAIN_DATA="$REPO_ROOT/data/processed/covunibind_stage3_1_strict_schema_train.jsonl"
STAGE3_1_DEV_DATA="$REPO_ROOT/data/processed/covunibind_stage3_1_strict_schema_dev.jsonl"
STAGE3_1_CONFIG_PATH="$REPO_ROOT/configs/lora_stage3_1_covunibind_strict_104m_1x4090.json"

check() {
  local label="$1"
  local path="$2"
  if [[ -e "$path" ]]; then
    echo "[OK] $label - $path"
  else
    echo "[MISSING] $label - $path"
  fi
}

echo "=== MiniMind-FBTP Ready Check ==="
check "Upstream root" "$UPSTREAM_ROOT"
check "Base weight" "$WEIGHT_PATH"
check "104M base weight" "$WEIGHT_768_PATH"
check "Train dataset" "$TRAIN_DATA"
check "Dev dataset" "$DEV_DATA"
check "LoRA config" "$CONFIG_PATH"
check "Local CoVUniBind source" "$LOCAL_COVUNIBIND_SOURCE"
check "Stage-2 train dataset" "$STAGE2_TRAIN_DATA"
check "Stage-2 dev dataset" "$STAGE2_DEV_DATA"
check "Stage-2 LoRA config" "$STAGE2_CONFIG_PATH"
check "Stage-3 train dataset" "$STAGE3_TRAIN_DATA"
check "Stage-3 dev dataset" "$STAGE3_DEV_DATA"
check "Stage-3 LoRA config" "$STAGE3_CONFIG_PATH"
check "Stage-3.1 train dataset" "$STAGE3_1_TRAIN_DATA"
check "Stage-3.1 dev dataset" "$STAGE3_1_DEV_DATA"
check "Stage-3.1 LoRA config" "$STAGE3_1_CONFIG_PATH"

if command -v python >/dev/null 2>&1; then
  echo "[OK] Python on PATH - $(command -v python)"
else
  echo "[MISSING] Python on PATH"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[OK] nvidia-smi on PATH"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
else
  echo "[INFO] nvidia-smi not found on current node"
  echo "       If you are on a login node, this is usually expected."
  echo "       GPU visibility should be checked again inside the submitted job."
fi

echo
echo "Recommended next command (Linux/bash, login node):"
echo "  cd \"$PROJECT_ROOT/minimind-fbtp-lab/cluster\""
echo "  sbatch -p gpu_4090 --gpus=1 ./run_lora_stage3_1_covunibind_strict_104m_1x4090.sh"
