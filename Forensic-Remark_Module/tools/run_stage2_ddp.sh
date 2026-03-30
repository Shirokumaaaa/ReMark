#!/usr/bin/env bash
set -euo pipefail

# Multi-GPU launcher for Stage2 training.
# Example:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/run_stage2_ddp.sh

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

TORCHRUN_BIN="${TORCHRUN_BIN:-$(command -v torchrun || true)}"
if [[ -z "${TORCHRUN_BIN}" ]]; then
  echo "[error] torchrun not found. Please set TORCHRUN_BIN." >&2
  exit 1
fi

CONFIG="${CONFIG:-configs/stage2_unet.yaml}"
OVERRIDE="${OVERRIDE:-}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29871}"
SEED="${SEED:-42}"

CMD=(
  "${TORCHRUN_BIN}"
  --nproc_per_node="${NPROC_PER_NODE}"
  --master_port="${MASTER_PORT}"
  train_stage2.py
  --config "${CONFIG}"
  --seed "${SEED}"
)

if [[ -n "${OVERRIDE}" ]]; then
  CMD+=(--override "${OVERRIDE}")
fi

echo "[run] ${CMD[*]}"
"${CMD[@]}"
