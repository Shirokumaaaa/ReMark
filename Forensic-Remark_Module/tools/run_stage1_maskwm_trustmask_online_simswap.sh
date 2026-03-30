#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/personal_workspace/chenkeyu/ReMark/Forensic-Remark_Module"
PYTHON_BIN="${PYTHON_BIN:-/home/ldy/miniconda3/envs/sepmark/bin/python}"
LOG_DIR="${ROOT}/logging"
mkdir -p "${LOG_DIR}"

TS="$(date +%Y%m%d_%H%M%S)"

run_one () {
  local gpu="$1"
  local cfg="$2"
  local tag="$3"
  local log_file="${LOG_DIR}/${tag}_${TS}.log"

  echo "[launch] gpu=${gpu} tag=${tag}"
  nohup env CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${ROOT}/train_stage1.py" \
    --config "${ROOT}/configs/stage1_vae.yaml" \
    --override "${ROOT}/${cfg}" \
    > "${log_file}" 2>&1 &

  local pid=$!
  echo "[ok] pid=${pid} log=${log_file}"
}

run_one 4 "configs/experiments/stage1_maskwm_compact_simswap_10k_online.yaml" "stage1_maskwm_online_simswap"
run_one 5 "configs/experiments/stage1_trustmask_compact_simswap_10k_online.yaml" "stage1_trustmask_online_simswap"

echo "[done] launched 2 online stage1 jobs (MaskWM/TrustMask + SimSwap)."
