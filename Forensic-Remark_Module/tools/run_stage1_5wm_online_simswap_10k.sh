#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/personal_workspace/chenkeyu/ReMark/Forensic-Remark_Module"
PYTHON_BIN="${PYTHON_BIN:-/home/ldy/miniconda3/envs/sepmark/bin/python}"
LOG_DIR="${ROOT}/logging"
mkdir -p "${LOG_DIR}"

TS="$(date +%Y%m%d_%H%M%S)"

# 默认按 5 张卡一一对应启动；如需调整可在启动前覆写这些环境变量。
GPU_SEPMARK="${GPU_SEPMARK:-0}"
GPU_LAMPMARK="${GPU_LAMPMARK:-1}"
GPU_FIN="${GPU_FIN:-2}"
GPU_TRUSTMASK="${GPU_TRUSTMASK:-3}"
GPU_MASKWM="${GPU_MASKWM:-4}"

run_one() {
  local gpu="$1"
  local cfg="$2"
  local tag="$3"
  local log_file="${LOG_DIR}/${tag}_${TS}.log"

  echo "[launch] gpu=${gpu} tag=${tag}"
  nohup env CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${ROOT}/train_stage1.py" \
    --config "${ROOT}/configs/stage1_vae.yaml" \
    --override "${ROOT}/${cfg}" \
    --seed 42 \
    > "${log_file}" 2>&1 &

  local pid=$!
  echo "[ok] pid=${pid} log=${log_file}"
}

run_one "${GPU_SEPMARK}" "configs/experiments/stage1_sepmark_compact_simswap_10k.yaml" "stage1_sepmark_online_simswap_10k"
run_one "${GPU_LAMPMARK}" "configs/experiments/stage1_lampmark_compact_simswap_10k.yaml" "stage1_lampmark_online_simswap_10k"
run_one "${GPU_FIN}" "configs/experiments/stage1_fin_compact_simswap_10k_online.yaml" "stage1_fin_online_simswap_10k"
run_one "${GPU_TRUSTMASK}" "configs/experiments/stage1_trustmask_compact_simswap_10k_online.yaml" "stage1_trustmask_online_simswap_10k"
run_one "${GPU_MASKWM}" "configs/experiments/stage1_maskwm_compact_simswap_10k_online.yaml" "stage1_maskwm_online_simswap_10k"

echo "[done] launched 5 online SimSwap Stage1 jobs (SepMark/LampMark/FIN/TrustMark/MaskWM)."
