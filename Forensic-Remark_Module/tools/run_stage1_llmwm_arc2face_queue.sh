#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/personal_workspace/chenkeyu/ReMark/Forensic-Remark_Module"
TORCHRUN="${TORCHRUN:-/home/ldy/miniconda3/envs/sepmark/bin/torchrun}"
GPUS="${GPUS:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$(awk -F',' '{print NF}' <<< "${GPUS}")}"
LOG_DIR="${ROOT}/logging"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/llmwm_arc2face_queue_${STAMP}.log"

mkdir -p "${LOG_DIR}"
cd "${ROOT}"

run_one() {
  local cfg="$1"
  local tag="$2"
  echo "[$(date '+%F %T')] START ${tag} cfg=${cfg} gpus=${GPUS} nproc=${NPROC_PER_NODE}" | tee -a "${LOG_FILE}"
  CUDA_VISIBLE_DEVICES="${GPUS}" OMP_NUM_THREADS=4 \
    "${TORCHRUN}" --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    train_stage1.py --config "${cfg}" 2>&1 | tee -a "${LOG_FILE}"
  echo "[$(date '+%F %T')] DONE ${tag}" | tee -a "${LOG_FILE}"
}

run_one "configs/experiments/stage1_sleepermark_celebahq_simswap_llmwm.yaml" "sleepermark_arc2face"
run_one "configs/experiments/stage1_tagwm_celebahq_simswap_llmwm.yaml" "tagwm_arc2face"
run_one "configs/experiments/stage1_lawa_celebahq_simswap_llmwm.yaml" "lawa_arc2face"

echo "[$(date '+%F %T')] ALL_DONE llmwm_arc2face_queue" | tee -a "${LOG_FILE}"
echo "${LOG_FILE}"
