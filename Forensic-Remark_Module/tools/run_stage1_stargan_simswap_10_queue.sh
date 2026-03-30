#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/personal_workspace/chenkeyu/ReMark/Forensic-Remark_Module"
PYBIN="/home/ldy/miniconda3/envs/sepmark/bin/torchrun"
LOG_DIR="${ROOT}/logging"

# Default to currently free GPUs; can override via GPUS="1,4,5"
GPUS_CSV="${GPUS:-1,4,5}"
IFS=',' read -r -a GPUS <<< "${GPUS_CSV}"
NUM_WORKERS="${#GPUS[@]}"

STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

JOBS=(
  "stargan_sepmark|configs/experiments/stage1_sepmark_compact_stargan_replay2000_target65.yaml|29940"
  "stargan_lampmark|configs/experiments/stage1_lampmark_compact_stargan_replay2000_target65.yaml|29941"
  "stargan_fin|configs/experiments/stage1_fin_compact_stargan_replay2000_target65.yaml|29942"
  "stargan_trustmask|configs/experiments/stage1_trustmask_compact_stargan_replay2000_target65.yaml|29943"
  "stargan_maskwm|configs/experiments/stage1_maskwm_compact_stargan_replay2000_bs12_target65.yaml|29944"
  "simswap_sepmark|configs/experiments/stage1_sepmark_compact_simswap_replay2000.yaml|29950"
  "simswap_lampmark|configs/experiments/stage1_lampmark_compact_simswap_replay2000.yaml|29951"
  "simswap_fin|configs/experiments/stage1_fin_compact_simswap_replay2000.yaml|29952"
  "simswap_trustmask|configs/experiments/stage1_trustmask_compact_simswap_replay2000.yaml|29953"
  "simswap_maskwm|configs/experiments/stage1_maskwm_compact_simswap_replay2000_bs12.yaml|29954"
)

run_worker() {
  local worker_idx="$1"
  local gpu="${GPUS[$worker_idx]}"
  mkdir -p "${LOG_DIR}"
  cd "${ROOT}"
  echo "[worker-${worker_idx}] gpu=${gpu} started at $(date '+%F %T')"

  local i=0
  for spec in "${JOBS[@]}"; do
    if (( i % NUM_WORKERS != worker_idx )); then
      i=$((i + 1))
      continue
    fi
    IFS='|' read -r name cfg port <<< "${spec}"
    local log_path="logging/tmux_stage1_${name}_${STAMP}.log"
    echo "[worker-${worker_idx}] -> ${name} (${cfg}) port=${port} log=${log_path}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
      "${PYBIN}" --nproc_per_node=1 --master_port "${port}" \
      train_stage1.py --config configs/stage1_vae.yaml --override "${cfg}" --seed 42 \
      | tee "${log_path}"
    echo "[worker-${worker_idx}] <- ${name} done at $(date '+%F %T')"
    i=$((i + 1))
  done

  echo "[worker-${worker_idx}] all assigned jobs completed at $(date '+%F %T')"
}

if [[ "${1:-}" == "--worker" ]]; then
  if [[ -z "${WORKER_IDX:-}" ]]; then
    echo "WORKER_IDX is required in --worker mode" >&2
    exit 1
  fi
  run_worker "${WORKER_IDX}"
  exit 0
fi

mkdir -p "${LOG_DIR}"
for idx in "${!GPUS[@]}"; do
  session="q10_stg_sim_w${idx}_${STAMP}"
  tmux kill-session -t "${session}" 2>/dev/null || true
  tmux new-session -d -s "${session}" \
    "cd ${ROOT} && RUN_STAMP=${STAMP} GPUS=${GPUS_CSV} WORKER_IDX=${idx} bash tools/run_stage1_stargan_simswap_10_queue.sh --worker | tee logging/tmux_${session}.log"
done

echo "Launched queue sessions:"
tmux ls | grep "q10_stg_sim_w.*_${STAMP}" || true
echo
echo "Queue stamp: ${STAMP}"
echo "Per-worker logs:"
ls -1 "${LOG_DIR}"/tmux_q10_stg_sim_w*_"${STAMP}".log
