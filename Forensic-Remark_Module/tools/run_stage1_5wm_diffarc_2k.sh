#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/personal_workspace/chenkeyu/ReMark/Forensic-Remark_Module"
cd "${ROOT}"

CONFIG_BASE="configs/stage1_vae.yaml"
TORCHRUN_BIN="${TORCHRUN_BIN:-/home/ldy/miniconda3/envs/sepmark/bin/torchrun}"
SEED="${SEED:-42}"
GPUS="${GPUS:-0,1,2,3,4,5}"

if [[ ! -x "${TORCHRUN_BIN}" ]]; then
  TORCHRUN_BIN="$(command -v torchrun || true)"
fi
if [[ -z "${TORCHRUN_BIN}" ]]; then
  echo "[error] torchrun not found. Set TORCHRUN_BIN explicitly."
  exit 1
fi

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logging/stage1_5wm_diffarc_2k_${TS}"
mkdir -p "${LOG_DIR}"

IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
if [[ ${#GPU_ARR[@]} -eq 0 ]]; then
  echo "[error] no GPUs configured. Set GPUS, e.g. GPUS=0,1,2,3,4,5"
  exit 1
fi
MAX_PARALLEL="${MAX_PARALLEL:-${#GPU_ARR[@]}}"
BASE_PORT="${BASE_PORT:-29600}"

declare -a JOBS=(
  "sepmark_diffswap configs/experiments/stage1_sepmark_compact_diffswap_replay2000.yaml"
  "sepmark_arc2face configs/experiments/stage1_sepmark_compact_arc2face_replay2000.yaml"
  "lampmark_diffswap configs/experiments/stage1_lampmark_compact_diffswap_replay2000.yaml"
  "lampmark_arc2face configs/experiments/stage1_lampmark_compact_arc2face_replay2000.yaml"
  "fin_diffswap configs/experiments/stage1_fin_compact_diffswap_replay2000.yaml"
  "fin_arc2face configs/experiments/stage1_fin_compact_arc2face_replay2000.yaml"
  "maskwm_diffswap configs/experiments/stage1_maskwm_compact_diffswap_replay2000.yaml"
  "maskwm_arc2face configs/experiments/stage1_maskwm_compact_arc2face_replay2000.yaml"
  "trustmask_diffswap configs/experiments/stage1_trustmask_compact_diffswap_replay2000.yaml"
  "trustmask_arc2face configs/experiments/stage1_trustmask_compact_arc2face_replay2000.yaml"
)

echo "[info] log_dir=${LOG_DIR}"
echo "[info] gpus=${GPUS} max_parallel=${MAX_PARALLEL} seed=${SEED}"

launch_job() {
  local name="$1"
  local cfg="$2"
  local gpu="$3"
  local port="$4"
  local logf="${LOG_DIR}/${name}.log"
  echo "[launch] ${name} gpu=${gpu} port=${port} cfg=${cfg}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${TORCHRUN_BIN}" --nproc_per_node=1 --master_port "${port}" \
    train_stage1.py --config "${CONFIG_BASE}" --override "${cfg}" --seed "${SEED}" \
    > "${logf}" 2>&1 &
}

num_jobs() {
  local n
  n="$(jobs -rp 2>/dev/null | wc -l || true)"
  echo "${n}"
}

i=0
for item in "${JOBS[@]}"; do
  while [[ "$(num_jobs)" -ge "${MAX_PARALLEL}" ]]; do
    sleep 5
  done
  name="${item%% *}"
  cfg="${item#* }"
  gpu="${GPU_ARR[$((i % ${#GPU_ARR[@]}))]}"
  port="$((BASE_PORT + i))"
  launch_job "${name}" "${cfg}" "${gpu}" "${port}"
  i=$((i + 1))
done

wait
echo "[done] stage1 5wm x (diffswap, arc2face) 2k matrix finished. logs=${LOG_DIR}"
