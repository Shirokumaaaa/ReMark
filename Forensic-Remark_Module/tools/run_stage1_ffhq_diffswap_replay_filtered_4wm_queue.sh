#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/personal_workspace/chenkeyu/ReMark/Forensic-Remark_Module"
PY="/home/ldy/miniconda3/envs/sepmark/bin/python"
BASE_CFG="configs/stage1_vae.yaml"
GPU_ID="1"

cd "$ROOT"

OVERRIDES=(
  "configs/experiments/stage1_lampmark_ffhq_diffswap_replay_filtered.yaml"
  "configs/experiments/stage1_trustmask_ffhq_diffswap_replay_filtered.yaml"
  "configs/experiments/stage1_maskwm_ffhq_diffswap_replay_filtered.yaml"
  "configs/experiments/stage1_fin_ffhq_diffswap_replay_filtered.yaml"
)

for OV in "${OVERRIDES[@]}"; do
  NAME="$(basename "$OV" .yaml)"
  TS="$(date +%Y%m%d_%H%M%S)"
  LOG="logging/${NAME}_${TS}.log"
  echo "[$(date '+%F %T')] START ${OV}"
  echo "[$(date '+%F %T')] LOG   ${LOG}"

  CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONUNBUFFERED=1 \
    "$PY" train_stage1.py --config "$BASE_CFG" --override "$OV" 2>&1 | tee "$LOG"

  echo "[$(date '+%F %T')] DONE  ${OV}"
  sleep 2
done

echo "[$(date '+%F %T')] ALL DONE (remaining 4 WM models)."
