#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/personal_workspace/chenkeyu/ReMark/Forensic-Remark_Module"
PY="/home/ldy/miniconda3/envs/sepmark/bin/python"
GPU_ID="1"

TRAIN_CSV="$ROOT/data_manifests/ffhq_2k_train.csv"
VAL_CSV="$ROOT/data_manifests/ffhq_500_val.csv"
REPLAY_DIR="/mnt/personal_workspace/chenkeyu/ReMark/Attack-DiffSwap/outputs_replay_ffhq_2k500"
REPLAY_JSONL="$REPLAY_DIR/generation_results.jsonl"
CACHE_DIR="/tmp/remark_diffswap_online_cache"

mkdir -p "$REPLAY_DIR"
mkdir -p "$ROOT/logging"

cd "$ROOT"

echo "[$(date '+%F %T')] Building FFHQ DiffSwap replay index"
CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONUNBUFFERED=1 \
"$PY" tools/build_ffhq_diffswap_replay_index.py \
  --train-csv "$TRAIN_CSV" \
  --val-csv "$VAL_CSV" \
  --output-jsonl "$REPLAY_JSONL" \
  --diffswap-repo /mnt/personal_workspace/chenkeyu/ReMark/Attack-DiffSwap \
  --diffswap-python /home/ldy/miniconda3/envs/DiffSwap/bin/python \
  --cache-dir "$CACHE_DIR" \
  --source-csv "$TRAIN_CSV" \
  --timeout-sec 900 \
  --max-retries 3 \
  --tgt-scale 0.01 \
  --device cuda:0 \
  --progress-every 20

echo "[$(date '+%F %T')] Launching 5-WM DiffSwap replay training queue"
./tools/run_stage1_ffhq_diffswap_replay_5wm_queue.sh
