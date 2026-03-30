#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TORCHRUN="${TORCHRUN:-/home/ldy/miniconda3/envs/sepmark/bin/torchrun}"
PYTHON_BIN="${PYTHON_BIN:-/home/ldy/miniconda3/envs/sepmark/bin/python}"

GPU_SET="${GPU_SET:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

MASTER_PORT_STAGE1="${MASTER_PORT_STAGE1:-29771}"
MASTER_PORT_STAGE2_P1="${MASTER_PORT_STAGE2_P1:-29772}"
MASTER_PORT_STAGE2_P2="${MASTER_PORT_STAGE2_P2:-29773}"

RUN_TAG="${RUN_TAG:-nokl_stargan_chain_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$ROOT_DIR/logging"
mkdir -p "$LOG_DIR"

PIPELINE_LOG="$LOG_DIR/${RUN_TAG}.pipeline.log"
STAGE1_LOG="$LOG_DIR/${RUN_TAG}.stage1.log"
STAGE2_P1_LOG="$LOG_DIR/${RUN_TAG}.stage2_phase1.log"
STAGE2_P2_LOG="$LOG_DIR/${RUN_TAG}.stage2_phase2.log"
EVAL_LOG="$LOG_DIR/${RUN_TAG}.eval.log"

log() {
  echo "[$(date '+%F %T')] $*" >> "$PIPELINE_LOG"
}

extract_run_dir() {
  local log_file="$1"
  grep -m1 'Run dir' "$log_file" | sed -E 's/.*Run dir *: *//' | tr -d '\r' || true
}

log "Pipeline start: $RUN_TAG"
log "GPU_SET=$GPU_SET NPROC_PER_NODE=$NPROC_PER_NODE"
log "Logs: $PIPELINE_LOG"

log "[1/4] Stage1(no-KL) start"
CUDA_VISIBLE_DEVICES="$GPU_SET" "$TORCHRUN" \
  --master_port "$MASTER_PORT_STAGE1" \
  --nproc_per_node="$NPROC_PER_NODE" \
  train_stage1.py \
  --config configs/stage1_vae.yaml \
  --override configs/experiments/stage1_sepmark_stargan_10k_nokl_120.yaml \
  > "$STAGE1_LOG" 2>&1

STAGE1_RUN_DIR="$(extract_run_dir "$STAGE1_LOG")"
if [[ -z "$STAGE1_RUN_DIR" || ! -d "$STAGE1_RUN_DIR" ]]; then
  log "ERROR: cannot resolve Stage1 run dir from $STAGE1_LOG"
  exit 1
fi
STAGE1_CKPT="$STAGE1_RUN_DIR/checkpoints/vae/best.pth"
if [[ ! -f "$STAGE1_CKPT" ]]; then
  log "ERROR: Stage1 best checkpoint missing: $STAGE1_CKPT"
  exit 1
fi
log "Stage1 done: run_dir=$STAGE1_RUN_DIR"
log "Stage1 best ckpt: $STAGE1_CKPT"

log "[2/4] Stage2 Phase1 (stargan_fixed) start"
CUDA_VISIBLE_DEVICES="$GPU_SET" "$TORCHRUN" \
  --master_port "$MASTER_PORT_STAGE2_P1" \
  --nproc_per_node="$NPROC_PER_NODE" \
  train_stage2.py \
  --config configs/stage2_unet.yaml \
  --override configs/experiments/stage2_unet_sepmark_stargan_nokl_phase1.yaml \
  --stage1-ckpt "$STAGE1_CKPT" \
  > "$STAGE2_P1_LOG" 2>&1

STAGE2_P1_RUN_DIR="$(extract_run_dir "$STAGE2_P1_LOG")"
if [[ -z "$STAGE2_P1_RUN_DIR" || ! -d "$STAGE2_P1_RUN_DIR" ]]; then
  log "ERROR: cannot resolve Stage2 phase1 run dir from $STAGE2_P1_LOG"
  exit 1
fi
STAGE2_P1_RUN_NAME="$(basename "$STAGE2_P1_RUN_DIR")"
log "Stage2 phase1 done: run_dir=$STAGE2_P1_RUN_DIR"

log "[3/4] Stage2 Phase2 (stargan random-domain resume) start"
CUDA_VISIBLE_DEVICES="$GPU_SET" "$TORCHRUN" \
  --master_port "$MASTER_PORT_STAGE2_P2" \
  --nproc_per_node="$NPROC_PER_NODE" \
  train_stage2.py \
  --config configs/stage2_unet.yaml \
  --override configs/experiments/stage2_unet_sepmark_stargan_nokl_phase2.yaml \
  --stage1-ckpt "$STAGE1_CKPT" \
  --resume "$STAGE2_P1_RUN_NAME" \
  > "$STAGE2_P2_LOG" 2>&1

log "Stage2 phase2 done (resumed run=$STAGE2_P1_RUN_NAME)"

log "[4/4] Transfer eval (best checkpoint): stargan + simswap"
CUDA_VISIBLE_DEVICES="$GPU_SET" "$PYTHON_BIN" tools/eval_stage2_transfer.py \
  --run-dir "$STAGE2_P1_RUN_DIR" \
  --checkpoint best \
  --attacks stargan simswap \
  --max-batches 16 \
  > "$EVAL_LOG" 2>&1

log "Pipeline complete."
log "Stage1 log: $STAGE1_LOG"
log "Stage2 phase1 log: $STAGE2_P1_LOG"
log "Stage2 phase2 log: $STAGE2_P2_LOG"
log "Eval log: $EVAL_LOG"
