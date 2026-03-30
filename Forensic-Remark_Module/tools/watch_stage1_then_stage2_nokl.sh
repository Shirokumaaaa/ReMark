#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TORCHRUN="${TORCHRUN:-/home/ldy/miniconda3/envs/sepmark/bin/torchrun}"
PYTHON_BIN="${PYTHON_BIN:-/home/ldy/miniconda3/envs/sepmark/bin/python}"

STAGE1_RUN_NAME="${STAGE1_RUN_NAME:-stage1_20260312_221712}"
STAGE1_OVERRIDE_CFG="${STAGE1_OVERRIDE_CFG:-configs/experiments/stage1_sepmark_stargan_10k_nokl_120.yaml}"
GPU_SET="${GPU_SET:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
POLL_SECONDS="${POLL_SECONDS:-120}"

MASTER_PORT_STAGE2_P1="${MASTER_PORT_STAGE2_P1:-29772}"
MASTER_PORT_STAGE2_P2="${MASTER_PORT_STAGE2_P2:-29773}"

PHASE1_OVERRIDE="${PHASE1_OVERRIDE:-configs/experiments/stage2_unet_sepmark_stargan_nokl_phase1.yaml}"
PHASE2_OVERRIDE="${PHASE2_OVERRIDE:-configs/experiments/stage2_unet_sepmark_stargan_nokl_phase2.yaml}"

RUN_TAG="${RUN_TAG:-watch_nokl_chain_${STAGE1_RUN_NAME}_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$ROOT_DIR/logging"
mkdir -p "$LOG_DIR"

PIPELINE_LOG="$LOG_DIR/${RUN_TAG}.pipeline.log"
STAGE2_P1_LOG="$LOG_DIR/${RUN_TAG}.stage2_phase1.log"
STAGE2_P2_LOG="$LOG_DIR/${RUN_TAG}.stage2_phase2.log"
EVAL_LOG="$LOG_DIR/${RUN_TAG}.eval.log"

STAGE1_RUN_DIR="$ROOT_DIR/runs/$STAGE1_RUN_NAME"
STAGE1_LOG="$STAGE1_RUN_DIR/train.log"
STAGE1_BEST="$STAGE1_RUN_DIR/checkpoints/vae/best.pth"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$PIPELINE_LOG"
}

extract_run_dir() {
  local log_file="$1"
  grep -m1 'Run dir' "$log_file" | sed -E 's/.*Run dir *: *//' | tr -d '\r' || true
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    log "ERROR: missing file: $path"
    exit 1
  fi
}

last_finished_epoch() {
  local log_file="$1"
  if [[ ! -f "$log_file" ]]; then
    echo "-1"
    return 0
  fi
  local e
  e="$(grep -Eo 'Epoch [0-9]{3} \|' "$log_file" | awk '{print $2}' | sort -n | tail -n1 || true)"
  if [[ -z "$e" ]]; then
    echo "-1"
  else
    echo "$((10#$e))"
  fi
}

is_stage1_running() {
  pgrep -af "train_stage1.py.*--resume ${STAGE1_RUN_NAME}|train_stage1.py.*${STAGE1_RUN_NAME}" >/dev/null 2>&1
}

if [[ ! -d "$STAGE1_RUN_DIR" ]]; then
  log "ERROR: stage1 run dir not found: $STAGE1_RUN_DIR"
  exit 1
fi
require_file "$STAGE1_LOG"
require_file "$STAGE1_BEST"

if [[ -n "${TARGET_EPOCH:-}" ]]; then
  TARGET_EPOCH="$TARGET_EPOCH"
elif [[ -f "$STAGE1_OVERRIDE_CFG" ]]; then
  TARGET_EPOCH="$("$PYTHON_BIN" - <<'PY' "$STAGE1_OVERRIDE_CFG"
import sys, yaml
cfg_path = sys.argv[1]
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
epochs = int(cfg.get("training", {}).get("epochs", 0))
print(max(epochs - 1, 0))
PY
)"
else
  TARGET_EPOCH="$("$PYTHON_BIN" - <<'PY' "$STAGE1_RUN_DIR/config.yaml"
import sys, yaml
cfg_path = sys.argv[1]
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
epochs = int(cfg.get("training", {}).get("epochs", 0))
print(max(epochs - 1, 0))
PY
)"
fi

log "Watcher start: run=$STAGE1_RUN_NAME target_epoch=$TARGET_EPOCH gpu=$GPU_SET nproc=$NPROC_PER_NODE"
log "Stage1 override cfg: $STAGE1_OVERRIDE_CFG"
log "Phase1 override: $PHASE1_OVERRIDE"
log "Phase2 override: $PHASE2_OVERRIDE"

while true; do
  CUR_EPOCH="$(last_finished_epoch "$STAGE1_LOG")"
  if [[ "$CUR_EPOCH" =~ ^[0-9]+$ ]] && (( CUR_EPOCH >= TARGET_EPOCH )); then
    log "Stage1 finished: last_epoch=$CUR_EPOCH target_epoch=$TARGET_EPOCH"
    break
  fi
  if is_stage1_running; then
    log "Stage1 running... last_finished_epoch=$CUR_EPOCH"
  else
    log "Stage1 not running yet; waiting... last_finished_epoch=$CUR_EPOCH target_epoch=$TARGET_EPOCH"
  fi
  sleep "$POLL_SECONDS"
done

if [[ ! -f "$STAGE1_BEST" ]]; then
  log "ERROR: Stage1 best checkpoint missing before Stage2: $STAGE1_BEST"
  exit 1
fi
log "Stage1 best ckpt: $STAGE1_BEST"

log "[1/3] Stage2 Phase1 start"
CUDA_VISIBLE_DEVICES="$GPU_SET" "$TORCHRUN" \
  --master_port "$MASTER_PORT_STAGE2_P1" \
  --nproc_per_node="$NPROC_PER_NODE" \
  train_stage2.py \
  --config configs/stage2_unet.yaml \
  --override "$PHASE1_OVERRIDE" \
  --stage1-ckpt "$STAGE1_BEST" \
  > "$STAGE2_P1_LOG" 2>&1

STAGE2_RUN_DIR="$(extract_run_dir "$STAGE2_P1_LOG")"
if [[ -z "$STAGE2_RUN_DIR" || ! -d "$STAGE2_RUN_DIR" ]]; then
  log "ERROR: cannot resolve Stage2 run dir from $STAGE2_P1_LOG"
  exit 1
fi
STAGE2_RUN_NAME="$(basename "$STAGE2_RUN_DIR")"
log "Stage2 Phase1 done: run_dir=$STAGE2_RUN_DIR"

log "[2/3] Stage2 Phase2 resume start"
CUDA_VISIBLE_DEVICES="$GPU_SET" "$TORCHRUN" \
  --master_port "$MASTER_PORT_STAGE2_P2" \
  --nproc_per_node="$NPROC_PER_NODE" \
  train_stage2.py \
  --config configs/stage2_unet.yaml \
  --override "$PHASE2_OVERRIDE" \
  --stage1-ckpt "$STAGE1_BEST" \
  --resume "$STAGE2_RUN_NAME" \
  > "$STAGE2_P2_LOG" 2>&1

log "Stage2 Phase2 done: resumed_run=$STAGE2_RUN_NAME"

log "[3/3] Transfer eval start (stargan+simswap)"
CUDA_VISIBLE_DEVICES="$GPU_SET" "$PYTHON_BIN" tools/eval_stage2_transfer.py \
  --run-dir "$STAGE2_RUN_DIR" \
  --checkpoint best \
  --attacks stargan simswap \
  --max-batches 16 \
  > "$EVAL_LOG" 2>&1

log "Pipeline complete."
log "Stage2 run dir: $STAGE2_RUN_DIR"
log "Phase1 log: $STAGE2_P1_LOG"
log "Phase2 log: $STAGE2_P2_LOG"
log "Eval log: $EVAL_LOG"
