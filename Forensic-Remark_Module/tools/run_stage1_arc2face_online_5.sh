#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/personal_workspace/chenkeyu/ReMark/Forensic-Remark_Module"
PYBIN="/home/ldy/miniconda3/envs/sepmark/bin/torchrun"

mkdir -p "${ROOT}/logging"

run_one() {
  local session="$1"
  local gpu="$2"
  local port="$3"
  local override="$4"
  local log="$5"

  tmux kill-session -t "${session}" 2>/dev/null || true
  tmux new-session -d -s "${session}" \
    "cd ${ROOT} && CUDA_VISIBLE_DEVICES=${gpu} ${PYBIN} --nproc_per_node=1 --master_port ${port} train_stage1.py --config configs/stage1_vae.yaml --override ${override} --seed 42 | tee ${log}"
}

run_one "arc2_sepmark"   "0" "29820" "configs/experiments/stage1_sepmark_compact_arc2face_online2000.yaml"      "logging/tmux_stage1_arc2face_sepmark_online2k.log"
run_one "arc2_lampmark"  "2" "29821" "configs/experiments/stage1_lampmark_compact_arc2face_online2000.yaml"     "logging/tmux_stage1_arc2face_lampmark_online2k.log"
run_one "arc2_fin"       "3" "29822" "configs/experiments/stage1_fin_compact_arc2face_online2000.yaml"          "logging/tmux_stage1_arc2face_fin_online2k.log"
run_one "arc2_trustmask" "4" "29823" "configs/experiments/stage1_trustmask_compact_arc2face_online2000.yaml"    "logging/tmux_stage1_arc2face_trustmask_online2k.log"
run_one "arc2_maskwm"    "5" "29824" "configs/experiments/stage1_maskwm_compact_arc2face_online2000_bs12.yaml"  "logging/tmux_stage1_arc2face_maskwm_online2k.log"

echo "Started tmux sessions:"
tmux ls | grep '^arc2_' || true
echo
echo "Logs:"
ls -1 "${ROOT}"/logging/tmux_stage1_arc2face_*_online2k.log
