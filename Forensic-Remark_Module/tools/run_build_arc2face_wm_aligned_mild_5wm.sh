#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/personal_workspace/chenkeyu/ReMark
MOD=${ROOT}/Forensic-Remark_Module
OUT=${ROOT}/Attack-arc2face_wrapper/outputs_replay_large
IN_JSONL=${OUT}/generation_results_inpaint.jsonl

# mild profile: reduce face replacement strength while preserving non-face area.
MODE=hybrid
ALPHA=0.70
DIFF_THR=0.10
DIFF_SOFT=0.04
DIFF_BLUR=9

cd "${MOD}"

build_one() {
  local wm="$1"
  local cfg="$2"
  local gpu="$3"
  CUDA_VISIBLE_DEVICES="${gpu}" /home/ldy/miniconda3/envs/sepmark/bin/python tools/build_wm_aligned_replay.py \
    --config configs/stage1_vae.yaml \
    --override "${cfg}" \
    --input-jsonl "${IN_JSONL}" \
    --output-jsonl "${OUT}/generation_results_wm_aligned_${wm}_mild.jsonl" \
    --output-dir "${OUT}/wm_aligned_${wm}_mild" \
    --outputs-base /mnt/personal_workspace/chenkeyu/ReMark/Attack-arc2face_wrapper \
    --source-key source_image \
    --message-seed-salt remark_v1 \
    --mode "${MODE}" \
    --alpha "${ALPHA}" \
    --match-stats \
    --diff-threshold "${DIFF_THR}" \
    --diff-softness "${DIFF_SOFT}" \
    --diff-blur-ks "${DIFF_BLUR}" \
    --progress-every 200
}

# adjust GPU mapping as needed
build_one sepmark   configs/experiments/stage1_sepmark_compact_arc2face_replay2000.yaml 1
build_one lampmark  configs/experiments/stage1_lampmark_compact_arc2face_replay2000.yaml 1
build_one fin       configs/experiments/stage1_fin_compact_arc2face_replay2000.yaml 1
build_one trustmask configs/experiments/stage1_trustmask_compact_arc2face_replay2000.yaml 1
build_one maskwm    configs/experiments/stage1_maskwm_compact_arc2face_replay2000_bs12.yaml 1
