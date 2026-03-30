#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/personal_workspace/chenkeyu/ReMark"
TOOL="${ROOT}/Forensic-Remark_Module/tools/build_inpainted_replay.py"
PYTHON_BIN="${PYTHON_BIN:-python3}"

run_one() {
  local in_jsonl="$1"
  local out_jsonl="$2"
  local out_dir="$3"
  local outputs_base="$4"

  if [[ ! -f "${in_jsonl}" ]]; then
    echo "[skip] missing jsonl: ${in_jsonl}"
    return
  fi

  echo "[run] ${in_jsonl} -> ${out_jsonl}"
  "${PYTHON_BIN}" "${TOOL}" \
    --input-jsonl "${in_jsonl}" \
    --output-jsonl "${out_jsonl}" \
    --output-dir "${out_dir}" \
    --outputs-base "${outputs_base}" \
    --mode hybrid \
    --alpha 1.0 \
    --match-stats \
    --diff-threshold 0.06 \
    --diff-softness 0.03 \
    --diff-blur-ks 9 \
    --progress-every 500
}

# Face-Adapter
run_one \
  "${ROOT}/Attack-Face-Adapter/outputs/generation_results.jsonl" \
  "${ROOT}/Attack-Face-Adapter/outputs/generation_results_inpaint.jsonl" \
  "${ROOT}/Attack-Face-Adapter/outputs/inpainted_replay" \
  "${ROOT}/Attack-Face-Adapter"

# REFace
run_one \
  "${ROOT}/Attack-REFace/outputs/generation_results.jsonl" \
  "${ROOT}/Attack-REFace/outputs/generation_results_inpaint.jsonl" \
  "${ROOT}/Attack-REFace/outputs/inpainted_replay" \
  "${ROOT}/Attack-REFace"

# DiffSwap (small + large replay)
run_one \
  "${ROOT}/Attack-DiffSwap/outputs/generation_results.jsonl" \
  "${ROOT}/Attack-DiffSwap/outputs/generation_results_inpaint.jsonl" \
  "${ROOT}/Attack-DiffSwap/outputs/inpainted_replay" \
  "${ROOT}/Attack-DiffSwap"

run_one \
  "${ROOT}/Attack-DiffSwap/outputs_replay_large/generation_results.jsonl" \
  "${ROOT}/Attack-DiffSwap/outputs_replay_large/generation_results_inpaint.jsonl" \
  "${ROOT}/Attack-DiffSwap/outputs_replay_large/inpainted_replay" \
  "${ROOT}/Attack-DiffSwap"

# Arc2Face wrapper (small + large replay)
run_one \
  "${ROOT}/Attack-arc2face_wrapper/outputs/generation_results.jsonl" \
  "${ROOT}/Attack-arc2face_wrapper/outputs/generation_results_inpaint.jsonl" \
  "${ROOT}/Attack-arc2face_wrapper/outputs/inpainted_replay" \
  "${ROOT}/Attack-arc2face_wrapper"

run_one \
  "${ROOT}/Attack-arc2face_wrapper/outputs_replay_large/generation_results.jsonl" \
  "${ROOT}/Attack-arc2face_wrapper/outputs_replay_large/generation_results_inpaint.jsonl" \
  "${ROOT}/Attack-arc2face_wrapper/outputs_replay_large/inpainted_replay" \
  "${ROOT}/Attack-arc2face_wrapper"

echo "[done] replay inpaint generation finished for configured diffusion attacks."
