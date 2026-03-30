#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/personal_workspace/chenkeyu/ReMark/Forensic-Remark_Module"
cd "${ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"

# Rebuild manifests from inpaint jsonl to ensure source list is aligned.
"${PYTHON_BIN}" tools/build_manifest_from_replay_jsonl.py \
  --input-jsonl /mnt/personal_workspace/chenkeyu/ReMark/Attack-Face-Adapter/outputs/generation_results_inpaint.jsonl \
  --output-csv data_manifests/preflight_face_adapter_inpaint.csv

"${PYTHON_BIN}" tools/build_manifest_from_replay_jsonl.py \
  --input-jsonl /mnt/personal_workspace/chenkeyu/ReMark/Attack-REFace/outputs/generation_results_inpaint.jsonl \
  --output-csv data_manifests/preflight_reface_inpaint.csv

"${PYTHON_BIN}" tools/build_manifest_from_replay_jsonl.py \
  --input-jsonl /mnt/personal_workspace/chenkeyu/ReMark/Attack-DiffSwap/outputs/generation_results_inpaint.jsonl \
  --output-csv data_manifests/preflight_diffswap_inpaint.csv

"${PYTHON_BIN}" tools/build_manifest_from_replay_jsonl.py \
  --input-jsonl /mnt/personal_workspace/chenkeyu/ReMark/Attack-arc2face_wrapper/outputs/generation_results_inpaint.jsonl \
  --output-csv data_manifests/preflight_arc2face_inpaint.csv

declare -a WMS=(sepmark lampmark fin)
declare -a ATTACKS=(face_adapter reface diffswap arc2face)

for wm in "${WMS[@]}"; do
  for atk in "${ATTACKS[@]}"; do
    cfg="configs/experiments/preflight_${wm}_${atk}_inpaint.yaml"
    echo "[run] wm=${wm} attack=${atk} cfg=${cfg}"
    "${TORCHRUN_BIN}" --nproc_per_node=1 train_stage1.py \
      --config configs/stage1_vae.yaml \
      --override "${cfg}" \
      --seed 42
  done
done

echo "[done] all inpaint preflights finished (wm: sepmark/lampmark/fin; attacks: face_adapter/reface/diffswap/arc2face)."
