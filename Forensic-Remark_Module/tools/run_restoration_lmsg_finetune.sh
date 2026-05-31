#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

MODEL="${1:-all}"
CONFIG="configs/stage1_vae.yaml"
BASE="configs/experiments/restoration_lmsg_base.yaml"
PYTHON_BIN="${PYTHON_BIN:-python3}"

declare -A MODEL_OVERRIDES=(
  [bdg36m]="configs/experiments/restoration_lmsg_bdg36m.yaml"
  [defusion]="configs/experiments/restoration_lmsg_defusion.yaml"
  [fape_ir]="configs/experiments/restoration_lmsg_fape_ir.yaml"
  [rar]="configs/experiments/restoration_lmsg_rar.yaml"
  [bdg_sd2]="configs/experiments/restoration_lmsg_bdg_sd2.yaml"
)

run_one() {
  local name="$1"
  local override="${MODEL_OVERRIDES[$name]}"
  echo "[run] ${name}: ${BASE},${override}"
  "${PYTHON_BIN}" train_stage1.py \
    --config "${CONFIG}" \
    --override "${BASE},${override}"
}

if [[ "${MODEL}" == "all" ]]; then
  for name in bdg36m defusion fape_ir rar bdg_sd2; do
    run_one "${name}"
  done
else
  if [[ -z "${MODEL_OVERRIDES[$MODEL]:-}" ]]; then
    echo "Unknown model '${MODEL}'. Use one of: all bdg36m defusion fape_ir rar bdg_sd2" >&2
    exit 2
  fi
  run_one "${MODEL}"
fi
