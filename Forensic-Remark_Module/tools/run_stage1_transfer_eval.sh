#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <run_dir> [checkpoint(best|last|/abs/path)] [max_batches]"
  exit 1
fi

RUN_DIR="$(python3 - <<'PY' "$1"
import os,sys
print(os.path.abspath(sys.argv[1]))
PY
)"
CKPT="${2:-best}"
MAXB="${3:-0}"
PYBIN="${PYBIN:-/home/ldy/miniconda3/envs/sepmark/bin/python}"

if [[ ! -f "${RUN_DIR}/config.yaml" ]]; then
  echo "config missing: ${RUN_DIR}/config.yaml"
  exit 1
fi

WM_MODEL="$(python3 - <<'PY' "${RUN_DIR}/config.yaml"
import sys,yaml
cfg=yaml.safe_load(open(sys.argv[1],'r',encoding='utf-8'))
print(str(cfg.get('wm_model','sepmark')).strip())
PY
)"

if [[ "${CKPT}" == "best" || "${CKPT}" == "last" ]]; then
  CKPT_PATH="${RUN_DIR}/checkpoints/vae/${CKPT}.pth"
else
  CKPT_PATH="${CKPT}"
fi

if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "checkpoint missing: ${CKPT_PATH}"
  exit 1
fi

pick_jsonl() {
  local CAND1="$1"
  local CAND2="$2"
  local CAND3="$3"
  if [[ -f "${CAND1}" ]]; then
    echo "${CAND1}"
    return 0
  fi
  if [[ -f "${CAND2}" ]]; then
    echo "${CAND2}"
    return 0
  fi
  if [[ -f "${CAND3}" ]]; then
    echo "${CAND3}"
    return 0
  fi
  return 1
}

FACE_JSONL="${FACE_JSONL_OVERRIDE:-$(pick_jsonl \
  "/mnt/personal_workspace/chenkeyu/ReMark/Attack-Face-Adapter/outputs_replay_large/generation_results_wm_aligned_${WM_MODEL}_large.jsonl" \
  "/mnt/personal_workspace/chenkeyu/ReMark/Attack-Face-Adapter/outputs/generation_results_wm_aligned_${WM_MODEL}.jsonl" \
  "/mnt/personal_workspace/chenkeyu/ReMark/Attack-Face-Adapter/outputs_replay_large/generation_results.jsonl" \
)}"

REFACE_JSONL="${REFACE_JSONL_OVERRIDE:-$(pick_jsonl \
  "/mnt/personal_workspace/chenkeyu/ReMark/Attack-REFace/outputs_replay_large/generation_results_wm_aligned_${WM_MODEL}_large.jsonl" \
  "/mnt/personal_workspace/chenkeyu/ReMark/Attack-REFace/outputs/generation_results_wm_aligned_${WM_MODEL}.jsonl" \
  "/mnt/personal_workspace/chenkeyu/ReMark/Attack-REFace/outputs_replay_large/generation_results.jsonl" \
)}"

run_one() {
  local ATTACK="$1"
  local JSONL_PATH="$2"
  local BASE_PATH="$3"

  local TMP_DIR
  TMP_DIR="$(mktemp -d /tmp/remark_transfer_${ATTACK}_XXXXXX)"
  local TMP_CFG="${TMP_DIR}/config.yaml"
  local TMP_CSV="${TMP_DIR}/manifest_${ATTACK}.csv"

  ${PYBIN} tools/build_manifest_from_replay_jsonl.py \
    --input-jsonl "${JSONL_PATH}" \
    --output-csv "${TMP_CSV}" \
    --source-key source_image \
    --require-ok

  python3 - <<'PY' "${RUN_DIR}/config.yaml" "${TMP_CFG}" "${ATTACK}" "${TMP_CSV}" "${JSONL_PATH}" "${BASE_PATH}"
import os,sys,yaml
src,dst,attack,csv_path,jsonl_path,base = sys.argv[1:7]
with open(src,'r',encoding='utf-8') as f:
    cfg=yaml.safe_load(f)
cfg.setdefault('attacks',{})['online']=[attack]
cfg['attacks']['offline']=[]
cfg.setdefault('data',{})['val_csv']=csv_path
cfg.setdefault('training',{})
cfg['training']['deterministic_messages']=True
cfg['training']['message_seed_salt']='remark_v1'
cfg.setdefault('attack_options',{})
opts=cfg['attack_options']
if attack.lower()=='face_adapter':
    opts['face_adapter_results_jsonl']=jsonl_path
    opts['face_adapter_outputs_base']=base
    opts['face_adapter_replay_key']='source_image'
    opts['face_adapter_allow_missing']=False
elif attack.lower()=='reface':
    opts['reface_results_jsonl']=jsonl_path
    opts['reface_outputs_base']=base
    opts['reface_replay_key']='source_image'
    opts['reface_allow_missing']=False
opts['enforce_nontrivial_swap']=True
opts['nontrivial_swap_eps']=1.0e-4
with open(dst,'w',encoding='utf-8') as f:
    yaml.safe_dump(cfg,f,allow_unicode=True,sort_keys=False)
PY

  ${PYBIN} tools/eval_attacked_recon_acc.py \
    --run-dir "${TMP_DIR}" \
    --checkpoint "${CKPT_PATH}" \
    --split val \
    --max-batches "${MAXB}" \
    --batch-size 0 \
    --num-workers 0

  local SRC_CSV="${TMP_DIR}/attacked_recon_eval_val.csv"
  local DST_CSV="${RUN_DIR}/transfer_eval_${ATTACK}.csv"
  cp "${SRC_CSV}" "${DST_CSV}"
  echo "[done] ${ATTACK} -> ${DST_CSV}"
  rm -rf "${TMP_DIR}"
}

run_one \
  face_adapter \
  "${FACE_JSONL}" \
  /mnt/personal_workspace/chenkeyu/ReMark/Attack-Face-Adapter

run_one \
  reface \
  "${REFACE_JSONL}" \
  /mnt/personal_workspace/chenkeyu/ReMark/Attack-REFace

echo "[all done] transfer eval finished for ${RUN_DIR}"
