#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/personal_workspace/chenkeyu/ReMark/Forensic-Remark_Module"
PY="${PY:-/home/ldy/miniconda3/envs/sepmark/bin/python}"

TRAIN_CSV="${TRAIN_CSV:-${ROOT}/data_manifests/celeba_hq_128_faceonly500_train_with_prompts.csv}"
VAL_CSV="${VAL_CSV:-${ROOT}/data_manifests/celeba_hq_128_faceonly50_val_with_prompts.csv}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
PROGRESS_EVERY="${PROGRESS_EVERY:-20}"
MAXB="${MAXB:-0}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MAX_REPLAY_ROUNDS="${MAX_REPLAY_ROUNDS:-6}"

REPLAY_DIR="${REPLAY_DIR:-/mnt/personal_workspace/chenkeyu/ReMark/Attack-Face-Adapter/outputs_replay_celebahq_faceonly550}"
REPLAY_JSONL="${REPLAY_DIR}/generation_results.jsonl"
SHARD_DIR="${REPLAY_DIR}/shards_parallel"

SLEEPER_RUN="/mnt/personal_workspace/chenkeyu/ReMark/Forensic-Remark_Module/runs/stage1_20260326_201745_125898"
SLEEPER_CKPT="${SLEEPER_RUN}/checkpoints/vae/epoch_010.pth"
LAWA_RUN="/mnt/personal_workspace/chenkeyu/ReMark/Forensic-Remark_Module/runs/stage1_20260326_232153_339605"
LAWA_CKPT="${LAWA_RUN}/checkpoints/vae/best.pth"

mkdir -p "${REPLAY_DIR}" "${SHARD_DIR}"
cd "${ROOT}"

echo "[$(date '+%F %T')] Building Face-Adapter replay index (multi-GPU)"
echo "[$(date '+%F %T')] train_csv=${TRAIN_CSV}"
echo "[$(date '+%F %T')] val_csv=${VAL_CSV}"
echo "[$(date '+%F %T')] gpu_list=${GPU_LIST}"

TOTAL="$(TRAIN_CSV="${TRAIN_CSV}" VAL_CSV="${VAL_CSV}" "${PY}" - <<'PY'
import csv, os
from pathlib import Path
train = os.environ['TRAIN_CSV']
val = os.environ['VAL_CSV']
seen = set()
for c in (train, val):
    with open(c, 'r', encoding='utf-8') as f:
        rd = csv.DictReader(f)
        for r in rd:
            p = (r.get('img_path') or r.get('image_path') or '').strip()
            if not p:
                continue
            ap = str(Path(p).resolve())
            if os.path.exists(ap):
                seen.add(ap)
print(len(seen))
PY
)"

if [[ -z "${TOTAL}" || "${TOTAL}" -le 0 ]]; then
  echo "[$(date '+%F %T')] ERROR: no valid samples from manifests"
  exit 1
fi
echo "[$(date '+%F %T')] replay target total=${TOTAL}"

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
N="${#GPUS[@]}"
if [[ "${N}" -le 0 ]]; then
  echo "[$(date '+%F %T')] ERROR: empty GPU_LIST"
  exit 1
fi

rm -f "${SHARD_DIR}"/replay_shard_*.jsonl "${SHARD_DIR}"/replay_shard_*.log "${REPLAY_JSONL}" || true

merge_replay_jsonl() {
  "${PY}" - <<'PY'
import json
from pathlib import Path

replay_dir = Path('/mnt/personal_workspace/chenkeyu/ReMark/Attack-Face-Adapter/outputs_replay_celebahq_faceonly550')
shard_dir = replay_dir / 'shards_parallel'
out_path = replay_dir / 'generation_results.jsonl'

rows = {}
for p in sorted(shard_dir.glob('replay_shard_*.jsonl')):
    with p.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            src = str(obj.get('source_image', '')).strip()
            outs = obj.get('outputs') or []
            if src and outs:
                rows[src] = obj

with out_path.open('w', encoding='utf-8') as w:
    for _, obj in sorted(rows.items()):
        w.write(json.dumps(obj, ensure_ascii=False) + '\n')
print(f'[merge] wrote {len(rows)} rows -> {out_path}')
PY
}

run_replay_shards_once() {
  local round="$1"
  declare -a PIDS=()
  for ((i=0; i<N; i++)); do
    start=$(( i * TOTAL / N ))
    end=$(( (i + 1) * TOTAL / N ))
    limit=$(( end - start ))
    gpu="${GPUS[$i]}"
    shard_jsonl="${SHARD_DIR}/replay_shard_${i}.jsonl"
    shard_log="${SHARD_DIR}/replay_shard_${i}.log"

    echo "[$(date '+%F %T')] round=${round} shard=${i} gpu=${gpu} range=[${start},${end}) limit=${limit}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" tools/build_ffhq_faceadapter_replay_index.py \
      --train-csv "${TRAIN_CSV}" \
      --val-csv "${VAL_CSV}" \
      --output-jsonl "${shard_jsonl}" \
      --face-repo /mnt/personal_workspace/chenkeyu/ReMark/Attack-Face-Adapter \
      --face-python /home/ldy/miniconda3/envs/FaceAdapter/bin/python \
      --cache-dir /tmp/remark_face_adapter_online_cache \
      --hf-cache /home/ldy/.cache/huggingface/hub \
      --source-csv "${TRAIN_CSV}" \
      --timeout-sec 600 \
      --num-steps 12 \
      --guidance-scale 3.5 \
      --crop-ratio 0.81 \
      --start-index "${start}" \
      --limit "${limit}" \
      --progress-every "${PROGRESS_EVERY}" \
      >> "${shard_log}" 2>&1 &
    PIDS+=("$!")
  done

  local fail=0
  for pid in "${PIDS[@]}"; do
    if ! wait "${pid}"; then
      fail=1
    fi
  done
  if [[ "${fail}" -ne 0 ]]; then
    echo "[$(date '+%F %T')] ERROR: one or more replay shards failed in round=${round}"
    return 1
  fi
  return 0
}

prev_count=0
for round in $(seq 1 "${MAX_REPLAY_ROUNDS}"); do
  run_replay_shards_once "${round}"
  echo "[$(date '+%F %T')] Merging shard jsonl -> ${REPLAY_JSONL} (round=${round})"
  merge_replay_jsonl

  merged_count="$("${PY}" - <<'PY'
import json
from pathlib import Path
p = Path('/mnt/personal_workspace/chenkeyu/ReMark/Attack-Face-Adapter/outputs_replay_celebahq_faceonly550/generation_results.jsonl')
cnt = 0
if p.exists():
    for line in p.open('r', encoding='utf-8'):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if obj.get('source_image') and (obj.get('outputs') or []):
            cnt += 1
print(cnt)
PY
)"
  echo "[$(date '+%F %T')] replay merged_count=${merged_count}/${TOTAL} (round=${round})"

  if [[ "${merged_count}" -ge "${TOTAL}" ]]; then
    echo "[$(date '+%F %T')] replay completed: ${merged_count}/${TOTAL}"
    break
  fi
  if [[ "${merged_count}" -le "${prev_count}" ]]; then
    echo "[$(date '+%F %T')] replay no progress in round=${round} (${merged_count} <= ${prev_count})"
  fi
  prev_count="${merged_count}"
done

final_count="$("${PY}" - <<'PY'
import json
from pathlib import Path
p = Path('/mnt/personal_workspace/chenkeyu/ReMark/Attack-Face-Adapter/outputs_replay_celebahq_faceonly550/generation_results.jsonl')
cnt = 0
if p.exists():
    for line in p.open('r', encoding='utf-8'):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if obj.get('source_image') and (obj.get('outputs') or []):
            cnt += 1
print(cnt)
PY
)"
if [[ "${final_count}" -lt "${TOTAL}" ]]; then
  echo "[$(date '+%F %T')] ERROR: replay incomplete after ${MAX_REPLAY_ROUNDS} rounds: ${final_count}/${TOTAL}"
  exit 3
fi

eval_one() {
  local run_dir="$1"
  local ckpt_path="$2"
  local dst_name="$3"
  local label="$4"

  if [[ ! -f "${run_dir}/config.yaml" ]]; then
    echo "[skip] missing config: ${run_dir}/config.yaml"
    return 0
  fi
  if [[ ! -f "${ckpt_path}" ]]; then
    echo "[skip] missing ckpt: ${ckpt_path}"
    return 0
  fi

  local tmp_dir
  tmp_dir="$(mktemp -d /tmp/remark_faceadapter_replay_eval_XXXXXX)"
  local tmp_cfg="${tmp_dir}/config.yaml"
  local tmp_run="${tmp_dir}/run"
  mkdir -p "${tmp_run}"

  echo "[$(date '+%F %T')] START eval ${label}"
  "${PY}" - <<'PY' "${run_dir}/config.yaml" "${tmp_cfg}" "${REPLAY_JSONL}"
import sys, yaml
src, dst, jsonl = sys.argv[1:4]
with open(src, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
cfg.setdefault('attacks', {})['online'] = ['face_adapter']
cfg['attacks']['offline'] = []
cfg.setdefault('training', {})
cfg['training']['deterministic_messages'] = True
cfg['training']['message_seed_salt'] = 'remark_v1'
cfg.setdefault('attack_options', {})
opts = cfg['attack_options']
opts['face_adapter_mode'] = 'replay'
opts['face_adapter_results_jsonl'] = jsonl
opts['face_adapter_outputs_base'] = '/mnt/personal_workspace/chenkeyu/ReMark/Attack-Face-Adapter'
opts['face_adapter_replay_key'] = 'source_image'
opts['face_adapter_allow_missing'] = False
opts['enforce_nontrivial_swap'] = False
with open(dst, 'w', encoding='utf-8') as f:
    yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
PY

  cp "${tmp_cfg}" "${tmp_run}/config.yaml"
  "${PY}" tools/eval_attacked_recon_acc.py \
    --run-dir "${tmp_run}" \
    --checkpoint "${ckpt_path}" \
    --split val \
    --max-batches "${MAXB}" \
    --batch-size 0 \
    --num-workers "${NUM_WORKERS}"

  cp "${tmp_run}/attacked_recon_eval_val.csv" "${run_dir}/${dst_name}"
  "${PY}" - <<'PY' "${run_dir}/${dst_name}" "${label}"
import csv, sys
p, label = sys.argv[1], sys.argv[2]
raw = rec = None
with open(p, 'r', encoding='utf-8') as f:
    for r in csv.DictReader(f):
        if r.get('attack') != 'face_adapter':
            continue
        if r.get('path') == 'raw_attacked':
            raw = r
        elif r.get('path') == 'vae_recon':
            rec = r
if raw is None or rec is None:
    print(f"[warn] {label} missing face_adapter rows: {p}")
else:
    rb = float(raw['bit_acc']); cb = float(rec['bit_acc'])
    print(f"[done] {label} raw_bit_acc={rb:.4f} recon_bit_acc={cb:.4f} delta={cb-rb:+.4f} csv={p}")
PY

  rm -rf "${tmp_dir}"
}

eval_one "${SLEEPER_RUN}" "${SLEEPER_CKPT}" "transfer_eval_face_adapter_replay_epoch010.csv" "sleepermark_epoch010"
eval_one "${LAWA_RUN}" "${LAWA_CKPT}" "transfer_eval_face_adapter_replay_best.csv" "lawa_best"

echo "[$(date '+%F %T')] ALL DONE sleeper+lawa face_adapter replay generalization"
