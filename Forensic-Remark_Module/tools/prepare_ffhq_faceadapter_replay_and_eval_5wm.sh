#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/personal_workspace/chenkeyu/ReMark/Forensic-Remark_Module"
PY="/home/ldy/miniconda3/envs/sepmark/bin/python"
REPLAY_DIR="/mnt/personal_workspace/chenkeyu/ReMark/Attack-Face-Adapter/outputs_replay_ffhq_2k500"
REPLAY_JSONL="${REPLAY_DIR}/generation_results.jsonl"
SHARD_DIR="${REPLAY_DIR}/shards_parallel"

TRAIN_CSV="${TRAIN_CSV:-${ROOT}/data_manifests/ffhq_2k_train.csv}"
VAL_CSV="${VAL_CSV:-${ROOT}/data_manifests/ffhq_500_val.csv}"
PROGRESS_EVERY="${PROGRESS_EVERY:-20}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"

mkdir -p "${REPLAY_DIR}" "${SHARD_DIR}"
cd "${ROOT}"

echo "[$(date '+%F %T')] Building FFHQ Face-Adapter replay index (multi-GPU)"

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
  echo "[$(date '+%F %T')] ERROR: no valid samples found from manifests"
  exit 1
fi

echo "[$(date '+%F %T')] Replay target total=${TOTAL}"

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
N="${#GPUS[@]}"
if [[ "${N}" -le 0 ]]; then
  echo "[$(date '+%F %T')] ERROR: empty GPU_LIST"
  exit 1
fi

rm -f "${SHARD_DIR}"/replay_shard_*.jsonl "${SHARD_DIR}"/replay_shard_*.log || true

declare -a PIDS=()
for ((i=0; i<N; i++)); do
  start=$(( i * TOTAL / N ))
  end=$(( (i + 1) * TOTAL / N ))
  limit=$(( end - start ))
  gpu="${GPUS[$i]}"
  shard_jsonl="${SHARD_DIR}/replay_shard_${i}.jsonl"
  shard_log="${SHARD_DIR}/replay_shard_${i}.log"

  echo "[$(date '+%F %T')] shard=${i} gpu=${gpu} range=[${start},${end}) limit=${limit}"

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
    > "${shard_log}" 2>&1 &
  PIDS+=("$!")
done

fail=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    fail=1
  fi
done

if [[ "${fail}" -ne 0 ]]; then
  echo "[$(date '+%F %T')] ERROR: one or more replay shards failed"
  exit 2
fi

"${PY}" - <<'PY'
import json
from pathlib import Path
replay_dir = Path('/mnt/personal_workspace/chenkeyu/ReMark/Attack-Face-Adapter/outputs_replay_ffhq_2k500')
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

echo "[$(date '+%F %T')] Launching 5-WM Face-Adapter replay generalization eval"
FACE_JSONL_OVERRIDE="${REPLAY_JSONL}" MAXB=0 NUM_WORKERS=0 ./tools/run_stage1_faceadapter_replay_generalization_5wm.sh

echo "[$(date '+%F %T')] ALL DONE FFHQ Face-Adapter replay + 5-WM eval"
