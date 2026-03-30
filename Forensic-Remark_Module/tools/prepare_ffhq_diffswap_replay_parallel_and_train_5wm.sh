#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/personal_workspace/chenkeyu/ReMark/Forensic-Remark_Module"
PY="/home/ldy/miniconda3/envs/sepmark/bin/python"
TRAIN_CSV="$ROOT/data_manifests/ffhq_2k_train.csv"
VAL_CSV="$ROOT/data_manifests/ffhq_500_val.csv"
OUT_BASE="/mnt/personal_workspace/chenkeyu/ReMark/Attack-DiffSwap/outputs_replay_ffhq_2k500"
FINAL_JSONL="$OUT_BASE/generation_results.jsonl"
SHARD_DIR="$OUT_BASE/shards_parallel"
CACHE_BASE="/tmp/remark_diffswap_online_cache_parallel"

# Use 4 GPUs in parallel for replay build.
GPUS=(1 2 3 4)
N_SHARDS=${#GPUS[@]}

mkdir -p "$OUT_BASE" "$SHARD_DIR" "$ROOT/logging" "$CACHE_BASE"
cd "$ROOT"

STAMP="$(date +%Y%m%d_%H%M%S)"

echo "[$(date '+%F %T')] Accelerated replay build start (shards=${N_SHARDS}, retries=1, timeout=300)"

TOTAL="$($PY - <<'PY'
import csv
from pathlib import Path
train=Path('/mnt/personal_workspace/chenkeyu/ReMark/Forensic-Remark_Module/data_manifests/ffhq_2k_train.csv')
val=Path('/mnt/personal_workspace/chenkeyu/ReMark/Forensic-Remark_Module/data_manifests/ffhq_500_val.csv')
seen=set()
for p in (train,val):
    with p.open('r',encoding='utf-8') as f:
        rd=csv.DictReader(f)
        for r in rd:
            ip=(r.get('img_path') or r.get('image_path') or '').strip()
            if ip:
                seen.add(str(Path(ip).resolve()))
print(len(seen))
PY
)"

if [[ -z "$TOTAL" || "$TOTAL" -le 0 ]]; then
  echo "[fatal] invalid TOTAL=$TOTAL"
  exit 1
fi

SHARD_SIZE=$(( (TOTAL + N_SHARDS - 1) / N_SHARDS ))
echo "[$(date '+%F %T')] TOTAL=${TOTAL}, SHARD_SIZE=${SHARD_SIZE}"

PIDS=()
LOGS=()
for i in "${!GPUS[@]}"; do
  gpu="${GPUS[$i]}"
  start=$(( i * SHARD_SIZE ))
  if [[ "$start" -ge "$TOTAL" ]]; then
    continue
  fi
  limit="$SHARD_SIZE"
  shard_jsonl="$SHARD_DIR/replay_shard_${i}.jsonl"
  shard_cache="$CACHE_BASE/shard_${i}"
  shard_log="$ROOT/logging/diffswap_replay_shard${i}_${STAMP}.log"
  mkdir -p "$shard_cache"

  echo "[$(date '+%F %T')] shard=${i} gpu=${gpu} start=${start} limit=${limit} -> ${shard_jsonl}"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
  "$PY" tools/build_ffhq_diffswap_replay_index.py \
    --train-csv "$TRAIN_CSV" \
    --val-csv "$VAL_CSV" \
    --output-jsonl "$shard_jsonl" \
    --diffswap-repo /mnt/personal_workspace/chenkeyu/ReMark/Attack-DiffSwap \
    --diffswap-python /home/ldy/miniconda3/envs/DiffSwap/bin/python \
    --cache-dir "$shard_cache" \
    --source-csv "$TRAIN_CSV" \
    --timeout-sec 300 \
    --max-retries 1 \
    --tgt-scale 0.01 \
    --start-index "$start" \
    --limit "$limit" \
    --device cuda:0 \
    --progress-every 20 \
    > "$shard_log" 2>&1 &

  PIDS+=("$!")
  LOGS+=("$shard_log")
done

FAIL=0
for p in "${PIDS[@]}"; do
  if ! wait "$p"; then
    FAIL=1
  fi
done

if [[ "$FAIL" -ne 0 ]]; then
  echo "[$(date '+%F %T')] [warn] one or more replay shards exited with non-zero status; continuing to merge available outputs."
fi

echo "[$(date '+%F %T')] Merging shard jsonl -> ${FINAL_JSONL}"
"$PY" - <<'PY'
import json
from pathlib import Path

out = Path('/mnt/personal_workspace/chenkeyu/ReMark/Attack-DiffSwap/outputs_replay_ffhq_2k500/generation_results.jsonl')
shard_dir = Path('/mnt/personal_workspace/chenkeyu/ReMark/Attack-DiffSwap/outputs_replay_ffhq_2k500/shards_parallel')

inputs = []
if out.exists():
    inputs.append(out)
inputs.extend(sorted(shard_dir.glob('replay_shard_*.jsonl')))

records = {}
for p in inputs:
    if not p.exists():
        continue
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
            outs = obj.get('outputs', [])
            if not src or not outs:
                continue
            records[src] = obj

tmp = out.with_suffix('.jsonl.tmp')
with tmp.open('w', encoding='utf-8') as w:
    for k in sorted(records.keys()):
        w.write(json.dumps(records[k], ensure_ascii=False) + '\n')

tmp.replace(out)
print(f'[merge] unique_records={len(records)} -> {out}')
PY

echo "[$(date '+%F %T')] Replay build done; launching 5-WM replay training queue"
./tools/run_stage1_ffhq_diffswap_replay_5wm_queue.sh
