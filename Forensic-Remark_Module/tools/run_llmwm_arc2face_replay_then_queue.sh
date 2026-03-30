#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/personal_workspace/chenkeyu/ReMark"
FRM="${ROOT}/Forensic-Remark_Module"
A2F="${ROOT}/Attack-arc2face_wrapper"
PYBIN="${PYBIN:-/home/ldy/miniconda3/envs/sepmark/bin/python}"
LOG_DIR="${FRM}/logging"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/llmwm_arc2face_replay_then_queue_${STAMP}.log"

TRAIN_CSV="${FRM}/data_manifests/celeba_hq_128_faceonly500_train_with_prompts.csv"
VAL_CSV="${FRM}/data_manifests/celeba_hq_128_faceonly50_val_with_prompts.csv"
MERGED_CSV="${FRM}/data_manifests/celeba_hq_128_faceonly550_for_arc2face.csv"
TRIPLETS_CSV="${FRM}/data_manifests/celeba_hq_128_faceonly550_arc2face_triplets.csv"
REPLAY_OUT="${A2F}/outputs_replay_celebahq_faceonly550"
REPLAY_GPUS="${REPLAY_GPUS:-1,2,3}"
IFS=',' read -r -a REPLAY_GPU_ARR <<< "${REPLAY_GPUS}"

mkdir -p "${LOG_DIR}" "${REPLAY_OUT}"

echo "[$(date '+%F %T')] START pipeline" | tee -a "${LOG_FILE}"
echo "LOG_FILE=${LOG_FILE}" | tee -a "${LOG_FILE}"

echo "[$(date '+%F %T')] build merged csv: ${MERGED_CSV}" | tee -a "${LOG_FILE}"
"${PYBIN}" - <<'PY' 2>&1 | tee -a "${LOG_FILE}"
import csv
from pathlib import Path

train_csv = Path("/mnt/personal_workspace/chenkeyu/ReMark/Forensic-Remark_Module/data_manifests/celeba_hq_128_faceonly500_train_with_prompts.csv")
val_csv = Path("/mnt/personal_workspace/chenkeyu/ReMark/Forensic-Remark_Module/data_manifests/celeba_hq_128_faceonly50_val_with_prompts.csv")
out_csv = Path("/mnt/personal_workspace/chenkeyu/ReMark/Forensic-Remark_Module/data_manifests/celeba_hq_128_faceonly550_for_arc2face.csv")

seen = set()
rows = []
for p in (train_csv, val_csv):
    with p.open("r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            ip = r.get("img_path", "").strip()
            if not ip:
                continue
            if ip in seen:
                continue
            seen.add(ip)
            rows.append({"img_path": ip})

out_csv.parent.mkdir(parents=True, exist_ok=True)
with out_csv.open("w", newline="", encoding="utf-8") as f:
    wr = csv.DictWriter(f, fieldnames=["img_path"])
    wr.writeheader()
    wr.writerows(rows)
print(f"[OK] merged rows={len(rows)} -> {out_csv}")
PY

echo "[$(date '+%F %T')] build triplets: ${TRIPLETS_CSV}" | tee -a "${LOG_FILE}"
"${PYBIN}" "${A2F}/scripts/build_triplets_random_source.py" \
  --input_csv "${MERGED_CSV}" \
  --output_csv "${TRIPLETS_CSV}" \
  --seed 1234 2>&1 | tee -a "${LOG_FILE}"

TOTAL_ROWS="$("${PYBIN}" - <<'PY'
import csv
from pathlib import Path
p = Path("/mnt/personal_workspace/chenkeyu/ReMark/Forensic-Remark_Module/data_manifests/celeba_hq_128_faceonly550_arc2face_triplets.csv")
with p.open("r", encoding="utf-8") as f:
    rd = csv.DictReader(f)
    print(sum(1 for _ in rd))
PY
)"
if [[ -z "${TOTAL_ROWS}" || "${TOTAL_ROWS}" -le 0 ]]; then
  echo "[ERROR] invalid TOTAL_ROWS=${TOTAL_ROWS}" | tee -a "${LOG_FILE}"
  exit 1
fi

NUM_SHARDS="${#REPLAY_GPU_ARR[@]}"
CHUNK_SIZE="$(( (TOTAL_ROWS + NUM_SHARDS - 1) / NUM_SHARDS ))"
SHARD_DIR="${REPLAY_OUT}/shards_parallel"
mkdir -p "${SHARD_DIR}"
rm -f "${REPLAY_OUT}/generation_results.jsonl"

echo "[$(date '+%F %T')] generate replay on multi-GPU: gpus=${REPLAY_GPUS} total_rows=${TOTAL_ROWS} shards=${NUM_SHARDS} chunk=${CHUNK_SIZE}" | tee -a "${LOG_FILE}"

pids=()
for i in "${!REPLAY_GPU_ARR[@]}"; do
  gpu="${REPLAY_GPU_ARR[$i]}"
  start="$(( i * CHUNK_SIZE ))"
  if [[ "${start}" -ge "${TOTAL_ROWS}" ]]; then
    continue
  fi
  limit="${CHUNK_SIZE}"
  if [[ "$(( start + limit ))" -gt "${TOTAL_ROWS}" ]]; then
    limit="$(( TOTAL_ROWS - start ))"
  fi
  out_dir="${SHARD_DIR}/shard_${i}"
  shard_log="${SHARD_DIR}/shard_${i}.log"
  mkdir -p "${out_dir}"
  echo "[$(date '+%F %T')] shard=${i} gpu=${gpu} start=${start} limit=${limit}" | tee -a "${LOG_FILE}"
  (
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYBIN}" "${A2F}/scripts/generate_from_manifest.py" \
      --manifest "${TRIPLETS_CSV}" \
      --output_dir "${out_dir}" \
      --limit "${limit}" \
      --start_index "${start}" \
      --num_steps 25 \
      --guidance_scale 3.0 \
      --num_images 1 \
      --exp_adapter_scale 1.0 \
      --output_size 256 \
      --lora_ref_scale 1.0 \
      --models_dir /mnt/personal_workspace/chenkeyu/Arc2Face/models \
      --allow_self_source
  ) > "${shard_log}" 2>&1 &
  pids+=("$!")
done

fail=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    fail=1
  fi
done
if [[ "${fail}" -ne 0 ]]; then
  echo "[ERROR] one or more replay shards failed. Check ${SHARD_DIR}/shard_*.log" | tee -a "${LOG_FILE}"
  exit 1
fi

echo "[$(date '+%F %T')] merge shard jsonl -> ${REPLAY_OUT}/generation_results.jsonl" | tee -a "${LOG_FILE}"
"${PYBIN}" - <<'PY' 2>&1 | tee -a "${LOG_FILE}"
import json
from pathlib import Path

base = Path("/mnt/personal_workspace/chenkeyu/ReMark/Attack-arc2face_wrapper/outputs_replay_celebahq_faceonly550")
shards = sorted((base / "shards_parallel").glob("shard_*/generation_results.jsonl"))
out = base / "generation_results.jsonl"

rows = []
for p in shards:
    for line in p.open("r", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        rows.append(rec)
rows.sort(key=lambda x: int(x.get("index", -1)))
with out.open("w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"[OK] merged_jsonl={out} rows={len(rows)}")
PY

echo "[$(date '+%F %T')] replay summary" | tee -a "${LOG_FILE}"
"${PYBIN}" - <<'PY' 2>&1 | tee -a "${LOG_FILE}"
import json
from pathlib import Path
p = Path("/mnt/personal_workspace/chenkeyu/ReMark/Attack-arc2face_wrapper/outputs_replay_celebahq_faceonly550/generation_results.jsonl")
ok=0
tot=0
for line in p.open("r", encoding="utf-8"):
    line=line.strip()
    if not line:
        continue
    tot += 1
    try:
        rec=json.loads(line)
    except Exception:
        continue
    if rec.get("ok", False):
        ok += 1
print(f"[OK] replay records total={tot} ok={ok} fail={tot-ok}")
if ok <= 0:
    raise SystemExit("No valid replay records generated.")
PY

echo "[$(date '+%F %T')] start 3-model queue (GPUS=1,2,3)" | tee -a "${LOG_FILE}"
cd "${FRM}"
GPUS=1,2,3 ./tools/run_stage1_llmwm_arc2face_queue.sh 2>&1 | tee -a "${LOG_FILE}"

echo "[$(date '+%F %T')] ALL_DONE pipeline" | tee -a "${LOG_FILE}"
