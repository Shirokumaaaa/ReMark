#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/personal_workspace/chenkeyu/ReMark
OUT=$ROOT/Dataset-FF++
PY=/home/ldy/miniconda3/envs/Lampmark/bin/python
DL=$ROOT/download_ffpp.py
LOG=$ROOT/ffpp_full_pipeline.log

DATASETS=(original DeepFakeDetection_original Deepfakes DeepFakeDetection Face2Face FaceShifter FaceSwap NeuralTextures)
SERVERS=(EU EU2 CA)

mkdir -p "$OUT"

echo "[$(date '+%F %T')] PIPELINE START" >> "$LOG"

run_download_with_retry() {
  local ds="$1"
  local done=0
  local round=0
  while [[ $done -eq 0 ]]; do
    round=$((round+1))
    for s in "${SERVERS[@]}"; do
      echo "[$(date '+%F %T')] Download dataset=$ds server=$s round=$round" >> "$LOG"
      if printf '\n' | "$PY" "$DL" "$OUT" -d "$ds" -c c23 -t videos --server "$s" >> "$LOG" 2>&1; then
        echo "[$(date '+%F %T')] Download success dataset=$ds server=$s" >> "$LOG"
        done=1
        break
      else
        echo "[$(date '+%F %T')] Download failed dataset=$ds server=$s (will retry)" >> "$LOG"
        sleep 8
      fi
    done
  done
}

extract_and_cleanup() {
  local ds="$1"
  local ds_root="$OUT"
  local count=0
  while IFS= read -r -d '' mp4; do
    # only process current dataset subtree
    case "$mp4" in
      *"/$ds/"*) ;;
      *) continue ;;
    esac
    count=$((count+1))
    dir=$(dirname "$mp4")
    base=$(basename "$mp4" .mp4)
    outdir="$dir/images/$base"
    mkdir -p "$outdir"
    if ffmpeg -hide_banner -loglevel error -i "$mp4" "$outdir/%06d.jpg"; then
      rm -f "$mp4"
    else
      echo "[$(date '+%F %T')] Extract failed: $mp4" >> "$LOG"
    fi
    if (( count % 20 == 0 )); then
      echo "[$(date '+%F %T')] Extracted $count videos for dataset=$ds" >> "$LOG"
    fi
  done < <(find "$ds_root" -type f -name '*.mp4' -print0)
  echo "[$(date '+%F %T')] Extract phase done for dataset=$ds, processed=$count" >> "$LOG"
}

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "[$(date '+%F %T')] ERROR: ffmpeg not found" >> "$LOG"
  exit 2
fi

for ds in "${DATASETS[@]}"; do
  run_download_with_retry "$ds"
  extract_and_cleanup "$ds"
  echo "[$(date '+%F %T')] Dataset completed: $ds" >> "$LOG"
done

echo "[$(date '+%F %T')] ALL DONE" >> "$LOG"
