#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/personal_workspace/chenkeyu/ReMark
OUT=$ROOT/Dataset-FF++
PY=/home/ldy/miniconda3/envs/Lampmark/bin/python
DL=$ROOT/download_ffpp.py
LOG=$ROOT/ffpp_full_pipeline.log

{
  echo "[$(date '+%F %T')] Step1: download FF++ all datasets (c23/videos)"
  printf '\n' | "$PY" "$DL" "$OUT" -d all -c c23 -t videos --server EU
  echo "[$(date '+%F %T')] Step1 done"

  echo "[$(date '+%F %T')] Step2: extract frames and remove mp4"
  if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ffmpeg not found; aborting extraction" >&2
    exit 2
  fi

  mapfile -t MP4S < <(find "$OUT" -type f -name '*.mp4' | sort)
  TOTAL=${#MP4S[@]}
  echo "Found $TOTAL mp4 files"
  i=0
  for mp4 in "${MP4S[@]}"; do
    i=$((i+1))
    dir=$(dirname "$mp4")
    base=$(basename "$mp4" .mp4)
    outdir="$dir/images/$base"
    mkdir -p "$outdir"
    if ffmpeg -hide_banner -loglevel error -i "$mp4" "$outdir/%06d.jpg"; then
      rm -f "$mp4"
      if (( i % 20 == 0 )) || (( i == TOTAL )); then
        echo "[$(date '+%F %T')] Extracted $i/$TOTAL"
      fi
    else
      echo "[$(date '+%F %T')] Failed extraction: $mp4" >&2
    fi
  done

  echo "[$(date '+%F %T')] Step2 done"
} >> "$LOG" 2>&1
