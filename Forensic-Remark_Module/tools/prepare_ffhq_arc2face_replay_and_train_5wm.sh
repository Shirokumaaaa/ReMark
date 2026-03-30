#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/personal_workspace/chenkeyu/ReMark/Forensic-Remark_Module"
WRAP="/mnt/personal_workspace/chenkeyu/ReMark/Attack-arc2face_wrapper"
PY="/home/ldy/miniconda3/envs/sepmark/bin/python"
GPU_ID="3"

TRAIN_CSV="$ROOT/data_manifests/ffhq_2k_train.csv"
VAL_CSV="$ROOT/data_manifests/ffhq_500_val.csv"
MERGED_CSV="$WRAP/outputs_replay_ffhq_2k500/ffhq_2k500_all.csv"
TRIPLET_CSV="$WRAP/outputs_replay_ffhq_2k500/ffhq_2k500_triplets.csv"
REPLAY_DIR="$WRAP/outputs_replay_ffhq_2k500"
REPLAY_JSONL="$REPLAY_DIR/generation_results.jsonl"

mkdir -p "$REPLAY_DIR"

cd "$ROOT"

echo "[$(date '+%F %T')] Building merged FFHQ manifest (train+val)"
"$PY" - <<'PY'
import csv
from pathlib import Path

train = Path('/mnt/personal_workspace/chenkeyu/ReMark/Forensic-Remark_Module/data_manifests/ffhq_2k_train.csv')
val = Path('/mnt/personal_workspace/chenkeyu/ReMark/Forensic-Remark_Module/data_manifests/ffhq_500_val.csv')
out = Path('/mnt/personal_workspace/chenkeyu/ReMark/Attack-arc2face_wrapper/outputs_replay_ffhq_2k500/ffhq_2k500_all.csv')
out.parent.mkdir(parents=True, exist_ok=True)

seen = set()
rows = []
for p in (train, val):
    with p.open('r', encoding='utf-8') as f:
        rd = csv.DictReader(f)
        for r in rd:
            ip = str(Path(r['img_path']).resolve())
            if ip in seen:
                continue
            seen.add(ip)
            rows.append({'img_path': ip})

with out.open('w', newline='', encoding='utf-8') as f:
    wr = csv.DictWriter(f, fieldnames=['img_path'])
    wr.writeheader()
    wr.writerows(rows)

print(f'[OK] merged rows={len(rows)} -> {out}')
PY

echo "[$(date '+%F %T')] Building Arc2Face triplets"
CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONUNBUFFERED=1 \
"$PY" "$WRAP/scripts/build_triplets_random_source.py" \
  --input_csv "$MERGED_CSV" \
  --output_csv "$TRIPLET_CSV" \
  --seed 1234

echo "[$(date '+%F %T')] Generating Arc2Face replay images (this may take a while)"
CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONUNBUFFERED=1 \
"$PY" "$WRAP/scripts/generate_from_manifest.py" \
  --manifest "$TRIPLET_CSV" \
  --output_dir "$REPLAY_DIR" \
  --start_index 0 \
  --limit 1000000 \
  --num_steps 20 \
  --guidance_scale 3.0 \
  --num_images 1 \
  --exp_adapter_scale 1.0 \
  --lora_ref_scale 1.0 \
  --output_size 512 \
  --seed 42 \
  --allow_self_source

echo "[$(date '+%F %T')] Updating 5 arc2face configs to replay index"
"$PY" - <<'PY'
from pathlib import Path
import yaml

root = Path('/mnt/personal_workspace/chenkeyu/ReMark/Forensic-Remark_Module/configs/experiments')
files = sorted(root.glob('stage1_*_ffhq_arc2face.yaml'))
replay_jsonl = '/mnt/personal_workspace/chenkeyu/ReMark/Attack-arc2face_wrapper/outputs_replay_ffhq_2k500/generation_results.jsonl'
outputs_base = '/mnt/personal_workspace/chenkeyu/ReMark/Attack-arc2face_wrapper'
for p in files:
    cfg = yaml.safe_load(p.read_text())
    opts = cfg.setdefault('attack_options', {})
    opts['arc2face_mode'] = 'replay'
    opts['arc2face_results_jsonl'] = replay_jsonl
    opts['arc2face_outputs_base'] = outputs_base
    opts['arc2face_replay_key'] = 'source_image'
    opts['arc2face_allow_missing'] = True
    opts['enforce_nontrivial_swap'] = False
    p.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print('[OK] updated', p.name)
PY

echo "[$(date '+%F %T')] Launching 5-WM replay training queue"
cd "$ROOT"
./tools/run_stage1_ffhq_arc2face_5wm_queue.sh
