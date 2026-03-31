#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/personal_workspace/chenkeyu/ReMark"
CACHE_DIR="$ROOT/landmark_bit_cache/celeba_hq_native_10k_clean_20260324_sharded"
SHARD_DIR="$ROOT/tmp_landmark_shards_20260324"
SHARD_LOG_DIR="$ROOT/logging/landmark_cache_shards_20260324"
CLEAN_ROOT="$ROOT/Dataset-CelebA_HQ_10k_landmark_clean_20260324"
LAMPMARK_MANIFEST_DIR="$ROOT/Forensic-LampMark/data_manifests"
LAMPMARK_WM_ROOT="$ROOT/Forensic-LampMark/watermark_data_dlib68_10k_clean_20260324/celeba_hq"
GENERATED_CFG_DIR="$ROOT/Forensic-Remark_Module/configs/generated_20260324"
MASKWM_BASE_CFG="$ROOT/Forensic-MaskWM/configs/train/train_celebahq_landmark_clean_10k.yaml"
MASKWM_RUNTIME_CFG="$ROOT/Forensic-MaskWM/configs/train/generated_train_celebahq_landmark_clean_current.yaml"
PREDICTOR_PATH="$ROOT/Attack-DiffSwap/checkpoints/shape_predictor_68_face_landmarks.dat"
IMMEDIATE_START="${IMMEDIATE_START_WITH_AVAILABLE_CACHE:-0}"

wait_tmux_session() {
  local session="$1"
  while tmux has-session -t "$session" 2>/dev/null; do
    sleep 60
  done
}

if [[ "$IMMEDIATE_START" == "1" ]]; then
  echo "[pipeline] immediate mode: using current cache snapshot"
else
  echo "[pipeline] waiting shard tmux sessions..."
  while tmux ls 2>/dev/null | grep -q 'lmcache_'; do
    sleep 30
  done

  if [[ ! -f "$SHARD_LOG_DIR/shard_00_failures.csv" ]]; then
    echo "[pipeline] shard_00 failure csv missing; rerunning shard_00 serially"
    /home/ldy/miniconda3/envs/REFace/bin/python "$ROOT/tools/build_landmark_bit_cache.py" \
      --csv "$SHARD_DIR/shard_00.csv" \
      --cache-dir "$CACHE_DIR" \
      --failure-csv "$SHARD_LOG_DIR/shard_00_failures.csv" \
      --predictor-path "$PREDICTOR_PATH" \
      --canonical-bits 128 \
      --bits-per-value 4 \
      --allow-failures | tee "$SHARD_LOG_DIR/shard_00_rerun.log"
  fi
fi

echo "[pipeline] preparing clean subset"
/home/ldy/miniconda3/envs/REFace/bin/python "$ROOT/tools/prepare_landmark_clean_subset.py" \
  --train-csv "$ROOT/Dataset-CelebA_HQ_10k_native/train.csv" \
  --val-csv "$ROOT/Dataset-CelebA_HQ_10k_native/val.csv" \
  --failure-csv-glob-dir "$SHARD_LOG_DIR" \
  --cache-dir "$CACHE_DIR" \
  --predictor-path "$PREDICTOR_PATH" \
  --canonical-bits 128 \
  --bits-per-value 4 \
  --out-root "$CLEAN_ROOT" \
  --lampmark-manifest-dir "$LAMPMARK_MANIFEST_DIR" | tee "$ROOT/logging/prepare_landmark_clean_subset_20260324.log"

TRAIN_COUNT="$(python - <<PY
import csv
with open("${CLEAN_ROOT}/train.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    print(sum(1 for _ in reader))
PY
)"
VAL_COUNT="$(python - <<PY
import csv
with open("${CLEAN_ROOT}/val.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    print(sum(1 for _ in reader))
PY
)"
export TRAIN_COUNT VAL_COUNT
echo "[pipeline] clean subset snapshot: train=${TRAIN_COUNT} val=${VAL_COUNT}"

export MASKWM_BASE_CFG MASKWM_RUNTIME_CFG CLEAN_ROOT
/home/ldy/miniconda3/envs/REFace/bin/python - <<'PY' | tee "$ROOT/logging/maskwm_runtime_cfg_20260324.log"
import os
import yaml

base_cfg = os.environ["MASKWM_BASE_CFG"]
runtime_cfg = os.environ["MASKWM_RUNTIME_CFG"]
train_count = int(os.environ["TRAIN_COUNT"])

with open(base_cfg, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

batch_size = int(cfg["batch_size"])
steps_per_epoch = train_count // batch_size
cfg["num_training_steps"] = steps_per_epoch * 20
cfg["dataset_path"] = os.environ["CLEAN_ROOT"]

with open(runtime_cfg, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

print(f"maskwm_steps_per_epoch={steps_per_epoch}")
print("maskwm_num_training_steps=%d" % cfg["num_training_steps"])
print(f"runtime_cfg={runtime_cfg}")
PY

echo "[pipeline] building LampMark landmark watermarks"
cd "$ROOT/Forensic-LampMark"
/home/ldy/miniconda3/envs/Lampmark/bin/python scripts/build_landmark_watermarks.py \
  --manifest-pattern data_manifests/celeba_hq_128_10k_landmark_clean_{split}.csv \
  --splits train,val \
  --out-root "$LAMPMARK_WM_ROOT" \
  --img-size 128 \
  --num-bits 64 \
  --predictor-path "$PREDICTOR_PATH" \
  --cache-dir "$CACHE_DIR" \
  --canonical-bits 128 \
  --bits-per-value 4 | tee "$ROOT/logging/build_lampmark_landmark_wm_20260324.log"

echo "[pipeline] launching SepMark / LampMark / MaskWM finetunes"
tmux new-session -d -s sepmark_landmark_clean_10k \
  "cd $ROOT/Forensic-SepMark && CUDA_VISIBLE_DEVICES=0 SEPMARK_CFG_PATH=cfg/train_DualMark_landmark_clean_10k.yaml SEPMARK_MESSAGE_MODE=landmark_bits SEPMARK_LANDMARK_PREDICTOR=$PREDICTOR_PATH SEPMARK_LANDMARK_CACHE_DIR=$CACHE_DIR SEPMARK_INIT_EC=$ROOT/Forensic-SepMark/results/baseline/Dual_watermark_256_128_0.1_0.0002_0.5_se_se_1_10_10_10_0.1_2023_04_18_16_29_54/models/EC_90.pth SEPMARK_INIT_D=$ROOT/Forensic-SepMark/results/baseline/Dual_watermark_256_128_0.1_0.0002_0.5_se_se_1_10_10_10_0.1_2023_04_18_16_29_54/models/D_90.pth SEPMARK_RUN_TAG=landmark_clean_10k_20ep_20260324 /home/ldy/miniconda3/envs/REFace/bin/python Dual_Mark_main.py > $ROOT/Forensic-SepMark/train_landmark_clean_10k_20260324.log 2>&1"

tmux new-session -d -s lampmark_landmark_clean_10k \
  "cd $ROOT/Forensic-LampMark && CUDA_VISIBLE_DEVICES=1 LAMPMARK_RUN_CONFIG=configurations/run_pretrain_dlib68_64_clean_10k_20ep.json LAMPMARK_PRETRAIN_CONFIG=configurations/pretrain_dlib68_64_clean_10k_20ep.json /home/ldy/miniconda3/envs/Lampmark/bin/python main.py > $ROOT/Forensic-LampMark/train_landmark_clean_10k_20260324.log 2>&1"

tmux new-session -d -s maskwm_landmark_clean_10k \
  "cd $ROOT/Forensic-MaskWM && CUDA_VISIBLE_DEVICES=3 /home/ldy/miniconda3/envs/REFace/bin/python train.py --model_name D_128bits_landmark_clean_10k_20260324 --train_config_path configs/train/generated_train_celebahq_landmark_clean_current.yaml --model_config_path configs/model/D_128bits.yaml > $ROOT/Forensic-MaskWM/train_landmark_clean_10k_20260324.log 2>&1"

echo "[pipeline] launched sessions:"
tmux ls | grep 'landmark_clean_10k' || true

echo "[pipeline] waiting native finetunes to finish"
wait_tmux_session sepmark_landmark_clean_10k
wait_tmux_session lampmark_landmark_clean_10k
wait_tmux_session maskwm_landmark_clean_10k

SEPMARK_DIR="$(find "$ROOT/Forensic-SepMark/results" -maxdepth 1 -mindepth 1 -type d -name '*landmark_clean_10k_20ep_20260324*' | sort | tail -1)"
SEPMARK_CKPT="$SEPMARK_DIR/models/EC_20.pth"
LAMPMARK_ENC="$ROOT/Forensic-LampMark/weights/128_64/wm-img/encoder_epoch_20.pth"
LAMPMARK_DEC="$ROOT/Forensic-LampMark/weights/128_64/wm-img/decoder_epoch_20.pth"
MASKWM_RUN_DIR="$(find "$ROOT/Forensic-MaskWM/checkpoints/D_128bits_landmark_clean_10k_20260324" -maxdepth 1 -mindepth 1 -type d | sort | tail -1)"
MASKWM_CKPT="$(find "$MASKWM_RUN_DIR/models" -maxdepth 1 -type f -name 'ckpt_*.pth' | sort -V | tail -1)"

test -f "$SEPMARK_CKPT"
test -f "$LAMPMARK_ENC"
test -f "$LAMPMARK_DEC"
test -f "$MASKWM_CKPT"

mkdir -p "$GENERATED_CFG_DIR"
export ROOT GENERATED_CFG_DIR CACHE_DIR CLEAN_ROOT SEPMARK_CKPT LAMPMARK_ENC LAMPMARK_DEC MASKWM_CKPT
/home/ldy/miniconda3/envs/REFace/bin/python - <<'PY'
import os
import yaml

root = os.environ["ROOT"]
generated = os.environ["GENERATED_CFG_DIR"]
cache_dir = os.environ["CACHE_DIR"]
train_csv = os.path.join(os.environ["CLEAN_ROOT"], "train.csv")
val_csv = os.path.join(os.environ["CLEAN_ROOT"], "val.csv")

jobs = [
    (
        os.path.join(root, "Forensic-Remark_Module/configs/experiments/stage1_sepmark_landmarkbits_simswap_clean_10k.yaml"),
        os.path.join(generated, "stage1_sepmark_landmarkbits_simswap_clean_10k_from_native.yaml"),
        {
            "wm_adapter_sepmark": {
                "ckpt": os.environ["SEPMARK_CKPT"],
            },
        },
    ),
    (
        os.path.join(root, "Forensic-Remark_Module/configs/experiments/stage1_lampmark_landmarkbits_simswap_clean_10k.yaml"),
        os.path.join(generated, "stage1_lampmark_landmarkbits_simswap_clean_10k_from_native.yaml"),
        {
            "wm_adapter_lampmark": {
                "encoder_ckpt": os.environ["LAMPMARK_ENC"],
                "decoder_ckpt": os.environ["LAMPMARK_DEC"],
            },
        },
    ),
    (
        os.path.join(root, "Forensic-Remark_Module/configs/experiments/stage1_maskwm_landmarkbits_simswap_clean_10k.yaml"),
        os.path.join(generated, "stage1_maskwm_landmarkbits_simswap_clean_10k_from_native.yaml"),
        {
            "wm_adapter_maskwm": {
                "ckpt": os.environ["MASKWM_CKPT"],
            },
        },
    ),
]

for src, dst, extra in jobs:
    with open(src, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("training", {})
    cfg["training"]["landmark_bits_cache_dir"] = cache_dir
    cfg.setdefault("data", {})
    cfg["data"]["train_csv"] = train_csv
    cfg["data"]["val_csv"] = val_csv
    for key, value in extra.items():
        cfg.setdefault(key, {})
        cfg[key].update(value)
    with open(dst, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    print(dst)
PY

echo "[pipeline] launching 3 Stage1 VAE SimSwap trainings"
tmux new-session -d -s stage1_vae_sepmark_from_native_10k \
  "cd $ROOT/Forensic-Remark_Module && CUDA_VISIBLE_DEVICES=0 /home/ldy/miniconda3/envs/Lampmark/bin/python train_stage1.py --config configs/stage1_vae.yaml --override configs/generated_20260324/stage1_sepmark_landmarkbits_simswap_clean_10k_from_native.yaml > $ROOT/Forensic-Remark_Module/logging/stage1_sepmark_from_native_10k_20260324.log 2>&1"

tmux new-session -d -s stage1_vae_lampmark_from_native_10k \
  "cd $ROOT/Forensic-Remark_Module && CUDA_VISIBLE_DEVICES=1 /home/ldy/miniconda3/envs/Lampmark/bin/python train_stage1.py --config configs/stage1_vae.yaml --override configs/generated_20260324/stage1_lampmark_landmarkbits_simswap_clean_10k_from_native.yaml > $ROOT/Forensic-Remark_Module/logging/stage1_lampmark_from_native_10k_20260324.log 2>&1"

tmux new-session -d -s stage1_vae_maskwm_from_native_10k \
  "cd $ROOT/Forensic-Remark_Module && CUDA_VISIBLE_DEVICES=3 /home/ldy/miniconda3/envs/Lampmark/bin/python train_stage1.py --config configs/stage1_vae.yaml --override configs/generated_20260324/stage1_maskwm_landmarkbits_simswap_clean_10k_from_native.yaml > $ROOT/Forensic-Remark_Module/logging/stage1_maskwm_from_native_10k_20260324.log 2>&1"

echo "[pipeline] launched Stage1 VAE sessions:"
tmux ls | grep 'stage1_vae_.*from_native_10k' || true
