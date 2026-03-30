#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/personal_workspace/chenkeyu/ReMark/Forensic-Remark_Module"
PYBIN="${PYBIN:-/home/ldy/miniconda3/envs/sepmark/bin/python}"
FACE_PY="${FACE_PY:-/home/ldy/miniconda3/envs/FaceAdapter/bin/python}"
MAXB="${MAXB:-0}"
NUM_WORKERS="${NUM_WORKERS:-0}"

RUNS=(
  "/mnt/personal_workspace/chenkeyu/ReMark/Forensic-Remark_Module/runs/stage1_20260323_223130_849999"
  "/mnt/personal_workspace/chenkeyu/ReMark/Forensic-Remark_Module/runs/stage1_20260324_001053_361694"
  "/mnt/personal_workspace/chenkeyu/ReMark/Forensic-Remark_Module/runs/stage1_20260324_015038_360189"
  "/mnt/personal_workspace/chenkeyu/ReMark/Forensic-Remark_Module/runs/stage1_20260324_033053_827814"
  "/mnt/personal_workspace/chenkeyu/ReMark/Forensic-Remark_Module/runs/stage1_20260324_051043_669020"
)

cd "${ROOT}"

for RUN_DIR in "${RUNS[@]}"; do
  if [[ ! -f "${RUN_DIR}/config.yaml" ]]; then
    echo "[skip] missing config: ${RUN_DIR}/config.yaml"
    continue
  fi
  if [[ ! -f "${RUN_DIR}/checkpoints/vae/best.pth" ]]; then
    echo "[skip] missing ckpt: ${RUN_DIR}/checkpoints/vae/best.pth"
    continue
  fi

  NAME="$(basename "${RUN_DIR}")"
  TMP_DIR="$(mktemp -d /tmp/remark_faceadapter_online_eval_XXXXXX)"
  TMP_CFG="${TMP_DIR}/config.yaml"
  TMP_RUN_DIR="${TMP_DIR}/run"
  mkdir -p "${TMP_RUN_DIR}"

  echo "[$(date '+%F %T')] START ${NAME}"

  "${PYBIN}" - <<'PY' "${RUN_DIR}/config.yaml" "${TMP_CFG}" "${FACE_PY}"
import os
import sys
import yaml

src, dst, face_py = sys.argv[1:4]
with open(src, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cfg.setdefault("attacks", {})
cfg["attacks"]["online"] = ["face_adapter"]
cfg["attacks"]["offline"] = []

cfg.setdefault("training", {})
cfg["training"]["deterministic_messages"] = True
cfg["training"]["message_seed_salt"] = "remark_v1"

cfg.setdefault("attack_options", {})
opts = cfg["attack_options"]
opts["face_adapter_mode"] = "online"
opts["face_adapter_online_fallback_replay"] = False
opts["face_adapter_allow_missing"] = True
opts["face_adapter_python_bin"] = face_py
opts["face_adapter_repo_root"] = "/mnt/personal_workspace/chenkeyu/ReMark/Attack-Face-Adapter"
opts["face_adapter_online_hf_cache"] = "/home/ldy/.cache/huggingface/hub"
opts["face_adapter_online_local_files_only"] = True
opts["face_adapter_online_timeout_sec"] = 300
opts["enforce_nontrivial_swap"] = False

source_csv = str(opts.get("face_adapter_source_csv", "")).strip()
if not source_csv:
    source_csv = str(cfg.get("data", {}).get("train_csv", "")).strip()
    opts["face_adapter_source_csv"] = source_csv

fixed_source = str(opts.get("face_adapter_fixed_source_path", "")).strip()
if fixed_source and (not os.path.exists(fixed_source)):
    opts["face_adapter_fixed_source_path"] = ""

with open(dst, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
PY

  cp "${TMP_CFG}" "${TMP_RUN_DIR}/config.yaml"
  t0="$(date +%s)"
  "${PYBIN}" tools/eval_attacked_recon_acc.py \
    --run-dir "${TMP_RUN_DIR}" \
    --checkpoint "${RUN_DIR}/checkpoints/vae/best.pth" \
    --split val \
    --max-batches "${MAXB}" \
    --batch-size 0 \
    --num-workers "${NUM_WORKERS}"
  t1="$(date +%s)"
  dt="$((t1 - t0))"

  SRC_CSV="${TMP_RUN_DIR}/attacked_recon_eval_val.csv"
  DST_CSV="${RUN_DIR}/transfer_eval_face_adapter_online.csv"
  cp "${SRC_CSV}" "${DST_CSV}"

  "${PYBIN}" - <<'PY' "${DST_CSV}" "${NAME}" "${dt}"
import csv
import sys

csv_path, name, dt = sys.argv[1], sys.argv[2], int(sys.argv[3])
raw = None
rec = None
with open(csv_path, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row.get("attack") != "face_adapter":
            continue
        if row.get("path") == "raw_attacked":
            raw = row
        elif row.get("path") == "vae_recon":
            rec = row
if raw is None or rec is None:
    print(f"[warn] {name} missing face_adapter rows in {csv_path}")
    raise SystemExit(0)
n = int(float(rec.get("num_samples", "0") or 0))
spd = (n / dt) if dt > 0 else 0.0
print(
    f"[done] {name} elapsed={dt}s n={n} speed={spd:.2f} samples/s "
    f"raw_bit_acc={float(raw['bit_acc']):.4f} recon_bit_acc={float(rec['bit_acc']):.4f} "
    f"delta={float(rec['bit_acc'])-float(raw['bit_acc']):+.4f}"
)
PY

  rm -rf "${TMP_DIR}"
done

echo "[$(date '+%F %T')] ALL DONE face_adapter online generalization (5 WM runs)."
