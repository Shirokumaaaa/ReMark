#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/personal_workspace/chenkeyu/ReMark/Forensic-Remark_Module"
PYBIN="${PYBIN:-/home/ldy/miniconda3/envs/TAG-WM/bin/python}"
JSONL="${JSONL:-/mnt/personal_workspace/chenkeyu/ReMark/Attack-Face-Adapter/outputs_replay_celebahq_faceonly550/generation_results.jsonl}"
MAXB="${MAXB:-1}"
BATCH_SIZE="${BATCH_SIZE:-50}"
NUM_WORKERS="${NUM_WORKERS:-0}"

RUNS=(
  "/mnt/personal_workspace/chenkeyu/ReMark/Forensic-Remark_Module/runs/stage1_20260326_131438_480607|/mnt/personal_workspace/chenkeyu/ReMark/Forensic-Remark_Module/runs/stage1_20260326_131438_480607/checkpoints/vae/last.pth|transfer_eval_face_adapter_replay_last_20260328.csv|tagwm_131438_last"
  "/mnt/personal_workspace/chenkeyu/ReMark/Forensic-Remark_Module/runs/stage1_20260326_222424_688144|/mnt/personal_workspace/chenkeyu/ReMark/Forensic-Remark_Module/runs/stage1_20260326_222424_688144/checkpoints/vae/best.pth|transfer_eval_face_adapter_replay_best_20260328.csv|tagwm_222424_best"
  "/mnt/personal_workspace/chenkeyu/ReMark/Forensic-Remark_Module/runs/stage1_20260328_033607_520323|/mnt/personal_workspace/chenkeyu/ReMark/Forensic-Remark_Module/runs/stage1_20260328_033607_520323/checkpoints/vae/best.pth|transfer_eval_face_adapter_replay_best_20260328.csv|tagwm_mix_best"
)

cd "${ROOT}"

echo "[$(date '+%F %T')] START tagwm face-adapter replay compare 3 ckpts"
echo "[cfg] PYBIN=${PYBIN}"
echo "[cfg] JSONL=${JSONL}"
echo "[cfg] MAXB=${MAXB} BATCH_SIZE=${BATCH_SIZE} NUM_WORKERS=${NUM_WORKERS}"

for item in "${RUNS[@]}"; do
  IFS='|' read -r RUN_DIR CKPT DST_NAME LABEL <<< "${item}"
  if [[ ! -f "${RUN_DIR}/config.yaml" ]]; then
    echo "[skip] missing config: ${RUN_DIR}/config.yaml"
    continue
  fi
  if [[ ! -f "${CKPT}" ]]; then
    echo "[skip] missing ckpt: ${CKPT}"
    continue
  fi

  TMP_DIR="$(mktemp -d /tmp/remark_tagwm_faceadapter_eval_XXXXXX)"
  TMP_RUN="${TMP_DIR}/run"
  mkdir -p "${TMP_RUN}"

  echo "[$(date '+%F %T')] START ${LABEL}"
  "${PYBIN}" - <<'PY' "${RUN_DIR}/config.yaml" "${TMP_RUN}/config.yaml" "${JSONL}"
import sys, yaml
src, dst, jsonl = sys.argv[1:4]
with open(src, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
cfg.setdefault("attacks", {})["online"] = ["face_adapter"]
cfg["attacks"]["offline"] = []
cfg.setdefault("training", {})
cfg["training"]["deterministic_messages"] = True
cfg["training"]["message_seed_salt"] = "remark_v1"
cfg.setdefault("attack_options", {})
opts = cfg["attack_options"]
opts["face_adapter_mode"] = "replay"
opts["face_adapter_results_jsonl"] = jsonl
opts["face_adapter_outputs_base"] = "/mnt/personal_workspace/chenkeyu/ReMark/Attack-Face-Adapter"
opts["face_adapter_replay_key"] = "source_image"
opts["face_adapter_allow_missing"] = False
opts["enforce_nontrivial_swap"] = False
with open(dst, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
PY

  t0="$(date +%s)"
  "${PYBIN}" tools/eval_attacked_recon_acc.py \
    --run-dir "${TMP_RUN}" \
    --checkpoint "${CKPT}" \
    --split val \
    --max-batches "${MAXB}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}"
  t1="$(date +%s)"
  dt="$((t1 - t0))"

  SRC_CSV="${TMP_RUN}/attacked_recon_eval_val.csv"
  DST_CSV="${RUN_DIR}/${DST_NAME}"
  cp "${SRC_CSV}" "${DST_CSV}"

  "${PYBIN}" - <<'PY' "${DST_CSV}" "${LABEL}" "${dt}"
import csv, sys
csv_path, label, dt = sys.argv[1], sys.argv[2], int(sys.argv[3])
raw = rec = None
with open(csv_path, "r", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if r.get("attack") != "face_adapter":
            continue
        if r.get("path") == "raw_attacked":
            raw = r
        elif r.get("path") == "vae_recon":
            rec = r
if raw is None or rec is None:
    print(f"[warn] {label} missing face_adapter rows in {csv_path}")
else:
    rb = float(raw["bit_acc"])
    rc = float(rec["bit_acc"])
    print(f"[done] {label} elapsed={dt}s raw={rb:.6f} recon={rc:.6f} delta={rc-rb:+.6f} csv={csv_path}")
PY

  rm -rf "${TMP_DIR}"
done

echo "[$(date '+%F %T')] ALL DONE"
