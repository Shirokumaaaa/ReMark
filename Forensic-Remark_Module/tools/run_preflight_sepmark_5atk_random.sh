#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/personal_workspace/chenkeyu/ReMark/Forensic-Remark_Module"
PYBIN="${PYBIN:-/home/ldy/miniconda3/envs/sepmark/bin/python}"
STAMP="${1:-$(date +%Y%m%d_%H%M%S)}"

ARC2FACE_JSONL="/mnt/personal_workspace/chenkeyu/ReMark/Attack-arc2face_wrapper/outputs_replay_large/generation_results_wm_aligned_sepmark.jsonl"
DIFFSWAP_JSONL="/mnt/personal_workspace/chenkeyu/ReMark/Attack-DiffSwap/outputs_replay_large/generation_results_inpaint.jsonl"
REFACE_JSONL="/mnt/personal_workspace/chenkeyu/ReMark/Attack-REFace/outputs_replay_large/generation_results_wm_aligned_sepmark_large.jsonl"
FACEADAPTER_JSONL="/mnt/personal_workspace/chenkeyu/ReMark/Attack-Face-Adapter/outputs/generation_results_wm_aligned_sepmark.jsonl"

MANI_DIR="${ROOT}/data_manifests/preflight_rand_${STAMP}"
OUT_DIR="${ROOT}/runs/preflight5_sepmark_${STAMP}"
LOG_DIR="${ROOT}/logging"
mkdir -p "${MANI_DIR}" "${OUT_DIR}/samples" "${LOG_DIR}"

make_csv_from_jsonl() {
  local name="$1"
  local jsonl="$2"
  local maxn="$3"
  local seed="$4"
  local out_csv="${MANI_DIR}/${name}.csv"
  "${PYBIN}" - <<'PY' "${jsonl}" "${maxn}" "${seed}" "${out_csv}"
import csv, json, os, random, sys
jsonl, maxn, seed, out_csv = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
s = []
seen = set()
with open(jsonl, 'r', encoding='utf-8') as f:
    for ln in f:
        rec = json.loads(ln)
        if not rec.get('ok', False):
            continue
        p = rec.get('source_image')
        if not p:
            continue
        p = os.path.abspath(os.path.expanduser(str(p)))
        if p in seen:
            continue
        seen.add(p)
        s.append(p)
rng = random.Random(seed)
rng.shuffle(s)
if maxn > 0:
    s = s[:maxn]
with open(out_csv, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['img_path'])
    for p in s:
        w.writerow([p])
print(f'[manifest] {out_csv} count={len(s)} seed={seed}')
PY
}

make_csv_from_img_csv() {
  local name="$1"
  local in_csv="$2"
  local maxn="$3"
  local seed="$4"
  local out_csv="${MANI_DIR}/${name}.csv"
  "${PYBIN}" - <<'PY' "${in_csv}" "${maxn}" "${seed}" "${out_csv}"
import csv, os, random, sys, pandas as pd
in_csv, maxn, seed, out_csv = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
df = pd.read_csv(in_csv)
paths = [os.path.abspath(os.path.expanduser(str(x))) for x in df['img_path'].tolist()]
rng = random.Random(seed)
rng.shuffle(paths)
if maxn > 0:
    paths = paths[:maxn]
with open(out_csv, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['img_path'])
    for p in paths:
        w.writerow([p])
print(f'[manifest] {out_csv} count={len(paths)} seed={seed}')
PY
}

check_jsonl_health() {
  local name="$1"
  local jsonl="$2"
  local min_ratio="$3"
  "${PYBIN}" - <<'PY' "${name}" "${jsonl}" "${min_ratio}"
import json, os, sys
name, jsonl, min_ratio = sys.argv[1], sys.argv[2], float(sys.argv[3])
if not os.path.exists(jsonl):
    raise SystemExit(f"[fatal] missing jsonl for {name}: {jsonl}")
ok = 0
uniq_out = set()
for ln in open(jsonl, 'r', encoding='utf-8'):
    ln = ln.strip()
    if not ln:
        continue
    try:
        rec = json.loads(ln)
    except Exception:
        continue
    if not rec.get('ok', False):
        continue
    ok += 1
    outs = rec.get('outputs', [])
    if outs:
        uniq_out.add(str(outs[0]))
if ok <= 0:
    raise SystemExit(f"[fatal] no ok records in {jsonl}")
ratio = len(uniq_out) / float(ok)
print(f"[check] {name}: ok={ok} uniq_out={len(uniq_out)} ratio={ratio:.4f} min={min_ratio:.4f}")
if ratio < min_ratio:
    raise SystemExit(
        f"[fatal] replay mapping for {name} looks corrupted: ratio={ratio:.4f} < {min_ratio:.4f}"
    )
PY
}

run_one() {
  local atk="$1"
  local csv_path="$2"
  local seed="$3"
  local tmp_cfg="${OUT_DIR}/cfg_${atk}.yaml"
  local log_file="${LOG_DIR}/tmux_preflight5_sepmark_${STAMP}_${atk}.log"

  cat > "${tmp_cfg}" <<YAML
wm_model: sepmark

training:
  deterministic_messages: true
  message_seed_salt: remark_v1
  epochs: 0
  batch_size: 16
  lr: 3.0e-4
  betas: [0.9, 0.999]
  weight_decay: 1.0e-4
  kl_warmup_epochs: 20
  save_freq: 10
  val_freq: 1
  best_select: val_attack_avg_acc

validation:
  max_batches: 0

progress:
  enabled: true
  log_interval_steps: 20

losses:
  l1:    { weight: 1.0 }
  lpips: { weight: 0.1, enabled: true }
  bce:   { weight: 3.0 }
  kl:    { weight: 0, target_weight: 0.001, warmup: false }

attacks:
  online: [${atk}]
  offline: []

attack_options:
  enforce_nontrivial_swap: true
  nontrivial_swap_eps: 1.0e-4
  simswap_source_mode: cover_roll

  arc2face_mode: replay
  arc2face_min_unique_output_ratio: 0.3
  arc2face_results_jsonl: ${ARC2FACE_JSONL}
  arc2face_outputs_base: /mnt/personal_workspace/chenkeyu/ReMark/Attack-arc2face_wrapper
  arc2face_replay_key: source_image
  arc2face_allow_missing: false

  diffswap_mode: online
  diffswap_online_fallback_replay: false
  diffswap_online_python_bin: /home/ldy/miniconda3/envs/DiffSwap/bin/python
  diffswap_online_repo_root: /mnt/personal_workspace/chenkeyu/ReMark/Attack-DiffSwap
  diffswap_online_pipeline_script: /mnt/personal_workspace/chenkeyu/ReMark/Attack-DiffSwap/pipeline.py
  diffswap_online_timeout_sec: 1800
  diffswap_source_csv: ${csv_path}
  diffswap_results_jsonl: ${DIFFSWAP_JSONL}
  diffswap_outputs_base: /mnt/personal_workspace/chenkeyu/ReMark/Attack-DiffSwap
  diffswap_replay_key: source_image
  diffswap_allow_missing: false
  preserve_background: true

  reface_mode: online
  reface_online_fallback_replay: false
  reface_python_bin: /home/ldy/miniconda3/envs/REFace/bin/python
  reface_online_ddim_steps: 10
  reface_online_scale: 2.0
  reface_online_timeout_sec: 300
  reface_results_jsonl: ${REFACE_JSONL}
  reface_outputs_base: /mnt/personal_workspace/chenkeyu/ReMark/Attack-REFace
  reface_replay_key: source_image
  reface_allow_missing: false
  reface_source_csv: ${csv_path}

  face_adapter_mode: online
  face_adapter_online_fallback_replay: false
  face_adapter_python_bin: /home/ldy/miniconda3/envs/FaceAdapter/bin/python
  face_adapter_online_hf_cache: /home/ldy/.cache/huggingface/hub
  face_adapter_online_local_files_only: true
  face_adapter_online_timeout_sec: 300
  face_adapter_results_jsonl: ${FACEADAPTER_JSONL}
  face_adapter_outputs_base: /mnt/personal_workspace/chenkeyu/ReMark/Attack-Face-Adapter
  face_adapter_replay_key: source_image
  face_adapter_allow_missing: false
  face_adapter_source_csv: ${csv_path}

alternating_train:
  enabled: false

efficiency:
  cache_wm_images: false
  use_amp: false
  wm_loss_freq: 1
  attack_num_workers: 2

data:
  train_csv: ${csv_path}
  val_csv: ${csv_path}
  image_size: 128

preflight_eval:
  enabled: true
  max_batches: 2
  save_csv: false
  sample_nrow: 4
YAML

  echo "[run] attack=${atk} seed=${seed} csv=${csv_path}" | tee -a "${OUT_DIR}/summary.txt"
  (cd "${ROOT}" && "${PYBIN}" train_stage1.py --config configs/stage1_vae.yaml --override "${tmp_cfg}" --seed "${seed}" | tee "${log_file}")

  local run_dir
  run_dir="$(grep -E 'Run dir[[:space:]]*:' "${log_file}" | tail -n 1 | sed 's/.*Run dir[[:space:]]*:[[:space:]]*//')"
  if [[ -z "${run_dir}" || ! -d "${run_dir}" ]]; then
    echo "[warn] cannot parse run dir for ${atk}" | tee -a "${OUT_DIR}/summary.txt"
    return 0
  fi

  local src_png="${run_dir}/samples/preflight_${atk}.png"
  local dst_png="${OUT_DIR}/samples/${atk}.png"
  if [[ -f "${src_png}" ]]; then
    cp "${src_png}" "${dst_png}"
    echo "[sample] ${atk} -> ${dst_png}" | tee -a "${OUT_DIR}/summary.txt"
  else
    echo "[warn] sample missing for ${atk}: ${src_png}" | tee -a "${OUT_DIR}/summary.txt"
  fi
}

# Randomized manifests (fresh seed per attack)
S0=$(( $(date +%s) + 11 ))
S1=$(( $(date +%s) + 23 ))
S2=$(( $(date +%s) + 37 ))
S3=$(( $(date +%s) + 53 ))
S4=$(( $(date +%s) + 71 ))

make_csv_from_img_csv "simswap" "${ROOT}/data_manifests/celeba_hq_128_val.csv" 128 "${S0}"
check_jsonl_health "arc2face" "${ARC2FACE_JSONL}" 0.95
check_jsonl_health "diffswap" "${DIFFSWAP_JSONL}" 0.95
check_jsonl_health "reface(replay-fallback-only)" "${REFACE_JSONL}" 0.95
check_jsonl_health "face_adapter(replay-fallback-only)" "${FACEADAPTER_JSONL}" 0.95
make_csv_from_jsonl "arc2face" "${ARC2FACE_JSONL}" 128 "${S1}"
make_csv_from_jsonl "diffswap" "${DIFFSWAP_JSONL}" 128 "${S2}"
make_csv_from_img_csv "reface" "${ROOT}/data_manifests/celeba_hq_128_val.csv" 4 "${S3}"
make_csv_from_img_csv "face_adapter" "${ROOT}/data_manifests/celeba_hq_128_val.csv" 4 "${S4}"

run_one "simswap"      "${MANI_DIR}/simswap.csv" "${S0}"
run_one "arc2face"     "${MANI_DIR}/arc2face.csv" "${S1}"
run_one "diffswap"     "${MANI_DIR}/diffswap.csv" "${S2}"
run_one "reface"       "${MANI_DIR}/reface.csv" "${S3}"
run_one "face_adapter" "${MANI_DIR}/face_adapter.csv" "${S4}"

echo "[done] samples in: ${OUT_DIR}/samples" | tee -a "${OUT_DIR}/summary.txt"
echo "[done] summary: ${OUT_DIR}/summary.txt"
