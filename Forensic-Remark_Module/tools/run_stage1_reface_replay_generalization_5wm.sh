#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/personal_workspace/chenkeyu/ReMark/Forensic-Remark_Module"
PYBIN="${PYBIN:-/home/ldy/miniconda3/envs/sepmark/bin/python}"
MAXB="${MAXB:-0}"
NUM_WORKERS="${NUM_WORKERS:-0}"

RUNS=(
  "/mnt/personal_workspace/chenkeyu/ReMark/Forensic-Remark_Module/runs/stage1_20260323_223130_849999"
  "/mnt/personal_workspace/chenkeyu/ReMark/Forensic-Remark_Module/runs/stage1_20260324_001053_361694"
  "/mnt/personal_workspace/chenkeyu/ReMark/Forensic-Remark_Module/runs/stage1_20260324_015038_360189"
  "/mnt/personal_workspace/chenkeyu/ReMark/Forensic-Remark_Module/runs/stage1_20260324_033053_827814"
  "/mnt/personal_workspace/chenkeyu/ReMark/Forensic-Remark_Module/runs/stage1_20260324_051043_669020"
)

pick_jsonl() {
  local c1="/mnt/personal_workspace/chenkeyu/ReMark/Attack-REFace/outputs_replay_ffhq_2k500/generation_results.jsonl"
  local c2="/mnt/personal_workspace/chenkeyu/ReMark/Attack-REFace/outputs/generation_results.jsonl"
  if [[ -f "$c1" ]]; then echo "$c1"; return 0; fi
  if [[ -f "$c2" ]]; then echo "$c2"; return 0; fi
  return 1
}

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
  WM="$(${PYBIN} - <<'PY' "${RUN_DIR}/config.yaml"
import yaml,sys
cfg=yaml.safe_load(open(sys.argv[1],'r',encoding='utf-8'))
print(str(cfg.get('wm_model','sepmark')).strip())
PY
)"

  if [[ -n "${REFACE_JSONL_OVERRIDE:-}" ]]; then
    JSONL="${REFACE_JSONL_OVERRIDE}"
  else
    JSONL="$(pick_jsonl)"
  fi
  if [[ -z "${JSONL}" ]]; then
    echo "[skip] no reface replay jsonl for wm=${WM} run=${NAME}"
    continue
  fi

  TMP_DIR="$(mktemp -d /tmp/remark_reface_replay_eval_XXXXXX)"
  TMP_CFG="${TMP_DIR}/config.yaml"
  TMP_RUN_DIR="${TMP_DIR}/run"
  mkdir -p "${TMP_RUN_DIR}"

  echo "[$(date '+%F %T')] START ${NAME} wm=${WM} jsonl=${JSONL}"

  "${PYBIN}" - <<'PY' "${RUN_DIR}/config.yaml" "${TMP_CFG}" "${JSONL}"
import sys,yaml
src,dst,jsonl = sys.argv[1:4]
with open(src,'r',encoding='utf-8') as f:
    cfg=yaml.safe_load(f)
cfg.setdefault('attacks',{})['online']=['reface']
cfg['attacks']['offline']=[]
cfg.setdefault('training',{})
cfg['training']['deterministic_messages']=True
cfg['training']['message_seed_salt']='remark_v1'
cfg.setdefault('attack_options',{})
opts=cfg['attack_options']
opts['reface_mode']='replay'
opts['reface_results_jsonl']=jsonl
opts['reface_outputs_base']='/mnt/personal_workspace/chenkeyu/ReMark/Attack-REFace'
opts['reface_replay_key']='source_image'
opts['reface_allow_missing']=True
opts['enforce_nontrivial_swap']=False
with open(dst,'w',encoding='utf-8') as f:
    yaml.safe_dump(cfg,f,allow_unicode=True,sort_keys=False)
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
  DST_CSV="${RUN_DIR}/transfer_eval_reface_replay.csv"
  cp "${SRC_CSV}" "${DST_CSV}"

  "${PYBIN}" - <<'PY' "${DST_CSV}" "${NAME}" "${dt}"
import csv,sys
csv_path,name,dt=sys.argv[1],sys.argv[2],int(sys.argv[3])
raw=rec=None
with open(csv_path,'r',encoding='utf-8') as f:
    for r in csv.DictReader(f):
        if r.get('attack')!='reface':
            continue
        if r.get('path')=='raw_attacked': raw=r
        elif r.get('path')=='vae_recon': rec=r
if raw is None or rec is None:
    print(f"[warn] {name} missing reface rows in {csv_path}")
    raise SystemExit(0)
n=int(float(rec.get('num_samples','0') or 0))
spd=(n/dt) if dt>0 else 0.0
print(f"[done] {name} elapsed={dt}s n={n} speed={spd:.2f} samples/s raw_bit_acc={float(raw['bit_acc']):.4f} recon_bit_acc={float(rec['bit_acc']):.4f} delta={float(rec['bit_acc'])-float(raw['bit_acc']):+.4f}")
PY

  rm -rf "${TMP_DIR}"
done

echo "[$(date '+%F %T')] ALL DONE reface replay generalization (5 WM runs)."
