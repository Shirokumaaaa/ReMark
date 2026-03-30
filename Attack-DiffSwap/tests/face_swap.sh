CKPT=checkpoints/diffswap.pth
PORT=$(comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)
PYTHON_BIN=${DIFFSWAP_PYTHON:-python3}
TGT_SCALE=${DIFFSWAP_TGT_SCALE:-0.01}

PYTHONPATH=./:$PYTHONPATH "$PYTHON_BIN" -m torch.distributed.launch --nproc_per_node=1 --master_port=$PORT tests/faceswap_portrait.py $CKPT --save_img True --tgt_scale "$TGT_SCALE"
