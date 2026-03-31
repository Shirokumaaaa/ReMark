# Legacy Repair Pipeline Index

Updated by Codex on 2026-03-12.

## 1) Core entry scripts

- Stage1-style legacy training script:
  - `/home/ldy/..workspace/zhou/repair/train.py`
- Stage2-style U-Net training (single GPU/multi-stage StarGAN pipeline):
  - `/home/ldy/..workspace/zhou/repair/train_denoise_step.py`
- Stage2-style U-Net training (DDP variant):
  - `/home/ldy/..workspace/zhou/repair/train_denoise_step_DDP.py`
- VAE standalone trainer:
  - `/home/ldy/..workspace/zhou/repair/VAE_train.py`

## 2) Legacy dependency files (present)

- Dataset/setting:
  - `/home/ldy/..workspace/zhou/repair/data_loader.py`
  - `/home/ldy/..workspace/zhou/repair/utils/load_train_setting.py`
  - `/home/ldy/..workspace/zhou/repair/train_settings.json`
- Core network modules:
  - `/home/ldy/..workspace/zhou/repair/network/Network.py`
  - `/home/ldy/..workspace/zhou/repair/network/Network_old.py`
  - `/home/ldy/..workspace/zhou/repair/network/autoencoder.py`
  - `/home/ldy/..workspace/zhou/repair/network/Encoder_MP.py`
  - `/home/ldy/..workspace/zhou/repair/network/Decoder.py`
  - `/home/ldy/..workspace/zhou/repair/network/Denoise.py`
  - `/home/ldy/..workspace/zhou/repair/network/EncoderDecoder.py`

## 3) Checkpoints (key)

- Watermark/attack backbone checkpoints:
  - `/home/ldy/..workspace/zhou/repair/modelckpt/EC_100.pth`
  - `/home/ldy/..workspace/zhou/repair/modelckpt/D_8.pth`
  - `/home/ldy/..workspace/zhou/repair/modelckpt/200000-G.ckpt`
  - `/home/ldy/..workspace/zhou/repair/modelckpt/AE.pth`
- VAE checkpoints:
  - `/home/ldy/..workspace/zhou/repair/VAE_model/autoencoder_epoch_*.pth`
  - `/home/ldy/..workspace/zhou/repair/VAE_model/autoencoder_epoch_100.pth` (symlink)
- U-Net checkpoints:
  - `/home/ldy/..workspace/zhou/repair/unet_train_results/unet_model/denoise_model_epoch_*.pth`

## 4) Logs and useful outputs

- Main U-Net training logs with useful epoch metrics:
  - `/home/ldy/..workspace/zhou/repair/unet_train_results/model_logs/logs.txt`
  - `/home/ldy/..workspace/zhou/repair/unet_train_results/model_logs/slerp_log.txt`
- Older aggregate metric file:
  - `/home/ldy/..workspace/zhou/repair/metrics_log.txt`

## 5) Important note about `results/*` folders

- Most folders under `/home/ldy/..workspace/zhou/repair/results/` only contain date headers in:
  - `train_log.txt`
  - `val_log.txt`
- They are mostly run stubs created by legacy setting bootstrap, not full training histories.
- For real per-epoch U-Net metrics, prefer `unet_train_results/model_logs/*`.

## 6) Suggested reproduction starting points

- Legacy single-GPU U-Net path:
  - `train_denoise_step.py` + `unet_train_results/model_logs/*` + `unet_model/denoise_model_epoch_*.pth`
- Legacy DDP path (later variant):
  - `train_denoise_step_DDP.py`

## 7) File timestamps (for chronology)

- `train.py`: 2025-09-18
- `network/Network.py`: 2025-09-29
- `train_denoise_step_DDP.py`: 2025-11-12
- `train_denoise_step.py`: 2025-12-04

