Stage2 Eval Guide
=================

Prereqs
--------
- A working conda env with the Stage2 deps (diffusers, torch, torchvision, transformers, kornia, wandb, etc.).
- Downloaded model artifacts:
  - `Output/` contains the watermarked UNet files:
    - `Output/config.json`
    - `Output/diffusion_pytorch_model.safetensors`
  - `pretrainedWM/` contains watermark encoder/decoder and fixed secret/residual:
    - `pretrainedWM/encoder.pth`
    - `pretrainedWM/decoder.pth`
    - `pretrainedWM/secret.pt`
    - `pretrainedWM/res.pt`
- Prompt file: `sampled_captions2014.jsonl` in this directory.

Quick Eval
----------
Run a small eval (recommended first):
```
cd /mnt/personal_workspace/chenkeyu/ReMark/Forensic-SleeperMark/Stage2
PYTHONNOUSERSITE=1 /home/ldy/miniconda3/envs/SleeperMark/bin/python eval.py \
  --unet_dir Output \
  --pretrainedWM_dir pretrainedWM \
  --num_samples 10
```

Notes
-----
- `--unet_dir` should point to the folder that contains `config.json`.
- If Hugging Face access is required and you use a proxy, make sure the host proxy is enabled.
- Increase `--num_samples` for a larger evaluation (default is 10 in code).
