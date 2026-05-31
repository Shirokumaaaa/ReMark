# Restoration + Lmsg Baselines

This repo reproduces the listed restoration baselines as trainable ReMark
Stage-1-compatible adapters.  Each model is fine-tuned with the frozen
LampMark decoder message loss (`Lmsg`) plus image reconstruction loss.

Run one baseline:

```bash
cd Forensic-Remark_Module
bash tools/run_restoration_lmsg_finetune.sh bdg36m
```

Run all:

```bash
cd Forensic-Remark_Module
bash tools/run_restoration_lmsg_finetune.sh all
```

Available names:

- `bdg36m`: generic restoration UNet approximation for BDG-36M + Lmsg
- `defusion`: one-step visual-instructed diffusion surrogate + Lmsg
- `fape_ir`: frequency-aware planner/executor surrogate + Lmsg
- `rar`: restore-assess-repeat recurrent restorer + Lmsg
- `bdg_sd2`: compact latent-bridge surrogate for BDG-SD2 + Lmsg

The common training override is
`configs/experiments/restoration_lmsg_base.yaml`; the model-specific
overrides are `configs/experiments/restoration_lmsg_*.yaml`.
