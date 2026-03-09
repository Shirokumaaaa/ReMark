# Forensic-SepMark Collaboration README

This document is for fast onboarding of new agents/collaborators on this branch.
It focuses on what was changed, where to look, how to run, and what to watch out for.

## 1) Project Scope in This Branch

This branch extends SepMark with DiffSwap-related evaluation, including:
- A lightweight face-swap-like noise (`DiffSwap()`), mainly for quick stress tests.
- A stricter Attack-DiffSwap wrapper (`AttackDiffSwap()`), intended to follow original DiffSwap attack chain as closely as possible inside SepMark test flow.

Main evaluation entry is still:
- `test_Dual_Mark.py`

## 2) Key File Map (Feature -> File)

Core test/eval orchestration:
- `test_Dual_Mark.py`
- `cfg/test_DualMark.yaml`

Noise layer registry:
- `network/noise_layers/__init__.py`

Lightweight approximation DiffSwap:
- `network/noise_layers/diffswap/main.py`

Attack-DiffSwap wrapper inside SepMark:
- `network/noise_layers/attack_diffswap/main.py`
- `network/noise_layers/attack_diffswap/__init__.py`

One-off utility scripts (not main test entry):
- `tools/real_diffswap_once.py`
- `tools/eval_real_diffswap_tracer.py`

## 3) AttackDiffSwap() Design Summary

`AttackDiffSwap()` (in `network/noise_layers/attack_diffswap/main.py`) does:
- Import Attack-DiffSwap components (`ldm`, `Portrait`, `DDIMSampler`, etc.).
- Load official DiffSwap checkpoint and config.
- Use EMA scope during sampling (`ema_scope("Plotting Inpaint")`).
- Use DDIM sampling with configurable steps and target-preserve-scale.
- For each encoded image, build target geometry via:
  - MTCNN 5-point detection -> affine theta conversion (DiffSwap-style).
  - dlib 68-point landmarks -> convex-hull face masks.
- Replace target geometry fields in batch before calling DiffSwap model.

Runtime controls (env vars):
- `SEPMARK_ATTACK_DIFFSWAP_ROOT`
- `SEPMARK_ATTACK_DIFFSWAP_CKPT`
- `SEPMARK_ATTACK_DIFFSWAP_CONFIG`
- `SEPMARK_ATTACK_DIFFSWAP_TGT_SCALE` (default `0.01`)
- `SEPMARK_ATTACK_DIFFSWAP_STEPS` (default `200`)

## 4) Commands You Will Reuse

### 4.1 Quick sanity check (1 sample)

```bash
cd ReMark/Forensic-SepMark
SEPMARK_TEST_RESULT_FOLDER='baseline/Dual_watermark_256_128_0.1_0.0002_0.5_se_se_1_10_10_10_0.1_2023_04_18_16_29_54/' \
SEPMARK_TEST_MODEL_EPOCH=90 \
SEPMARK_DATASET_PATH='../Dataset-CelebA_HQ' \
SEPMARK_TEST_NOISE_LAYER='AttackDiffSwap()' \
SEPMARK_TEST_BATCH_SIZE=1 \
SEPMARK_TEST_MAX_STEPS=1 \
SEPMARK_SAVE_IMAGES_NUMBER=1 \
SEPMARK_ATTACK_DIFFSWAP_STEPS=30 \
SEPMARK_ATTACK_DIFFSWAP_TGT_SCALE=0.01 \
conda run -n DiffSwap python test_Dual_Mark.py
```

### 4.2 Full-quality small eval (recommended visual check)

```bash
cd ReMark/Forensic-SepMark
SEPMARK_TEST_RESULT_FOLDER='baseline/Dual_watermark_256_128_0.1_0.0002_0.5_se_se_1_10_10_10_0.1_2023_04_18_16_29_54/' \
SEPMARK_TEST_MODEL_EPOCH=90 \
SEPMARK_DATASET_PATH='../Dataset-CelebA_HQ' \
SEPMARK_TEST_NOISE_LAYER='AttackDiffSwap()' \
SEPMARK_TEST_BATCH_SIZE=1 \
SEPMARK_TEST_MAX_STEPS=50 \
SEPMARK_SAVE_IMAGES_NUMBER=8 \
SEPMARK_ATTACK_DIFFSWAP_STEPS=200 \
SEPMARK_ATTACK_DIFFSWAP_TGT_SCALE=0.01 \
conda run -n DiffSwap python test_Dual_Mark.py
```

## 5) Latest Verified Result Snapshot

Most recent 50-sample run (scale=0.01, steps=200):
- CSV: `results/baseline/Dual_watermark_256_128_0.1_0.0002_0.5_se_se_1_10_10_10_0.1_2023_04_18_16_29_54/noise_eval_2026_03_08__13_27_33.csv`
- Grid: `results/baseline/Dual_watermark_256_128_0.1_0.0002_0.5_se_se_1_10_10_10_0.1_2023_04_18_16_29_54/images/epoch-test_AttackDiffSwap.png`

Metrics:
- `tracer_decoder_acc = 0.9875`
- `tracer_decoder_ber = 0.0125`
- `detector_decoder_acc = 0.976875`
- `detector_decoder_ber = 0.023125`

## 6) Known Issues / Caveats

1. Visual quality vs metric quality
- Message ACC can remain high while face realism is not perfect.
- Bit-level decode accuracy is not equivalent to human-perceived swap naturalness.

2. `swap_res` style output vs full compositing
- Current SepMark noise flow evaluates attack effect in encoded image domain directly.
- It does not fully replicate all external post-processing stages (repair/paste) used in standalone DiffSwap pipelines.

3. CPU fallback can be slow
- If CUDA is unavailable, `steps=200` is very slow.
- TensorFlow/CUDA warnings can appear even when run succeeds.

4. External dependency coupling
- `AttackDiffSwap()` depends on sibling repo: `../Attack-DiffSwap` (from `Forensic-SepMark`).
- Missing checkpoints or preprocessing assets in that repo will break the wrapper.

## 7) Important Branch-Specific Code Changes

Inside SepMark repo:
- Added `AttackDiffSwap` module and registration:
  - `network/noise_layers/attack_diffswap/main.py`
  - `network/noise_layers/__init__.py`
- Added `AttackDiffSwap()` note in config comment:
  - `cfg/test_DualMark.yaml`
- Made model loading robust to CPU-only execution:
  - `network/Dual_Mark.py` (`torch.load(..., map_location=self.device)`)

Outside SepMark repo (but required for this flow):
- `Attack-DiffSwap/ldm/models/diffusion/ddim.py` was patched to avoid hard forcing CUDA buffers when CUDA is unavailable.

## 8) Suggested Handoff Checklist for Next Agent

1. Confirm data and model paths exist:
- SepMark result folder and `EC_90.pth`
- CelebA-HQ dataset CSV/test split
- Attack-DiffSwap checkpoint and preprocessing artifacts

2. Start with 1-step smoke test.

3. Run 10-20 samples before larger runs.

4. Report both:
- `tracer/detector ACC-BER`
- Grid image path for visual inspection

5. If visuals look odd:
- First verify `SEPMARK_ATTACK_DIFFSWAP_TGT_SCALE` and `SEPMARK_ATTACK_DIFFSWAP_STEPS`
- Then inspect target landmark/mask generation in `AttackDiffSwap` wrapper.

## 9) Collaboration Notes

- Keep original project structure intact; add new methods as sibling modules under `network/noise_layers/`.
- Prefer reproducible env-var driven runs over hardcoded edits.
- When changing attack logic, update this document and include:
  - What changed
  - Why changed
  - One exact command to reproduce
