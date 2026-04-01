# ReMark

A watermark restoration framework designed against deepfake attacks.

This repository is not a lightweight release of a single model, but rather an experimental workbench that currently includes:

- The main `ReMark` module: restores watermarked images corrupted by deepfakes in latent space
- Multiple generative watermarking methods
- Multiple deepfake attack backends
- Multiple datasets: CelebA-HQ, FFHQ, Stable Diffusion Prompt

The main pipeline can be summarized as:

```text
Original image -> Watermark encoding -> Watermarked image -> Deepfake / Attack -> Corrupted image
                                                                        |
                                                                        v
                                                                ReMark restoration module
                                                                        |
                                                                        v
                                                                Watermark decoding / verification
````

The core idea of ReMark is not to rewrite existing watermarking models, but to add an independent restorer after the attack, so as to pull the corrupted image back toward the original watermarked image as much as possible, thereby improving watermark extraction accuracy.

## Repository Overview

### 1. Main Control Module

* `Forensic-Remark_Module/`

  * The main entry point of the whole repository.
  * Contains `train_stage1.py`, `train_stage2.py`, `configs/`, `data_manifests/`, `wm_adapters/`, `attacks/`, `tools/`.
  * Responsible for connecting different watermarking models and attack models into the same ReMark training/evaluation pipeline.

### 2. Watermarking Methods

These directories can be used independently or integrated into ReMark through `Forensic-Remark_Module/wm_adapters/`:

* `Forensic-FIN/`
* `Forensic-LaWa/`
* `Forensic-LampMark/`
* `Forensic-MaskWM/`
* `Forensic-SepMark/`
* `Forensic-SleeperMark/`
* `Forensic-TAG-WM/`
* `Forensic-TrustMark/`

The watermark adapters currently registered in ReMark include:

* [`FIN`](https://ojs.aaai.org/index.php/AAAI/article/view/25633)
* [`SepMark`](https://doi.org/10.1145/3581783.3612471)
* [`LampMark`](https://dl.acm.org/doi/10.1145/3664647.3680869)
* [`LaWa`](https://arxiv.org/abs/2408.05868)
* [`SleeperMark`](https://arxiv.org/abs/2412.04852)
* [`TAG-WM`](https://openaccess.thecvf.com/content/ICCV2025/html/Chen_TAG-WM_Tamper-Aware_Generative_Image_Watermarking_via_Diffusion_Inversion_Sensitivity_ICCV_2025_paper.html)
* [`MaskWM`](https://arxiv.org/abs/2504.12739)
* [`TrustMark`](https://openaccess.thecvf.com/content/ICCV2025/html/Bui_TrustMark_Robust_Watermarking_and_Watermark_Removal_for_Arbitrary_Resolution_Images_ICCV_2025_paper.html)

### 3. Attack Methods

These directories provide deepfake / face swap / reenactment backends for ReMark:

* `Attack-DiffSwap/`
* `Attack-Face-Adapter/`
* `Attack-REFace/`
* `Attack-StarGAN/`
* `Attack-arc2face_wrapper/`

The attack adapters currently registered in ReMark include:

* [`stargan2`](https://openaccess.thecvf.com/content_CVPR_2020/html/Choi_StarGAN_v2_Diverse_Image_Synthesis_for_Multiple_Domains_CVPR_2020_paper.html)
* [`SimSwap`](https://dl.acm.org/doi/abs/10.1145/3394171.3413630)
* [`Arc2face_wrapper`](https://openaccess.thecvf.com/content/ICCV2025W/I-HFM/html/Papantoniou_ID-Consistent_Precise_Expression_Generation_with_Blendshape-Guided_Diffusion_ICCVW_2025_paper.html)
* [`DiffSwap`](https://openaccess.thecvf.com/content/CVPR2023/html/Zhao_DiffSwap_High-Fidelity_and_Controllable_Face_Swapping_via_3D-Aware_Masked_Diffusion_CVPR_2023_paper.html)
* [`REFace`](https://ieeexplore.ieee.org/abstract/document/10943471)
* [`Face_Adapter`](https://link.springer.com/chapter/10.1007/978-3-031-72973-7_2)

### 4. Data and General Tools

* `Dataset-CelebA_HQ/`
* `Dataset-FFHQ/`
* `common/`

  * Shared logic, currently mainly the landmark -> bit encoding tools.
* `tools/`

  * Top-level data cleaning scripts, such as landmark bit cache generation and clean subset construction.

## Recommended Reading Order

If this is your first time working with this repository, it is recommended to read in the following order:

1. Root `README.md`: first understand the division of roles across the whole repository.
2. `Forensic-Remark_Module/README.md`: more detailed design notes and experimental context.
3. `Forensic-Remark_Module/readme_research.md`: research notes on latent geometry.
4. The README files of the specific watermarking method and attack method you actually plan to use.

## ReMark Main Pipeline

`Forensic-Remark_Module/` is currently the part most worth running first.

### Stage 1

Train a VAE-style reconstructor to map the corrupted image after deepfake manipulation back into a recoverable latent-space representation.

Entry:

```bash
cd Forensic-Remark_Module
python train_stage1.py --config configs/stage1_vae.yaml
```

Multi-GPU:

```bash
cd Forensic-Remark_Module
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_stage1.py \
  --config configs/stage1_vae.yaml
```

### Stage 2

Train a U-Net restorer in the Stage 1 latent space to gradually push the corrupted latent back toward the original watermarked latent, ultimately achieving watermark restoration.

Entry:

```bash
cd Forensic-Remark_Module
python train_stage2.py --config configs/stage2_unet.yaml
```

Multi-GPU:

```bash
cd Forensic-Remark_Module
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_stage2.py \
  --config configs/stage2_unet.yaml
```

Manually specify the Stage 1 checkpoint:

```bash
cd Forensic-Remark_Module
python train_stage2.py \
  --config configs/stage2_unet.yaml \
  --stage1-ckpt runs/<stage1_run>/checkpoints/vae/best.pth
```

Resume training:

```bash
cd Forensic-Remark_Module
python train_stage1.py --config configs/stage1_vae.yaml --resume <run_name>
python train_stage2.py --config configs/stage2_unet.yaml --resume <run_name>
```

Training outputs are written by default to:

```text
Forensic-Remark_Module/runs/<stage>_<timestamp>/
  config.yaml
  train.log
  samples/
  checkpoints/
```

## Configurations That Must Be Modified Before Running

The default YAML files in the current repository still contain machine-specific paths and local experimental states. Before running for the first time, be sure to check:

* `Forensic-Remark_Module/configs/stage1_vae.yaml`
* `Forensic-Remark_Module/configs/stage2_unet.yaml`

Also note in particular: these two base YAML files currently correspond to local experiment snapshots from different time points, so they are not guaranteed to be a matched pair by default. When switching watermarking models, attack models, or datasets, make sure to check Stage 1 and Stage 2 together.

Pay particular attention to the following fields:

* `data.train_csv`
* `data.val_csv`
* `wm_model`
* `attacks.online`
* `attacks.offline`
* `attack_options.*`
* `paths.stage1_checkpoint` (Stage 2)

The `configs/experiments/` directory is currently empty, so for now the main workflow is to edit the base config files directly.

## Data Organization and Manifest Format

ReMark training data is driven by CSV manifests. For different training requirements, there is no need to move image files around; only the CSV files need to be modified. The typical formats supported by `Forensic-Remark_Module/data/dataset.py` are as follows.

Minimal format:

```csv
img_path
/abs/path/to/image1.png
/abs/path/to/image2.png
```

With offline attack images:

```csv
img_path,fake_path
/abs/path/to/image1.png,/abs/path/to/fake1.png
```

With watermark cache:

```csv
img_path,wm_path
/abs/path/to/image1.png,/abs/path/to/wm1.npy
```

If the CSV also contains additional columns:

* Numeric columns will be treated as attribute labels, such as `Black_Hair, Blond_Hair, Brown_Hair, Male, Young`
* String columns will be retained as text conditions, such as prompts or replay metadata

This is also why manifests such as `ffhq_10k_train_with_attrs.csv` can directly support conditional attacks.

## Online Attacks and Replay Attacks

In ReMark, attacks are divided into two categories:

* Online attacks

  * Fake images are generated directly inside the training loop
* Replay / offline attacks

  * Results are generated offline first, then mapped back to training samples through CSV or JSONL

Some deepfake attack methods, such as `diffswap` and `arc2face`, are relatively slow when used as online attacks, so offline attacks were designed for them. Although this approach reduces attack diversity to some extent, it is a good trade-off when time is limited and compute resources are constrained.

Therefore, `Forensic-Remark_Module/tools/` contains many helper scripts specifically used for:

* Building replay indices
* Converting offline results into manifests
* Generating training lists for FFHQ / CelebA-HQ
* Performing transfer evaluation, false positive evaluation, and latent geometry analysis

Commonly used scripts include:

* `Forensic-Remark_Module/tools/analyze_latent_geometry.py`
* `Forensic-Remark_Module/tools/build_manifest_from_replay_jsonl.py`
* `Forensic-Remark_Module/tools/build_ffhq_diffswap_replay_index.py`
* `Forensic-Remark_Module/tools/build_ffhq_reface_replay_index.py`
* `Forensic-Remark_Module/tools/build_ffhq_faceadapter_replay_index.py`
* `Forensic-Remark_Module/tools/prepare_ffhq_ffpp_10k.py`

## Data Preprocessing Scripts

### Landmark Bit Cache and Clean Subset

The top-level `common/landmark_bits.py` provides unified landmark-bit encoding logic, which is typically used in landmark-conditioned watermarking experiments.

Build cache:

```bash
python tools/build_landmark_bit_cache.py \
  --csv /path/to/train.csv \
  --cache-dir landmark_bit_cache
```

Filter failed samples and generate a clean subset:

```bash
python tools/prepare_landmark_clean_subset.py \
  --train-csv /path/to/train.csv \
  --val-csv /path/to/val.csv \
  --cache-dir landmark_bit_cache \
  --out-root Dataset-CelebA_HQ_10k_landmark_clean_20260324 \
  --lampmark-manifest-dir Forensic-LampMark/data_manifests
```

## Environment and Dependencies

This repository currently does not provide a unified root-level `requirements.txt`, because it is essentially a combination of multiple projects. It is recommended to think of dependencies in two layers:

### 1. The main module you want to run

If you only want to run the ReMark main pipeline, prioritize making sure the following are all available:

* `Forensic-Remark_Module/`
* The watermark backend you selected
* The attack backend you selected

The dependencies of all three must be available simultaneously.

### 2. Environment files of individual subprojects

Dependency files currently available in the repository include:

* `Attack-DiffSwap/requirements.txt`
* `Attack-Face-Adapter/requirements.txt`
* `Attack-REFace/environment.yml`
* `Attack-REFace/requirements.txt`
* `Forensic-LaWa/environment.yml`
* `Forensic-TAG-WM/requirements.txt`
* `Forensic-TrustMark/python/requirements.txt`
* `Forensic-TrustMark/python/pyproject.toml`

In actual use, pay special attention to the following:

* Many scripts and YAML files still contain absolute paths, such as `/mnt/personal_workspace/...` and `/home/ldy/miniconda3/...`
* Model weights are generally not distributed with the repository and must be downloaded separately according to the README of each subproject
* Some attack or watermark modules depend on independent conda environments

## Current Positioning of the Repository

To avoid misunderstanding, the current status is clarified here:

* This is a research integration repository, not a one-click release.
* The goal of the root README is to give you a map, not to replace the original documentation of each subproject.
* Many subdirectories are combinations of upstream projects, reproduction code, or locally adapted versions.
* `Forensic-Remark_Module/README.md` still contains a large amount of experimental records and research logs, which are suitable as contextual references, but should not be treated as the only quick-start document.

## Suggested Onboarding Route

If you want to become familiar with the whole repository as quickly as possible, it is recommended to proceed in the following order:

1. First determine which watermarking method you want to use, such as `lampmark` or `sepmark`.
2. Then determine which attack backend you want to use, such as `simswap`, `stargan2`, or `diffswap`.
3. Install the corresponding dependencies.
4. Modify `Forensic-Remark_Module/configs/stage1_vae.yaml` and run Stage 1 first.
5. After Stage 1 produces a stable checkpoint, start Stage 2.
6. If you need to study interpretability or transferability, then look into `tools/analyze_latent_geometry.py` and the various replay / eval tools.

## Citation and Acknowledgments

If you use this repository for research, please cite:

* The ReMark paper or project description
* The original paper of the watermarking method you actually use
* The original paper of the attack method you actually use

Each subdirectory usually already includes its own README, paper links, and license notes. Please refer to the corresponding subproject for the final authoritative information.

