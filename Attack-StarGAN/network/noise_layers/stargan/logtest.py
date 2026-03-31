#!/usr/bin/env python
"""
Delta‑Analysis (Log‑Domain) for Real vs. Fake Image Pairs
========================================================
Given two folders — one containing real (true) images and the other containing
fake (Deepfake / StarGAN) images — this script pairs images with the same
filename stem, converts them to linear‑RGB, computes the log‑domain difference

    Δ = log(fake + ε) − log(real + ε)

and reports per‑pair as well as aggregated statistics.

Main outputs
------------
* Pearson |r| between log(real) and Δ for every pair
* Variance of Δ for every pair
* Overall averages printed to console
* Optional detailed visualisation (Δ‑map / histogram / scatter / FFT) for the
  first pair, enabled by --show

Usage
-----
    python delta_analysis_batch.py \
        --real_dir   /path/to/true  \
        --fake_dir   /path/to/fake  \
        --show

Dependencies
------------
    pip install numpy scipy tqdm opencv-python matplotlib torch

Notes
-----
* Images are assumed to be sRGB (8‑bit PNG/JPG).  They are linearised with a
  simple γ = 2.2 inverse (sufficient for comparative analysis). If your data
  are already linear, set --linear to skip this step.
* ε is fixed at 1e-6; adjust via --epsilon if necessary.
"""

import os, glob, argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import pearsonr

# ---------------------------  Constants  --------------------------- #
DEFAULT_EPS = 1e-6
GAMMA = 2.2  # sRGB ⇢ linear approximation

# ---------------------------  Utilities  --------------------------- #

def srgb_to_linear(img: torch.Tensor) -> torch.Tensor:
    """Approximate sRGB → linear‑RGB (γ=2.2). img in [0,1]."""
    return img ** GAMMA


def load_image(path: str) -> torch.Tensor:
    """Read image via cv2 (BGR), convert to RGB float32 [0,1], shape C×H×W."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(rgb).permute(2, 0, 1)  # C,H,W


def compute_delta(real_lin: torch.Tensor, fake_lin: torch.Tensor, eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return Δ (C×H×W) and Δ_gray (H×W)."""
    delta = torch.log(fake_lin + eps) - torch.log(real_lin + eps)
    delta_gray = delta.mean(dim=0)  # 转为灰度
    return delta, delta_gray


def analyse_pair(real: torch.Tensor, fake: torch.Tensor, eps: float) -> Tuple[float, float]:
    """Return |r| and var(Δ_gray). Both real/fake already linear RGB in [0,1]."""
    _, delta_gray = compute_delta(real, fake, eps)
    real_gray = real.mean(dim=0)  # 计算 real 的灰度图
    r, _ = pearsonr(real_gray.flatten().numpy(), delta_gray.flatten().numpy())  # 计算相关性
    return abs(r), delta_gray.var().item()


def visualise(real_lin: torch.Tensor, fake_lin: torch.Tensor, eps: float, title_prefix: str = "") -> None:
    """Four‑plot visualisation for the first pair."""
    delta, delta_gray = compute_delta(real_lin, fake_lin, eps)
    real_gray = real_lin.mean(dim=0)  # 计算 real 的灰度图
    r, p = pearsonr(real_gray.flatten().numpy(), delta_gray.flatten().numpy())

    print(f"[{title_prefix}]  Pearson r = {r:.4f}  (p={p:.2e})  |  Δ variance = {delta_gray.var().item():.4f}")

    delta_np = delta_gray.numpy()
    real_np = real_gray.numpy()

    plt.figure(figsize=(12, 5))
    plt.suptitle(title_prefix)

    plt.subplot(1, 4, 1)
    plt.imshow(delta_np, cmap='seismic')
    plt.colorbar(); plt.title('Δ‑map')

    plt.subplot(1, 4, 2)
    plt.hist(delta_np.flatten(), bins=256, color='royalblue', alpha=0.85)
    plt.title('Histogram Δ')

    plt.subplot(1, 4, 3)
    idx = np.random.choice(delta_np.size, delta_np.size // 10, replace=False)
    plt.scatter(real_np.flatten()[idx], delta_np.flatten()[idx], s=1, alpha=0.25)
    plt.xlabel('real'); plt.ylabel('Δ'); plt.title('Scatter')

    plt.subplot(1, 4, 4)
    fft_img = np.log(np.abs(np.fft.fftshift(np.fft.fft2(delta_np))) + eps)
    plt.imshow(fft_img, cmap='viridis')
    plt.colorbar(); plt.title('FFT |Δ|')

    plt.tight_layout()
    plt.savefig('/home/ldy/..workspace/kei/stargan/FFT.png')  # Save the figure before showing it
    plt.show()  # Optionally display the figure

# ---------------------------  Main  --------------------------- #

def main(cfg):
    real_paths = sorted(glob.glob(os.path.join(cfg.real_dir, '*')))
    fake_paths = sorted(glob.glob(os.path.join(cfg.fake_dir, '*')))
    assert real_paths and fake_paths, 'No images found in the provided directories.'

    # Debugging: Print the paths found
    print(f"Real image paths: {real_paths}")
    print(f"Fake image paths: {fake_paths}")

    # build stem→path dict for quick pairing
    stem2real = {Path(p).stem: p for p in real_paths}
    stem2fake = {Path(p).stem: p for p in fake_paths}
    common_stems = sorted(set(stem2real) & set(stem2fake))

    # Debugging: Print the common stems
    print(f"Common stems: {common_stems}")

    if not common_stems:
        raise ValueError('No matching filename stems between real and fake folders.')

    r_vals: List[float] = []
    var_vals: List[float] = []

    first_vis_done = False

    for stem in tqdm(common_stems, desc='Analysing'):
        real_img = load_image(stem2real[stem]).float()
        fake_img = load_image(stem2fake[stem]).float()

        # Optional linearisation
        if not cfg.linear:
            real_lin, fake_lin = srgb_to_linear(real_img), srgb_to_linear(fake_img)
        else:
            real_lin, fake_lin = real_img, fake_img

        r, var = analyse_pair(real_lin, fake_lin, cfg.epsilon)
        r_vals.append(r); var_vals.append(var)

        if cfg.show and not first_vis_done:
            visualise(real_lin, fake_lin, cfg.epsilon, title_prefix=stem)
            first_vis_done = True

    print('\n==========  Aggregated Statistics  ==========', flush=True)
    print(f'Total pairs analysed : {len(r_vals)}')
    print(f'Average |r|          : {np.mean(r_vals):.4f}')
    print(f'Average Δ variance   : {np.mean(var_vals):.4f}')

# ---------------------------  CLI  --------------------------- #
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Log‑domain Δ analysis for GAN‑generated images')
    ap.add_argument('--real_dir', help='Folder with REAL / true images', 
                    default='/home/ldy/..workspace/kei/stargan/stargan/results/true')
    ap.add_argument('--fake_dir', help='Folder with FAKE images generated by StarGAN / GAN', 
                    default='/home/ldy/..workspace/kei/stargan/stargan/results/fake')
    ap.add_argument('--epsilon', type=float, default=DEFAULT_EPS, help='Epsilon to avoid log(0)')
    ap.add_argument('--linear', action='store_true', help='Skip sRGB→linear γ‑correction if images already linear')
    ap.add_argument('--show', action='store_true', help='Show visualisation for the first matched pair')
    cfg = ap.parse_args()

    main(cfg)
