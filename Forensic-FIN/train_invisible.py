"""
train_invisible.py
Fine-tune a pre-trained FIN checkpoint toward invisible watermarking.

Strategy:
  - Load from fed_best_auto.pt (or explicit --init-ckpt)
  - Gradually shift from message-priority to visual-quality-priority
  - Add SSIM loss (kornia) as perceptual quality constraint
  - Use gradient clipping to keep INN stable
  - Three phases with increasing stego_weight / ssim_weight

Target: PSNR > 40 dB with bit-acc > 90%

Usage:
    conda activate sepmark
    cd /mnt/personal_workspace/chenkeyu/ReMark/Forensic-FIN
    python train_invisible.py

    # Or explicit init checkpoint:
    python train_invisible.py --init-ckpt experiments/celeba_hq_128/fed_best_auto.pt
"""

import argparse
import logging
import os
from dataclasses import dataclass

import kornia
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config as c
from models.encoder_decoder import FED
from utils.datasets import INN_Dataset, transform, transform_val
from utils.jpeg import JpegSS, JpegTest
from utils.metric import decoded_message_error_rate_batch, psnr


# -----------------------------------------------------------------------
# Losses
# -----------------------------------------------------------------------

def mse_loss(a, b):
    return nn.functional.mse_loss(a, b)


def ssim_loss(stego, cover):
    """1 - SSIM, clamped to [0,1]. Inputs in [-1,1]."""
    # kornia.losses.ssim_loss expects [0,1] or [-1,1]; values already clipped.
    return kornia.losses.ssim_loss(stego, cover, window_size=11, reduction='mean')


# -----------------------------------------------------------------------
# Checkpoint helpers
# -----------------------------------------------------------------------

def load_checkpoint(model, optim, path, device, logger):
    ckpt = torch.load(path, map_location=device)
    net_state = {k: v for k, v in ckpt['net'].items() if 'tmp_var' not in k}
    model.load_state_dict(net_state)
    if optim is not None and 'opt' in ckpt:
        try:
            optim.load_state_dict(ckpt['opt'])
            for state in optim.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)
            logger.info("Optimizer state restored from checkpoint.")
        except Exception as e:
            logger.warning(f"Optimizer state could not be restored ({e}); using fresh optimizer.")
    logger.info(f"Loaded weights from {path}")


def save_checkpoint(model, optim, path):
    torch.save({'net': model.state_dict(), 'opt': optim.state_dict()}, path)


# -----------------------------------------------------------------------
# Phase definition
# -----------------------------------------------------------------------

@dataclass
class Phase:
    name: str
    epochs: int
    lr: float
    mw: float   # message_weight
    sw: float   # stego MSE weight
    aw: float   # ssim weight  (1 - SSIM)


DEFAULT_PHASES = [
    # Phase 1: gentle balance shift (keep accuracy, start improving PSNR)
    Phase('p1_balance',   epochs=20, lr=2e-5, mw=12.0, sw=8.0,  aw=8.0),
    # Phase 2: push visual quality (PSNR target ~35+ dB)
    Phase('p2_quality',   epochs=30, lr=1e-5, mw=8.0,  sw=15.0, aw=12.0),
    # Phase 3: maximize invisibility (PSNR target ~40+ dB)
    Phase('p3_invisible', epochs=20, lr=5e-6, mw=6.0,  sw=22.0, aw=18.0),
]


# -----------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------

def run(args):
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs('logging', exist_ok=True)

    log_path = os.path.join('logging', 'train_invisible.log')
    logger = logging.getLogger('invisible')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s %(levelname)s: %(message)s', datefmt='%H:%M:%S')
    logger.addHandler(logging.FileHandler(log_path, mode='w'))
    logger.addHandler(logging.StreamHandler())
    for h in logger.handlers:
        h.setFormatter(fmt)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')

    fed = FED(c.diff, c.message_length).to(device)
    params = list(filter(lambda p: p.requires_grad, fed.parameters()))
    optim = torch.optim.Adam(params, lr=DEFAULT_PHASES[0].lr,
                             betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)

    init_ckpt = args.init_ckpt or os.path.join(c.MODEL_PATH, 'fed_best_auto.pt')
    load_checkpoint(fed, optim, init_ckpt, device, logger)

    noise_layer      = JpegSS(50)
    test_noise_layer = JpegTest(50)

    trainloader = DataLoader(
        INN_Dataset(transforms=transform, mode='train'),
        batch_size=c.batch_size, shuffle=True,
        pin_memory=True, num_workers=0, drop_last=True,
    )
    valloader = DataLoader(
        INN_Dataset(transforms=transform_val, mode='val'),
        batch_size=c.batchsize_val, shuffle=False,
        pin_memory=True, num_workers=0, drop_last=True,
    )

    best_psnr = 0.0
    best_path = os.path.join(args.save_dir, 'fed_invisible_best.pt')

    phases = DEFAULT_PHASES
    step = 0
    global_epoch = 217  # continues from epoch 216

    logger.info('=== Invisible Fine-Tuning Schedule ===')
    for p in phases:
        logger.info(f'  {p.name}: {p.epochs} epochs  lr={p.lr:.1e}  '
                    f'mw={p.mw}  sw={p.sw}  ssim_w={p.aw}')
    logger.info('======================================')

    for phase in phases:
        for g in optim.param_groups:
            g['lr'] = phase.lr

        for _ in range(phase.epochs):
            # ---- Train ----
            fed.train()
            t_loss, t_sloss, t_mloss, t_psnr, t_acc = [], [], [], [], []

            for cover in trainloader:
                cover = cover.to(device)
                msg = torch.Tensor(
                    np.random.choice([-0.5, 0.5], (cover.shape[0], c.message_length))
                ).to(device)

                stego, left_noise = fed([cover, msg])
                # Clamp stego: prevent values drifting out of [-1,1] which causes
                # visible saturation artifacts
                stego_clamped = stego.clamp(-1.0, 1.0)
                stego_noised = noise_layer(stego_clamped.clone())

                zeros = torch.zeros_like(left_noise)
                _, re_msg = fed([stego_noised, zeros], rev=True)

                s_loss = mse_loss(stego_clamped, cover)
                m_loss = mse_loss(re_msg, msg)
                a_loss = ssim_loss(stego_clamped, cover)

                loss = phase.mw * m_loss + phase.sw * s_loss + phase.aw * a_loss
                loss.backward()

                # Gradient clipping: critical for INN stability during fine-tuning
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)

                optim.step()
                optim.zero_grad()

                t_psnr.append(psnr(cover, stego_clamped, 255))
                t_acc.append(1 - decoded_message_error_rate_batch(msg, re_msg))
                t_loss.append(loss.item())
                t_sloss.append(s_loss.item())
                t_mloss.append(m_loss.item())

            # ---- Val ----
            fed.eval()
            v_psnr, v_acc = [], []
            with torch.no_grad():
                for cover in valloader:
                    cover = cover.to(device)
                    msg = torch.Tensor(
                        np.random.choice([-0.5, 0.5], (cover.shape[0], c.message_length))
                    ).to(device)
                    stego, left_noise = fed([cover, msg])
                    stego_clamped = stego.clamp(-1.0, 1.0)
                    stego_noised = test_noise_layer(stego_clamped.clone())
                    zeros = torch.zeros_like(left_noise)
                    _, re_msg = fed([stego_noised, zeros], rev=True)
                    v_psnr.append(psnr(cover, stego_clamped, 255))
                    v_acc.append(1 - decoded_message_error_rate_batch(msg, re_msg))

            ep_val_psnr = float(np.mean(v_psnr))
            ep_val_acc  = float(np.mean(v_acc))

            logger.info(
                f'[{phase.name}] Epoch {global_epoch:04d} | '
                f'lr={phase.lr:.1e} mw={phase.mw} sw={phase.sw} ssim_w={phase.aw} | '
                f'train_loss={np.mean(t_loss):.4f}  '
                f'stego_loss={np.mean(t_sloss):.5f}  msg_loss={np.mean(t_mloss):.5f}  '
                f'train_psnr={np.mean(t_psnr):.3f}  train_acc={np.mean(t_acc):.4f} | '
                f'val_psnr={ep_val_psnr:.3f}  val_acc={ep_val_acc:.4f}'
            )

            # Save every 5 epochs
            if step % 5 == 0:
                ckpt_path = os.path.join(
                    args.save_dir, f'fed_inv_{ep_val_psnr:.3f}_{global_epoch:05d}.pt'
                )
                save_checkpoint(fed, optim, ckpt_path)

            # Track best PSNR checkpoint
            if ep_val_psnr > best_psnr:
                best_psnr = ep_val_psnr
                save_checkpoint(fed, optim, best_path)
                logger.info(f'  >> New best PSNR: {best_psnr:.3f} dB  ->  {best_path}')

            step += 1
            global_epoch += 1

    # Final checkpoint
    final_path = os.path.join(args.save_dir, 'fed_invisible_final.pt')
    save_checkpoint(fed, optim, final_path)
    # Update FED.pt symlink to point to new best
    fed_pt = os.path.join(args.save_dir, 'FED.pt')
    if os.path.islink(fed_pt) or os.path.exists(fed_pt):
        os.remove(fed_pt)
    os.symlink(best_path, fed_pt)

    logger.info('=== Training complete ===')
    logger.info(f'Best val_psnr : {best_psnr:.3f} dB')
    logger.info(f'Best checkpoint: {best_path}')
    logger.info(f'FED.pt symlink -> {best_path}')


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Fine-tune FIN for invisible watermarking')
    p.add_argument('--init-ckpt', default='', type=str,
                   help='Path to starting checkpoint (default: experiments/celeba_hq_128/fed_best_auto.pt)')
    p.add_argument('--save-dir', default=c.MODEL_PATH, type=str,
                   help='Directory to save checkpoints')
    return p.parse_args()


if __name__ == '__main__':
    run(parse_args())
