#!/usr/bin/env python3

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from train_stage1 import build_loaders, cleanup_ddp, is_main, set_seed, setup_ddp, Trainer
from utils.config import load_config
from utils.logger import RunLogger


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/stage1_vae.yaml")
    p.add_argument("--override", default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    rank, local_rank, world_size = setup_ddp()
    args = parse_args()
    cfg = load_config(args.config, args.override)
    set_seed(args.seed, rank)

    import torch

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    logger = None
    if is_main(rank):
        logger = RunLogger(
            cfg,
            config_path=args.config,
            override_path=args.override,
            stage="stage1_preflight",
            resume_run=None,
        )
        logger.info(f"Run dir   : {logger.run_dir}")
        logger.info(f"Device    : {device}  |  world_size: {world_size}")

    train_loader, val_loader, train_sampler = build_loaders(cfg, rank, world_size)
    del train_loader, train_sampler

    if is_main(rank):
        n_val = len(val_loader.dataset)
        logger.info(f"Val: {n_val} samples")

    trainer = Trainer(cfg, logger, device, rank, world_size, resume_path=None)

    if is_main(rank):
        n_params = sum(p.numel() for p in trainer._vae_module.parameters())
        logger.info(f"VAE: {n_params/1e6:.1f}M params")
        logger.info("Preflight-only run: training loop is skipped.")

    trainer._run_preflight_baseline(val_loader)

    if is_main(rank):
        logger.info("Preflight complete.")
        logger.close()

    cleanup_ddp()


if __name__ == "__main__":
    main()
