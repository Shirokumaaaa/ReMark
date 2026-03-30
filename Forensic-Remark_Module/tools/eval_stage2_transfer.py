#!/usr/bin/env python3
"""
Evaluate Stage2 transfer across attacks: raw attacked ACC vs Stage2 repaired ACC.

Example:
  CUDA_VISIBLE_DEVICES=0 python tools/eval_stage2_transfer.py \
      --run-dir runs/stage2_20260311_133024 \
      --checkpoint best \
      --attacks simswap stargan \
      --max-batches 8
"""

import argparse
import csv
import os
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.dataset import ReMark_Dataset
from network.unet import build_unet
from wm_adapters.registry import build_wm_adapter
from attacks.registry import build_attack
from train_stage2 import _infer_stage1_cfg, load_stage1_vae

# Register attacks/adapters.
import attacks  # noqa: F401
import wm_adapters  # noqa: F401


def dict_to_ns(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_ns(v) for k, v in d.items()})
    if isinstance(d, list):
        return [dict_to_ns(v) for v in d]
    return d


def load_cfg(run_dir: str):
    cfg_path = os.path.join(run_dir, "config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return dict_to_ns(yaml.safe_load(f))


def resolve_unet_ckpt(run_dir: str, checkpoint: str):
    if os.path.isabs(checkpoint) and os.path.isfile(checkpoint):
        return checkpoint
    if os.path.isfile(checkpoint):
        return os.path.abspath(checkpoint)
    if checkpoint in ("best", "last"):
        p = os.path.join(run_dir, "checkpoints", "unet", f"{checkpoint}.pth")
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"UNet checkpoint not found: {checkpoint}")


def build_loader(cfg, split: str, batch_size: int, num_workers: int):
    csv_path = cfg.data.val_csv if split == "val" else cfg.data.train_csv
    ds = ReMark_Dataset(
        csv_path=csv_path,
        image_size=cfg.data.image_size,
        mode="val",
    )
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def decode_latent(vae, z, reference):
    x_hat = vae.decode(z)
    if bool(getattr(vae, "residual_output", False)):
        scale = float(getattr(vae, "residual_scale", 1.0))
        x_hat = torch.clamp(reference + scale * x_hat, -1.0, 1.0)
    return x_hat


def step_timestep(step_idx: int, total_steps: int, time_min: float, time_max: float, batch_size: int, device):
    if total_steps <= 1:
        t = time_max
    else:
        alpha = step_idx / float(total_steps - 1)
        t = time_max + (time_min - time_max) * alpha
    return torch.full((batch_size,), float(t), device=device, dtype=torch.float32)


def unet_step(unet, z, k, total_steps, time_cond, time_min, time_max, delta_scale, out_blend):
    if time_cond:
        t = step_timestep(k, total_steps, time_min, time_max, z.shape[0], z.device)
        delta = unet(z, timestep=t)
    else:
        delta = unet(z)
    if out_blend < 1.0:
        delta = out_blend * delta
    return z + delta_scale * delta


def apply_attack(attack, wm_images, images, batch):
    if hasattr(attack, "attack_with_cover"):
        try:
            return attack.attack_with_cover(wm_images, images, batch=batch)
        except TypeError:
            return attack.attack_with_cover(wm_images, images)
    return attack(wm_images)


def maybe_filter_replay_batch(attack, wm_images, images, batch):
    """
    Replay attacks may fail for only part of a batch due to missing path keys.
    Fall back to per-sample replay and skip missing entries.
    """
    try:
        attacked = apply_attack(attack, wm_images, images, batch)
        idx = torch.arange(wm_images.shape[0], device=wm_images.device)
        return attacked, idx, 0
    except Exception:
        fake_list = []
        idx_list = []
        skipped = 0
        b = wm_images.shape[0]
        for i in range(b):
            one_batch = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    one_batch[k] = v[i:i + 1]
                elif isinstance(v, (list, tuple)):
                    one_batch[k] = [v[i]]
                else:
                    one_batch[k] = v
            try:
                fake_i = apply_attack(
                    attack, wm_images[i:i + 1], images[i:i + 1], one_batch
                )
                fake_list.append(fake_i)
                idx_list.append(i)
            except Exception:
                skipped += 1
        if not idx_list:
            return None, None, skipped
        attacked = torch.cat(fake_list, dim=0)
        idx = torch.tensor(idx_list, dtype=torch.long, device=wm_images.device)
        return attacked, idx, skipped


@torch.no_grad()
def evaluate(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    run_dir = os.path.abspath(args.run_dir)
    cfg = load_cfg(run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    unet_ckpt = resolve_unet_ckpt(run_dir, args.checkpoint)
    cfg_s1 = _infer_stage1_cfg(cfg.paths.stage1_checkpoint, cfg)
    vae, latent_channels, _ = load_stage1_vae(cfg.paths.stage1_checkpoint, cfg_s1, device)
    vae.eval()

    unet = build_unet(cfg, latent_channels).to(device)
    state = torch.load(unet_ckpt, map_location=device, weights_only=False)
    unet.load_state_dict(state["net"])
    unet.eval()

    wm_adapter = build_wm_adapter(cfg.wm_model, cfg)
    msg_len = int(wm_adapter.message_length)
    max_infer = int(getattr(cfg.training, "max_infer_steps", 10))
    time_cond = bool(getattr(getattr(cfg, "model", None), "time_cond", False))
    slerp_cfg = getattr(cfg, "slerp", None)
    time_min = float(getattr(slerp_cfg, "time_min", 0.0))
    time_max = float(getattr(slerp_cfg, "time_max", 1000.0))
    delta_scale = float(getattr(getattr(cfg, "training", None), "delta_scale", 1.0))
    out_blend = float(getattr(getattr(cfg, "training", None), "unet_output_blend", 1.0))

    batch_size = args.batch_size if args.batch_size > 0 else int(cfg.training.batch_size)
    loader = build_loader(cfg, split=args.split, batch_size=batch_size, num_workers=args.num_workers)

    rows = []
    for attack_name in args.attacks:
        try:
            attack = build_attack(attack_name, cfg)
        except Exception as e:
            rows.append({
                "attack": attack_name,
                "status": "build_failed",
                "num_samples": 0,
                "skipped_samples": 0,
                "step0_acc": float("nan"),
                "step_last_acc": float("nan"),
                "raw_acc": float("nan"),
                "rec_acc": float("nan"),
                "delta": float("nan"),
                "rec_l1": float("nan"),
                "error": str(e),
            })
            print(f"[SKIP] {attack_name}: build failed -> {e}")
            continue

        correct_raw = 0.0
        correct_rec = 0.0
        total_bits = 0.0
        l1_sum = 0.0
        sample_count = 0
        skipped_total = 0

        for bidx, batch in enumerate(loader):
            if args.max_batches > 0 and bidx >= args.max_batches:
                break

            images = batch["image"].to(device, non_blocking=True)
            bs = images.shape[0]
            messages = torch.randint(0, 2, (bs, msg_len), dtype=torch.float32, device=device)
            wm_images = wm_adapter.encode(images, messages)

            attacked, use_idx, skipped = maybe_filter_replay_batch(
                attack, wm_images, images, batch
            )
            skipped_total += skipped
            if attacked is None:
                continue

            msg_sel = messages.index_select(0, use_idx)
            wm_sel = wm_images.index_select(0, use_idx)

            # raw_acc 口径对齐 train_stage2：
            # Stage1-only baseline = VAE resample (no U-Net).
            z_src, _ = vae.encode(attacked)
            x_stage1 = decode_latent(vae, z_src, attacked)
            raw_logits = wm_adapter.decode(x_stage1)
            raw_pred = (raw_logits > 0).float()
            correct_raw += raw_pred.eq(msg_sel).float().sum().item()

            z = z_src
            for k in range(max_infer):
                z = unet_step(
                    unet, z, k, max_infer,
                    time_cond=time_cond,
                    time_min=time_min,
                    time_max=time_max,
                    delta_scale=delta_scale,
                    out_blend=out_blend,
                )
            x_hat = decode_latent(vae, z, attacked)
            rec_logits = wm_adapter.decode(x_hat)
            rec_pred = (rec_logits > 0).float()
            correct_rec += rec_pred.eq(msg_sel).float().sum().item()

            z_gt, _ = vae.encode(wm_sel)
            l1_sum += torch.nn.functional.l1_loss(
                z, z_gt, reduction="mean"
            ).item() * attacked.shape[0]

            total_bits += float(msg_sel.numel())
            sample_count += int(attacked.shape[0])

        if sample_count == 0 or total_bits <= 0:
            rows.append({
                "attack": attack_name,
                "status": "no_valid_samples",
                "num_samples": 0,
                "skipped_samples": skipped_total,
                "step0_acc": float("nan"),
                "step_last_acc": float("nan"),
                "raw_acc": float("nan"),
                "rec_acc": float("nan"),
                "delta": float("nan"),
                "rec_l1": float("nan"),
                "error": "",
            })
            print(f"[SKIP] {attack_name}: no valid samples")
            continue

        raw_acc = correct_raw / total_bits
        rec_acc = correct_rec / total_bits
        delta = rec_acc - raw_acc
        rec_l1 = l1_sum / max(sample_count, 1)
        rows.append({
            "attack": attack_name,
            "status": "ok",
            "num_samples": sample_count,
            "skipped_samples": skipped_total,
            "step0_acc": raw_acc,
            "step_last_acc": rec_acc,
            "raw_acc": raw_acc,
            "rec_acc": rec_acc,
            "delta": delta,
            "rec_l1": rec_l1,
            "error": "",
        })
        print(
            f"[{attack_name}] n={sample_count} skipped={skipped_total} "
            f"raw_acc={raw_acc:.4f} rec_acc={rec_acc:.4f} delta={delta:+.4f} rec_l1={rec_l1:.4f}"
        )

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_name = f"transfer_eval_stage2_{os.path.basename(unet_ckpt).replace('.pth', '')}_{args.split}_{stamp}.csv"
    out_csv = os.path.join(run_dir, out_name)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "attack", "status", "num_samples", "skipped_samples",
                "step0_acc", "step_last_acc",
                "raw_acc", "rec_acc", "delta", "rec_l1", "error"
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved CSV: {out_csv}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--checkpoint", default="best", help="best|last|/abs/path/to/unet.pth")
    p.add_argument("--split", choices=["train", "val"], default="val")
    p.add_argument("--attacks", nargs="+", required=True)
    p.add_argument("--max-batches", type=int, default=8, help="0 means full split")
    p.add_argument("--batch-size", type=int, default=0, help="0 means cfg.training.batch_size")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
