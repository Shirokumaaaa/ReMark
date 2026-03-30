#!/usr/bin/env python3
"""
Evaluate watermark bit-ACC/BER on attacked images before and after VAE reconstruction.

Usage:
  python tools/eval_attacked_recon_acc.py \
    --run-dir runs/stage1_20260310_134226 \
    --checkpoint best \
    --split val \
    --max-batches 8
"""

import argparse
import csv
import os
import random
import sys
from types import SimpleNamespace

import numpy as np
import torch

# Allow "python tools/xxx.py" to import project packages.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.dataset import ReMark_Dataset
from network.vae import build_vae
from wm_adapters.registry import build_wm_adapter
from attacks.registry import build_attack, ATTACK_REGISTRY
from utils.message_bits import deterministic_messages_from_paths

# Ensure adapter/attack modules register themselves.
import wm_adapters  # noqa: F401
import attacks  # noqa: F401


def dict_to_ns(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_ns(v) for k, v in d.items()})
    if isinstance(d, list):
        return [dict_to_ns(v) for v in d]
    return d


def load_run_config(run_dir: str):
    import yaml

    cfg_path = os.path.join(run_dir, "config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return dict_to_ns(cfg)


def resolve_ckpt_path(run_dir: str, checkpoint: str) -> str:
    if os.path.isabs(checkpoint) and os.path.isfile(checkpoint):
        return checkpoint
    if os.path.isfile(checkpoint):
        return os.path.abspath(checkpoint)
    if checkpoint in ("best", "last"):
        path = os.path.join(run_dir, "checkpoints", "vae", f"{checkpoint}.pth")
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"checkpoint not found: {checkpoint}")


def build_loader(cfg, split: str, batch_size: int, num_workers: int):
    csv_path = cfg.data.val_csv if split == "val" else cfg.data.train_csv
    dataset = ReMark_Dataset(
        csv_path=csv_path,
        image_size=cfg.data.image_size,
        mode="val",
        use_wm_cache=False,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def _build_messages_for_batch(cfg, wm_adapter, batch, images, message_len, device):
    bs = images.shape[0]
    training_cfg = getattr(cfg, "training", None)
    deterministic_messages = bool(getattr(training_cfg, "deterministic_messages", False))
    message_seed_salt = str(getattr(training_cfg, "message_seed_salt", "remark_v1"))

    if hasattr(wm_adapter, "build_messages"):
        m = wm_adapter.build_messages(batch_size=bs, device=device, batch=batch)
        if m is not None:
            return m.float().to(device)

    if deterministic_messages and ("img_path" in batch):
        paths = batch["img_path"]
        if isinstance(paths, (list, tuple)) and len(paths) == bs:
            return deterministic_messages_from_paths(
                paths=paths,
                message_len=message_len,
                device=device,
                salt=message_seed_salt,
            )
    return torch.randint(0, 2, (bs, message_len), device=device).float()


@torch.no_grad()
def evaluate(cfg, run_dir, ckpt_path, split, max_batches, batch_size, num_workers, seed, per_image=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vae = build_vae(cfg).to(device)
    state = torch.load(ckpt_path, map_location=device)
    vae.load_state_dict(state["vae"], strict=True)
    vae.eval()

    wm_adapter = build_wm_adapter(cfg.wm_model, cfg)
    message_len = int(wm_adapter.message_length)
    attack_names = list(getattr(cfg.attacks, "online", []))
    if not attack_names:
        raise RuntimeError("No online attacks configured.")

    attack_objects = {}
    for name in attack_names:
        if name not in ATTACK_REGISTRY:
            raise KeyError(f'Attack "{name}" not registered. Available: {list(ATTACK_REGISTRY.keys())}')
        attack_objects[name] = build_attack(name, cfg)

    loader = build_loader(cfg, split=split, batch_size=batch_size, num_workers=num_workers)

    rows = []
    per_image_rows = []
    for attack_name, attack in attack_objects.items():
        correct_raw = 0.0
        total_raw = 0.0
        sample_acc_sum_raw = 0.0
        sample_count_raw = 0.0

        correct_recon = 0.0
        total_recon = 0.0
        sample_acc_sum_recon = 0.0
        sample_count_recon = 0.0
        recon_l1_sum = 0.0
        recon_count = 0.0
        kl_sum = 0.0
        mu_abs_sum = 0.0
        logvar_mean_sum = 0.0
        latent_count = 0.0

        sample_index = 0
        for bidx, batch in enumerate(loader):
            if max_batches > 0 and bidx >= max_batches:
                break

            images = batch["image"].to(device, non_blocking=True)
            bs = images.shape[0]
            messages = _build_messages_for_batch(
                cfg=cfg,
                wm_adapter=wm_adapter,
                batch=batch,
                images=images,
                message_len=message_len,
                device=device,
            )
            if hasattr(wm_adapter, "encode_with_batch"):
                wm_images = wm_adapter.encode_with_batch(images, messages, batch=batch)
            else:
                wm_images = wm_adapter.encode(images, messages)
            if hasattr(attack, "attack_with_cover"):
                try:
                    attacked = attack.attack_with_cover(wm_images, images, batch=batch)
                except TypeError:
                    attacked = attack.attack_with_cover(wm_images, images)
            else:
                attacked = attack(wm_images)

            logits_raw = wm_adapter.decode(attacked)
            pred_raw = (torch.sigmoid(logits_raw) > 0.5).float()
            bit_eq_raw = (pred_raw == messages).float()
            sample_acc_raw = bit_eq_raw.mean(dim=1)
            correct_raw += float(bit_eq_raw.sum().item())
            total_raw += float(messages.numel())
            sample_acc_sum_raw += float(sample_acc_raw.sum().item())
            sample_count_raw += float(bs)

            recon, mu, logvar = vae(attacked)
            logits_recon = wm_adapter.decode(recon)
            pred_recon = (torch.sigmoid(logits_recon) > 0.5).float()
            bit_eq_recon = (pred_recon == messages).float()
            sample_acc_recon = bit_eq_recon.mean(dim=1)
            correct_recon += float(bit_eq_recon.sum().item())
            total_recon += float(messages.numel())
            sample_acc_sum_recon += float(sample_acc_recon.sum().item())
            sample_count_recon += float(bs)
            recon_l1_sum += float((recon - attacked).abs().mean().item()) * float(bs)
            recon_count += float(bs)

            # Raw latent KL to N(0,1), same formula as training kl loss.
            kl_val = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            kl_sum += float(kl_val.item()) * float(bs)
            mu_abs_sum += float(mu.abs().mean().item()) * float(bs)
            logvar_mean_sum += float(logvar.mean().item()) * float(bs)
            latent_count += float(bs)

            if per_image:
                img_paths = batch.get("img_path", [f"sample_{bidx}_{i}" for i in range(bs)])
                if not isinstance(img_paths, (list, tuple)):
                    img_paths = [str(img_paths)] * bs
                for i in range(bs):
                    row = {
                        "attack": attack_name,
                        "index": sample_index,
                        "img_path": str(img_paths[i]),
                        "raw_acc": float(sample_acc_raw[i].item()),
                        "recon_acc": float(sample_acc_recon[i].item()),
                        "ckpt": os.path.basename(ckpt_path),
                        "split": split,
                    }
                    per_image_rows.append(row)
                    print(
                        f"[{attack_name}] idx={sample_index:03d} "
                        f"raw_acc={row['raw_acc']:.6f} recon_acc={row['recon_acc']:.6f} "
                        f"img={row['img_path']}"
                    )
                    sample_index += 1

        bit_acc_raw = correct_raw / max(total_raw, 1.0)
        ber_raw = 1.0 - bit_acc_raw
        sample_acc_raw = sample_acc_sum_raw / max(sample_count_raw, 1.0)
        rows.append(
            {
                "attack": attack_name,
                "path": "raw_attacked",
                "bit_acc": bit_acc_raw,
                "ber": ber_raw,
                "sample_acc": sample_acc_raw,
                "num_samples": int(sample_count_raw),
                "num_bits": int(total_raw),
                "ckpt": os.path.basename(ckpt_path),
                "split": split,
                "recon_l1": None,
                "latent_kl": None,
                "mu_abs_mean": None,
                "logvar_mean": None,
            }
        )

        bit_acc_recon = correct_recon / max(total_recon, 1.0)
        ber_recon = 1.0 - bit_acc_recon
        sample_acc_recon = sample_acc_sum_recon / max(sample_count_recon, 1.0)
        recon_l1 = recon_l1_sum / max(recon_count, 1.0)
        latent_kl = kl_sum / max(latent_count, 1.0)
        mu_abs_mean = mu_abs_sum / max(latent_count, 1.0)
        logvar_mean = logvar_mean_sum / max(latent_count, 1.0)
        rows.append(
            {
                "attack": attack_name,
                "path": "vae_recon",
                "bit_acc": bit_acc_recon,
                "ber": ber_recon,
                "sample_acc": sample_acc_recon,
                "num_samples": int(sample_count_recon),
                "num_bits": int(total_recon),
                "ckpt": os.path.basename(ckpt_path),
                "split": split,
                "recon_l1": recon_l1,
                "latent_kl": latent_kl,
                "mu_abs_mean": mu_abs_mean,
                "logvar_mean": logvar_mean,
            }
        )

        print(
            f"[{attack_name}] raw_bit_acc={bit_acc_raw:.6f}  recon_bit_acc={bit_acc_recon:.6f}  "
            f"delta={bit_acc_recon - bit_acc_raw:+.6f}  "
            f"recon_l1={recon_l1:.6f}  latent_kl={latent_kl:.6f}  "
            f"mu_abs={mu_abs_mean:.6f}  logvar_mean={logvar_mean:.6f}"
        )

    out_csv = os.path.join(run_dir, f"attacked_recon_eval_{split}.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "attack",
                "path",
                "bit_acc",
                "ber",
                "sample_acc",
                "num_samples",
                "num_bits",
                "ckpt",
                "split",
                "recon_l1",
                "latent_kl",
                "mu_abs_mean",
                "logvar_mean",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {out_csv}")

    if per_image and per_image_rows:
        out_csv_detail = os.path.join(run_dir, f"attacked_recon_eval_{split}_per_image.csv")
        with open(out_csv_detail, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "attack",
                    "index",
                    "img_path",
                    "raw_acc",
                    "recon_acc",
                    "ckpt",
                    "split",
                ],
            )
            writer.writeheader()
            writer.writerows(per_image_rows)
        print(f"Saved: {out_csv_detail}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, help="Path to run dir containing config.yaml")
    p.add_argument("--checkpoint", default="best", help='best | last | /abs/path/to/ckpt.pth')
    p.add_argument("--split", choices=["train", "val"], default="val")
    p.add_argument("--max-batches", type=int, default=8, help="0 means full split")
    p.add_argument("--batch-size", type=int, default=0, help="0 means use config batch size")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--per-image", action="store_true", help="Save and print per-image acc rows")
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = os.path.abspath(args.run_dir)
    cfg = load_run_config(run_dir)
    ckpt_path = resolve_ckpt_path(run_dir, args.checkpoint)
    batch_size = args.batch_size if args.batch_size > 0 else int(cfg.training.batch_size)
    evaluate(
        cfg=cfg,
        run_dir=run_dir,
        ckpt_path=ckpt_path,
        split=args.split,
        max_batches=args.max_batches,
        batch_size=batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        per_image=args.per_image,
    )


if __name__ == "__main__":
    main()
