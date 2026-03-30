#!/usr/bin/env python3
"""
Evaluate false-positive behavior on SleeperMark no-trigger generations.

Pipeline:
  1) Use native SleeperMark generation with EMPTY trigger (prompt only).
  2) Decode raw no-trigger images against fixed secret bits.
  3) Reconstruct with trained Stage1 VAE, then decode again.
  4) Report ACC stats and false-positive hit rates.
"""

import argparse
import csv
import json
import os
import random
import sys
from types import SimpleNamespace

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.dataset import ReMark_Dataset
from network.vae import build_vae
from wm_adapters.registry import build_wm_adapter

import wm_adapters  # noqa: F401


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


def _sample_acc_and_hits(logits: torch.Tensor, messages: torch.Tensor, thresholds):
    pred = (torch.sigmoid(logits) > 0.5).float()
    bit_eq = (pred == messages).float()
    sample_acc = bit_eq.mean(dim=1)
    hits = {}
    for t in thresholds:
        hits[f"hit@{t}"] = float((sample_acc >= t).float().sum().item())
    return sample_acc, float(bit_eq.sum().item()), float(messages.numel()), hits


@torch.no_grad()
def evaluate(cfg, run_dir, ckpt_path, split, max_batches, batch_size, num_workers, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    thresholds = [0.75, 0.9, 0.95]

    vae = build_vae(cfg).to(device)
    state = torch.load(ckpt_path, map_location=device)
    vae.load_state_dict(state["vae"], strict=True)
    vae.eval()

    wm_adapter = build_wm_adapter(cfg.wm_model, cfg)
    encode_mode = str(getattr(getattr(cfg, "wm_adapter_sleepermark", None), "encode_mode", "residual")).lower()
    if cfg.wm_model.lower() != "sleepermark" or encode_mode != "native_trigger":
        raise RuntimeError("This evaluator expects wm_model=sleepermark with native_trigger mode.")

    # Force no-trigger generation.
    wm_adapter._trigger = ""

    loader = build_loader(cfg, split=split, batch_size=batch_size, num_workers=num_workers)

    total_samples = 0.0
    bit_correct_raw = 0.0
    bit_total_raw = 0.0
    bit_correct_recon = 0.0
    bit_total_recon = 0.0
    sample_acc_sum_raw = 0.0
    sample_acc_sum_recon = 0.0
    recon_l1_sum = 0.0
    hit_counts_raw = {f"hit@{t}": 0.0 for t in thresholds}
    hit_counts_recon = {f"hit@{t}": 0.0 for t in thresholds}

    per_sample_rows = []

    for bidx, batch in enumerate(loader):
        if max_batches > 0 and bidx >= max_batches:
            break

        images = batch["image"].to(device, non_blocking=True)
        bs = images.shape[0]
        messages = wm_adapter.build_messages(batch_size=bs, device=device, batch=batch)
        if messages is None:
            raise RuntimeError("SleeperMark native_trigger should provide fixed secret messages.")

        no_trigger_images = wm_adapter.encode_with_batch(images, messages, batch=batch)
        logits_raw = wm_adapter.decode(no_trigger_images)
        sample_acc_raw, corr_raw, total_raw, hits_raw = _sample_acc_and_hits(logits_raw, messages, thresholds)

        recon, _, _ = vae(no_trigger_images)
        logits_recon = wm_adapter.decode(recon)
        sample_acc_recon, corr_recon, total_recon, hits_recon = _sample_acc_and_hits(
            logits_recon, messages, thresholds
        )

        bit_correct_raw += corr_raw
        bit_total_raw += total_raw
        bit_correct_recon += corr_recon
        bit_total_recon += total_recon
        sample_acc_sum_raw += float(sample_acc_raw.sum().item())
        sample_acc_sum_recon += float(sample_acc_recon.sum().item())
        recon_l1_sum += float((recon - no_trigger_images).abs().mean().item()) * float(bs)
        total_samples += float(bs)
        for k in hit_counts_raw:
            hit_counts_raw[k] += hits_raw[k]
            hit_counts_recon[k] += hits_recon[k]

        img_paths = batch.get("img_path", [f"sample_{bidx}_{i}" for i in range(bs)])
        prompts = batch.get("prompt", [""] * bs)
        for i in range(bs):
            per_sample_rows.append(
                {
                    "img_path": str(img_paths[i]),
                    "prompt": str(prompts[i]),
                    "sample_acc_raw": float(sample_acc_raw[i].item()),
                    "sample_acc_recon": float(sample_acc_recon[i].item()),
                }
            )

    if total_samples <= 0:
        raise RuntimeError("No samples evaluated. Check dataset/split/max_batches settings.")

    summary = {
        "run_dir": os.path.abspath(run_dir),
        "checkpoint": os.path.abspath(ckpt_path),
        "split": split,
        "num_samples": int(total_samples),
        "message_length": int(wm_adapter.message_length),
        "raw_bit_acc": bit_correct_raw / max(bit_total_raw, 1.0),
        "raw_sample_acc_mean": sample_acc_sum_raw / max(total_samples, 1.0),
        "recon_bit_acc": bit_correct_recon / max(bit_total_recon, 1.0),
        "recon_sample_acc_mean": sample_acc_sum_recon / max(total_samples, 1.0),
        "delta_recon_minus_raw": (bit_correct_recon / max(bit_total_recon, 1.0))
        - (bit_correct_raw / max(bit_total_raw, 1.0)),
        "recon_l1_mean": recon_l1_sum / max(total_samples, 1.0),
        "raw_false_positive_rates": {k: hit_counts_raw[k] / max(total_samples, 1.0) for k in hit_counts_raw},
        "recon_false_positive_rates": {k: hit_counts_recon[k] / max(total_samples, 1.0) for k in hit_counts_recon},
    }
    return summary, per_sample_rows


def write_outputs(run_dir: str, split: str, summary: dict, rows: list):
    out_prefix = f"eval_no_trigger_false_positive_{split}"
    out_json = os.path.join(run_dir, f"{out_prefix}_summary.json")
    out_csv = os.path.join(run_dir, f"{out_prefix}_samples.csv")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["img_path", "prompt", "sample_acc_raw", "sample_acc_recon"])
        writer.writeheader()
        writer.writerows(rows)
    return out_json, out_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="best")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    ckpt_path = resolve_ckpt_path(run_dir, args.checkpoint)
    cfg = load_run_config(run_dir)

    summary, rows = evaluate(
        cfg=cfg,
        run_dir=run_dir,
        ckpt_path=ckpt_path,
        split=args.split,
        max_batches=args.max_batches,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    out_json, out_csv = write_outputs(run_dir, args.split, summary, rows)

    print("=== No-trigger false-positive evaluation ===")
    print(f"split={summary['split']}  samples={summary['num_samples']}  ckpt={summary['checkpoint']}")
    print(
        f"raw_bit_acc={summary['raw_bit_acc']:.6f}  recon_bit_acc={summary['recon_bit_acc']:.6f}  "
        f"delta={summary['delta_recon_minus_raw']:+.6f}"
    )
    print(
        f"raw_sample_acc_mean={summary['raw_sample_acc_mean']:.6f}  "
        f"recon_sample_acc_mean={summary['recon_sample_acc_mean']:.6f}  "
        f"recon_l1_mean={summary['recon_l1_mean']:.6f}"
    )
    print(f"raw_false_positive_rates={summary['raw_false_positive_rates']}")
    print(f"recon_false_positive_rates={summary['recon_false_positive_rates']}")
    print(f"summary_json={out_json}")
    print(f"samples_csv={out_csv}")


if __name__ == "__main__":
    main()

