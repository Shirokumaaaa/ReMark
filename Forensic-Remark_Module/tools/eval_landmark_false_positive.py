#!/usr/bin/env python3
"""
Evaluate landmark false-positive behavior without embedding any watermark.

For each clean or attacked image:
  1) do NOT watermark the image
  2) decode raw image directly
  3) reconstruct with Stage1 VAE, then decode again
  4) compare decoded bits against the image's own dlib68 landmark bits
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

REPO_ROOT = os.path.dirname(ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from common.landmark_bits import bits_from_image_paths, bits_from_image_tensors
from data.dataset import ReMark_Dataset
from network.vae import build_vae
from wm_adapters.registry import build_wm_adapter
from attacks.registry import build_attack, ATTACK_REGISTRY

import attacks  # noqa: F401
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


def build_loader(cfg, split: str, batch_size: int, num_workers: int, csv_path_override: str = ""):
    if csv_path_override:
        csv_path = os.path.abspath(csv_path_override)
    elif split == "val":
        csv_path = cfg.data.val_csv
    elif split == "train":
        csv_path = cfg.data.train_csv
    else:
        raise ValueError(f"Unsupported split without csv override: {split}")
    dataset = ReMark_Dataset(
        csv_path=csv_path,
        image_size=cfg.data.image_size,
        mode="val",
        use_wm_cache=False,
        center_crop=getattr(cfg.data, "center_crop", 0),
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


def _parse_attacks(arg: str, cfg):
    if arg.strip():
        names = [x.strip() for x in arg.split(",") if x.strip()]
    else:
        names = list(getattr(getattr(cfg, "validation", None), "attacks", []) or [])
        if not names:
            names = ["clean"] + list(getattr(cfg.attacks, "online", []) or [])
    dedup = []
    seen = set()
    for name in names:
        if name not in seen:
            dedup.append(name)
            seen.add(name)
    return dedup


@torch.no_grad()
def evaluate(
    cfg,
    run_dir,
    ckpt_path,
    split,
    max_batches,
    batch_size,
    num_workers,
    seed,
    thresholds,
    attack_names,
    csv_path_override="",
    verbose_every_batch=0,
    per_sample_csv_path="",
):
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
    landmark_predictor_path = str(getattr(getattr(cfg, "training", None), "landmark_predictor_path", ""))
    landmark_bits_cache_dir = str(getattr(getattr(cfg, "training", None), "landmark_bits_cache_dir", ""))
    landmark_canonical_bits = int(getattr(getattr(cfg, "training", None), "landmark_canonical_bits", 128))
    landmark_bits_per_value = int(getattr(getattr(cfg, "training", None), "landmark_bits_per_value", 4))

    attacks_map = {}
    for name in attack_names:
        if name == "clean":
            attacks_map[name] = None
            continue
        if name not in ATTACK_REGISTRY:
            raise KeyError(f'Attack "{name}" not registered. Available: {list(ATTACK_REGISTRY.keys())}')
        attacks_map[name] = build_attack(name, cfg)

    loader = build_loader(
        cfg,
        split=split,
        batch_size=batch_size,
        num_workers=num_workers,
        csv_path_override=csv_path_override,
    )

    rows = []
    per_sample_rows = []
    summary = {
        "run_dir": os.path.abspath(run_dir),
        "checkpoint": os.path.abspath(ckpt_path),
        "split": split,
        "message_length": message_len,
        "thresholds": thresholds,
        "attacks": {},
    }

    for attack_name, attack in attacks_map.items():
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

        for bidx, batch in enumerate(loader):
            if max_batches > 0 and bidx >= max_batches:
                break

            images = batch["image"].to(device, non_blocking=True)
            bs = images.shape[0]

            if attack is None:
                attacked = images
            elif hasattr(attack, "attack_with_cover"):
                try:
                    attacked = attack.attack_with_cover(images, images, batch=batch)
                except TypeError:
                    attacked = attack.attack_with_cover(images, images)
            else:
                attacked = attack(images)

            # False-positive GT should stay aligned with the original clean image.
            # We evaluate whether a non-embedded sample is falsely decoded as the
            # source image's landmark message after attack / VAE reconstruction.
            img_paths = batch.get("img_path")
            if img_paths is not None:
                target_bits = bits_from_image_paths(
                    image_paths=img_paths,
                    num_bits=message_len,
                    device=device,
                    predictor_path=landmark_predictor_path or None,
                    cache_dir=landmark_bits_cache_dir,
                    canonical_bits=landmark_canonical_bits,
                    bits_per_value=landmark_bits_per_value,
                )
            else:
                target_bits = bits_from_image_tensors(
                    images=images,
                    num_bits=message_len,
                    device=device,
                    predictor_path=landmark_predictor_path or None,
                    cache_dir=landmark_bits_cache_dir,
                    canonical_bits=landmark_canonical_bits,
                    bits_per_value=landmark_bits_per_value,
                )

            logits_raw = wm_adapter.decode(attacked)
            sample_acc_raw, corr_raw, total_raw, hits_raw = _sample_acc_and_hits(logits_raw, target_bits, thresholds)

            recon, _, _ = vae(attacked)
            logits_recon = wm_adapter.decode(recon)
            sample_acc_recon, corr_recon, total_recon, hits_recon = _sample_acc_and_hits(
                logits_recon, target_bits, thresholds
            )

            bit_correct_raw += corr_raw
            bit_total_raw += total_raw
            bit_correct_recon += corr_recon
            bit_total_recon += total_recon
            sample_acc_sum_raw += float(sample_acc_raw.sum().item())
            sample_acc_sum_recon += float(sample_acc_recon.sum().item())
            recon_l1_sum += float((recon - attacked).abs().mean().item()) * float(bs)
            total_samples += float(bs)
            for k in hit_counts_raw:
                hit_counts_raw[k] += hits_raw[k]
                hit_counts_recon[k] += hits_recon[k]

            img_paths = list(batch.get("img_path", [""] * bs))
            for i in range(bs):
                row = {
                    "attack": attack_name,
                    "index": int(total_samples - bs + i),
                    "img_path": str(img_paths[i]),
                    "raw_acc": float(sample_acc_raw[i].item()),
                    "recon_acc": float(sample_acc_recon[i].item()),
                }
                for t in thresholds:
                    row[f"raw_hit@{t}"] = float(sample_acc_raw[i].item() >= t)
                    row[f"recon_hit@{t}"] = float(sample_acc_recon[i].item() >= t)
                per_sample_rows.append(row)

            if verbose_every_batch > 0 and ((bidx + 1) % verbose_every_batch == 0):
                start_idx = int(total_samples - bs)
                end_idx = int(total_samples) - 1
                print(
                    f"[progress] attack={attack_name} batch={bidx + 1} "
                    f"samples={int(total_samples)} range={start_idx}-{end_idx} "
                    f"raw_acc_mean={float(sample_acc_raw.mean().item()):.6f} "
                    f"recon_acc_mean={float(sample_acc_recon.mean().item()):.6f}",
                    flush=True,
                )
                for i in range(bs):
                    print(
                        f"[sample] attack={attack_name} idx={start_idx + i} "
                        f"img={img_paths[i]} raw_acc={float(sample_acc_raw[i].item()):.6f} "
                        f"recon_acc={float(sample_acc_recon[i].item()):.6f}",
                        flush=True,
                    )

        if total_samples <= 0:
            raise RuntimeError(f"No samples evaluated for attack={attack_name}")

        attack_summary = {
            "num_samples": int(total_samples),
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
        summary["attacks"][attack_name] = attack_summary

        rows.append(
            {
                "attack": attack_name,
                "path": "raw",
                "bit_acc": attack_summary["raw_bit_acc"],
                "sample_acc_mean": attack_summary["raw_sample_acc_mean"],
                "recon_l1_mean": "",
                **attack_summary["raw_false_positive_rates"],
            }
        )
        rows.append(
            {
                "attack": attack_name,
                "path": "vae_recon",
                "bit_acc": attack_summary["recon_bit_acc"],
                "sample_acc_mean": attack_summary["recon_sample_acc_mean"],
                "recon_l1_mean": attack_summary["recon_l1_mean"],
                **attack_summary["recon_false_positive_rates"],
            }
        )

    if per_sample_csv_path:
        fieldnames = sorted({k for row in per_sample_rows for k in row.keys()}) if per_sample_rows else [
            "attack", "index", "img_path", "raw_acc", "recon_acc"
        ]
        with open(per_sample_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_sample_rows)

    return summary, rows


def write_outputs(run_dir: str, split: str, summary: dict, rows: list):
    out_prefix = f"eval_landmark_false_positive_{split}"
    out_json = os.path.join(run_dir, f"{out_prefix}_summary.json")
    out_csv = os.path.join(run_dir, f"{out_prefix}_summary.csv")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out_json, out_csv


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--checkpoint", default="best")
    p.add_argument("--split", default="val")
    p.add_argument("--csv-path", default="", help="Optional manifest csv to evaluate instead of cfg.data.(train|val)_csv")
    p.add_argument("--max-batches", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--attacks", type=str, default="", help="Comma separated, e.g. clean,simswap")
    p.add_argument("--thresholds", type=str, default="0.55,0.6,0.65,0.7")
    p.add_argument("--verbose-every-batch", type=int, default=0)
    p.add_argument("--per-sample-csv", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = os.path.abspath(args.run_dir)
    cfg = load_run_config(run_dir)
    ckpt_path = resolve_ckpt_path(run_dir, args.checkpoint)
    batch_size = args.batch_size if args.batch_size > 0 else int(cfg.training.batch_size)
    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    attack_names = _parse_attacks(args.attacks, cfg)

    summary, rows = evaluate(
        cfg=cfg,
        run_dir=run_dir,
        ckpt_path=ckpt_path,
        split=args.split,
        max_batches=args.max_batches,
        batch_size=batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        thresholds=thresholds,
        attack_names=attack_names,
        csv_path_override=args.csv_path,
        verbose_every_batch=args.verbose_every_batch,
        per_sample_csv_path=args.per_sample_csv,
    )
    out_json, out_csv = write_outputs(run_dir, args.split, summary, rows)

    print("=== Landmark false-positive evaluation ===")
    print(f"run_dir={run_dir}")
    print(f"checkpoint={ckpt_path}")
    print(f"split={args.split} attacks={attack_names}")
    print(f"summary_json={out_json}")
    print(f"summary_csv={out_csv}")
    for attack_name, attack_summary in summary["attacks"].items():
        print(
            f"[{attack_name}] raw_bit_acc={attack_summary['raw_bit_acc']:.6f} "
            f"recon_bit_acc={attack_summary['recon_bit_acc']:.6f} "
            f"delta={attack_summary['delta_recon_minus_raw']:+.6f}"
        )
        print(f"  raw_fpr={attack_summary['raw_false_positive_rates']}")
        print(f"  recon_fpr={attack_summary['recon_false_positive_rates']}")


if __name__ == "__main__":
    main()
