#!/usr/bin/env python3

import argparse
import json
import math
import os
import random
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torchvision.utils import make_grid, save_image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.dataset import ReMark_Dataset
from wm_adapters.registry import build_wm_adapter

import wm_adapters  # noqa: F401


def dict_to_ns(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_ns(v) for k, v in d.items()})
    if isinstance(d, list):
        return [dict_to_ns(v) for v in d]
    return d


def load_cfg(path):
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return dict_to_ns(yaml.safe_load(f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cfg = load_cfg(args.config)
    csv_path = cfg.data.val_csv if args.split == "val" else cfg.data.train_csv
    dataset = ReMark_Dataset(csv_path=csv_path, image_size=cfg.data.image_size, mode="val", use_wm_cache=False)
    count = min(args.max_samples, len(dataset))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    adapter = build_wm_adapter(cfg.wm_model, cfg)
    os.makedirs(args.out_dir, exist_ok=True)

    images = []
    prompts = []
    sample_rows = []

    for idx in range(count):
        sample = dataset[idx]
        batch = {
            "image": sample["image"].unsqueeze(0),
            "img_path": [sample["img_path"]],
            "prompt": [sample["prompt"]],
        }
        prompt = str(sample["prompt"])
        msg = torch.randint(0, 2, (1, adapter.message_length), device=device).float()
        wm = adapter.encode_with_batch(batch["image"].to(device), msg, batch=batch)
        logits = adapter.decode(wm)
        pred = (torch.sigmoid(logits) > 0.5).float()
        acc = float((pred == msg).float().mean().item())

        images.append(wm.detach().cpu())
        prompts.append(prompt)
        sample_rows.append(
            {
                "index": idx,
                "img_path": sample["img_path"],
                "prompt": prompt,
                "acc": acc,
            }
        )

    grid = make_grid(torch.cat(images, dim=0), nrow=max(1, min(4, count)), normalize=True, value_range=(-1, 1))
    save_image(grid, os.path.join(args.out_dir, "generated_grid.png"))

    with open(os.path.join(args.out_dir, "prompts.txt"), "w", encoding="utf-8") as f:
        for i, prompt in enumerate(prompts):
            f.write(f"[{i}] {prompt}\n")

    summary = {
        "config": os.path.abspath(args.config),
        "split": args.split,
        "num_samples": count,
        "mean_acc": float(np.mean([r["acc"] for r in sample_rows])) if sample_rows else 0.0,
        "min_acc": float(np.min([r["acc"] for r in sample_rows])) if sample_rows else 0.0,
        "max_acc": float(np.max([r["acc"] for r in sample_rows])) if sample_rows else 0.0,
        "samples": sample_rows,
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
