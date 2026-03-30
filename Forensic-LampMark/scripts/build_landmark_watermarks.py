#!/usr/bin/env python3
import argparse
import csv
import os
import sys
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LAMPMARK_ROOT = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(LAMPMARK_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from common.landmark_bits import build_landmark_bit_encoder


def load_manifest(manifest_path: str) -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = (row.get("img_path") or row.get("img") or row.get("image") or "").strip()
            if not img_path:
                continue
            items.append((os.path.abspath(img_path), os.path.basename(img_path)))
    return items


def write_manifest(out_manifest: str, rows: List[Tuple[str, str]]):
    os.makedirs(os.path.dirname(out_manifest), exist_ok=True)
    with open(out_manifest, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["img_path", "wm_path"])
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-pattern", type=str, required=True, help="e.g. data_manifests/celeba_hq_128_{split}.csv")
    parser.add_argument("--splits", type=str, default="train,val,test")
    parser.add_argument("--out-root", type=str, required=True)
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--num-bits", type=int, default=64)
    parser.add_argument("--predictor-path", type=str, default="")
    parser.add_argument("--cache-dir", type=str, default="")
    parser.add_argument("--canonical-bits", type=int, default=128)
    parser.add_argument("--bits-per-value", type=int, default=4)
    args = parser.parse_args()

    encoder = build_landmark_bit_encoder(
        predictor_path=args.predictor_path or None,
        cache_dir=args.cache_dir,
        canonical_bits=args.canonical_bits,
        bits_per_value=args.bits_per_value,
    )

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    for split in splits:
        manifest_path = args.manifest_pattern.format(split=split)
        if not os.path.isabs(manifest_path):
            manifest_path = os.path.join(LAMPMARK_ROOT, manifest_path)
        items = load_manifest(manifest_path)
        out_dir = os.path.join(args.out_root, str(args.img_size), split)
        os.makedirs(out_dir, exist_ok=True)

        manifest_rows: List[Tuple[str, str]] = []
        failures: List[Tuple[str, str]] = []
        for img_path, base_name in tqdm(items, desc=f"LampMark {split}"):
            try:
                bits = encoder.bits_from_path(img_path, num_bits=args.num_bits)
                wm_name = os.path.splitext(base_name)[0] + ".npy"
                wm_path = os.path.join(out_dir, wm_name)
                np.save(wm_path, bits.astype(np.float32))
                manifest_rows.append((img_path, wm_path))
            except Exception as exc:
                failures.append((img_path, str(exc)))

        out_manifest = os.path.join(
            LAMPMARK_ROOT,
            "data_manifests",
            f"dlib68_landmark_{args.img_size}_{split}.csv",
        )
        write_manifest(out_manifest, manifest_rows)
        if failures:
            failure_csv = os.path.join(
                LAMPMARK_ROOT,
                "data_manifests",
                f"dlib68_landmark_{args.img_size}_{split}_failures.csv",
            )
            with open(failure_csv, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["img_path", "error"])
                writer.writerows(failures)
            print(f"split={split} failures={len(failures)} failure_csv={failure_csv}")
        print(f"split={split} manifest={out_manifest} out_dir={out_dir}")


if __name__ == "__main__":
    main()
