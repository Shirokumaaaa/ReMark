#!/usr/bin/env python3
import argparse
import csv
import os
import sys
from typing import List

from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from common.landmark_bits import build_landmark_bit_encoder


def collect_paths(args) -> List[str]:
    paths: List[str] = []
    if args.csv:
        with open(args.csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_path = (row.get("img_path") or row.get("image") or row.get("path") or "").strip()
                if image_path:
                    paths.append(os.path.abspath(image_path))
    elif args.root:
        for root, _, files in os.walk(args.root):
            for name in sorted(files):
                if name.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
                    paths.append(os.path.abspath(os.path.join(root, name)))
    else:
        raise ValueError("Either --csv or --root is required.")
    return sorted(set(paths))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="")
    parser.add_argument("--root", type=str, default="")
    parser.add_argument("--cache-dir", type=str, required=True)
    parser.add_argument("--failure-csv", type=str, default="")
    parser.add_argument("--predictor-path", type=str, default="")
    parser.add_argument("--canonical-bits", type=int, default=128)
    parser.add_argument("--bits-per-value", type=int, default=4)
    parser.add_argument("--allow-failures", action="store_true")
    args = parser.parse_args()

    paths = collect_paths(args)
    encoder = build_landmark_bit_encoder(
        predictor_path=args.predictor_path or None,
        cache_dir=args.cache_dir,
        canonical_bits=args.canonical_bits,
        bits_per_value=args.bits_per_value,
    )

    num_ok = 0
    failures = []
    for path in tqdm(paths, desc="Build landmark cache"):
        try:
            encoder.canonical_bits_from_path(path)
            num_ok += 1
        except Exception as e:
            failures.append((path, str(e)))

    print(f"cache_dir={os.path.abspath(args.cache_dir)}")
    print(f"processed={len(paths)} ok={num_ok} failed={len(failures)}")
    if failures:
        err_path = args.failure_csv or os.path.join(args.cache_dir, "landmark_cache_failures.csv")
        os.makedirs(args.cache_dir, exist_ok=True)
        with open(err_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["img_path", "error"])
            writer.writerows(failures)
        print(f"failure_csv={err_path}")
        if not args.allow_failures:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
