#!/usr/bin/env python3
"""
Prepare ~10k manifests for FFHQ and FF++.

Outputs CSV files with `img_path` only, compatible with ReMark_Dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import pandas as pd


def list_images(root: Path) -> list[str]:
    exts = {".png", ".jpg", ".jpeg"}
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    files.sort()
    return [str(p.resolve()) for p in files]


def make_split(paths: list[str], n_total: int, n_val: int, seed: int):
    if len(paths) < n_total:
        raise ValueError(f"Need {n_total} images, found {len(paths)}.")
    rng = random.Random(seed)
    picked = paths[:]
    rng.shuffle(picked)
    picked = picked[:n_total]
    val = picked[:n_val]
    train = picked[n_val:]
    return train, val


def save_manifest(paths: list[str], out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"img_path": paths})
    df.to_csv(out_csv, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ffhq-root",
        type=Path,
        default=Path("/mnt/personal_workspace/chenkeyu/ReMark/Dataset-FFHQ/images"),
    )
    parser.add_argument(
        "--ffpp-roots",
        type=Path,
        nargs="+",
        default=[
            Path("/mnt/personal_workspace/chenkeyu/ReMark/Dataset-FF++/original_sequences/youtube/c23/frames_retina_512"),
            Path("/mnt/personal_workspace/chenkeyu/ReMark/Dataset-FF++/original_sequences/actors/c23/frames_retina_512"),
        ],
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/mnt/personal_workspace/chenkeyu/ReMark/Forensic-Remark_Module/data_manifests"),
    )
    parser.add_argument("--n-total", type=int, default=10000)
    parser.add_argument("--n-val", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--strict-ffpp",
        action="store_true",
        help="If set, exit with non-zero when FF++ is still below target.",
    )
    args = parser.parse_args()

    ffhq = list_images(args.ffhq_root)
    ffpp = []
    for r in args.ffpp_roots:
        ffpp.extend(list_images(r))
    ffpp = sorted(set(ffpp))

    print(f"[prepare] ffhq images: {len(ffhq)}")
    print(f"[prepare] ffpp images: {len(ffpp)}")

    ffhq_train, ffhq_val = make_split(ffhq, args.n_total, args.n_val, args.seed)

    save_manifest(ffhq_train, args.out_dir / "ffhq_10k_train.csv")
    save_manifest(ffhq_val, args.out_dir / "ffhq_10k_val.csv")
    print("[prepare] wrote:")
    print(f"  - {args.out_dir / 'ffhq_10k_train.csv'} ({len(ffhq_train)})")
    print(f"  - {args.out_dir / 'ffhq_10k_val.csv'} ({len(ffhq_val)})")

    if len(ffpp) < args.n_total:
        msg = f"[prepare] FF++ still below target: need {args.n_total}, found {len(ffpp)}."
        if args.strict_ffpp:
            raise ValueError(msg)
        print(msg)
        return

    ffpp_train, ffpp_val = make_split(ffpp, args.n_total, args.n_val, args.seed)
    save_manifest(ffpp_train, args.out_dir / "ffpp_10k_train.csv")
    save_manifest(ffpp_val, args.out_dir / "ffpp_10k_val.csv")
    print(f"  - {args.out_dir / 'ffpp_10k_train.csv'} ({len(ffpp_train)})")
    print(f"  - {args.out_dir / 'ffpp_10k_val.csv'} ({len(ffpp_val)})")


if __name__ == "__main__":
    main()
